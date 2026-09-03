#include <core.p4>
#include <v1model.p4>
#include "../common/statedt_headers.p4"
#include "../common/statedt_model.p4inc"
#include "../common/statedt_layout.p4inc"
#include "../common/statedt_entry_type.p4inc"

const bit<32> FLOW_BANK_SIZE = 32w32768;

struct metadata_t {
    bit<32> low_addr;
    bit<32> high_addr;
    bit<16> low_port;
    bit<16> high_port;
    bit<1> packet_low_to_high;
    bit<32> index0;
    bit<32> index1;
    bit<16> fingerprint;
    bit<16> packet_length;
    bit<1> forward;
    bit<1> state_valid;
    bit<2> state_status;
    bit<8> class_id;
    statedt_entry_t state;
    flow_features_t features;
}

parser StateDTParser(
        packet_in packet,
        out headers_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) {
    state start {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.ihl, hdr.ipv4.fragment_offset, hdr.ipv4.protocol) {
            (4w5, 13w0, IP_PROTOCOL_TCP): parse_tcp;
            (4w5, 13w0, IP_PROTOCOL_UDP): parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }
}

control StateDTVerifyChecksum(
        inout headers_t hdr,
        inout metadata_t meta) {
    apply { }
}

control StateDTIngress(
        inout headers_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) {
    register<statedt_entry_t>(FLOW_BANK_SIZE) flow_bank0;
    register<statedt_entry_t>(FLOW_BANK_SIZE) flow_bank1;
    counter(1, CounterType.packets) statedt_allocations;
    counter(1, CounterType.packets) statedt_fingerprint_mismatches;
    counter(1, CounterType.packets) statedt_collisions;
    counter(1, CounterType.packets) statedt_fallbacks;

    action set_class(bit<8> class_id) {
        meta.class_id = class_id;
    }

    table classify {
        key = {
            meta.features.packet_length_max : range;
            meta.features.psh_flag_count : range;
            meta.features.total_fwd_length : range;
            meta.features.fin_flag_count : range;
            meta.features.fwd_packet_length_max : range;
            meta.features.total_bwd_packets : range;
        }
        actions = {
            set_class;
        }
        size = STATEDT_RULE_COUNT;
        default_action = set_class(CLASS_BENIGN);
        const entries = {
#include "../common/statedt_entries.p4inc"
        }
    }

    action canonicalize() {
        if (hdr.ipv4.src_addr < hdr.ipv4.dst_addr ||
            (hdr.ipv4.src_addr == hdr.ipv4.dst_addr &&
             (hdr.tcp.isValid() && hdr.tcp.src_port <= hdr.tcp.dst_port ||
              hdr.udp.isValid() && hdr.udp.src_port <= hdr.udp.dst_port))) {
            meta.low_addr = hdr.ipv4.src_addr;
            meta.high_addr = hdr.ipv4.dst_addr;
            meta.packet_low_to_high = 1;
            if (hdr.tcp.isValid()) {
                meta.low_port = hdr.tcp.src_port;
                meta.high_port = hdr.tcp.dst_port;
            } else {
                meta.low_port = hdr.udp.src_port;
                meta.high_port = hdr.udp.dst_port;
            }
        } else {
            meta.low_addr = hdr.ipv4.dst_addr;
            meta.high_addr = hdr.ipv4.src_addr;
            meta.packet_low_to_high = 0;
            if (hdr.tcp.isValid()) {
                meta.low_port = hdr.tcp.dst_port;
                meta.high_port = hdr.tcp.src_port;
            } else {
                meta.low_port = hdr.udp.dst_port;
                meta.high_port = hdr.udp.src_port;
            }
        }
    }

    action update_state() {
        bit<16> incoming_region;
        bit<17> total;

        if (hdr.tcp.isValid()) {
            meta.packet_length = hdr.ipv4.total_len - ((bit<16>) hdr.ipv4.ihl << 2) -
                ((bit<16>) hdr.tcp.data_offset << 2);
        } else {
            meta.packet_length = hdr.udp.length - 8;
        }

#include "../common/statedt_unpack_features.p4inc"

        // Maximum features store threshold-region identifiers, not concrete
        // packet lengths. The model-specific encoder is compiler generated.
#include "../common/statedt_update_packet_max.p4inc"
        if (incoming_region > meta.features.packet_length_max) {
            meta.features.packet_length_max = incoming_region;
        }

        if (hdr.tcp.isValid() && hdr.tcp.psh == 1 &&
            meta.features.psh_flag_count < CAP_PSH_FLAG_COUNT) {
            meta.features.psh_flag_count = meta.features.psh_flag_count + 1;
        }
        if (hdr.tcp.isValid() && hdr.tcp.fin == 1 &&
            meta.features.fin_flag_count < CAP_FIN_FLAG_COUNT) {
            meta.features.fin_flag_count = meta.features.fin_flag_count + 1;
        }

        if (meta.forward == 1) {
            total = (bit<17>) meta.features.total_fwd_length + (bit<17>) meta.packet_length;
            if (total > (bit<17>) CAP_TOTAL_FWD_LENGTH) {
                meta.features.total_fwd_length = CAP_TOTAL_FWD_LENGTH;
            } else {
                meta.features.total_fwd_length = (bit<16>) total;
            }
            incoming_region = 0;
#include "../common/statedt_update_fwd_packet_max.p4inc"
            if (incoming_region > meta.features.fwd_packet_length_max) {
                meta.features.fwd_packet_length_max = incoming_region;
            }
        } else if (meta.features.total_bwd_packets < CAP_TOTAL_BWD_PACKETS) {
            meta.features.total_bwd_packets = meta.features.total_bwd_packets + 1;
        }

#include "../common/statedt_pack_features.p4inc"
    }

    action try_bank0() {
        flow_bank0.read(meta.state, meta.index0);
        if (meta.state[STATEDT_VALID_BIT:STATEDT_VALID_BIT] == 0) {
            meta.state = 0;
            meta.state[STATEDT_VALID_BIT:STATEDT_VALID_BIT] = 1;
            meta.state[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] = meta.packet_low_to_high;
            meta.state[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] = meta.fingerprint;
            meta.forward = 1;
            meta.state_valid = 1;
            meta.state_status = STATEDT_STATUS_ALLOCATED;
            statedt_allocations.count(0);
            update_state();
            flow_bank0.write(meta.index0, meta.state);
        } else if (meta.state[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] == meta.fingerprint) {
            meta.forward = (bit<1>)(meta.state[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] == meta.packet_low_to_high);
            meta.state_valid = 1;
            meta.state_status = STATEDT_STATUS_MATCH;
            update_state();
            flow_bank0.write(meta.index0, meta.state);
        } else {
            statedt_fingerprint_mismatches.count(0);
        }
    }

    action try_bank1() {
        flow_bank1.read(meta.state, meta.index1);
        if (meta.state[STATEDT_VALID_BIT:STATEDT_VALID_BIT] == 0) {
            meta.state = 0;
            meta.state[STATEDT_VALID_BIT:STATEDT_VALID_BIT] = 1;
            meta.state[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] = meta.packet_low_to_high;
            meta.state[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] = meta.fingerprint;
            meta.forward = 1;
            meta.state_valid = 1;
            meta.state_status = STATEDT_STATUS_ALLOCATED;
            statedt_allocations.count(0);
            update_state();
            flow_bank1.write(meta.index1, meta.state);
        } else if (meta.state[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] == meta.fingerprint) {
            meta.forward = (bit<1>)(meta.state[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] == meta.packet_low_to_high);
            meta.state_valid = 1;
            meta.state_status = STATEDT_STATUS_MATCH;
            update_state();
            flow_bank1.write(meta.index1, meta.state);
        } else {
            meta.state_status = STATEDT_STATUS_FALLBACK_COLLISION;
            statedt_fingerprint_mismatches.count(0);
            statedt_collisions.count(0);
            statedt_fallbacks.count(0);
        }
    }

    apply {
        standard_metadata.egress_spec = standard_metadata.ingress_port;
        meta.class_id = CLASS_BENIGN;
        meta.state_valid = 0;
        meta.state_status = STATEDT_STATUS_NOT_PROCESSED;

        if (hdr.ipv4.isValid() && (hdr.tcp.isValid() || hdr.udp.isValid())) {
            canonicalize();
            hash(meta.index0, HashAlgorithm.crc16, 32w0,
                 { meta.low_addr, meta.high_addr, hdr.ipv4.protocol,
                   meta.low_port, meta.high_port, 8w0x31 }, 16w32768);
            hash(meta.index1, HashAlgorithm.crc16, 32w0,
                 { meta.low_addr, meta.high_addr, hdr.ipv4.protocol,
                   meta.low_port, meta.high_port, 8w0xa7 }, 16w32768);
            hash(meta.fingerprint, HashAlgorithm.crc16, 16w0,
                 { meta.low_addr, meta.high_addr, hdr.ipv4.protocol,
                   meta.low_port, meta.high_port }, 16w65535);

            try_bank0();
            if (meta.state_valid == 0) {
                try_bank1();
            }
            if (meta.state_valid == 1) {
                classify.apply();
            }

            hdr.result.setValid();
            hdr.result.original_ether_type = ETHERTYPE_IPV4;
            hdr.result.class_id = meta.class_id;
            hdr.result.state_valid = meta.state_valid;
            hdr.result.state_status = meta.state_status;
            hdr.result.reserved = 0;
            hdr.ethernet.ether_type = ETHERTYPE_STATEDT;
        }
    }
}

control StateDTEgress(
        inout headers_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) {
    apply { }
}

control StateDTComputeChecksum(
        inout headers_t hdr,
        inout metadata_t meta) {
    apply { }
}

control StateDTDeparser(packet_out packet, in headers_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.result);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

V1Switch(
    StateDTParser(),
    StateDTVerifyChecksum(),
    StateDTIngress(),
    StateDTEgress(),
    StateDTComputeChecksum(),
    StateDTDeparser()
) main;

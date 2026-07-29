#include <core.p4>
#include <v1model.p4>
#include "../common/statedt_headers.p4"
#include "../common/statedt_model.p4inc"

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
    bit<1> claimed;
    bit<1> forward;
    bit<1> state_valid;
    bit<8> class_id;
    bit<114> state;
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
    register<bit<114>>(FLOW_BANK_SIZE) flow_bank0;
    register<bit<114>>(FLOW_BANK_SIZE) flow_bank1;

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
        bit<16> packet_length;
        bit<17> total;

        if (hdr.tcp.isValid()) {
            packet_length = hdr.ipv4.total_len - ((bit<16>) hdr.ipv4.ihl << 2) -
                ((bit<16>) hdr.tcp.data_offset << 2);
        } else {
            packet_length = hdr.udp.length - 8;
        }

        meta.features.packet_length_max = meta.state[95:80];
        meta.features.psh_flag_count = meta.state[79:64];
        meta.features.total_fwd_length = meta.state[63:48];
        meta.features.fin_flag_count = meta.state[47:32];
        meta.features.fwd_packet_length_max = meta.state[31:16];
        meta.features.total_bwd_packets = meta.state[15:0];

        if (packet_length > meta.features.packet_length_max) {
            meta.features.packet_length_max = packet_length;
        }
        if (meta.features.packet_length_max > CAP_PACKET_LENGTH_MAX) {
            meta.features.packet_length_max = CAP_PACKET_LENGTH_MAX;
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
            total = (bit<17>) meta.features.total_fwd_length + (bit<17>) packet_length;
            if (total > (bit<17>) CAP_TOTAL_FWD_LENGTH) {
                meta.features.total_fwd_length = CAP_TOTAL_FWD_LENGTH;
            } else {
                meta.features.total_fwd_length = (bit<16>) total;
            }
            if (packet_length > meta.features.fwd_packet_length_max) {
                meta.features.fwd_packet_length_max = packet_length;
            }
            if (meta.features.fwd_packet_length_max > CAP_FWD_PACKET_LENGTH_MAX) {
                meta.features.fwd_packet_length_max = CAP_FWD_PACKET_LENGTH_MAX;
            }
        } else if (meta.features.total_bwd_packets < CAP_TOTAL_BWD_PACKETS) {
            meta.features.total_bwd_packets = meta.features.total_bwd_packets + 1;
        }

        meta.state[95:80] = meta.features.packet_length_max;
        meta.state[79:64] = meta.features.psh_flag_count;
        meta.state[63:48] = meta.features.total_fwd_length;
        meta.state[47:32] = meta.features.fin_flag_count;
        meta.state[31:16] = meta.features.fwd_packet_length_max;
        meta.state[15:0] = meta.features.total_bwd_packets;
    }

    action try_bank0() {
        flow_bank0.read(meta.state, meta.index0);
        if (meta.state[113:113] == 0) {
            meta.state = 0;
            meta.state[113:113] = 1;
            meta.state[112:112] = meta.packet_low_to_high;
            meta.state[111:96] = meta.fingerprint;
            meta.forward = 1;
            meta.claimed = 1;
            update_state();
            flow_bank0.write(meta.index0, meta.state);
        } else if (meta.state[111:96] == meta.fingerprint) {
            meta.forward = (bit<1>)(meta.state[112:112] == meta.packet_low_to_high);
            meta.claimed = 1;
            update_state();
            flow_bank0.write(meta.index0, meta.state);
        }
    }

    action try_bank1() {
        flow_bank1.read(meta.state, meta.index1);
        if (meta.state[113:113] == 0) {
            meta.state = 0;
            meta.state[113:113] = 1;
            meta.state[112:112] = meta.packet_low_to_high;
            meta.state[111:96] = meta.fingerprint;
            meta.forward = 1;
            meta.claimed = 1;
            update_state();
            flow_bank1.write(meta.index1, meta.state);
        } else if (meta.state[111:96] == meta.fingerprint) {
            meta.forward = (bit<1>)(meta.state[112:112] == meta.packet_low_to_high);
            meta.claimed = 1;
            update_state();
            flow_bank1.write(meta.index1, meta.state);
        }
    }

    apply {
        standard_metadata.egress_spec = standard_metadata.ingress_port;
        meta.class_id = CLASS_BENIGN;
        meta.state_valid = 0;
        meta.claimed = 0;

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
            if (meta.claimed == 0) {
                try_bank1();
            }
            if (meta.claimed == 1) {
                meta.state_valid = 1;
                classify.apply();
            }

            hdr.result.setValid();
            hdr.result.original_ether_type = ETHERTYPE_IPV4;
            hdr.result.class_id = meta.class_id;
            hdr.result.state_valid = meta.state_valid;
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

#include <core.p4>
#include <tna.p4>
#include "../common/statedt_headers.p4"
#include "../common/statedt_model.p4inc"
#include "../common/statedt_layout.p4inc"
#include "../common/statedt_entry_type.p4inc"

@pragma pa_auto_init_metadata

const bit<32> FLOW_BANK_SIZE = 32w32768;

struct metadata_t {
    bit<32> low_addr;
    bit<32> high_addr;
    bit<16> low_port;
    bit<16> high_port;
    bit<1> packet_low_to_high;
    bit<15> index0;
    bit<15> index1;
    bit<16> fingerprint;
    bit<1> forward;
    bit<1> state_valid;
    bit<2> state_status;
    bit<1> packet_psh;
    bit<1> packet_fin;
    bit<16> packet_length;
    bit<8> class_id;
    bit<2> mismatch_increment;
    bit<32> counter_value;
    flow_features_t features;
}

struct empty_headers_t { }
struct empty_metadata_t { }

parser SwitchIngressParser(
        packet_in pkt,
        out headers_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag) {
            0: parse_port_metadata;
            default: reject;
        }
    }
    state parse_port_metadata {
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.ihl, hdr.ipv4.fragment_offset, hdr.ipv4.protocol) {
            (4w5, 13w0, IP_PROTOCOL_TCP): parse_tcp;
            (4w5, 13w0, IP_PROTOCOL_UDP): parse_udp;
            default: accept;
        }
    }
    state parse_tcp { pkt.extract(hdr.tcp); transition accept; }
    state parse_udp { pkt.extract(hdr.udp); transition accept; }
}

control SwitchIngress(
        inout headers_t hdr,
        inout metadata_t meta,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {
    Hash<bit<15>>(HashAlgorithm_t.CRC16) index_hash0;
    Hash<bit<15>>(HashAlgorithm_t.CRC16) index_hash1;
    Hash<bit<16>>(HashAlgorithm_t.CRC16) fingerprint_hash;

    // One canonical packed entry per choice makes ownership validation and
    // update one atomic RegisterAction operation.
    Register<statedt_entry_t, bit<15>>(FLOW_BANK_SIZE, 0) flow_bank0;
    Register<statedt_entry_t, bit<15>>(FLOW_BANK_SIZE, 0) flow_bank1;

    Register<bit<32>, bit<1>>(1, 32w0) statedt_allocations;
    Register<bit<32>, bit<1>>(1, 32w0) statedt_fingerprint_mismatches;
    Register<bit<32>, bit<1>>(1, 32w0) statedt_collisions;
    Register<bit<32>, bit<1>>(1, 32w0) statedt_fallbacks;

    RegisterAction<bit<32>, bit<1>, bit<32>>(statedt_allocations) count_allocation = {
        void apply(inout bit<32> value, out bit<32> result) {
            value = value + 1; result = value;
        }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(statedt_fingerprint_mismatches) count_mismatch = {
        void apply(inout bit<32> value, out bit<32> result) {
            value = value + (bit<32>) meta.mismatch_increment; result = value;
        }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(statedt_collisions) count_collision = {
        void apply(inout bit<32> value, out bit<32> result) {
            value = value + 1; result = value;
        }
    };
    RegisterAction<bit<32>, bit<1>, bit<32>>(statedt_fallbacks) count_fallback = {
        void apply(inout bit<32> value, out bit<32> result) {
            value = value + 1; result = value;
        }
    };

    RegisterAction<statedt_entry_t, bit<15>, bit<2>>(flow_bank0) access_bank0 = {
        void apply(inout statedt_entry_t value, out bit<2> result) {
            bit<16> incoming_region;
            bit<17> total;
            result = STATEDT_STATUS_NOT_PROCESSED;
            if (value[STATEDT_VALID_BIT:STATEDT_VALID_BIT] == 0) {
                value = 0;
                value[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] = meta.fingerprint;
                value[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] = meta.packet_low_to_high;
                value[STATEDT_VALID_BIT:STATEDT_VALID_BIT] = 1;
                meta.forward = 1;
#include "../common/statedt_update_value.p4inc"
                result = STATEDT_STATUS_ALLOCATED;
            } else if (value[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] == meta.fingerprint) {
                meta.forward = (bit<1>)(value[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] == meta.packet_low_to_high);
#include "../common/statedt_update_value.p4inc"
                result = STATEDT_STATUS_MATCH;
            }
        }
    };

    RegisterAction<statedt_entry_t, bit<15>, bit<2>>(flow_bank1) access_bank1 = {
        void apply(inout statedt_entry_t value, out bit<2> result) {
            bit<16> incoming_region;
            bit<17> total;
            result = STATEDT_STATUS_NOT_PROCESSED;
            if (value[STATEDT_VALID_BIT:STATEDT_VALID_BIT] == 0) {
                value = 0;
                value[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] = meta.fingerprint;
                value[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] = meta.packet_low_to_high;
                value[STATEDT_VALID_BIT:STATEDT_VALID_BIT] = 1;
                meta.forward = 1;
#include "../common/statedt_update_value.p4inc"
                result = STATEDT_STATUS_ALLOCATED;
            } else if (value[STATEDT_FINGERPRINT_MSB:STATEDT_FINGERPRINT_LSB] == meta.fingerprint) {
                meta.forward = (bit<1>)(value[STATEDT_DIRECTION_BIT:STATEDT_DIRECTION_BIT] == meta.packet_low_to_high);
#include "../common/statedt_update_value.p4inc"
                result = STATEDT_STATUS_MATCH;
            }
        }
    };

    action set_class(bit<8> class_id) { meta.class_id = class_id; }

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

    action tcp_payload_40() { meta.packet_length = hdr.ipv4.total_len - 40; }
    action tcp_payload_44() { meta.packet_length = hdr.ipv4.total_len - 44; }
    action tcp_payload_48() { meta.packet_length = hdr.ipv4.total_len - 48; }
    action tcp_payload_52() { meta.packet_length = hdr.ipv4.total_len - 52; }
    action tcp_payload_56() { meta.packet_length = hdr.ipv4.total_len - 56; }
    action tcp_payload_60() { meta.packet_length = hdr.ipv4.total_len - 60; }
    action tcp_payload_64() { meta.packet_length = hdr.ipv4.total_len - 64; }
    action tcp_payload_68() { meta.packet_length = hdr.ipv4.total_len - 68; }
    action tcp_payload_72() { meta.packet_length = hdr.ipv4.total_len - 72; }
    action tcp_payload_76() { meta.packet_length = hdr.ipv4.total_len - 76; }
    action tcp_payload_80() { meta.packet_length = hdr.ipv4.total_len - 80; }

    table tcp_payload_length {
        key = { hdr.tcp.data_offset : exact; }
        actions = {
            tcp_payload_40; tcp_payload_44; tcp_payload_48; tcp_payload_52;
            tcp_payload_56; tcp_payload_60; tcp_payload_64; tcp_payload_68;
            tcp_payload_72; tcp_payload_76; tcp_payload_80;
        }
        size = 11;
        default_action = tcp_payload_40();
        const entries = {
            4w5: tcp_payload_40(); 4w6: tcp_payload_44();
            4w7: tcp_payload_48(); 4w8: tcp_payload_52();
            4w9: tcp_payload_56(); 4w10: tcp_payload_60();
            4w11: tcp_payload_64(); 4w12: tcp_payload_68();
            4w13: tcp_payload_72(); 4w14: tcp_payload_76();
            4w15: tcp_payload_80();
        }
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
        actions = { set_class; }
        size = STATEDT_RULE_COUNT;
        default_action = set_class(CLASS_BENIGN);
        const entries = {
#include "../common/statedt_entries.p4inc"
        }
    }

    apply {
        bit<2> probe_result;
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        meta.class_id = CLASS_BENIGN;
        meta.state_valid = 0;
        meta.mismatch_increment = 0;
        meta.state_status = STATEDT_STATUS_NOT_PROCESSED;

        if (hdr.ipv4.isValid() && (hdr.tcp.isValid() || hdr.udp.isValid())) {
            meta.packet_psh = 0;
            meta.packet_fin = 0;
            if (hdr.tcp.isValid()) {
                tcp_payload_length.apply();
                meta.packet_psh = hdr.tcp.psh;
                meta.packet_fin = hdr.tcp.fin;
            } else {
                meta.packet_length = hdr.udp.length - 8;
            }
            canonicalize();
            meta.index0 = index_hash0.get({ meta.low_addr, meta.high_addr,
                hdr.ipv4.protocol, meta.low_port, meta.high_port, 8w0x31 });
            meta.index1 = index_hash1.get({ meta.low_addr, meta.high_addr,
                hdr.ipv4.protocol, meta.low_port, meta.high_port, 8w0xa7 });
            meta.fingerprint = fingerprint_hash.get({ meta.low_addr, meta.high_addr,
                hdr.ipv4.protocol, meta.low_port, meta.high_port });

            probe_result = access_bank0.execute(meta.index0);
            if (probe_result == STATEDT_STATUS_NOT_PROCESSED) {
                meta.mismatch_increment = 1;
                probe_result = access_bank1.execute(meta.index1);
                if (probe_result == STATEDT_STATUS_NOT_PROCESSED) {
                    meta.mismatch_increment = 2;
                    meta.counter_value = count_collision.execute(1w0);
                    meta.counter_value = count_fallback.execute(1w0);
                    meta.state_status = STATEDT_STATUS_FALLBACK_COLLISION;
                }
            }
            if (meta.mismatch_increment != 0) {
                meta.counter_value = count_mismatch.execute(1w0);
            }
            if (probe_result != STATEDT_STATUS_NOT_PROCESSED) {
                meta.state_valid = 1;
                meta.state_status = probe_result;
                if (probe_result == STATEDT_STATUS_ALLOCATED) {
                    meta.counter_value = count_allocation.execute(1w0);
                }
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

control SwitchIngressDeparser(
        packet_out packet,
        inout headers_t hdr,
        in metadata_t meta,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.result);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

parser EmptyEgressParser(
        packet_in pkt,
        out empty_headers_t hdr,
        out empty_metadata_t meta,
        out egress_intrinsic_metadata_t eg_intr_md) {
    state start { transition accept; }
}

control EmptyEgress(
        inout empty_headers_t hdr,
        inout empty_metadata_t meta,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_prsr_md,
        inout egress_intrinsic_metadata_for_deparser_t eg_dprsr_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {
    apply { }
}

control EmptyEgressDeparser(
        packet_out pkt,
        inout empty_headers_t hdr,
        in empty_metadata_t meta,
        in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    apply { }
}

Pipeline(
    SwitchIngressParser(),
    SwitchIngress(),
    SwitchIngressDeparser(),
    EmptyEgressParser(),
    EmptyEgress(),
    EmptyEgressDeparser()
) pipe;

Switch(pipe) main;

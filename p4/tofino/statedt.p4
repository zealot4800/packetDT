#include <core.p4>
#include <tna.p4>
#include "../common/statedt_headers.p4"
#include "../common/statedt_model.p4inc"

@pragma pa_auto_init_metadata

const bit<32> FLOW_BANK_SIZE = 32w32768;

struct metadata_t {
    bit<32> low_addr;
    bit<32> high_addr;
    bit<16> low_port;
    bit<16> high_port;
    bit<16> src_port;
    bit<15> index0;
    bit<15> index1;
    bit<16> fingerprint;
    bit<16> endpoint_fingerprint;
    bit<1> claimed;
    bit<1> selected_bank;
    bit<1> forward;
    bit<1> state_valid;
    bit<1> packet_psh;
    bit<1> packet_fin;
    bit<16> packet_length;
    bit<16> fwd_packet_length;
    bit<16> total_fwd_increment;
    bit<16> bwd_packet_increment;
    bit<8> class_id;
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

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition accept;
    }
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
    Hash<bit<16>>(HashAlgorithm_t.CRC16) endpoint_hash;

    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) flow_id_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) flow_id_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) flow_dir_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) flow_dir_bank1;

    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) packet_max_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) psh_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) total_fwd_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) fin_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) fwd_max_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) bwd_packets_bank0;

    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) packet_max_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) psh_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) total_fwd_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) fin_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) fwd_max_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE, 16w0) bwd_packets_bank1;

    RegisterAction<bit<16>, bit<15>, bit<1>>(flow_id_bank0) claim_bank0 = {
        void apply(inout bit<16> value, out bit<1> result) {
            result = 0;
            if (value == 0) {
                value = meta.fingerprint;
                result = 1;
            } else if (value == meta.fingerprint) {
                result = 1;
            }
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<1>>(flow_id_bank1) claim_bank1 = {
        void apply(inout bit<16> value, out bit<1> result) {
            result = 0;
            if (value == 0) {
                value = meta.fingerprint;
                result = 1;
            } else if (value == meta.fingerprint) {
                result = 1;
            }
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<1>>(flow_dir_bank0) direction_bank0 = {
        void apply(inout bit<16> value, out bit<1> result) {
            result = 0;
            if (value == 0) {
                value = meta.endpoint_fingerprint;
                result = 1;
            } else if (value == meta.endpoint_fingerprint) {
                result = 1;
            }
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<1>>(flow_dir_bank1) direction_bank1 = {
        void apply(inout bit<16> value, out bit<1> result) {
            result = 0;
            if (value == 0) {
                value = meta.endpoint_fingerprint;
                result = 1;
            } else if (value == meta.endpoint_fingerprint) {
                result = 1;
            }
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<16>>(packet_max_bank0) update_packet_max0 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (meta.packet_length > value) { value = meta.packet_length; }
            if (value > CAP_PACKET_LENGTH_MAX) { value = CAP_PACKET_LENGTH_MAX; }
            result = value;
        }
    };
    RegisterAction<bit<16>, bit<15>, bit<16>>(packet_max_bank1) update_packet_max1 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (meta.packet_length > value) { value = meta.packet_length; }
            if (value > CAP_PACKET_LENGTH_MAX) { value = CAP_PACKET_LENGTH_MAX; }
            result = value;
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<16>>(psh_bank0) update_psh0 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (value < CAP_PSH_FLAG_COUNT) { value = value + (bit<16>) meta.packet_psh; }
            result = value;
        }
    };
    RegisterAction<bit<16>, bit<15>, bit<16>>(psh_bank1) update_psh1 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (value < CAP_PSH_FLAG_COUNT) { value = value + (bit<16>) meta.packet_psh; }
            result = value;
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<16>>(total_fwd_bank0) update_total_fwd0 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (meta.total_fwd_increment > CAP_TOTAL_FWD_LENGTH) {
                value = CAP_TOTAL_FWD_LENGTH;
            } else if (value >= CAP_TOTAL_FWD_LENGTH - meta.total_fwd_increment) {
                value = CAP_TOTAL_FWD_LENGTH;
            } else {
                value = value + meta.total_fwd_increment;
            }
            result = value;
        }
    };
    RegisterAction<bit<16>, bit<15>, bit<16>>(total_fwd_bank1) update_total_fwd1 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (meta.total_fwd_increment > CAP_TOTAL_FWD_LENGTH) {
                value = CAP_TOTAL_FWD_LENGTH;
            } else if (value >= CAP_TOTAL_FWD_LENGTH - meta.total_fwd_increment) {
                value = CAP_TOTAL_FWD_LENGTH;
            } else {
                value = value + meta.total_fwd_increment;
            }
            result = value;
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<16>>(fin_bank0) update_fin0 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (value < CAP_FIN_FLAG_COUNT) { value = value + (bit<16>) meta.packet_fin; }
            result = value;
        }
    };
    RegisterAction<bit<16>, bit<15>, bit<16>>(fin_bank1) update_fin1 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (value < CAP_FIN_FLAG_COUNT) { value = value + (bit<16>) meta.packet_fin; }
            result = value;
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<16>>(fwd_max_bank0) update_fwd_max0 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (meta.fwd_packet_length > value) {
                value = meta.fwd_packet_length;
            }
            if (value > CAP_FWD_PACKET_LENGTH_MAX) { value = CAP_FWD_PACKET_LENGTH_MAX; }
            result = value;
        }
    };
    RegisterAction<bit<16>, bit<15>, bit<16>>(fwd_max_bank1) update_fwd_max1 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (meta.fwd_packet_length > value) {
                value = meta.fwd_packet_length;
            }
            if (value > CAP_FWD_PACKET_LENGTH_MAX) { value = CAP_FWD_PACKET_LENGTH_MAX; }
            result = value;
        }
    };

    RegisterAction<bit<16>, bit<15>, bit<16>>(bwd_packets_bank0) update_bwd0 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (value < CAP_TOTAL_BWD_PACKETS) { value = value + meta.bwd_packet_increment; }
            result = value;
        }
    };
    RegisterAction<bit<16>, bit<15>, bit<16>>(bwd_packets_bank1) update_bwd1 = {
        void apply(inout bit<16> value, out bit<16> result) {
            if (value < CAP_TOTAL_BWD_PACKETS) { value = value + meta.bwd_packet_increment; }
            result = value;
        }
    };

    action set_class(bit<8> class_id) {
        meta.class_id = class_id;
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
            4w5: tcp_payload_40();
            4w6: tcp_payload_44();
            4w7: tcp_payload_48();
            4w8: tcp_payload_52();
            4w9: tcp_payload_56();
            4w10: tcp_payload_60();
            4w11: tcp_payload_64();
            4w12: tcp_payload_68();
            4w13: tcp_payload_72();
            4w14: tcp_payload_76();
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
        bit<1> identity_result;
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        meta.class_id = CLASS_BENIGN;
        meta.state_valid = 0;
        meta.claimed = 0;

        if (hdr.ipv4.isValid() && (hdr.tcp.isValid() || hdr.udp.isValid())) {
            if (hdr.tcp.isValid()) {
                tcp_payload_length.apply();
                meta.low_port = hdr.tcp.src_port ^ hdr.tcp.dst_port;
                meta.high_port = hdr.tcp.src_port + hdr.tcp.dst_port;
                meta.src_port = hdr.tcp.src_port;
            } else {
                meta.packet_length = hdr.udp.length - 8;
                meta.low_port = hdr.udp.src_port ^ hdr.udp.dst_port;
                meta.high_port = hdr.udp.src_port + hdr.udp.dst_port;
                meta.src_port = hdr.udp.src_port;
            }
            meta.low_addr = hdr.ipv4.src_addr ^ hdr.ipv4.dst_addr;
            meta.high_addr = hdr.ipv4.src_addr + hdr.ipv4.dst_addr;
            meta.endpoint_fingerprint = endpoint_hash.get({ hdr.ipv4.src_addr,
                hdr.ipv4.protocol, meta.src_port }) | 16w1;
            meta.packet_psh = 0;
            meta.packet_fin = 0;
            if (hdr.tcp.isValid()) {
                meta.packet_psh = hdr.tcp.psh;
                meta.packet_fin = hdr.tcp.fin;
            }
            meta.index0 = index_hash0.get({ meta.low_addr, meta.high_addr,
                hdr.ipv4.protocol, meta.low_port, meta.high_port, 8w0x31 });
            meta.index1 = index_hash1.get({ meta.low_addr, meta.high_addr,
                hdr.ipv4.protocol, meta.low_port, meta.high_port, 8w0xa7 });
            meta.fingerprint = fingerprint_hash.get({ meta.low_addr, meta.high_addr,
                hdr.ipv4.protocol, meta.low_port, meta.high_port }) | 16w1;

            identity_result = claim_bank0.execute(meta.index0);
            if (identity_result == 1) {
                meta.claimed = 1;
                meta.selected_bank = 0;
                meta.forward = direction_bank0.execute(meta.index0);
            } else {
                identity_result = claim_bank1.execute(meta.index1);
                if (identity_result == 1) {
                    meta.claimed = 1;
                    meta.selected_bank = 1;
                    meta.forward = direction_bank1.execute(meta.index1);
                }
            }

            meta.fwd_packet_length = 0;
            meta.total_fwd_increment = 0;
            meta.bwd_packet_increment = 0;
            if (meta.forward == 1) {
                meta.fwd_packet_length = meta.packet_length;
                meta.total_fwd_increment = meta.packet_length;
            } else {
                meta.bwd_packet_increment = 1;
            }

            if (meta.claimed == 1 && meta.selected_bank == 0) {
                meta.features.packet_length_max = update_packet_max0.execute(meta.index0);
                meta.features.psh_flag_count = update_psh0.execute(meta.index0);
                meta.features.total_fwd_length = update_total_fwd0.execute(meta.index0);
                meta.features.fin_flag_count = update_fin0.execute(meta.index0);
                meta.features.fwd_packet_length_max = update_fwd_max0.execute(meta.index0);
                meta.features.total_bwd_packets = update_bwd0.execute(meta.index0);
            } else if (meta.claimed == 1) {
                meta.features.packet_length_max = update_packet_max1.execute(meta.index1);
                meta.features.psh_flag_count = update_psh1.execute(meta.index1);
                meta.features.total_fwd_length = update_total_fwd1.execute(meta.index1);
                meta.features.fin_flag_count = update_fin1.execute(meta.index1);
                meta.features.fwd_packet_length_max = update_fwd_max1.execute(meta.index1);
                meta.features.total_bwd_packets = update_bwd1.execute(meta.index1);
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

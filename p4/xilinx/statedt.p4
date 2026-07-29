#include <core.p4>
#include <xsa.p4>
#include "../common/statedt_headers.p4"
#include "../common/statedt_model.p4inc"

const bit<32> FLOW_BANK_SIZE = 32w32768;

struct metadata_t {
    bit<32> addr_xor;
    bit<32> addr_sum;
    bit<16> port_xor;
    bit<16> port_sum;
    bit<16> src_port;
    bit<32> mix0;
    bit<32> mix1;
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
    bit<8> class_id;
    bit<16> identity;
    bit<16> direction;
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

control StateDTMatchAction(
        inout headers_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) {
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) flow_id_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) flow_id_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) flow_dir_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) flow_dir_bank1;

    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) packet_max_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) psh_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) total_fwd_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) fin_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) fwd_max_bank0;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) bwd_packets_bank0;

    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) packet_max_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) psh_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) total_fwd_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) fin_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) fwd_max_bank1;
    Register<bit<16>, bit<15>>(FLOW_BANK_SIZE) bwd_packets_bank1;

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
        actions = { set_class; }
        size = STATEDT_RULE_COUNT;
        default_action = set_class(CLASS_BENIGN);
        const entries = {
#include "../common/statedt_entries.p4inc"
        }
    }

    apply {
        meta.class_id = CLASS_BENIGN;
        meta.state_valid = 0;
        meta.claimed = 0;

        if (hdr.ipv4.isValid() && (hdr.tcp.isValid() || hdr.udp.isValid())) {
            meta.packet_psh = 0;
            meta.packet_fin = 0;
            if (hdr.tcp.isValid()) {
                meta.packet_length = hdr.ipv4.total_len - ((bit<16>) hdr.ipv4.ihl << 2) -
                    ((bit<16>) hdr.tcp.data_offset << 2);
                meta.src_port = hdr.tcp.src_port;
                meta.port_xor = hdr.tcp.src_port ^ hdr.tcp.dst_port;
                meta.port_sum = hdr.tcp.src_port + hdr.tcp.dst_port;
                meta.packet_psh = hdr.tcp.psh;
                meta.packet_fin = hdr.tcp.fin;
            } else {
                meta.packet_length = hdr.udp.length - 8;
                meta.src_port = hdr.udp.src_port;
                meta.port_xor = hdr.udp.src_port ^ hdr.udp.dst_port;
                meta.port_sum = hdr.udp.src_port + hdr.udp.dst_port;
            }

            // Both mixes are commutative, so reverse-direction packets use the
            // same two register slots without requiring a variable comparison.
            meta.addr_xor = hdr.ipv4.src_addr ^ hdr.ipv4.dst_addr;
            meta.addr_sum = hdr.ipv4.src_addr + hdr.ipv4.dst_addr;
            meta.mix0 = meta.addr_xor ^ (meta.addr_sum << 7) ^
                ((bit<32>) meta.port_xor << 16) ^ (bit<32>) meta.port_sum ^
                (bit<32>) hdr.ipv4.protocol;
            meta.mix1 = meta.addr_sum ^ (meta.addr_xor << 11) ^
                ((bit<32>) meta.port_sum << 16) ^ (bit<32>) meta.port_xor ^
                ((bit<32>) hdr.ipv4.protocol << 24);
            meta.index0 = meta.mix0[14:0] ^ meta.mix0[29:15];
            meta.index1 = meta.mix1[14:0] ^ meta.mix1[29:15];
            meta.fingerprint = (meta.mix0[31:16] ^ meta.mix1[15:0]) | 16w1;
            meta.endpoint_fingerprint = (hdr.ipv4.src_addr[31:16] ^
                hdr.ipv4.src_addr[15:0] ^ meta.src_port ^
                (bit<16>) hdr.ipv4.protocol) | 16w1;

            flow_id_bank0.read(meta.index0, meta.identity);
            if (meta.identity == 0 || meta.identity == meta.fingerprint) {
                meta.identity = meta.fingerprint;
                flow_id_bank0.write(meta.index0, meta.identity);
                meta.claimed = 1;
                meta.selected_bank = 0;
            } else {
                flow_id_bank1.read(meta.index1, meta.identity);
                if (meta.identity == 0 || meta.identity == meta.fingerprint) {
                    meta.identity = meta.fingerprint;
                    flow_id_bank1.write(meta.index1, meta.identity);
                    meta.claimed = 1;
                    meta.selected_bank = 1;
                }
            }

            if (meta.claimed == 1 && meta.selected_bank == 0) {
                flow_dir_bank0.read(meta.index0, meta.direction);
                if (meta.direction == 0) {
                    meta.direction = meta.endpoint_fingerprint;
                    meta.forward = 1;
                } else if (meta.direction == meta.endpoint_fingerprint) {
                    meta.forward = 1;
                } else {
                    meta.forward = 0;
                }
                flow_dir_bank0.write(meta.index0, meta.direction);

                packet_max_bank0.read(meta.index0, meta.features.packet_length_max);
                if (meta.packet_length > meta.features.packet_length_max) {
                    meta.features.packet_length_max = meta.packet_length;
                }
                if (meta.features.packet_length_max > CAP_PACKET_LENGTH_MAX) {
                    meta.features.packet_length_max = CAP_PACKET_LENGTH_MAX;
                }
                packet_max_bank0.write(meta.index0, meta.features.packet_length_max);

                psh_bank0.read(meta.index0, meta.features.psh_flag_count);
                if (meta.packet_psh == 1 && meta.features.psh_flag_count < CAP_PSH_FLAG_COUNT) {
                    meta.features.psh_flag_count = meta.features.psh_flag_count + 1;
                }
                psh_bank0.write(meta.index0, meta.features.psh_flag_count);

                total_fwd_bank0.read(meta.index0, meta.features.total_fwd_length);
                if (meta.forward == 1) {
                    if (meta.packet_length >= CAP_TOTAL_FWD_LENGTH - meta.features.total_fwd_length) {
                        meta.features.total_fwd_length = CAP_TOTAL_FWD_LENGTH;
                    } else {
                        meta.features.total_fwd_length =
                            meta.features.total_fwd_length + meta.packet_length;
                    }
                }
                total_fwd_bank0.write(meta.index0, meta.features.total_fwd_length);

                fin_bank0.read(meta.index0, meta.features.fin_flag_count);
                if (meta.packet_fin == 1 && meta.features.fin_flag_count < CAP_FIN_FLAG_COUNT) {
                    meta.features.fin_flag_count = meta.features.fin_flag_count + 1;
                }
                fin_bank0.write(meta.index0, meta.features.fin_flag_count);

                fwd_max_bank0.read(meta.index0, meta.features.fwd_packet_length_max);
                if (meta.forward == 1 && meta.packet_length > meta.features.fwd_packet_length_max) {
                    meta.features.fwd_packet_length_max = meta.packet_length;
                }
                if (meta.features.fwd_packet_length_max > CAP_FWD_PACKET_LENGTH_MAX) {
                    meta.features.fwd_packet_length_max = CAP_FWD_PACKET_LENGTH_MAX;
                }
                fwd_max_bank0.write(meta.index0, meta.features.fwd_packet_length_max);

                bwd_packets_bank0.read(meta.index0, meta.features.total_bwd_packets);
                if (meta.forward == 0 &&
                    meta.features.total_bwd_packets < CAP_TOTAL_BWD_PACKETS) {
                    meta.features.total_bwd_packets = meta.features.total_bwd_packets + 1;
                }
                bwd_packets_bank0.write(meta.index0, meta.features.total_bwd_packets);
            } else if (meta.claimed == 1) {
                flow_dir_bank1.read(meta.index1, meta.direction);
                if (meta.direction == 0) {
                    meta.direction = meta.endpoint_fingerprint;
                    meta.forward = 1;
                } else if (meta.direction == meta.endpoint_fingerprint) {
                    meta.forward = 1;
                } else {
                    meta.forward = 0;
                }
                flow_dir_bank1.write(meta.index1, meta.direction);

                packet_max_bank1.read(meta.index1, meta.features.packet_length_max);
                if (meta.packet_length > meta.features.packet_length_max) {
                    meta.features.packet_length_max = meta.packet_length;
                }
                if (meta.features.packet_length_max > CAP_PACKET_LENGTH_MAX) {
                    meta.features.packet_length_max = CAP_PACKET_LENGTH_MAX;
                }
                packet_max_bank1.write(meta.index1, meta.features.packet_length_max);

                psh_bank1.read(meta.index1, meta.features.psh_flag_count);
                if (meta.packet_psh == 1 && meta.features.psh_flag_count < CAP_PSH_FLAG_COUNT) {
                    meta.features.psh_flag_count = meta.features.psh_flag_count + 1;
                }
                psh_bank1.write(meta.index1, meta.features.psh_flag_count);

                total_fwd_bank1.read(meta.index1, meta.features.total_fwd_length);
                if (meta.forward == 1) {
                    if (meta.packet_length >= CAP_TOTAL_FWD_LENGTH - meta.features.total_fwd_length) {
                        meta.features.total_fwd_length = CAP_TOTAL_FWD_LENGTH;
                    } else {
                        meta.features.total_fwd_length =
                            meta.features.total_fwd_length + meta.packet_length;
                    }
                }
                total_fwd_bank1.write(meta.index1, meta.features.total_fwd_length);

                fin_bank1.read(meta.index1, meta.features.fin_flag_count);
                if (meta.packet_fin == 1 && meta.features.fin_flag_count < CAP_FIN_FLAG_COUNT) {
                    meta.features.fin_flag_count = meta.features.fin_flag_count + 1;
                }
                fin_bank1.write(meta.index1, meta.features.fin_flag_count);

                fwd_max_bank1.read(meta.index1, meta.features.fwd_packet_length_max);
                if (meta.forward == 1 && meta.packet_length > meta.features.fwd_packet_length_max) {
                    meta.features.fwd_packet_length_max = meta.packet_length;
                }
                if (meta.features.fwd_packet_length_max > CAP_FWD_PACKET_LENGTH_MAX) {
                    meta.features.fwd_packet_length_max = CAP_FWD_PACKET_LENGTH_MAX;
                }
                fwd_max_bank1.write(meta.index1, meta.features.fwd_packet_length_max);

                bwd_packets_bank1.read(meta.index1, meta.features.total_bwd_packets);
                if (meta.forward == 0 &&
                    meta.features.total_bwd_packets < CAP_TOTAL_BWD_PACKETS) {
                    meta.features.total_bwd_packets = meta.features.total_bwd_packets + 1;
                }
                bwd_packets_bank1.write(meta.index1, meta.features.total_bwd_packets);
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

control StateDTDeparser(
        packet_out packet,
        in headers_t hdr,
        inout metadata_t meta,
        inout standard_metadata_t standard_metadata) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.result);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

XilinxPipeline(
    StateDTParser(),
    StateDTMatchAction(),
    StateDTDeparser()
) main;

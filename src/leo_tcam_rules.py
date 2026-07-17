import argparse

import switch_resources as sw_resources


def calculate_leaves_full_tree(nodes):
    """
    Calculate the number of leaf nodes in a full binary decision tree
    based on the total number of nodes.
    """
    if nodes % 2 == 0:
        return "Not a valid full binary tree node count"
    leaves = (nodes + 1) // 2
    return leaves


def calculate_leaves_general_tree(nodes):
    """
    Estimate the number of leaf nodes in a general binary decision tree.
    For a general binary tree, the number of leaves can vary widely,
    so we can only make estimates without specific tree structure information.
    """
    if nodes == 0:
        return 0
    elif nodes == 1:
        return 1
    else:
        # Estimating the number of leaves assuming a balanced binary tree
        return nodes // 2


def leo_model(alu_config, is_sram, transient, log=False):
    num_alu_layers = len(alu_config)
    # print('Number of ALU layers:', num_alu_layers)
    single_table_sizes = []

    total_size = 0
    curr_layer_result_combos = 1
    prev_layer_tcam = 1
    curr_layer_tcam = 1

    if log:
        print("{:>12}  {:>12}  {:>12}".format("Layer #", "Single Table Size", "Total Layer Size"))
    for l in range(1, num_alu_layers + 2):
        if l == num_alu_layers + 1:
            num_mux_next_layer = 1
        else:
            num_mux_next_layer = alu_config[l - 1]

        if l > 1:
            curr_layer_tcam = alu_config[l - 2] + 1
            if is_sram:
                curr_layer_result_combos = 2 ** alu_config[l - 2]
            else:
                curr_layer_result_combos = curr_layer_tcam

        single_table_size = curr_layer_result_combos * prev_layer_tcam
        layer_size = single_table_size * num_mux_next_layer
        if transient:
            layer_size = layer_size * 2
            single_table_size = single_table_size * 2

        total_size += layer_size

        single_table_sizes.append(single_table_size)
        if log:
            print("{:>12}  {:>12}  {:>12}".format(l, single_table_size, layer_size))

        prev_layer_tcam = curr_layer_tcam * prev_layer_tcam

    if log:
        print("Total Size:", total_size)
    return l, single_table_sizes, total_size


# Rules in layer i of LEO
def R_i(i, L, R, K):
    if i == 1:
        return 1
    if L != 0:
        return min(L, R[i - 1] * (K[i] + 1))
    else:
        return R[i - 1] * (K[i] + 1)


def args_type_for_number_list(arg):
    try:
        return [int(num) for num in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid list of integers: "{}"'.format(arg))


def partition_tree_nodes(depth):
    def generate_partitions(remaining_depth, current_partition):
        # Base case: if no remaining depth, add the current partition to results
        if remaining_depth == 0:
            partitions.append(current_partition)
            return

        # Try partitions uptil 8 levels
        for levels_in_stage in range(1, 9):
            if remaining_depth >= levels_in_stage:
                generate_partitions(
                    remaining_depth - levels_in_stage, current_partition + [levels_in_stage]
                )

    partitions = []
    generate_partitions(depth, [])

    return partitions


def calculate_nodes_per_stage(partitions):
    partitioned_nodes_list = []
    for partition in partitions:
        nodes = []
        for levels in partition:
            # Calculate number of nodes in `levels` tree levels
            nodes_in_stage = 2**levels - 1
            nodes.append(nodes_in_stage)
        partitioned_nodes_list.append(nodes)
    return partitioned_nodes_list


def get_leo_tcam_entries(tree_depth):
    partitions = partition_tree_nodes(tree_depth)
    partitioned_nodes_list = calculate_nodes_per_stage(partitions)
    layers, single_table_entries, tcam_entries = leo_model(
        partitioned_nodes_list[0], False, False, False
    )
    min_layer = 100
    # Print results
    for i, nodes in enumerate(partitioned_nodes_list):
        # print(f"Partition {i + 1}: {nodes}")
        out_keys = calculate_leaves_full_tree(max(nodes))
        # print(f"The nuber of outkeys are {out_keys}")

        # alu_config, is_sram, transient, log=False
        layers, single_table_entries, tcam_entries = leo_model(nodes, False, False, False)

        if layers < min_layer and out_keys + 1 <= 8:
            # print(nodes)
            required_out_keys = out_keys
            min_layer = layers

        # print(f"TCAM entries: {tcam_entries} and layers: {layers}")
    return min_layer, tcam_entries, required_out_keys


def get_num_stages(tree_depth, features, iat=False):
    layers, tcam_entries, outkeys = get_leo_tcam_entries(tree_depth)
    # set data width and feature width
    data_width = 16
    feature_width = 32
    num_mau_tofino1 = 12
    # calaulate number of stages
    feature_tcam_entries_per_key = sw_resources.calculate_allocated_tcam_entries(
        num_tcam_entries=tcam_entries, width_per_key=data_width + feature_width * outkeys
    )
    # print(f"Total number of TCAM entries: {feature_tcam_entries_per_key}")

    stages_used = sw_resources.tcam_entries_to_stages(
        num_tcam_entries=feature_tcam_entries_per_key,
        width_per_key=data_width + feature_width * outkeys,
    )
    stages_used = max(stages_used, layers)
    remaining_stages = num_mau_tofino1 - stages_used
    num_flows = sw_resources.stages_to_registers(
        num_features=features, remaining_stages=remaining_stages
    )

    return feature_tcam_entries_per_key, num_flows


# test the function
if __name__ == "__main__":
    depth = 11
    features = 15
    min_features = min(features, 2 ** (depth + 1) - 1)
    get_num_stages(tree_depth=depth, features=min_features)

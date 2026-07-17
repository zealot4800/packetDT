# *****************************************************************************
#
# Copyright 2026
#   Murayyiam Parvez (Purdue University),
#   Annus Zulfiqar (University of Michigan),
#   Roman Beltiukov (University of California, Santa Barbara),
#   Shir Landau Feibish (The Open University of Israel),
#   Walter Willinger (NIKSUN Inc.),
#   Arpit Gupta (University of California, Santa Barbara),
#   Muhammad Shahbaz (University of Michigan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *****************************************************************************


import math


def calculate_allocated_tcam_entries(num_tcam_entries, width_per_key, entry_width_per_block=44):
    """
    Calculate the number of TCAM entries allocated per key based on the number of TCAM entries and keys.

    Args:
    - num_tcam_entries (int): The total number of TCAM entries.
    - width_per_key (int): The width of each key in bits.
    - entry_width_per_block (int): The width of each TCAM entry in bits (default is 44).
    """

    num_tcam_entries = math.ceil(num_tcam_entries / 512) * 512
    total_width = width_per_key
    num_tcam_entries_per_key = num_tcam_entries * (math.ceil(total_width / entry_width_per_block))
    return num_tcam_entries_per_key


def tcam_entries_to_stages(
    num_tcam_entries,
    width_per_key=32,
    entry_width_per_block=44,
    num_entries_per_block=512,
    num_blocks_per_mau=24,
):
    """
    Calculate the number of TCAM stages used based on the number of TCAM entries and keys.

    Args:
    - num_tcam_entries (int): The total number of TCAM entries.
    - num_keys (int): The total number of keys to be stored in the TCAM.
    - width_per_key (list): The width of each key in bits (default is [8,16,8,8,4,16,16]).
    - entry_width_per_block (int): The width of each TCAM entry in bits (default is 44).
    - num_entries_per_block (int): The number of entries per TCAM block (default is 512).
    - num_blocks_per_mau (int): The number of blocks per memory access unit (default is 24).
    - num_stages (int): The number of TCAM stages (default is 12).

    Returns:
    - int: The number of TCAM stages required for the model table.
    """

    num_tcam_entries = math.ceil(num_tcam_entries / 512) * 512
    total_width = width_per_key

    # Calculate the total number of keys that can be stored in one stage
    max_entires_per_stage = num_entries_per_block * num_blocks_per_mau

    # take the mod of the total number of entries and the max entries per stage
    max_entires_per_stage_per_key = max_entires_per_stage / math.ceil(
        total_width / entry_width_per_block
    )
    # Calculate the number of stages required to handle all the keys
    required_stages = math.ceil(num_tcam_entries / max_entires_per_stage_per_key)

    # Ensure that at least one stage is used, even if the number of keys is small
    return max(required_stages, 1)


def stages_to_registers(num_features, remaining_stages, feature_width=32):
    FEATURE_WIDTH = feature_width
    MAX_32BIT_SLOTS_PER_STAGE = 128 * 1024  # two full registers 94208 * 2
    MEMORY_BLOCKS_PER_STAGE = 80 - 32  # number of SRAM memory blocks available per stage
    TOTAL_SRAM_PER_MAU = MAX_32BIT_SLOTS_PER_STAGE * MEMORY_BLOCKS_PER_STAGE / FEATURE_WIDTH

    full_stages = remaining_stages // num_features
    partial_stages = remaining_stages % num_features

    full_stage_flows = TOTAL_SRAM_PER_MAU * full_stages
    partial_stage_flows = 0
    if partial_stages > 0 and num_features / partial_stages <= 4:
        partial_stage_flows += (
            TOTAL_SRAM_PER_MAU * 1 / (math.floor(4 * partial_stages / num_features))
        )

    if full_stage_flows + partial_stage_flows < 0:
        return 0

    return full_stage_flows + partial_stage_flows


def get_resubmission_traffic_Gbps(switch_model, num_flows, num_partitions):
    # assume all flows update their node ID at the same time
    concurrent_update_bits = num_flows * switch_model.resubmission_bits_per_pkt
    # assume all partitions are tested in one second and convert to Gbps
    concurrent_update_bw = round(num_partitions * concurrent_update_bits / 1e9, 3)
    return concurrent_update_bw


def get_tcam_blocks(
    feature_count,
    feature_entries,
    table_entries,
    FEATURE_WIDTH=32,
    NODE_ID_WIDTH=16,
    TCAM_BLOCKS_PER_STAGE=24,
    NUM_MAU_STAGES=12,
):
    # calculate number of stages
    feature_tcam_entries_per_key = calculate_allocated_tcam_entries(
        num_tcam_entries=feature_entries, width_per_key=FEATURE_WIDTH
    )

    table_tcam_entries_per_key = calculate_allocated_tcam_entries(
        num_tcam_entries=table_entries, width_per_key=FEATURE_WIDTH * feature_count
    )

    feature_table_stages = tcam_entries_to_stages(
        num_tcam_entries=feature_tcam_entries_per_key,
        width_per_key=NODE_ID_WIDTH + (FEATURE_WIDTH * feature_count),
    )

    tree_table_stage = tcam_entries_to_stages(
        num_tcam_entries=table_tcam_entries_per_key,
        width_per_key=NODE_ID_WIDTH + (FEATURE_WIDTH * feature_count),
    )

    tcam_used_stages = feature_table_stages + tree_table_stage
    remaining_stages = NUM_MAU_STAGES - tcam_used_stages
    tcam_used_blocks = tcam_used_stages * TCAM_BLOCKS_PER_STAGE
    return tcam_used_blocks, remaining_stages


def main():
    features_and_table_entries = [
        (0, 16384),
        (0, 2048),
        (0, 2048),
        (0, 8192),
        (0, 2048),
        (0, 2048),
        (0, 16384),
        (0, 2048),
        (0, 2048),
        (0, 8192),
        (0, 8192),
        (0, 2048),
        (0, 8192),
        (0, 2048),
        (0, 2048),
        (0, 8192),
        (0, 8192),
        (0, 2048),
        (0, 2048),
        (0, 2048),
        (0, 2048),
    ]
    BOOKKEEPING = 1
    for feature_entries, table_entries in features_and_table_entries:
        for num_features in [1, 3, 7]:
            used_tcam_blocks, remaining_stages = get_tcam_blocks(
                feature_count=num_features,
                feature_entries=feature_entries,
                table_entries=table_entries,
            )
            num_flows = stages_to_registers(
                num_features=BOOKKEEPING + num_features, remaining_stages=remaining_stages
            )
            num_flows = int(num_flows)
            print(f"Feature Entries = {feature_entries}, Table Entries = {table_entries}, ", end="")
            print(f"Feature Count = {num_features}, Used TCAM blocks: {used_tcam_blocks}, ", end="")
            print(f"Remaining Stages = {remaining_stages}, Num Flows = {num_flows}")

        print()


if __name__ == "__main__":
    main()

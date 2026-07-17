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


import json
import os
from dataclasses import dataclass

from utils import Box


@dataclass
class Objectives:
    f1_score: bool = False
    num_flows: bool = False
    feasible: bool = False
    response_header: str = ""


def get_experiment_scenario(parsed_args, results_path):
    # experiment design
    scenario = Box({"application_name": parsed_args.dataset.name})
    scenario.update(parsed_args.hypermapper.scenario)

    # update log paths
    scenario.output_data_file = os.path.join(results_path, scenario.output_data_file)
    scenario.log_file = os.path.join(results_path, scenario.log_file)

    # collect optimization objectives
    objectives = Objectives()
    if "f1_score" in scenario.optimization_objectives:
        objectives.f1_score = True
    if "num_flows" in scenario.optimization_objectives:
        objectives.num_flows = True

    # drop feasibility if not needed
    check_feasibility = scenario.check_feasibility
    scenario.pop("check_feasibility", None)
    if not check_feasibility:
        scenario.pop("feasible_output", None)
    else:
        objectives.feasible = True

    # drop unwanted optimization parameters
    if scenario.optimization_method == "bayesian_optimization":
        inner_params = scenario.bayesian_optimization
    elif scenario.optimization_method == "local_search":
        inner_params = scenario.local_search
    elif scenario.optimization_method == "evolutionary_optimization":
        inner_params = scenario.evolutionary_optimization

    scenario.pop("local_search", None)
    scenario.pop("evolutionary_optimization", None)
    scenario.pop("bayesian_optimization", None)
    scenario.update(inner_params)

    # dump the scenario to a json file
    scenario_path = os.path.join(results_path, "scenario.json")
    with open(scenario_path, "w") as scenario_file:
        json.dump(scenario, scenario_file, indent=4)

    # create the response string for hypermapper
    objectives.response_header = "depth,features_per_partition,c1,c2,c3,c4,c5,c6,"
    objectives.response_header += ",".join(scenario.optimization_objectives)
    if check_feasibility:
        objectives.response_header += f",{scenario.feasible_output.name}"
    objectives.response_header += "\n"

    return scenario_path, objectives

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import pytest
from pandapower import pp_dir
from pandapower.test.shortcircuit.sce_tests.functions_tests import (compare_results, run_test_cases,
                                                                    load_test_case_data, create_parameter_list)
import warnings
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

testfiles_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests')

# Define common parameters
net_names = [
    "test_case_2_five_bus_radial_grid",
    "test_case_3_five_bus_meshed_grid",
    "test_case_4_twenty_bus_radial_grid"
]
faults = ["LLL", "LL", "LG", "LLG"]
cases = ["max", "min"]
values = [(0, 0), (5, 5)]
vector_groups = ['Dyn', 'Yyn', 'YNyn']
lv_tol_percents = [6, 10]
fault_location_buses = [0, 1, 2, 3]
is_branch_test = [True, False]

# parameters for WP 2.2 and WP 2.4
gen_idx = [1, [1, 3]]  # includes both, static generator and synchronous generator
is_active_current = [False, True]  # TODO: for gen not working yet (grids and results not created)
gen_mode = ['sgen', 'gen']  # TODO: mode 'all' is implemented but results are not created yet

# create parameter lists for WPs
param_wp21, param_vec_wp21, param_wp22, param_vec_wp22 = create_parameter_list(
    net_names, faults, cases, values, lv_tol_percents, fault_location_buses, is_branch_test, vector_groups,
    gen_idx, is_active_current, gen_mode)


@pytest.mark.slow
@pytest.mark.parametrize("fault, case, fault_values, lv_tol_percent, fault_location_bus, is_branch",
                         param_wp21, ids=lambda val: str(val))
def test_wp21_four_bus_radial_grid(fault, case, fault_values, lv_tol_percent, fault_location_bus, is_branch):
    net, dataframes = load_test_case_data("test_case_1_four_bus_radial_grid", fault_location_bus)
    results = run_test_cases(
        net,
        dataframes["branch" if is_branch else "bus"],
        fault,
        case,
        fault_values,
        lv_tol_percent,
        fault_location_bus,
        branch_results=is_branch
    )
    compare_results(*results)


@pytest.mark.slow
@pytest.mark.parametrize(
    "net_name, fault, case, fault_values, lv_tol_percent, vector_group, fault_location_bus, is_branch",
    param_vec_wp21, ids=lambda val: str(val))
def test_wp21_grids_with_trafo(net_name, fault, case, fault_values, lv_tol_percent, vector_group, fault_location_bus,
                               is_branch):
    net, dataframes = load_test_case_data(net_name, fault_location_bus, vector_group)
    results = run_test_cases(
        net,
        dataframes["branch" if is_branch else "bus"],
        fault,
        case,
        fault_values,
        lv_tol_percent,
        fault_location_bus,
        branch_results=is_branch
    )
    compare_results(*results)


@pytest.mark.slow
@pytest.mark.parametrize("fault, case, fault_values, lv_tol_percent, fault_location_bus, is_branch,"
                         "gen_loc, active_current, mode",
                         param_wp22, ids=lambda val: str(val))
def test_wp22_24_four_bus_radial_grid(fault, case, fault_values, lv_tol_percent, fault_location_bus, is_branch,
                                      gen_loc, active_current, mode):

    # TODO: remove when implemented
    if (mode in ('gen', 'all')) and active_current:
        logger.warning("results and grids for generators with active_current=True not created yet, skipping tests")
        return

    if active_current:
        net_name = "1_four_bus_radial_grid_sgen_act"
    else:
        net_name = "1_four_bus_radial_grid_gen"
    net, dataframes = load_test_case_data(net_name, fault_location_bus,
                                          gen_idx=gen_loc, is_active_current=active_current, gen_mode=mode)
    results = run_test_cases(
        net,
        dataframes["branch" if is_branch else "bus"],
        fault,
        case,
        fault_values,
        lv_tol_percent,
        fault_location_bus,
        branch_results=is_branch
    )
    compare_results(*results)


@pytest.mark.slow
@pytest.mark.parametrize(
    "net_name, fault, case, fault_values, lv_tol_percent, vector_group, fault_location_bus, is_branch,"
    "gen_loc, active_current, mode",
    param_vec_wp22, ids=lambda val: str(val))
def test_wp22_24_grids_with_trafo(net_name, fault, case, fault_values, lv_tol_percent, vector_group, fault_location_bus,
                                  is_branch, gen_loc, active_current, mode):
    # TODO: remove when implemented
    if mode == 'all':
        logger.warning("results for mode 'all' not created yet, skipping tests")
        return
    elif mode == 'gen' and active_current:
        logger.warning("results and grids for generators with active_current=True not created yet, skipping tests")
        return

    net, dataframes = load_test_case_data(net_name, fault_location_bus, vector_group,
                                          gen_idx=gen_loc, is_active_current=active_current, gen_mode=mode)
    results = run_test_cases(
        net,
        dataframes["branch" if is_branch else "bus"],
        fault,
        case,
        fault_values,
        lv_tol_percent,
        fault_location_bus,
        branch_results=is_branch
    )
    compare_results(*results)

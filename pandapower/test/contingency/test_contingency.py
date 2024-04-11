# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pandapower as pp
import pandapower.networks
import pandapower.control
import pandapower.timeseries
import pandapower.contingency
import pytest
from pandapower.contingency.contingency import _convert_trafo_phase_shifter

try:
    import lightsim2grid

    lightsim2grid_installed = True
except ImportError:
    lightsim2grid_installed = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_contingency():
    net = pp.networks.case9()

    element_limits = pp.contingency.get_element_limits(net)
    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases)
    pp.contingency.report_contingency_results(element_limits, res)

    pp.contingency.check_elements_within_limits(element_limits, res, True)

    net.line["max_loading_percent"] = np.nan
    net.line.loc[0:5, 'max_loading_percent'] = 70

    element_limits = pp.contingency.get_element_limits(net)
    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases)
    pp.contingency.report_contingency_results(element_limits, res)


def test_contingency_timeseries(get_net):
    nminus1_cases = {element: {"index": get_net[element].index.values}
                     for element in ("line", "trafo") if len(get_net[element]) > 0}

    contingency_functions = [pp.contingency.run_contingency]
    if lightsim2grid_installed:
        contingency_functions = [*contingency_functions, pp.contingency.run_contingency_ls2g]

    for contingency_function in contingency_functions:
        net0 = get_net.deepcopy()
        setup_timeseries(net0)
        ow = net0.output_writer.object.at[0]

        pp.timeseries.run_timeseries(net0, time_steps=range(2), run_control_fct=contingency_function,
                                     nminus1_cases=nminus1_cases,
                                     contingency_evaluation_function=run_for_from_bus_loading)

        # check for the last time step:
        res1 = pp.contingency.run_contingency(net0, nminus1_cases,
                                              contingency_evaluation_function=run_for_from_bus_loading)
        net1 = net0.deepcopy()

        # check for the first time step:
        for c in net0.controller.object.values:
            c.time_step(net0, 0)
            c.control_step(net0)
        res0 = pp.contingency.run_contingency(net0, nminus1_cases,
                                              contingency_evaluation_function=run_for_from_bus_loading)

        for var in ("vm_pu", "max_vm_pu", "min_vm_pu"):
            assert np.allclose(res1["bus"][var], net1.res_bus[var].values, atol=1e-9, rtol=0), var
            assert np.allclose(res0["bus"][var], net0.res_bus[var].values, atol=1e-9, rtol=0), var
            assert np.allclose(res1["bus"][var],
                               ow.output[f"res_bus.{var}"].iloc[-1, :].values, atol=1e-9, rtol=0), var
            assert np.allclose(res0["bus"][var],
                               ow.output[f"res_bus.{var}"].iloc[0, :].values, atol=1e-9, rtol=0), var
        for var in ("loading_percent", "max_loading_percent", "min_loading_percent"):
            for element in ("line", "trafo"):
                if len(net0.trafo) == 0:
                    continue
                assert np.allclose(res1[element][var], net1[f"res_{element}"][var].values, atol=1e-6, rtol=0), var
                assert np.allclose(res0[element][var], net0[f"res_{element}"][var].values, atol=1e-6, rtol=0), var
                assert np.allclose(res1[element][var],
                                   ow.output[f"res_{element}.{var}"].iloc[-1, :].values, atol=1e-6, rtol=0), var
                assert np.allclose(res0[element][var],
                                   ow.output[f"res_{element}.{var}"].iloc[0, :].values, atol=1e-6, rtol=0), var


@pytest.mark.skipif(not lightsim2grid_installed, reason="lightsim2grid package is not installed")
def test_with_lightsim2grid(get_net, get_case):
    net = get_net
    case = get_case
    rng = np.random.default_rng()

    if case == 0:
        nminus1_cases = {"line": {"index": net.line.index.values}}
    elif case == 1 and len(net.trafo) > 0:
        nminus1_cases = {"trafo": {"index": net.trafo.index.values}}
    else:
        nminus1_cases = {element: {"index": rng.choice(net[element].index.values,
                                                       rng.integers(1, len(net[element])), replace=False)}
                         for element in ("line", "trafo") if len(net[element]) > 0}

    net.line.max_loading_percent = 50
    res = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    pp.contingency.run_contingency_ls2g(net, nminus1_cases)

    assert np.array_equal(res["line"]["causes_overloading"], net.res_line.causes_overloading.values)
    if len(net.trafo) > 0:
        assert np.array_equal(res["trafo"]["causes_overloading"], net.res_trafo.causes_overloading.values)

    for s in ("min", "max"):
        assert np.allclose(res["bus"][f"{s}_vm_pu"], net.res_bus[f"{s}_vm_pu"].values, atol=1e-9, rtol=0), s
        assert np.allclose(np.nan_to_num(res["line"][f"{s}_loading_percent"]),
                           net.res_line[f"{s}_loading_percent"].values, atol=1e-6, rtol=0), s
        if len(net.trafo) > 0:
            assert np.allclose(np.nan_to_num(res["trafo"][f"{s}_loading_percent"]),
                               net.res_trafo[f"{s}_loading_percent"].values, atol=1e-6, rtol=0), s


@pytest.mark.skipif(not lightsim2grid_installed, reason="lightsim2grid package is not installed")
def test_lightsim2grid_distributed_slack():
    net = pp.networks.case9()
    net.gen["slack_weight"] = 1
    pp.replace_ext_grid_by_gen(net, slack=True, cols_to_keep=["slack_weight"])
    nminus1_cases = {"line": {"index": np.array([1, 2, 4, 5, 7, 8])}}

    net1 = net.deepcopy()
    net1.gen.loc[~net1.gen.slack, 'slack_weight'] = 0

    res = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading,
                                         distributed_slack=True)
    res1 = pp.contingency.run_contingency(net1, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    pp.contingency.run_contingency_ls2g(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading,
                                        distributed_slack=True)
    pp.contingency.run_contingency_ls2g(net1, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    assert np.array_equal(res["line"]["causes_overloading"], net.res_line.causes_overloading.values)
    if len(net.trafo) > 0:
        assert np.array_equal(res["trafo"]["causes_overloading"], net.res_trafo.causes_overloading.values)

    for var in ("loading_percent", "max_loading_percent", "min_loading_percent"):
        assert np.allclose(res1["line"][var], net1.res_line[var].values, atol=1e-6, rtol=0)
        assert np.allclose(res["line"][var], net.res_line[var].values, atol=1e-6, rtol=0)
    for var in ("vm_pu", "max_vm_pu", "min_vm_pu"):
        assert np.allclose(res1["bus"][var], net1.res_bus[var].values, atol=1e-9, rtol=0)
        assert np.allclose(res["bus"][var], net.res_bus[var].values, atol=1e-9, rtol=0)


@pytest.mark.skipif(not lightsim2grid_installed, reason="lightsim2grid package is not installed")
def test_lightsim2grid_phase_shifters():
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, calculate_voltage_angles=True)
    pp.create_buses(net, 4, 110)
    pp.create_gen(net, 0, 0, slack=True, slack_weight=1)

    pp.create_lines(net, [0, 0], [1, 1], 40, "243-AL1/39-ST1A 110.0", max_loading_percent=100)
    pp.create_transformer_from_parameters(net, 1, 2, 150, 110, 110, 0.5, 10, 15, 0.1, 150,
                                          'hv', 0, 10, -10, 0, 1, 5, True, max_loading_percent=100)
    pp.create_lines(net, [2, 2], [3, 3], 25, "243-AL1/39-ST1A 110.0", max_loading_percent=100)

    pp.create_load(net, 3, 110)

    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    pp.contingency.run_contingency_ls2g(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading,
                                        distributed_slack=True)

    assert net.trafo.shift_degree.values[0] == 150
    assert net.trafo.tap_pos.values[0] == 5
    assert net.trafo.tap_phase_shifter.values[0]

    assert np.array_equal(res["line"]["causes_overloading"], net.res_line.causes_overloading.values)
    if len(net.trafo) > 0:
        assert np.array_equal(res["trafo"]["causes_overloading"], net.res_trafo.causes_overloading.values)

    for var in ("loading_percent", "max_loading_percent", "min_loading_percent"):
        assert np.allclose(res["line"][var], net.res_line[var].values, atol=1e-6, rtol=0)
    for var in ("vm_pu", "max_vm_pu", "min_vm_pu"):
        assert np.allclose(res["bus"][var], net.res_bus[var].values, atol=1e-9, rtol=0)

    pp.runpp(net)
    bus_res = net.res_bus.copy()
    _convert_trafo_phase_shifter(net)
    pp.runpp(net)
    assert_frame_equal(bus_res, net.res_bus)


@pytest.mark.skipif(not lightsim2grid_installed, reason="lightsim2grid package is not installed")
def test_cause_congestion():
    net = pp.networks.case14()
    for c in ("tap_neutral", "tap_step_percent", "tap_pos", "tap_step_degree"):
        net.trafo[c] = 0
    net.trafo.sn_mva /= 200
    net.trafo.vk_percent /= 200
    net.line.max_i_ka /= 100
    net.gen["slack_weight"] = 1
    pp.replace_ext_grid_by_gen(net, slack=True, cols_to_keep=["slack_weight"])

    _randomize_indices(net)

    nminus1_cases = {"line": {"index": net.line.iloc[[4, 2, 1, 5, 7, 8]].index.values},
                     "trafo": {"index": net.trafo.iloc[[2, 1, 0, 4]].index.values}}
    # trafo with iloc index 3 causes 2 disconnected grid areas, which is handled differently by
    # lightsim2grid and pandapower, so the results do not match for the contingency defined by net.trafo.iloc[3]

    pp.contingency.run_contingency_ls2g(net, nminus1_cases,
                                        contingency_evaluation_function=run_for_from_bus_loading)
    res = {"trafo": net.res_trafo.copy(), "line": net.res_line.copy()}
    for element, val in nminus1_cases.items():
        for i in val["index"]:
            net[element].at[i, "in_service"] = False
            run_for_from_bus_loading(net)
            net[element].at[i, 'in_service'] = True
            idx_overloaded_tr = net.res_trafo.loc[net.res_trafo.loading_percent > net.trafo.max_loading_percent].index
            congestion_tr = (net.res_trafo.loc[idx_overloaded_tr, 'loading_percent'].values -
                             net.trafo.loc[idx_overloaded_tr, 'max_loading_percent'].values) * \
                            net.trafo.loc[idx_overloaded_tr, 'sn_mva'].values / 100
            idx_overloaded_ln = net.res_line.loc[net.res_line.loading_percent > net.line.max_loading_percent].index
            congestion_ln = (net.res_line.loc[idx_overloaded_ln, 'loading_percent'].values -
                             net.line.loc[idx_overloaded_ln, 'max_loading_percent'].values) * \
                            net.line.loc[idx_overloaded_ln, 'max_i_ka'].values * \
                            net.bus.loc[net.line.loc[idx_overloaded_ln, 'from_bus'].values, 'vn_kv'].values * \
                            np.sqrt(3) / 100
            if res[element].at[i, "congestion_caused_mva"] == 0:
                assert len(congestion_tr) == 0  # just to be sure...
                assert len(congestion_ln) == 0
            else:
                assert np.allclose(res[element].at[i, "congestion_caused_mva"],
                                   congestion_tr.sum() + congestion_ln.sum(), rtol=0, atol=1e-6)


def test_cause_element_index():
    net = pp.networks.case14()
    for c in ("tap_neutral", "tap_step_percent", "tap_pos", "tap_step_degree"):
        net.trafo[c] = 0
    net.gen["slack_weight"] = 1
    pp.replace_ext_grid_by_gen(net, slack=True, cols_to_keep=["slack_weight"])

    _randomize_indices(net)

    nminus1_cases = {"line": {"index": net.line.iloc[[4, 2, 1, 5, 7, 8]].index.values},
                     "trafo": {"index": net.trafo.iloc[[2, 3, 1, 0, 4]].index.values}}

    _ = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    cause_res_copy_line = net.res_line.copy()
    cause_res_copy_trafo = net.res_trafo.copy()

    check_cause_index(net, nminus1_cases)

    if lightsim2grid_installed:
        pp.contingency.run_contingency_ls2g(net, nminus1_cases,
                                            contingency_evaluation_function=run_for_from_bus_loading)

        columns = ["loading_percent", "max_loading_percent", "min_loading_percent", "causes_overloading",
                   "cause_element", "cause_index"]
        assert_frame_equal(net.res_line[columns], cause_res_copy_line[columns], rtol=0, atol=1e-6, check_dtype=False)
        assert_frame_equal(net.res_trafo[columns], cause_res_copy_trafo[columns], rtol=0, atol=1e-6, check_dtype=False)

        check_cause_index(net, nminus1_cases)


def check_cause_index(net, nminus1_cases):
    """
    This is a not so efficient but very easy to understand auxiliary function to test the "cause element" feature
    that is otherwise complicated to test properly.
    """
    net_copy = net.deepcopy()
    elements_to_check = [e for e in ("line", "trafo") if len(net[e]) > 0]
    for check_element in elements_to_check:
        result_table = net_copy[f"res_{check_element}"]
        # here we iterate over the "to check" elements
        for check_element_index in net[check_element].index.values:
            element_max_loading = 0
            element_cause_index = -1
            cause_element = None
            # here we run the n-1 calculation
            for nminus1_element, nminus1_element_index in nminus1_cases.items():
                for nminus1_idx in nminus1_element_index["index"]:
                    net[nminus1_element].at[nminus1_idx, 'in_service'] = False
                    run_for_from_bus_loading(net)
                    net[nminus1_element].at[nminus1_idx, 'in_service'] = True
                    if net[f"res_{check_element}"].at[check_element_index, 'loading_percent'] > element_max_loading:
                        element_max_loading = net[f"res_{check_element}"].at[check_element_index, 'loading_percent']
                        element_cause_index = nminus1_idx
                        cause_element = nminus1_element

            assert result_table.at[check_element_index, 'cause_index'] == element_cause_index
            assert result_table.at[check_element_index, 'cause_element'] == cause_element
            assert np.isclose(result_table.at[check_element_index, 'max_loading_percent'], element_max_loading,
                              rtol=0, atol=1e-6)


def run_for_from_bus_loading(net, **kwargs):
    pp.runpp(net, **kwargs)
    net.res_line["loading_percent"] = net.res_line.i_from_ka / net.line.max_i_ka * 100
    if len(net.trafo) > 0:
        max_i_ka_limit = net.trafo.sn_mva.values / (net.trafo.vn_hv_kv.values * np.sqrt(3))
        net.res_trafo["loading_percent"] = net.res_trafo.i_hv_ka / max_i_ka_limit * 100


def setup_timeseries(net):
    load_profiles = pd.DataFrame(net.load.p_mw.values * (np.random.random((4, len(net.load))) * 0.4 + 0.8),
                                 index=np.arange(4), columns=net.load.index.values)
    dsl = pp.timeseries.DFData(load_profiles)
    pp.control.ConstControl(net, element="load", variable="p_mw", element_index=net.load.index.values,
                            profile_name=net.load.index.values, data_source=dsl)

    gen_profiles = pd.DataFrame(net.gen.p_mw.values * (np.random.random((4, len(net.gen))) * 0.4 + 0.8),
                                index=np.arange(4), columns=net.gen.index.values)
    dsg = pp.timeseries.DFData(gen_profiles)
    pp.control.ConstControl(net, element="gen", variable="p_mw", element_index=net.gen.index.values,
                            profile_name=net.gen.index.values, data_source=dsg)

    ow = pp.timeseries.OutputWriter(net)
    ow.log_variable("res_bus", "max_vm_pu")
    ow.log_variable("res_bus", "min_vm_pu")
    ow.log_variable("res_line", "max_loading_percent")
    ow.log_variable("res_line", "min_loading_percent")
    if len(net.trafo) > 0:
        ow.log_variable("res_trafo", "loading_percent")
        ow.log_variable("res_trafo", "max_loading_percent")
        ow.log_variable("res_trafo", "min_loading_percent")


def _randomize_indices(net):
    rng = np.random.default_rng()
    for element in ("line", "trafo", "trafo3w"):
        if len(net[element]) == 0:
            continue
        new_index = net[element].index.values + rng.integers(1, 10)
        rng.shuffle(new_index)
        pp.reindex_elements(net, element, new_index)


@pytest.fixture(params=["case9", "case14", "case118"])
def get_net(request):
    # pandapower and lightsim2grid behave differently when the grid becomes isolated from the ext_grid:
    # pandapower selects next gen and uses it as ext_grid, and lightsim2grid does not and therefore has nan for results
    # to circumvent this issue in this test, we add parallel lines to the grid

    net = pp.networks.__dict__[request.param]()
    pp.replace_ext_grid_by_gen(net, slack=True, cols_to_keep=["slack_weight"])

    add_parallel = True

    if add_parallel:
        pp.create_lines_from_parameters(net, net.line.from_bus.values, net.line.to_bus.values,
                                        net.line.length_km.values,
                                        net.line.r_ohm_per_km.values, net.line.x_ohm_per_km.values,
                                        net.line.c_nf_per_km.values,
                                        net.line.max_i_ka.values,
                                        max_loading_percent=net.line.max_loading_percent.values)

        if len(net.trafo) > 0:
            pp.create_transformers_from_parameters(net, net.trafo.hv_bus.values, net.trafo.lv_bus.values,
                                                   net.trafo.sn_mva.values, net.trafo.vn_hv_kv.values,
                                                   net.trafo.vn_lv_kv.values, net.trafo.vkr_percent.values,
                                                   net.trafo.vk_percent.values, net.trafo.pfe_kw.values,
                                                   net.trafo.i0_percent.values,
                                                   max_loading_percent=net.trafo.max_loading_percent.values)
    if len(net.trafo) > 0:
        for col in ("tap_neutral", "tap_step_percent", "tap_pos", "tap_step_degree"):
            net.trafo[col] = net.trafo[col].fillna(0)

    _randomize_indices(net)

    pp.create_continuous_bus_index(net)

    if np.any(net.line.max_i_ka > 10):
        net.line.max_i_ka = 1

    return net


@pytest.fixture(params=[0, 1, 2])
def get_case(request):
    return request.param


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

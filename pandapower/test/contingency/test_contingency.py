# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks
import pandapower.control
import pandapower.timeseries
import pandapower.contingency
import pytest

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

    for contingency_function in (pp.contingency.run_contingency, pp.contingency.run_contingency_ls2g):
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


def test_with_lightsim2grid(get_net, get_case):
    net = get_net
    case = get_case
    rng = np.random.default_rng()

    for element in ("line", "trafo"):
        new_index = net[element].index.values.copy()
        rng.shuffle(new_index)
        pp.reindex_elements(net, element, new_index)

    if case == 0:
        nminus1_cases = {"line": {"index": net.line.index.values}}
    elif case == 1 and len(net.trafo) > 0:
        nminus1_cases = {"trafo": {"index": net.trafo.index.values}}
    else:
        nminus1_cases = {element: {"index": rng.choice(net[element].index.values, rng.integers(1, len(net[element])))}
                         for element in ("line", "trafo") if len(net[element]) > 0}

    res = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    pp.contingency.run_contingency_ls2g(net, nminus1_cases)

    for s in ("min", "max"):
        assert np.allclose(res["bus"][f"{s}_vm_pu"], net.res_bus[f"{s}_vm_pu"].values, atol=1e-9, rtol=0), s
        assert np.allclose(res["line"][f"{s}_loading_percent"],
                           net.res_line[f"{s}_loading_percent"].values, atol=1e-6, rtol=0), s
        if len(net.trafo) > 0:
            assert np.allclose(res["trafo"][f"{s}_loading_percent"],
                               net.res_trafo[f"{s}_loading_percent"].values, atol=1e-6, rtol=0), s


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

    for var in ("loading_percent", "max_loading_percent", "min_loading_percent"):
        assert np.allclose(res1["line"][var], net1.res_line[var].values, atol=1e-6, rtol=0)
        assert np.allclose(res["line"][var], net.res_line[var].values, atol=1e-6, rtol=0)
    for var in ("vm_pu", "max_vm_pu", "min_vm_pu"):
        assert np.allclose(res1["bus"][var], net1.res_bus[var].values, atol=1e-9, rtol=0)
        assert np.allclose(res["bus"][var], net.res_bus[var].values, atol=1e-9, rtol=0)


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
                                        net.line.max_i_ka.values)

        if len(net.trafo) > 0:
            pp.create_transformers_from_parameters(net, net.trafo.hv_bus.values, net.trafo.lv_bus.values,
                                                   net.trafo.sn_mva.values, net.trafo.vn_hv_kv.values,
                                                   net.trafo.vn_lv_kv.values, net.trafo.vkr_percent.values,
                                                   net.trafo.vk_percent.values, net.trafo.pfe_kw.values,
                                                   net.trafo.i0_percent.values)
    if len(net.trafo) > 0:
        for col in ("tap_neutral", "tap_step_percent", "tap_pos", "tap_step_degree"):
            net.trafo[col].fillna(0, inplace=True)

    return net


@pytest.fixture(params=[0, 1, 2])
def get_case(request):
    return request.param


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

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


def test_contingency_timeseries():
    net = pp.networks.case9()
    nminus1_cases = {"line": {"index": net.line.index.values}}

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
    ow.log_variable("res_line", "max_loading_percent")
    ow.log_variable("res_line", "min_loading_percent")
    ow.log_variable("res_bus", "max_vm_pu")
    ow.log_variable("res_bus", "min_vm_pu")

    pp.timeseries.run_timeseries(net, time_steps=range(4), run=pp.contingency.run_contingency,
                                 nminus1_cases=nminus1_cases)

    # check for the last time step:
    res = pp.contingency.run_contingency(net, nminus1_cases)
    net1 = net.deepcopy()

    # check for the first time step:
    for c in net.controller.object.values:
        c.time_step(net, 0)
        c.control_step(net)
    res0 = pp.contingency.run_contingency(net, nminus1_cases)

    for var in ("vm_pu", "max_vm_pu", "min_vm_pu"):
        assert np.array_equal(res["bus"][var], net1.res_bus[var]), var
        assert np.array_equal(res0["bus"][var], net.res_bus[var]), var
        assert np.array_equal(res["bus"][var], ow.output[f"res_bus.{var}"].iloc[-1, :]), var
        assert np.array_equal(res0["bus"][var], ow.output[f"res_bus.{var}"].iloc[0, :]), var
    for var in ("loading_percent", "max_loading_percent", "min_loading_percent"):
        assert np.array_equal(res["line"][var], net1.res_line[var]), var
        assert np.array_equal(res0["line"][var], net.res_line[var]), var
        assert np.array_equal(res["line"][var], ow.output[f"res_line.{var}"].iloc[-1, :]), var
        assert np.array_equal(res0["line"][var], ow.output[f"res_line.{var}"].iloc[0, :]), var


def test_with_lightsim2grid():
    net = pp.networks.case9()

    # pandapower and lightsim2grid behave differently when the grid becomes isolated from the ext_grid:
    # pandapower selects next gen and uses it as ext_grid, and lightsim2grid does not and therefore has nan for results
    # to circumvent this issue in this test, we add parallel lines to the grid

    pp.create_lines_from_parameters(net, net.line.from_bus.values, net.line.to_bus.values, net.line.length_km.values,
                                    net.line.r_ohm_per_km.values, net.line.x_ohm_per_km.values,
                                    net.line.c_nf_per_km.values,
                                    net.line.max_i_ka.values)

    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    pp.contingency.run_contingency_ls2g(net)

    assert np.allclose(res["line"]["max_loading_percent"], net.res_line.max_loading_percent.values, atol=1e-9, rtol=0)
    assert np.allclose(res["line"]["min_loading_percent"], net.res_line.min_loading_percent.values, atol=1e-9, rtol=0)
    assert np.allclose(res["bus"]["max_vm_pu"], net.res_bus.max_vm_pu.values, atol=1e-9, rtol=0)
    assert np.allclose(res["bus"]["min_vm_pu"], net.res_bus.min_vm_pu.values, atol=1e-9, rtol=0)


def test_with_lightsim2grid_trafos():
    net = pp.networks.case14()

    # pandapower and lightsim2grid behave differently when the grid becomes isolated from the ext_grid:
    # pandapower selects next gen and uses it as ext_grid, and lightsim2grid does not and therefore has nan for results
    # to circumvent this issue in this test, we add parallel lines to the grid

    pp.create_lines_from_parameters(net, net.line.from_bus.values, net.line.to_bus.values, net.line.length_km.values,
                                    net.line.r_ohm_per_km.values, net.line.x_ohm_per_km.values,
                                    net.line.c_nf_per_km.values,
                                    net.line.max_i_ka.values)

    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases, contingency_evaluation_function=run_for_from_bus_loading)

    pp.contingency.run_contingency_ls2g(net)

    assert np.allclose(res["line"]["max_loading_percent"], net.res_line.max_loading_percent.values, atol=1e-9, rtol=0)
    assert np.allclose(res["line"]["min_loading_percent"], net.res_line.min_loading_percent.values, atol=1e-9, rtol=0)
    assert np.allclose(res["bus"]["max_vm_pu"], net.res_bus.max_vm_pu.values, atol=1e-9, rtol=0)
    assert np.allclose(res["bus"]["min_vm_pu"], net.res_bus.min_vm_pu.values, atol=1e-9, rtol=0)


def run_for_from_bus_loading(net, **kwargs):
    pp.runpp(net, **kwargs)
    net.res_line["loading_percent"] = net.res_line.i_from_ka / net.line.max_i_ka * 100

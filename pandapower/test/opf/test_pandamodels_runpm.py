# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import copy
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
import numpy as np
import pandas as pd
import pytest
import pandapower as pp
import pandapower.networks as nw
import pandapower.control
import pandapower.timeseries
from copy import deepcopy
from pandapower.converter.pandamodels.to_pm import init_ne_line
from pandapower.test.consistency_checks import consistency_checks
from pandapower.test.helper_functions import add_grid_connection, create_test_line
from pandapower.test.opf.test_basic import net_3w_trafo_opf
from pandapower.converter import convert_pp_to_pm
from pandapower import pp_dir
from pandapower.opf.pm_storage import read_pm_storage_results


try:
    from julia.core import UnsupportedPythonError
except ImportError:
    UnsupportedPythonError = Exception
try:
    from julia import Main
    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False


def create_cigre_grid_with_time_series(json_path, net=None, add_ts_constaints=False):
    if net is None:
        net = nw.create_cigre_network_mv("pv_wind")
        min_vm_pu = 0.95
        max_vm_pu = 1.05

        net["bus"].loc[:, "min_vm_pu"] = min_vm_pu
        net["bus"].loc[:, "max_vm_pu"] = max_vm_pu
        net["line"].loc[:, "max_loading_percent"] = 100.

        # close all switches
        net.switch.loc[:, "closed"] = True
        # add storage to bus 10
        pp.create_storage(net, 10, p_mw=0.5, max_e_mwh=.2, soc_percent=0., q_mvar=0., controllable=True)

    # set the load type in the cigre grid, since it is not specified
    net["load"].loc[:, "type"] = "residential"

    # set the sgen type in the cigre grid
    net.sgen.loc[:, "type"] = "pv"
    net.sgen.loc[8, "type"] = "wind"

    # read the example time series
    time_series = pd.read_json(json_path)
    time_series.sort_index(inplace=True)

    # this example time series has a 15min resolution with 96 time steps for one day
    n_timesteps = time_series.shape[0]

    # get rated power
    load_p = net["load"].loc[:, "p_mw"].values
    sgen_p = net["sgen"].loc[:7, "p_mw"].values
    wind_p = net["sgen"].loc[8, "p_mw"]

    load_ts = pd.DataFrame(index=time_series.index.tolist(), columns=net.load.index.tolist())
    sgen_ts = pd.DataFrame(index=time_series.index.tolist(), columns=net.sgen.index.tolist())
    for t in range(n_timesteps):
        load_ts.loc[t] = load_p * time_series.at[t, "residential"]
        sgen_ts.loc[t][:8] = sgen_p * time_series.at[t, "pv"]
        sgen_ts.loc[t][8] = wind_p * time_series.at[t, "wind"]

    # create time series controller for load and sgen
    ConstControl(net, element="load", variable="p_mw",
                 element_index=net.load.index.tolist(), profile_name=net.load.index.tolist(),
                 data_source=DFData(load_ts))
    ConstControl(net, element="sgen", variable="p_mw",
                 element_index=net.sgen.index.tolist(), profile_name=net.sgen.index.tolist(),
                 data_source=DFData(sgen_ts))

    if add_ts_constaints:
        df_qmax, df_qmin = sgen_ts.copy(), sgen_ts.copy()
        df_qmax[df_qmax.columns] = net.sgen.max_q_mvar
        df_qmin[df_qmin.columns] = net.sgen.min_q_mvar

        ConstControl(net, element="sgen", variable="max_p_mw",
                      element_index=net.sgen.index.tolist(), profile_name=net.sgen.index.tolist(),
                      data_source=DFData(sgen_ts))
        ConstControl(net, element="sgen", variable="min_p_mw",
                      element_index=net.sgen.index.tolist(), profile_name=net.sgen.index.tolist(),
                      data_source=DFData(sgen_ts))
        ConstControl(net, element="sgen", variable="max_q_mvar",
                      element_index=net.sgen.index.tolist(), profile_name=net.sgen.index.tolist(),
                      data_source=DFData(df_qmax))
        ConstControl(net, element="sgen", variable="min_q_mvar",
                      element_index=net.sgen.index.tolist(), profile_name=net.sgen.index.tolist(),
                      data_source=DFData(df_qmin))

    return net


def assert_pf(net, dc=False):
    if dc:
        model = "DCMPPowerModel"
    else:
        model = "ACPPowerModel"

    pp.runpm_pf(net, pm_model=model)
    va_pm = copy.deepcopy(net.res_bus.va_degree)
    vm_pm = copy.deepcopy(net.res_bus.vm_pu)

    if dc:
        pp.rundcpp(net, calculate_voltage_angles=True)
    else:
        pp.runpp(net, calculate_voltage_angles=True)

    va_pp = copy.deepcopy(net.res_bus.va_degree)
    vm_pp = copy.deepcopy(net.res_bus.vm_pu)

    assert np.allclose(va_pm, va_pp)

    if not dc:
        assert np.allclose(vm_pm, vm_pp)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_ac_powerflow_simple():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=False)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_dc_powerflow_simple():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=True)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_ac_powerflow_shunt():
    net = nw.simple_four_bus_system()
    pp.create_shunt(net, 2, q_mvar=-0.5)
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=False)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_dc_powerflow_shunt():
    net = nw.simple_four_bus_system()
    pp.create_shunt(net, 2, q_mvar=-0.5)
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=True)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_ac_powerflow_tap():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 30.
    net.trafo.loc[0, "tap_pos"] = -2.
    assert_pf(net, dc=False)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_dc_powerflow_tap():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=True)

    net.trafo.loc[0, "shift_degree"] = 30.
    net.trafo.loc[0, "tap_pos"] = -2.
    assert_pf(net, dc=True)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_compare_pwl_and_poly(net_3w_trafo_opf):
    net = net_3w_trafo_opf
    net.ext_grid.loc[:, "min_p_mw"] = -999.
    net.ext_grid.loc[:, "max_p_mw"] = 999.
    net.ext_grid.loc[:, "max_q_mvar"] = 999.
    net.ext_grid.loc[:, "min_q_mvar"] = -999.
    pp.create_pwl_cost(net, 0, 'ext_grid', [[0, 1, 1]])
    pp.create_pwl_cost(net, 0, 'gen', [[0, 30, 3], [30, 80, 3]])
    pp.create_pwl_cost(net, 1, 'gen', [[0, 80, 2]])
    net.bus.loc[:, "max_vm_pu"] = 1.1
    net.bus.loc[:, "min_vm_pu"] = .9
    pp.runpm_ac_opf(net)
    consistency_checks(net)

    p_gen = net.res_gen.p_mw.values
    q_gen = net.res_gen.q_mvar.values
    vm_bus = net.res_bus.vm_pu.values
    va_bus = net.res_bus.va_degree.values

    net.pwl_cost = net.pwl_cost.drop(net.pwl_cost.index)

    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=1)
    pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=3)
    pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=2)

    # pp.runopp(net)
    pp.runpm_ac_opf(net, correct_pm_network_data=False)
    consistency_checks(net)

    np.allclose(p_gen, net.res_gen.p_mw.values)
    np.allclose(q_gen, net.res_gen.q_mvar.values)
    np.allclose(vm_bus, net.res_bus.vm_pu.values)
    np.allclose(va_bus, net.res_bus.va_degree.values)

    # pp.rundcopp(net)
    pp.runpm_dc_opf(net, correct_pm_network_data=False)
    consistency_checks(net, test_q=False)

    np.allclose(p_gen, net.res_gen.p_mw.values)
    np.allclose(va_bus, net.res_bus.va_degree.values)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pwl():
    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus2 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus3 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)

    # create 110 kV lines
    pp.create_line(net, bus1, bus2, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus2, bus3, length_km=50., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=80, controllable=False)

    # create generators
    g1 = pp.create_gen(net, bus1, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01, slack=True)
    g2 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
    #    net.gen["controllable"] = False

    pp.create_pwl_cost(net, g1, 'gen', [[0, 2, 2], [2, 80, 5]])
    pp.create_pwl_cost(net, g2, 'gen', [[0, 2, 2], [2, 80, 5]])

    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.iloc[0], net.res_gen.p_mw.iloc[1])
    assert np.isclose(net.res_gen.q_mvar.iloc[0], net.res_gen.q_mvar.iloc[1])

    net.pwl_cost = net.pwl_cost.drop(net.pwl_cost.index)
    g3 = pp.create_gen(net, bus1, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)

    pp.create_pwl_cost(net, g1, 'gen', [[0, 2, 1.], [2, 80, 8.]])
    pp.create_pwl_cost(net, g2, 'gen', [[0, 3, 2.], [3, 80, 14]])
    pp.create_pwl_cost(net, g3, 'gen', [[0, 1, 3.], [1, 80, 10.]])

    net.load.p_mw = 1
    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g2], 0)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1], atol=1e-4)

    net.load.p_mw = 3
    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_gen.p_mw.at[g1], 2)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] + net.res_gen.p_mw.at[g2] * 2, atol=1e-4)

    net.load.p_mw = 5
    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g1], 2)
    assert np.isclose(net.res_gen.p_mw.at[g2], 3)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] + net.res_gen.p_mw.at[g2] * 2 +
                      net.res_gen.p_mw.at[g3] * 3, atol=1e-4)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_without_ext_grid():
    net = pp.create_empty_network()

    min_vm_pu = 0.95
    max_vm_pu = 1.05

    # create buses
    bus1 = pp.create_bus(net, vn_kv=220., geodata=(5, 9))
    bus2 = pp.create_bus(net, vn_kv=110., geodata=(6, 10), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus3 = pp.create_bus(net, vn_kv=110., geodata=(10, 9), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus4 = pp.create_bus(net, vn_kv=110., geodata=(8, 8), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus5 = pp.create_bus(net, vn_kv=110., geodata=(6, 8), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)

    # create 220/110/110 kV 3W-transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vk_hv_percent=10., vk_mv_percent=10.,
                                            vk_lv_percent=10., vkr_hv_percent=0.5,
                                            vkr_mv_percent=0.5, vkr_lv_percent=0.5, pfe_kw=100,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_mva=100, sn_mv_mva=50, sn_lv_mva=50)

    # create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=60, controllable=False)
    pp.create_load(net, bus3, p_mw=70, controllable=False)
    pp.create_load(net, bus4, p_mw=10, controllable=False)

    # create generators
    g1 = pp.create_gen(net, bus1, p_mw=40, min_p_mw=0, min_q_mvar=-20, max_q_mvar=20, slack=True, min_vm_pu=min_vm_pu,
                       max_vm_pu=max_vm_pu)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=1000)

    g2 = pp.create_gen(net, bus3, p_mw=40, min_p_mw=0, min_q_mvar=-20, max_q_mvar=20, vm_pu=1.01,
                       min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu, max_p_mw=40.)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=2000)

    g3 = pp.create_gen(net, bus4, p_mw=0.050, min_p_mw=0, min_q_mvar=-20, max_q_mvar=20, vm_pu=1.01,
                       min_vm_pu=min_vm_pu,
                       max_vm_pu=max_vm_pu, max_p_mw=0.05)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_mw=3000)

    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g2], 0, atol=1e-5, rtol=1e-5)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0, atol=1e-5, rtol=1e-5)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] * 1e3)
    net.trafo3w["max_loading_percent"] = 150.

    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert 149. < net.res_trafo3w.loading_percent.values[0] < 150.01
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] * 1e3 + net.res_gen.p_mw.at[g2] * 2e3)

    pp.runpm_dc_opf(net)
    consistency_checks(net, rtol=1e-3, test_q=False)
    assert 149. < net.res_trafo3w.loading_percent.values[0] < 150.01
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] * 1e3 + net.res_gen.p_mw.at[g2] * 2e3)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_multiple_ext_grids():
    net = pp.create_empty_network()
    # generate three ext grids
    b11, b12, l11 = add_grid_connection(net, vn_kv=110.)
    b21, b22, l21 = add_grid_connection(net, vn_kv=110.)
    b31, b32, l31 = add_grid_connection(net, vn_kv=110.)
    # connect them
    create_test_line(net, b12, b22)
    create_test_line(net, b22, b32)

    # create load and sgen to optimize
    pp.create_load(net, b12, p_mw=60)

    g3 = pp.create_sgen(net, b12, p_mw=50, min_p_mw=20, max_p_mw=200, controllable=True)
    pp.create_poly_cost(net, g3, 'sgen', cp1_eur_per_mw=10.)
    # set positive costs for ext_grid -> minimize ext_grid usage
    ext_grids = net.ext_grid.index
    net["ext_grid"].loc[0, "vm_pu"] = .99
    net["ext_grid"].loc[1, "vm_pu"] = 1.0
    net["ext_grid"].loc[2, "vm_pu"] = 1.01
    for idx in ext_grids:
        # eg = net["ext_grid"].loc[idx]
        pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=10.)

    pp.runpm_ac_opf(net)
    assert np.allclose(net.res_sgen.loc[0, "p_mw"], 60.)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_voltage_angles():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    b5 = pp.create_bus(net, vn_kv=10., in_service=False)
    tidx = pp.create_transformer3w(
        net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV', max_loading_percent=120)
    pp.create_load(net, b3, p_mw=5, controllable=False)
    load_id = pp.create_load(net, b4, p_mw=5, controllable=True, max_p_mw=25, min_p_mw=0, min_q_mvar=-1e-6,
                             max_q_mvar=1e-6)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=1)
    pp.create_poly_cost(net, load_id, "load", cp1_eur_per_mw=1000)
    net.trafo3w.at[tidx, "shift_lv_degree"] = 10
    net.trafo3w.at[tidx, "shift_mv_degree"] = 30
    net.bus.loc[:, "max_vm_pu"] = 1.1
    net.bus.loc[:, "min_vm_pu"] = .9

    # load is zero since costs are high. PF results should be the same as OPF
    net.load.loc[1, "p_mw"] = 0.

    pp.runpp(net, calculate_voltage_angles=True)
    va_degree = net.res_bus.loc[:, "va_degree"].values
    vm_pu = net.res_bus.loc[:, "vm_pu"].values
    loading3w = net.res_trafo3w.loc[:, "loading_percent"].values

    net_opf = copy.deepcopy(net)
    pp.runpm_ac_opf(net_opf)

    assert 30. < (net_opf.res_bus.va_degree.at[b1] - net_opf.res_bus.va_degree.at[b3]) % 360 < 32.
    assert 10. < (net_opf.res_bus.va_degree.at[b1] - net_opf.res_bus.va_degree.at[b4]) % 360 < 11.
    assert np.isnan(net_opf.res_bus.va_degree.at[b5])

    assert np.allclose(net_opf.res_bus.va_degree.values, va_degree, atol=1e-6, rtol=1e-6, equal_nan=True)
    assert np.allclose(net_opf.res_bus.vm_pu.values, vm_pu, atol=1e-6, rtol=1e-6, equal_nan=True)
    assert np.allclose(net_opf.res_trafo3w.loading_percent, loading3w, atol=1e-2, rtol=1e-2, equal_nan=True)


def tnep_grid():
    net = pp.create_empty_network()

    min_vm_pu = 0.95
    max_vm_pu = 1.05

    # create buses
    bus1 = pp.create_bus(net, vn_kv=110., geodata=(5, 9), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus2 = pp.create_bus(net, vn_kv=110., geodata=(6, 10), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus3 = pp.create_bus(net, vn_kv=110., geodata=(10, 9), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus4 = pp.create_bus(net, vn_kv=110., geodata=(8, 8), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)

    # create 110 kV lines
    pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus1, bus3, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus1, bus4, length_km=100., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=60)
    pp.create_load(net, bus3, p_mw=70)
    pp.create_load(net, bus4, p_mw=50)

    # create generators
    g1 = pp.create_gen(net, bus1, p_mw=9.513270, min_p_mw=0, max_p_mw=200, vm_pu=1.01, slack=True)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=1)

    g2 = pp.create_gen(net, bus2, p_mw=78.403291, min_p_mw=0, max_p_mw=200, vm_pu=1.01)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=3)

    g3 = pp.create_gen(net, bus3, p_mw=92.375601, min_p_mw=0, max_p_mw=200, vm_pu=1.01)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_mw=3)

    net.line["max_loading_percent"] = 20

    # possible new lines (set out of service in line DataFrame)
    l1 = pp.create_line(net, bus1, bus4, 10., std_type="305-AL1/39-ST1A 110.0", name="new_line1",
                        max_loading_percent=20., in_service=False)
    l2 = pp.create_line(net, bus2, bus4, 20., std_type="149-AL1/24-ST1A 110.0", name="new_line2",
                        max_loading_percent=20., in_service=False)
    l3 = pp.create_line(net, bus3, bus4, 30., std_type='149-AL1/24-ST1A 110.0', name="new_line3",
                        max_loading_percent=20., in_service=False)
    l4 = pp.create_line(net, bus3, bus4, 40., std_type='149-AL1/24-ST1A 110.0', name="new_line4",
                        max_loading_percent=20., in_service=False)

    new_line_index = [l1, l2, l3, l4]
    construction_costs = [10., 20., 30., 45.]
    # create new line dataframe
    init_ne_line(net, new_line_index, construction_costs)

    return net


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_pm_tnep():
    net = tnep_grid()
    # check if max line loading percent is violated (should be)
    pp.runpp(net)
    assert np.any(net["res_line"].loc[:, "loading_percent"] >
                  net["line"].loc[:, "max_loading_percent"])

    # run power models tnep optimization
    pp.runpm_tnep(net, pm_model="ACPPowerModel")
    # set lines to be built in service
    lines_to_built = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
    net["line"].loc[lines_to_built, "in_service"] = True
    # run a power flow calculation again and check if max_loading percent is still violated
    pp.runpp(net)
    # check max line loading results
    assert not np.any(net["res_line"].loc[:, "loading_percent"] >
                      net["line"].loc[:, "max_loading_percent"])


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
# @pytest.mark.xfail(reason="OTS does not correctly consider net.sn_mva. Probably the impedances [pu]"
#                    " are not correctly calculated.")
def test_ots_opt():
    net = nw.case5()
    net.sn_mva = 1.
    branch_status = net["line"].loc[:, "in_service"].values
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1]).astype(bool), branch_status.astype(bool))
    pp.runpm_ots(net)
    branch_status = net["res_line"].loc[:, "in_service"].values
    pp.runpp(net)
    net.line.loc[:, "in_service"] = branch_status.astype(bool)
    pp.runpp(net)
    try:
        assert np.array_equal(np.array([1, 1, 1, 0, 1, 0]).astype(bool), branch_status.astype(bool))
    except AssertionError:
        assert np.array_equal(np.array([0, 1, 1, 1, 1, 0]).astype(bool), branch_status.astype(bool))


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
@pytest.mark.xfail(reason="not complited yet")
def test_timeseries_pandamodels():
    profiles = pd.DataFrame()
    n_timesteps = 3
    profiles['load1'] = np.random.random(n_timesteps) * 2e1
    ds = pp.timeseries.DFData(profiles)

    net = nw.simple_four_bus_system()
    time_steps = range(3)
    pp.control.ConstControl(net, 'load', 'p_mw', element_index=0, data_source=ds, profile_name='load1',
                            scale_factor=0.85)
    net.load['controllable'] = False
    pp.timeseries.run_timeseries(net, time_steps, continue_on_divergence=True, verbose=False, recycle=False,
                                 run=pp.runpm_dc_opf)


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_runpm_vstab():
    net = nw.create_cigre_network_mv(with_der="pv_wind")
    net.sgen.p_mw = net.sgen.p_mw * 8
    net.sgen.sn_mva = net.sgen.sn_mva * 8
    pp.runpp(net)
    net_org = deepcopy(net)

    net.load['controllable'] = False
    net.sgen['controllable'] = True
    net.sgen["max_p_mw"] = net.sgen.p_mw.values
    net.sgen["min_p_mw"] = net.sgen.p_mw.values
    net.sgen["max_q_mvar"] = net.sgen.p_mw.values * 0.328
    net.sgen["min_q_mvar"] = -net.sgen.p_mw.values * 0.328
    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9
    net.ext_grid["max_q_mvar"] = 10000.0
    net.ext_grid["min_q_mvar"] = -10000.0
    net.ext_grid["max_p_mw"] = 10000.0
    net.ext_grid["min_p_mw"] = -10000.0
    net.gen["max_p_mw"] = net.gen.p_mw.values
    net.gen["min_p_mw"] = net.gen.p_mw.values
    net.gen["max_q_mvar"] = 10000.0
    net.gen["min_q_mvar"] = -10000.0
    net.trafo["max_loading_percent"] = 500.0
    net.line["max_loading_percent"] = 500.0

    for idx in net.sgen.index:
        pp.create_poly_cost(net, idx, "sgen", 1.0)
    for idx in net.gen.index:
        pp.create_poly_cost(net, idx, "gen", 1.0)
    for idx in net.ext_grid.index:
        pp.create_poly_cost(net, idx, "ext_grid", 1.0)

    net.bus["pm_param/setpoint_v"] = None
    net.bus["pm_param/setpoint_v"].loc[net.sgen.bus] = 0.99

    pp.runpm_vstab(net)

    assert np.allclose(net.res_bus.vm_pu[net.sgen.bus], 0.99, atol=1e-2, rtol=1e-2)
    assert np.not_equal(net_org.res_sgen.q_mvar.values.all(), net.res_sgen.q_mvar.values.all())


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_storage_opt():
    json_path = os.path.join(pp_dir, "test", "opf", "cigre_timeseries_15min.json")

    net = create_cigre_grid_with_time_series(json_path)
    pm = convert_pp_to_pm(net, from_time_step=5, to_time_step=26)
    assert "gen_and_controllable_sgen" not in pm["user_defined_params"]
    assert len(
        pm["time_series"]["gen"].keys()) == 0  # because all sgen are not controllable, they are treated as loads.
    assert len(pm["time_series"]["load"].keys()) == len(net.load) + len(net.sgen)
    assert set(pm["time_series"]["load"]["1"]["p_mw"].keys()) == set([str(i) for i in range(5, 26)])

    net = create_cigre_grid_with_time_series(json_path)
    pp.runpm_storage_opf(net, from_time_step=0, to_time_step=5)
    storage_results_1 = read_pm_storage_results(net)
    assert net._pm_org_result["multinetwork"]
    assert net._pm["pm_solver"] == "juniper"
    assert net._pm["pm_mip_solver"] == "cbc"
    assert len(net.res_ts_opt) == 5


    net2 = create_cigre_grid_with_time_series(json_path)
    net2.sn_mva = 100.0
    pp.runpm_storage_opf(net2, from_time_step=0, to_time_step=5)
    storage_results_100 = read_pm_storage_results(net2)

    assert abs(storage_results_100[0].values - storage_results_1[0].values).max() < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_runpm_multi_vstab():
    net = nw.create_cigre_network_mv(with_der="pv_wind")
    net.load['controllable'] = False
    net.sgen['controllable'] = True
    # lower and upper bounds for buses
    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9

    # lower and upper bounds for external grid
    net.ext_grid["max_q_mvar"] = 10000.0
    net.ext_grid["min_q_mvar"] = -10000.0
    net.ext_grid["max_p_mw"] = 10000.0
    net.ext_grid["min_p_mw"] = -10000.0

    # lower and upper bounds for DERs
    net.sgen["max_p_mw"] = net.sgen.p_mw.values
    net.sgen["min_p_mw"] = net.sgen.p_mw.values
    net.sgen["max_q_mvar"] = net.sgen.p_mw.values * 0.328
    net.sgen["min_q_mvar"] = -net.sgen.p_mw.values * 0.328

    net.trafo["max_loading_percent"] = 100.0
    net.line["max_loading_percent"] = 100.0

    net.bus["pm_param/setpoint_v"] = None # add extra column
    net.bus["pm_param/setpoint_v"].loc[net.sgen.bus] = 0.96

    # load time series data for 96 time steps
    json_path = os.path.join(pp_dir, "test", "opf", "cigre_timeseries_15min.json")
    create_cigre_grid_with_time_series(json_path, net, True)

    # run time series opf
    pp.runpm_multi_vstab(net, from_time_step=0, to_time_step=96)
    assert len(net.res_ts_opt) == 96

    # get opf-results
    y_multi = []
    for t in range(96):
        y_multi.append(net.res_ts_opt[str(t)].res_bus.vm_pu[net.sgen.bus].values.mean() - 0.96)

    assert np.array(y_multi).max() < 0.018
    assert np.array(y_multi).min() > -0.002


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_runpm_qflex_and_multi_qflex():

    net = nw.create_cigre_network_mv(with_der="pv_wind")
    pp.runpp(net)

    net.load['controllable'] = False
    net.sgen['controllable'] = True
    # lower and upper bounds for buses
    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9

    # lower and upper bounds for external grid
    net.ext_grid["max_q_mvar"] = 10000.0
    net.ext_grid["min_q_mvar"] = -10000.0
    net.ext_grid["max_p_mw"] = 10000.0
    net.ext_grid["min_p_mw"] = -10000.0

    # lower and upper bounds for DERs
    net.sgen["max_p_mw"] = net.sgen.p_mw.values
    net.sgen["min_p_mw"] = net.sgen.p_mw.values
    net.sgen["max_q_mvar"] = net.sgen.p_mw.values * 0.828
    net.sgen["min_q_mvar"] = -net.sgen.p_mw.values * 0.828

    net.trafo["max_loading_percent"] = 100.0
    net.line["max_loading_percent"] = 100.0

    net.trafo["pm_param/setpoint_q"] = None # add extra column
    net.trafo["pm_param/setpoint_q"].loc[0] = -5
    net.trafo["pm_param/side"] = None
    net.trafo["pm_param/side"][0] = "lv"

    # run opf
    pp.runpm_qflex(net)
    opt_q = net.res_trafo.q_lv_mvar[0]
    assert  abs(opt_q + 5) < 1e-6

    # test for multi_qflex
    # load time series data for 96 time steps
    json_path = os.path.join(pp_dir, "test", "opf", "cigre_timeseries_15min.json")
    create_cigre_grid_with_time_series(json_path, net, True)
    pp.runpm_multi_qflex(net, from_time_step=0, to_time_step=96)
    # get opf-results
    y_multi = []
    for t in range(96):
        y_multi.append(abs(abs(net.res_ts_opt[str(t)].res_trafo.q_lv_mvar[0])-5))
    assert np.array(y_multi).max() < 1e-6


@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_runpm_ploss_loading():
    net = nw.create_cigre_network_mv(with_der="pv_wind")
    net.load['controllable'] = False
    net.sgen['controllable'] = True
    net.sgen["max_p_mw"] = net.sgen.p_mw.values
    net.sgen["min_p_mw"] = net.sgen.p_mw.values
    net.sgen["max_q_mvar"] = net.sgen.p_mw.values * 0.328
    net.sgen["min_q_mvar"] = -net.sgen.p_mw.values * 0.328
    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9
    net.ext_grid["max_q_mvar"] = 10000.0
    net.ext_grid["min_q_mvar"] = -10000.0
    net.ext_grid["max_p_mw"] = 10000.0
    net.ext_grid["min_p_mw"] = -10000.0
    net.trafo["max_loading_percent"] = 100.0
    net.line["max_loading_percent"] = 100.0
    net.line["pm_param/target_branch"] = True
    net.switch.loc[:, "closed"] = True
    pp.runpp(net)
    net_org = deepcopy(net)
    pp.runpm_ploss(net)

    ### test loss reduction with Q-optimierung
    assert net.res_line.pl_mw.values.sum() < net_org.res_line.pl_mw.values.sum()

    net.line = net.line.drop(columns=["pm_param/target_branch"])
    net.trafo["pm_param/target_branch"] = True
    pp.runpm_ploss(net)

    assert net.res_trafo.pl_mw.values.sum() < net_org.res_trafo.pl_mw.values.sum()

    ### test loading reduction with Q-optimierung
    net = deepcopy(net_org)
    pp.runpm_loading(net)
    assert (net.res_line.loading_percent.values - \
            net_org.res_line.loading_percent.values).sum() < 0


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_convergence_dc_opf():
    for cpnd in [True, False]:
        net = nw.case5()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case9()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case14()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case30()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case39()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case57()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case118()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case145()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)
        net = nw.case300()
        pp.runpm_dc_opf(net, correct_pm_network_data=cpnd)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_ac_opf_differnt_snmva():
    net = nw.case9()
    res = pd.DataFrame(columns=net.bus.index.tolist())
    for i, snmva in enumerate([1, 13, 45, 78, 98, 100]):
        net.sn_mva = snmva
        pp.runpm_ac_opf(net)
        res.loc[i] = net.res_bus.vm_pu.values
    for i in res.columns:
        assert res[i].values.min() - res[i].values.max() < 1e-10


if __name__ == '__main__':
    pytest.main(['-x', __file__])

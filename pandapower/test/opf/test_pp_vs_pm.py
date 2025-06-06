# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pytest
from numpy import array

from pandapower.converter.pypower import from_ppc
from pandapower.create import create_ext_grid
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_bus import BUS_I, VMAX, VMIN, BUS_TYPE, REF
from pandapower.run import runpp, runopp
from pandapower.runpm import runpm_ac_opf

try:
    from julia.core import UnsupportedPythonError
except ImportError:
    UnsupportedPythonError = Exception
try:
    from julia.api import Julia

    Julia(compiled_modules=False)
    from julia import Main

    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# test data from :https://github.com/lanl-ansi/PowerModels.jl/blob/master/test/data/matpower/case5_clm.m
def case5_pm_matfile_i():
    mpc = {"branch": array([
        [1, 2, 0.00281, 0.0281, 0.00712, 400.0, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
        [1, 4, 0.00304, 0.0304, 0.00658, 426, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
        [1, 10, 0.00064, 0.0064, 0.03126, 426, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
        [2, 3, 0.00108, 0.0108, 0.01852, 426, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
        [3, 4, 0.00297, 0.0297, 0.00674, 426, 0.0, 0.0, 1.05, 1.0, 1, -30.0, 30.0],
        [3, 4, 0.00297, 0.0297, 0.00674, 426, 0.0, 0.0, 1.05, - 1.0, 1, -30.0, 30.0],
        [4, 10, 0.00297, 0.0297, 0.00674, 240.0, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
    ]), "bus": array([
        [1, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.00000, 2.80377, 230.0, 1, 1.10000, 0.90000],
        [2, 1, 300.0, 98.61, 0.0, 0.0, 1, 1.08407, 0.73465, 230.0, 1, 1.10000, 0.90000],
        [3, 2, 300.0, 98.61, 0.0, 0.0, 1, 1.00000, 0.55972, 230.0, 1, 1.10000, 0.90000],
        [4, 3, 400.0, 131.47, 0.0, 0.0, 1, 1.00000, 0.00000, 230.0, 1, 1.10000, 0.90000],
        [10, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.00000, 3.59033, 230.0, 1, 1.10000, 0.90000],
    ]), "gen": array([
        [1, 40.0, 30.0, 30.0, -30.0, 1.07762, 100.0, 1, 40.0, 0.0],
        [1, 170.0, 127.5, 127.5, -127.5, 1.07762, 100.0, 1, 170.0, 0.0],
        [3, 324.49, 390.0, 390.0, -390.0, 1.1, 100.0, 1, 520.0, 0.0],
        [4, 0.0, -10.802, 150.0, -150.0, 1.06414, 100.0, 1, 200.0, 0.0],
        [10, 470.69, -165.039, 450.0, -450.0, 1.06907, 100.0, 1, 600.0, 0.0],
    ]), "gencost": array([
        [2, 0.0, 0.0, 3, 0.000000, 14.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 15.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 30.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 40.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 10.000000, 0.000000],
    ]), "version": 2, "baseMVA": 100.0}

    net = from_ppc(mpc, f_hz=50)

    return net


def test_case5_pm_pd2ppc():
    # load net
    net = case5_pm_matfile_i()
    # run pd2ppc with ext_grid controllable = False
    runpp(net)
    net.ext_grid = net.ext_grid.drop(columns=['controllable'])
    assert "controllable" not in net.ext_grid
    net["_options"]["mode"] = "opf"
    ppc = _pd2ppc(net)
    # check which one is the ref bus in ppc
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF].item())
    vmax = ppc[0]["bus"][ref_idx, VMAX]
    vmin = ppc[0]["bus"][ref_idx, VMIN]

    assert net.ext_grid.vm_pu[0] == vmin
    assert net.ext_grid.vm_pu[0] == vmax

    # run pd2ppc with ext_grd controllable = True
    net.ext_grid["controllable"] = True
    ppc = _pd2ppc(net)
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF].item())
    vmax = ppc[0]["bus"][ref_idx, VMAX]
    vmin = ppc[0]["bus"][ref_idx, VMIN]

    assert net.bus.min_vm_pu[net.ext_grid.bus].values[0] == vmin
    assert net.bus.max_vm_pu[net.ext_grid.bus].values[0] == vmax

    assert net.ext_grid["in_service"].values.dtype == bool
    assert net.ext_grid["bus"].values.dtype == "uint32"
    create_ext_grid(net, bus=4, vm_pu=net.res_bus.vm_pu.loc[4], controllable=False)

    assert net.ext_grid["bus"].values.dtype == "uint32"
    assert net.ext_grid["in_service"].values.dtype == bool

    ppc = _pd2ppc(net)
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF].item())

    vmax1 = ppc[0]["bus"][ref_idx, VMAX]
    vmin1 = ppc[0]["bus"][ref_idx, VMIN]

    assert net.ext_grid.vm_pu.values[1] == vmin1
    assert net.ext_grid.vm_pu.values[1] == vmax1


def test_opf_ext_grid_controllable():
    # load net
    net = case5_pm_matfile_i()
    net_old = copy.deepcopy(net)
    net_new = copy.deepcopy(net)
    # run pd2ppc with ext_grid controllable = False
    runopp(net_old, delta=1e-12)
    net_new.ext_grid["controllable"] = True
    runopp(net_new, delta=1e-12)
    eg_bus = net.ext_grid.bus.at[0]
    assert np.isclose(net_old.res_bus.vm_pu[eg_bus], 1.06414000007302)
    assert np.isclose(net_new.res_bus.vm_pu[eg_bus], net_new.res_bus.vm_pu[eg_bus])
    assert np.abs(net_new.res_cost - net_old.res_cost) / net_old.res_cost < 4.5e-3


# todo: it is unclear what is tested here, a fix and some additional comments are necessary
@pytest.mark.xfail
def test_opf_create_ext_grid_controllable():
    # load net
    net = case5_pm_matfile_i()
    # run pd2ppc with ext_grid controllable = False
    create_ext_grid(net, bus=1, controllable=True)
    # create_ext_grid(net, bus=4, controllable=True, min_p_mw=0, max_p_mw=200, min_q_mvar=-150, max_q_mvar=150)
    runopp(net)
    assert np.isclose(net.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0641399999827315)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_opf_ext_grid_controllable_pm():
    # load net
    net = case5_pm_matfile_i()

    net_old = copy.deepcopy(net)
    runpp(net_old)
    runpm_ac_opf(net_old, calculate_voltage_angles=True, correct_pm_network_data=False, opf_flow_lim="I")

    net_new = copy.deepcopy(net)
    net_new.ext_grid["controllable"] = True
    runpp(net_new)
    runpm_ac_opf(net_new, calculate_voltage_angles=True, correct_pm_network_data=False,
                 opf_flow_lim="I")

    eg_bus = net.ext_grid.bus.at[0]
    assert np.isclose(net_old.res_bus.vm_pu[eg_bus], 1.06414000007302)
    assert np.abs(net_new.res_bus.vm_pu[eg_bus] - net_new.res_bus.vm_pu[eg_bus]) < 0.0058
    assert np.abs(net_new.res_cost - net_old.res_cost) / net_old.res_cost < 1e-2


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

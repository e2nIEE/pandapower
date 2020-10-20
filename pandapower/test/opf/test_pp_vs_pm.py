# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.pypower.idx_bus import BUS_I, VMAX, VMIN, BUS_TYPE, REF
import numpy as np
import pytest
import copy
from numpy import array
from pandapower.converter.pypower import from_ppc
import pandapower as pp
from pandapower.pd2ppc import _pd2ppc

try:
    from julia.core import UnsupportedPythonError
except ImportError:
    UnsupportedPythonError = Exception
try:
    from julia import Main

    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False
    print(e)

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def case5_pm_matfile_I():
    
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
    

@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_case5_pm_pd2ppc():

    # load net
    net = case5_pm_matfile_I()
    # run pd2ppc with ext_grid controllable = False
    pp.runpp(net)
    assert "controllable" not in net.ext_grid
    net["_options"]["mode"] = "opf"
    ppc = _pd2ppc(net)
    # check which one is the ref bus in ppc
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF])
    vmax = ppc[0]["bus"][ref_idx, VMAX]
    vmin = ppc[0]["bus"][ref_idx, VMIN]

    assert net.ext_grid.vm_pu[0] == vmin
    assert net.ext_grid.vm_pu[0] == vmax

    # run pd2ppc with ext_grd controllable = True
    net.ext_grid["controllable"] = True
    ppc = _pd2ppc(net)
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF])
    vmax = ppc[0]["bus"][ref_idx, VMAX]
    vmin = ppc[0]["bus"][ref_idx, VMIN]

    assert net.bus.min_vm_pu[net.ext_grid.bus].values[0] == vmin
    assert net.bus.max_vm_pu[net.ext_grid.bus].values[0] == vmax

    assert net.ext_grid["in_service"].values.dtype == bool
    assert net.ext_grid["bus"].values.dtype == "uint32"
    pp.create_ext_grid(net, bus=4, vm_pu=net.res_bus.vm_pu.loc[4])

    assert net.ext_grid["bus"].values.dtype == "uint32"
    assert net.ext_grid["in_service"].values.dtype == bool

    ppc = _pd2ppc(net)
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF])

    bus2 = net._pd2ppc_lookups["bus"][net.ext_grid.bus[1]]
    vmax0 = ppc[0]["bus"][ref_idx, VMAX]
    vmin0 = ppc[0]["bus"][ref_idx, VMIN]

    vmax1 = ppc[0]["bus"][bus2, VMAX]
    vmin1 = ppc[0]["bus"][bus2, VMIN]

    assert net.bus.min_vm_pu[net.ext_grid.bus].values[0] == vmin0
    assert net.bus.max_vm_pu[net.ext_grid.bus].values[0] == vmax0

    assert net.ext_grid.vm_pu.values[1] == vmin1
    assert net.ext_grid.vm_pu.values[1] == vmax1


def test_opf_ext_grid_controllable():
    # load net
    net = case5_pm_matfile_I()
    net_old = copy.deepcopy(net)
    net_new = copy.deepcopy(net)
    # run pd2ppc with ext_grid controllable = False
    pp.runopp(net_old)
    net_new.ext_grid["controllable"] = True
    pp.runopp(net_new)
    assert np.isclose(net_new.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0586551789267864)
    assert np.isclose(net_old.res_bus.vm_pu[net.ext_grid.bus[0]], 1.06414000007302)

    assert np.isclose(net_old.res_cost, 17082.8)
    assert np.isclose(net_new.res_cost, 17015.5635)


def test_opf_ext_grid_controllable():
    # load net
    net = case5_pm_matfile_I()
    # run pd2ppc with ext_grid controllable = False
    pp.create_ext_grid(net, bus=0, controllable=True)
    pp.runopp(net)
    assert np.isclose(net.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0641399999827315)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_opf_ext_grid_controllable_pm():
    # load net
    net = case5_pm_matfile_I()
    net_old = copy.deepcopy(net)
    net_new = copy.deepcopy(net)
    # run pd2ppc with ext_grid controllable = False
    pp.runpm_ac_opf(net_old, calculate_voltage_angles=True, correct_pm_network_data=False, opf_flow_lim="I")
    net_new.ext_grid["controllable"] = True
    pp.runpm_ac_opf(net_new, calculate_voltage_angles=True, correct_pm_network_data=False,
                    delete_buffer_file=False, pm_file_path="buffer_file.json", opf_flow_lim="I")

    assert np.isclose(net_new.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0586551789267864)
    assert np.isclose(net_old.res_bus.vm_pu[net.ext_grid.bus[0]], 1.06414000007302)

    assert np.isclose(net_old.res_cost, 17082.8)
    assert np.isclose(net_new.res_cost, 17015.5635)


if __name__ == "__main__":
    test_opf_ext_grid_controllable()
    # pytest.main([__file__, "-xs"])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pytest
from pandapower import control
from pandapower.control import Characteristic


def test_continuous_p():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.)
    lv = pp.create_bus(net, vn_kv=20)
    t = pp.create_transformer(net, hv, lv, std_type='40 MVA 110/20 kV')
    pp.create_ext_grid(net, hv)
    eps = 0.0005

    c = control.ContinuousTapControl(net, t, 1., tol=eps)

    characteristic = Characteristic(net, [10, 20], [0.95, 1.05])
    tc = control.VmSetTapControl(net, 0, characteristic.index, tol=eps)

    # create 20kW load
    lid = pp.create_load(net, lv, 20)
    gid = pp.create_sgen(net, lv, 0)
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps

    # power sums up to 15kW
    net.sgen.at[gid, "p_mw"] = 5
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at 1.0 pu
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.) < eps

    # generation now cancels load
    net.sgen.at[gid, "p_mw"] = 10
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # testing limits
    # power flowing back
    net.sgen.at[gid, "p_mw"] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit and not drop even lower
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # excessive load
    net.sgen.at[gid, "p_mw"] = 0
    net.load.at[lid, "p_mw"] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit and not to go beyond
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps


def test_continuous_i():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.)
    lv = pp.create_bus(net, vn_kv=20)
    t = pp.create_transformer(net, hv, lv, std_type='40 MVA 110/20 kV')
    pp.create_ext_grid(net, hv)
    eps = 0.0005

    c = control.ContinuousTapControl(net, t, 1., tol=eps)

    # a different characteristic for i_lv_ka rather than for p_lv_mw
    characteristic = Characteristic(net, [0.315, 0.55], [0.95, 1.05])
    tc = control.VmSetTapControl(net, 0, characteristic.index, variable='i_lv_ka', tol=eps)

    # create 20kW load
    lid = pp.create_load(net, lv, 20)
    gid = pp.create_sgen(net, lv, 0)
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps

    # power sums up to 15kW
    net.sgen.at[gid, "p_mw"] = 5
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at 1.0 pu
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.) < eps

    # generation now cancels load
    net.sgen.at[gid, "p_mw"] = 10
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # testing limits
    # power flowing back
    net.sgen.at[gid, "p_mw"] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit and not drop even lower
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # excessive load
    net.sgen.at[gid, "p_mw"] = 0
    net.load.at[lid, "p_mw"] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit and not to go beyond
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps

if __name__ == '__main__':
    pytest.main(['-s', __file__])
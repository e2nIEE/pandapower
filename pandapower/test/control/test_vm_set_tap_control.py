# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.control.controller.trafo.ContinuousTapControl import (
    ContinuousTapControl,
)
from pandapower.control.controller.trafo.VmSetTapControl import VmSetTapControl
from pandapower.control.util.characteristic import Characteristic
from pandapower.create import (
    create_empty_network,
    create_bus,
    create_ext_grid,
    create_load,
    create_sgen,
    create_transformer,
)
from pandapower.run import runpp


def test_continuous_p():
    net = create_empty_network()
    hv = create_bus(net, vn_kv=110.0)
    lv = create_bus(net, vn_kv=20)
    t = create_transformer(net, hv, lv, std_type="40 MVA 110/20 kV")
    create_ext_grid(net, hv)
    eps = 0.0005

    c = ContinuousTapControl(net, t, 1.0, tol=eps)

    characteristic = Characteristic(net, [10, 20], [0.95, 1.05])
    tc = VmSetTapControl(net, 0, characteristic.index, tol=eps)

    # create 20kW load
    lid = create_load(net, lv, 20)
    gid = create_sgen(net, lv, 0)
    runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 1.05) < eps

    # power sums up to 15kW
    net.sgen.at[gid, "p_mw"] = 5
    runpp(net, run_control=True)

    # we expect the tap to converge at 1.0 pu
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 1.0) < eps

    # generation now cancels load
    net.sgen.at[gid, "p_mw"] = 10
    runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 0.95) < eps

    # testing limits
    # power flowing back
    net.sgen.at[gid, "p_mw"] = 30
    runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit and not drop even lower
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 0.95) < eps

    # excessive load
    net.sgen.at[gid, "p_mw"] = 0
    net.load.at[lid, "p_mw"] = 30
    runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit and not to go beyond
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 1.05) < eps


def test_continuous_i():
    net = create_empty_network()
    hv = create_bus(net, vn_kv=110.0)
    lv = create_bus(net, vn_kv=20)
    t = create_transformer(net, hv, lv, std_type="40 MVA 110/20 kV")
    create_ext_grid(net, hv)
    eps = 0.0005

    c = ContinuousTapControl(net, t, 1.0, tol=eps)

    # a different characteristic for i_lv_ka rather than for p_lv_mw
    characteristic = Characteristic(net, [0.315, 0.55], [0.95, 1.05])
    tc = VmSetTapControl(net, 0, characteristic.index, variable="i_lv_ka", tol=eps)

    # create 20kW load
    lid = create_load(net, lv, 20)
    gid = create_sgen(net, lv, 0)
    runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 1.05) < eps

    # power sums up to 15kW
    net.sgen.at[gid, "p_mw"] = 5
    runpp(net, run_control=True)

    # we expect the tap to converge at 1.0 pu
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 1.0) < eps

    # generation now cancels load
    net.sgen.at[gid, "p_mw"] = 10
    runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 0.95) < eps

    # testing limits
    # power flowing back
    net.sgen.at[gid, "p_mw"] = 30
    runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit and not drop even lower
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 0.95) < eps

    # excessive load
    net.sgen.at[gid, "p_mw"] = 0
    net.load.at[lid, "p_mw"] = 30
    runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit and not to go beyond
    assert abs(net.res_bus.vm_pu.at[c.trafobus] - 1.05) < eps


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

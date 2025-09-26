# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import (
    create_empty_network,
    create_bus,
    create_line,
    create_ext_grid,
    create_transformer,
    create_switch,
)
from pandapower.shortcircuit.calc_sc import calc_sc


def ring_network():
    net = create_empty_network(sn_mva=2.0)
    b0 = create_bus(net, 220)
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    create_ext_grid(
        net, b0, s_sc_max_mva=100.0, s_sc_min_mva=80.0, rx_min=0.4, rx_max=0.4
    )
    create_transformer(net, b0, b1, "100 MVA 220/110 kV")
    create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.0)
    l2 = create_line(
        net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=15.0
    )
    create_line(
        net, b3, b1, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=10.0
    )
    create_switch(net, b3, l2, closed=False, et="l")
    return net


def test_branch_results_open_ring():
    net = ring_network()
    calc_sc(net, branch_results=True, inverse_y=False)
    assert np.allclose(net.res_trafo_sc.ikss_lv_ka.values, [0.47705988])
    assert np.allclose(net.res_line_sc.ikss_ka.values, [0.45294928, 0.0, 0.47125418])


def test_branch_results_open_ring_with_impedance():
    net = ring_network()
    calc_sc(net, branch_results=True, inverse_y=False)
    res_line_no_imp = net.res_line_sc.ikss_ka.values.copy()

    # Make sure that with fault impedance, the total current should be smaller
    calc_sc(net, branch_results=True, inverse_y=False, r_fault_ohm=1, x_fault_ohm=5)
    non_null_flag = np.abs(res_line_no_imp) > 1e-10
    assert np.all(
        net.res_line_sc.ikss_ka.values[non_null_flag] < res_line_no_imp[non_null_flag]
    )


def test_branch_results_closed_ring():
    net = ring_network()
    net.switch.closed = True
    calc_sc(net, branch_results=True)

    assert np.allclose(net.res_trafo_sc.ikss_lv_ka.values, [0.47705988])
    assert np.allclose(
        net.res_line_sc.ikss_ka.values, [0.17559325, 0.29778739, 0.40286545]
    )


def test_kappa_methods():
    net = ring_network()
    net.switch.closed = True
    calc_sc(net, kappa_method="B", ip=True, inverse_y=False)
    assert np.allclose(
        net.res_bus_sc.ip_ka.values,
        [0.48810547956, 0.91192962511, 1.0264898716, 1.0360554521],
    )
    calc_sc(net, kappa_method="C", ip=True, topology="auto")
    assert np.allclose(
        net.res_bus_sc.ip_ka.values,
        [0.48810547956, 0.91192962511, 0.89331396461, 0.90103415924],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

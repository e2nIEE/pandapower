# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower.create import create_buses, create_ext_grid, create_gen, create_lines, create_switch
from pandapower.network import pandapowerNet
from pandapower.shortcircuit.calc_sc import calc_sc


@pytest.fixture
def one_line_one_generator():
    net = pandapowerNet(name="one_line_one_generator", sn_mva=23)
    b1, b2, b3 = create_buses(net, 3, vn_kv=10.0)
    create_buses(net, 1, vn_kv=0.4, in_service=False)
    create_gen(net, b1, vn_kv=10.5, xdss_pu=0.2, rdss_ohm=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    create_gen(net, b1, vn_kv=10.5, xdss_pu=0.2, rdss_ohm=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    create_lines(
        net,
        b2,
        b1,
        length_km=1.0,
        line_params={
            "max_i_ka": 0.29,
            "r_ohm_per_km": 0.1548,
            "x_ohm_per_km": 0.0816814,
            "c_nf_per_km": 165,
            "endtemp_degree": 165,
        },
    )
    create_switch(net, b3, b1, et="b")
    return net


@pytest.fixture
def gen_three_bus_example():
    net = pandapowerNet(name="gen_three_bus_example", sn_mva=12)
    b1, b2, b3 = create_buses(net, 3, vn_kv=10.0)
    create_gen(net, b2, vn_kv=10.5, xdss_pu=0.2, rdss_ohm=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    create_lines(
        net,
        [b1, b2],
        [b2, b3],
        length_km=1.0,
        line_params={
            "max_i_ka": 0.29,
            "r_ohm_per_km": 0.1548,
            "x_ohm_per_km": 0.0816814,
            "c_nf_per_km": 165,
            "endtemp_degree": 165,
        },
    )
    create_ext_grid(net, b1, s_sc_max_mva=10., s_sc_min_mva=8., rx_min=0.4, rx_max=0.4)
    return net


def test_max_gen(one_line_one_generator):
    net = one_line_one_generator
    calc_sc(net, case="max", inverse_y=False)
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3], [1.5395815, 1.5083952, 1.5395815], atol=1e-3)
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_branch_max_gen(gen_three_bus_example):
    net = gen_three_bus_example
    calc_sc(net, case="max", branch_results=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.76204252, 1.28698045]), atol=1e-3)


def test_min_gen(one_line_one_generator):
    net = one_line_one_generator
    calc_sc(net, case="min")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3], [1.3996195, 1.3697407, 1.3996195], atol=1e-3)
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_branch_min_gen(gen_three_bus_example):
    net = gen_three_bus_example
    calc_sc(net, case="min", branch_results=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.44487882, 1.10747517]), atol=1e-3)


def test_max_gen_fault_impedance(one_line_one_generator):
    net = one_line_one_generator
    calc_sc(net, case="max", r_fault_ohm=2, x_fault_ohm=10)
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3], [0.4450868, 0.4418823, 0.4450868], atol=1e-3)
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_gen_ext_grid_same_bus():
    net = pandapowerNet(name="test_gen_ext_grid_same_bus")
    (b,) = create_buses(net, 1, 110)

    net1 = copy.deepcopy(net)
    create_ext_grid(net1, b, s_sc_max_mva=1000, rx_max=0.4)
    calc_sc(net1)

    net2 = copy.deepcopy(net)
    create_gen(net2, b, 0, sn_mva=50, vn_kv=115, xdss_pu=0.2, rdss_ohm=20, cos_phi=0.8, pg_percent=0)
    calc_sc(net2)

    net3 = copy.deepcopy(net1)
    create_gen(net3, b, 0, sn_mva=50, vn_kv=115, xdss_pu=0.2, rdss_ohm=20, cos_phi=0.8, pg_percent=0)
    calc_sc(net3)

    # no idea why it is not close to 1e-6
    assert np.isclose(net3.res_bus_sc.at[0, "ikss_ka"],
                      net1.res_bus_sc.at[0, "ikss_ka"] + net2.res_bus_sc.at[0, "ikss_ka"], rtol=0, atol=2e-4)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

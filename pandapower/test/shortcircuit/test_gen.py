# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower.create import create_empty_network, create_bus, create_gen, create_line_from_parameters, create_switch, \
    create_ext_grid
from pandapower.shortcircuit.calc_sc import calc_sc


@pytest.fixture
def one_line_one_generator():
    net = create_empty_network(sn_mva=23)
    b1 = create_bus(net, vn_kv=10.)
    b2 = create_bus(net, vn_kv=10.)
    b3 = create_bus(net, vn_kv=10.)
    create_bus(net, vn_kv=0.4, in_service=False)
    create_gen(net, b1, vn_kv=10.5, xdss_pu=0.2, rdss_ohm=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    create_gen(net, b1, vn_kv=10.5, xdss_pu=0.2, rdss_ohm=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    line = create_line_from_parameters(net, b2, b1, length_km=1.0, max_i_ka=0.29, r_ohm_per_km=0.1548,
                                       x_ohm_per_km=0.0816814, c_nf_per_km=165)
    net.line.loc[line, "endtemp_degree"] = 165
    create_switch(net, b3, b1, et="b")
    return net


@pytest.fixture
def gen_three_bus_example():
    net = create_empty_network(sn_mva=12)
    b1 = create_bus(net, vn_kv=10.)
    b2 = create_bus(net, vn_kv=10.)
    b3 = create_bus(net, vn_kv=10.)
    # create_bus(net, vn_kv=0.4, in_service=False)
    create_gen(net, b2, vn_kv=10.5, xdss_pu=0.2, rdss_ohm=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    create_line_from_parameters(net, b1, b2, length_km=1.0, max_i_ka=0.29, r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814,
                                c_nf_per_km=165)
    create_line_from_parameters(net, b2, b3, length_km=1.0, max_i_ka=0.29, r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814,
                                c_nf_per_km=165)
    net.line["endtemp_degree"] = 165
    create_ext_grid(net, b1, s_sc_max_mva=10., s_sc_min_mva=8., rx_min=0.4, rx_max=0.4)
    # create_switch(net, b3, b1, et="b")
    return net


def test_max_gen(one_line_one_generator):
    net = one_line_one_generator
    calc_sc(net, case="max", inverse_y=False)
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3], [1.5395815, 1.5083952, 1.5395815], atol=1e-3)
    # assert abs(net.res_bus_sc.ikss_ka.at[0] - 1.5395815) < 1e-7
    # assert abs(net.res_bus_sc.ikss_ka.at[2] - 1.5395815) < 1e-7
    # assert abs(net.res_bus_sc.ikss_ka.at[1] - 1.5083952) < 1e-7
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_branch_max_gen(gen_three_bus_example):
    net = gen_three_bus_example
    calc_sc(net, case="max", branch_results=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.76204252, 1.28698045]), atol=1e-3)


def test_min_gen(one_line_one_generator):
    net = one_line_one_generator
    calc_sc(net, case="min")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3], [1.3996195, 1.3697407, 1.3996195], atol=1e-3)
    # assert abs(net.res_bus_sc.ikss_ka.at[0] - 1.3996195) < 1e-7
    # assert abs(net.res_bus_sc.ikss_ka.at[2] - 1.3996195) < 1e-7
    # assert abs(net.res_bus_sc.ikss_ka.at[1] - 1.3697407) < 1e-7
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_branch_min_gen(gen_three_bus_example):
    net = gen_three_bus_example
    calc_sc(net, case="min", branch_results=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.44487882, 1.10747517]), atol=1e-3)


def test_max_gen_fault_impedance(one_line_one_generator):
    net = one_line_one_generator
    calc_sc(net, case="max", r_fault_ohm=2, x_fault_ohm=10)
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3], [0.4450868, 0.4418823, 0.4450868], atol=1e-3)
    # assert abs(net.res_bus_sc.ikss_ka.at[0] - 0.4450868) < 1e-7
    # assert abs(net.res_bus_sc.ikss_ka.at[1] - 0.4418823) < 1e-7
    # assert abs(net.res_bus_sc.ikss_ka.at[2] - 0.4450868) < 1e-7
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_gen_ext_grid_same_bus():
    net = create_empty_network()
    b = create_bus(net, 110)

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


# def test_rdss_estimations():
#     net = create_empty_network(sn_mva=1)
#     b1 = create_bus(net, vn_kv=0.4)
#     g1 = create_gen(net, b1, vn_kv=0.4, xdss_pu=0.1, cos_phi=0.8, p_mw=0.1, sn_mva=0.1)
#     b2 = create_bus(net, vn_kv=20.)
#     g2 = create_gen(net, b2, vn_kv=21., xdss_pu=0.2, cos_phi=0.85, p_mw=0.1, sn_mva=2.5)
#     b3 = create_bus(net, vn_kv=20.)
#     g3 = create_gen(net, b3, vn_kv=30., xdss_pu=0.25, cos_phi=0.9, p_mw=0.1, sn_mva=150)

#     calc_sc(net, case="max")
#     assert np.isclose(net.res_bus_sc.ikss_ka.at[b1], 1.5130509845)
#     net.gen.at[g1, "rdss_pu"] = net.gen.xdss_pu.at[g1] * 0.15
#     calc_sc(net, case="max")
#     assert np.isclose(net.res_bus_sc.ikss_ka.at[b1], 1.5130509845)

#     calc_sc(net, case="max")
#     assert np.isclose(net.res_bus_sc.ikss_ka.at[b2], 0.37894052506)
#     net.gen.at[g2, "rdss_pu"] = net.gen.xdss_pu.at[g2] * 0.07
#     calc_sc(net, case="max")
#     assert np.isclose(net.res_bus_sc.ikss_ka.at[b2], 0.37894052506)

#     calc_sc(net, case="max")
#     assert np.isclose(net.res_bus_sc.ikss_ka.at[b3], 12.789334853)
#     net.gen.at[g3, "rdss_pu"] = net.gen.xdss_pu.at[g3] * 0.05
#     calc_sc(net, case="max")
#     assert np.isclose(net.res_bus_sc.ikss_ka.at[b3], 12.789334853)


# def test_close_to_gen_simple():
#     # from pandapower.shortcircuit import calc_sc
#     # vars = {name: getattr(calc_sc, name) for name in
#     #         dir(calc_sc) if not name.startswith('__')}
#     # globals().update(vars)
#     # del vars, calc_sc
#     # WIP
#     net = create_empty_network()
#     b1, b2, b3, b4, b5 = create_buses(net, 5, 20)
#     # skss = np.sqrt(3) * 400 * 40  # we assume 40 kA sc current in the 400-kV EHV grid
#     # create_ext_grid(net, b1, s_sc_max_mva=skss, s_sc_min_mva=0.8 * skss, rx_min=0.2, rx_max=0.4)
#     create_gen(net, b3, vn_kv=20, xdss_pu=0.2, rdss_pu=0.2*0.07, cos_phi=0.8, p_mw=5, sn_mva=5)
#     create_gen(net, b5, vn_kv=20, xdss_pu=0.2, rdss_pu=0.2*0.07, cos_phi=0.8, p_mw=10, sn_mva=10)

#     create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     create_line(net, b2, b3, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     create_line(net, b3, b4, std_type="305-AL1/39-ST1A 110.0", length_km=50)
#     create_line(net, b4, b5, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     # calc_single_sc(net, b5)
#     calc_sc(net, tk_s=5e-2)


# def test_close_to_gen_simple2():
#     # WIP
#     net = create_empty_network()
#     # b1, b2 = create_buses(net, 2, 110)
#     b1 = create_bus(net, 70)
#     # skss = np.sqrt(3) * 400 * 40  # we assume 40 kA sc current in the 400-kV EHV grid
#     # create_ext_grid(net, b1, s_sc_max_mva=skss, s_sc_min_mva=0.8 * skss, rx_min=0.2, rx_max=0.4)
#     create_gen(net, b1, vn_kv=70, xdss_pu=0.2, rdss_pu=0.2*0.07, cos_phi=0.8, p_mw=0, sn_mva=50)
#     # create_gen(net, b3, vn_kv=70, xdss_pu=0.2, rdss_pu=0.001, cos_phi=0.8, p_mw=50, sn_mva=60)

#     # create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     # create_line(net, b2, b3, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     # calc_single_sc(net, b2)
#     calc_sc(net, tk_s=5e-2)

# def test_generator_book():
#     net=create_empty_network()
#     b1= create_bus(net, 110)
#     b2= create_bus(net, 6)

#     create_ext_grid(net, b1, s_sc_max_mva=300, rx_max=0.1, s_sc_min_mva=250, rx_min=0.1)
#     create_transformer_from_parameters(net, b1, b2, 25, 110, 6, 0.5, 15, 15,0.1)
#     create_shunt(net, b2, 25, 0, 6)
#     create_gen(net, b2, 0, 1, sn_mva=25, vn_kv=6.3, xdss_pu=0.11, cos_phi=np.cos(np.arcsin(0.8)))
#     calc_sc(net, tk_s=2.5e-2)

# def test_shunt():
#     net=create_empty_network()
#     b1= create_bus(net, 110)
#     b2= create_bus(net, 110)

#     create_ext_grid(net, b1, s_sc_max_mva=300, rx_max=0.1)
#     # create_shunt(net, b2, 25, 0, 6)
#     create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=10)

#     calc_sc(net, tk_s=2.5e-2)

# def test_power_station_unit():
#     net = create_empty_network()
#     b1, b2, b3, b4 = create_buses(net, 4, 20)
#     b5 = create_bus(net, 10)

#     create_ext_grid(net, b1, s_sc_max_mva=250, rx_max=0.1)

#     create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     create_line(net, b2, b3, std_type="305-AL1/39-ST1A 110.0", length_km=10)
#     create_line(net, b3, b4, std_type="305-AL1/39-ST1A 110.0", length_km=50)

#     create_transformer_from_parameters(net, b4, b5, 25, 20, 10, 0.41104, 10.3, 0.1, 0.1)
#     create_gen(net, b5, vn_kv=10, xdss_pu=0.12, cos_phi=0.8, p_mw=0, sn_mva=10)

#     calc_sc(net)

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

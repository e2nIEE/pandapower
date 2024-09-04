# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.pypower.idx_brch import BR_R, BR_X


def simplest_test_grid(generator_type, step_up_trafo=False):
    net = pp.create_empty_network(sn_mva=6)
    if step_up_trafo:
        b0 = pp.create_bus(net, 20)
        b1 = pp.create_bus(net, 0.4)
        pp.create_transformer(net, b0, b1, "0.25 MVA 20/0.4 kV", parallel=10)
    else:
        b0 = b1 = pp.create_bus(net, 20)

    pp.create_ext_grid(net, b0, s_sc_max_mva=1e-12, rx_max=0)
    if generator_type == "async_doubly_fed":
        pp.create_sgen(net, b1, 0, 0, 2.5, current_source=False,
                       generator_type=generator_type, max_ik_ka=0.388, kappa=1.7, rx=0.1)
    elif generator_type == "current_source":
        pp.create_sgen(net, b1, 0, 0, 2.5, generator_type=generator_type, current_source=True, k=1.3, rx=0.1)
    elif generator_type == "async":
        pp.create_sgen(net, b1, 0, 0, 2.5, generator_type=generator_type, current_source=False, rx=0.1, lrc_pu=5)
    else:
        raise NotImplementedError(f"unknown sgen generator type {generator_type}, can be one of "
                                  f"'full_size_converter', 'async', 'async_doubly_fed'")
    return net


def wind_park_grid(case):
    net = pp.create_empty_network(sn_mva=7)
    pp.create_bus(net, 110, index=1)
    pp.create_buses(net, 13, 20)

    pp.create_ext_grid(net, 1, 1, s_sc_max_mva=10.5 * 110 * np.sqrt(3), rx_max=0.1)

    pp.create_transformer_from_parameters(net, 1, 2, 31.5, 110, 20, 0.6, 12, 0, 0)

    pp.create_line_from_parameters(net, 2, 3, 13.1, 0.0681, 0.102, 0, 1e3, 'L1', parallel=2)

    from_buses = np.array([3, 4, 3, 6, 7, 7, 3, 10, 11, 11, 12])
    to_buses = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13])
    length_km = [1.1, 0.55, 0.79, 0.17, 0.4, 0.55, 0.95, 0.24, 0.29, 0.15, 0.495]
    names = [f"L{i}" for i in range(2, 13)]
    pp.create_lines_from_parameters(net, from_buses, to_buses, length_km, 0.211, 0.122, 0, 1e3, names)

    sgen_buses = np.array([4, 5, 6, 8, 9, 10, 12, 13, 3, 14])
    if case=="all_async_doubly_fed":
        pp.create_sgens(net, sgen_buses, 0, 0, 2.5, rx=0.1, current_source=False,
                        generator_type="async_doubly_fed", max_ik_ka=0.388, kappa=1.7)
    elif case == "all_full_size_converter":
        pp.create_sgens(net, sgen_buses, 0, 0, 2.5, rx=0.1, k=1.3, current_source=True,
                        generator_type="current_source")
    elif case == "mixed":
        pp.create_sgens(net, sgen_buses[:5], 0, 0, 2.5, rx=0.1, current_source=False,
                        generator_type="async_doubly_fed", max_ik_ka=0.388, kappa=1.7)
        pp.create_sgens(net, sgen_buses[5:], 0, 0, 2.5, rx=0.1, k=1.3, current_source=True,
                        generator_type="current_source")
    else:
        raise NotImplementedError(f"case {case} not implemented")

    if len(net.sgen) > 0:
        net.sgen["current_angle_degree"] = -90
    return net


def wind_park_example():
    net = pp.create_empty_network(sn_mva=8)
    b1 = pp.create_bus(net, vn_kv=110., index=1)
    b2 = pp.create_bus(net, vn_kv=110., index=2)
    b3 = pp.create_bus(net, vn_kv=110., index=3)
    b4 = pp.create_bus(net, vn_kv=110., index=4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=20*110*np.sqrt(3), rx_max=0.1)

    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=100, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b3, to_bus=b4, length_km=25, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)

    pp.create_sgen(net, b2, p_mw=0.1e3, sn_mva=100)
    pp.create_sgen(net, b3, p_mw=0.050e3, sn_mva=50)
    pp.create_sgen(net, b4, p_mw=0.050e3, sn_mva=50)
    net.sgen["k"] = 1.2
    return net


def three_bus_example():
    net = pp.create_empty_network(sn_mva=9)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4, x0x_max=0.2, x0x_min=0.1,
                       r0x0_max=0.3, r0x0_min=0.2)
    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)
    return net


def big_sgen_three_bus_example():
    # ext_grid-bus1--line0--bus2--line1--bus3
    #                        |
    #                       sgen0
    net = pp.create_empty_network(sn_mva=2)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    net.ext_grid['x0x_max'] = 0.1
    net.ext_grid['r0x0_max'] = 0.1

    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=200., p_mw=0, k=1.2)
    return net


def test_max_3ph_branch_small_sgen():
    net = three_bus_example()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([0.53746061, 0.50852707, 0.4988896]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([ 0.49593034,  0.4988896]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([ 0.92787443,  0.9251165]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([ 0.49811957,  0.50106881]))


def test_max_3ph_branch_big_sgen():
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([1.78453722, 1.75560368, 1.72233192]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([1.25967331, 1.72233192]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([1.78144709, 2.65532524]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([1.26511638, 1.7298553]))


def test_min_3ph_branch_results_small_sgen():
    net = three_bus_example()
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([0.419891,  0.398969,  0.391938]), atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.391938,  0.391938]), atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([0.744387, 0.728265]), atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([0.400712, 0.393625]), atol=1e-6, rtol=0)


def test_min_3ph_branch_results_big_sgen():
    net = big_sgen_three_bus_example()
    # net.sn_mva = 110 * np.sqrt(3)
    # bacause in case "min" sgen is ignored, it does't matter if sgen is big or small -
    # the results here are the same as in the test test_min_3ph_branch_results_small_sgen
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([0.419891,  0.398969,  0.391938]), atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.391938,  0.391938]), atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([0.744387, 0.728265]), atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([0.400712, 0.393625]), atol=1e-6, rtol=0)


def test_max_1ph_branch_small_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Check coherence between bus result and branch results
    assert np.isclose(i_line_with_gen.ikss_ka.at[0], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)



def test_max_1ph_branch_big_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = big_sgen_three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contribution
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    # Check coherence between bus result and branch results
    # Since here sgen ikss_ka > ext_grid ikss_ka
    assert np.isclose(i_line_with_gen.ikss_ka.at[0], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_min_1ph_branch_small_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contribution
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    # Check coherence between bus result and branch results
    # in case 'min' sgen does not contribute so this check does not apply:
    # assert np.isclose(i_line_with_gen.ikss_ka.at[0], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(0, i_bus_only_sgen.ikss_ka.at[0], atol=1e-6)
    assert np.isclose(i_line_with_gen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-6)


def test_min_1ph_branch_big_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_sgen = net.res_line_sc.copy()

    # Without generator
    net = big_sgen_three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contribution
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    # Applying current divider: when sc is on bus 2 a small portion sgen ikss_ka flows along line_0 in the opposite
    # direction with respect of ext_grid ikss_ka
    i_line_1 = i_bus_without_sgen.ikss_ka.at[2] - (i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2])

    # Check coherence between bus result and branch results
    assert np.isclose(i_line_with_sgen.ikss_ka.at[0], i_line_1, atol=1e-4)
    assert np.isclose(i_line_with_sgen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_wind_park():
    net = wind_park_example()
    sc.calc_sc(net, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[2], 3.9034, rtol=1e-4)
    assert np.isclose(net.res_bus_sc.ip_ka.at[2], 7.3746, rtol=1e-4)


def test_wind_park_1():
    # example from IEC 60909-4 section 8.5
    net = wind_park_grid("all_async_doubly_fed")
    sc.calc_sc(net)
    ikss_ka = [10.745, 9.045, 6.978, 6.385, 6.095, 6.568, 6.478, 6.262,
               6.184, 6.513, 6.394, 6.247, 5.993, 6.313]
    assert np.allclose(net.res_bus_sc.ikss_ka, ikss_ka, atol=1e-3, rtol=0)


def test_wind_park_2():
    # example from IEC 60909-4 section 8.6
    net = wind_park_grid("all_full_size_converter")
    sc.calc_sc(net)
    ikss_ka = [10.671,  8.387,  6.161,  5.728,  5.522, 5.852, 5.787, 5.633,
               5.577, 5.797, 5.708, 5.600, 5.419, 5.651]
    assert np.allclose(net.res_bus_sc.ikss_ka, ikss_ka, atol=1.5e-3, rtol=0)


def test_wind_park_3():
    # example from IEC 60909-4 section 8.7
    net = wind_park_grid("mixed")
    sc.calc_sc(net)
    ikss_ka = [10.713,  8.734,  6.570,  6.078,  5.834, 6.232, 6.157, 5.976,
               5.910, 6.124, 6.015, 5.884, 5.666, 5.946]
    assert np.allclose(net.res_bus_sc.ikss_ka, ikss_ka, atol=1e-3, rtol=0)


def test_wind_power_station_unit():
    # full size converter (current source)
    net = simplest_test_grid('current_source')
    sc.calc_sc(net, ip=True)

    vn_kv = net.bus.vn_kv.at[0]
    sn_mva = net.sgen.sn_mva.at[0]
    k = net.sgen.k.at[0]
    ikss_ka = k * sn_mva / (np.sqrt(3) * vn_kv)
    skss_mw = ikss_ka * vn_kv * np.sqrt(3)
    assert np.isclose(net.res_bus_sc.at[0, 'ikss_ka'], ikss_ka, rtol=0, atol=1e-12)
    assert np.isclose(net.res_bus_sc.at[0, 'ip_ka'], ikss_ka * np.sqrt(2), rtol=0, atol=1e-12)
    assert np.isclose(net.res_bus_sc.at[0, 'skss_mw'], skss_mw, rtol=0, atol=1e-9)

    # doubly fed asynchronous generator (DFIG), kappa not calculated but provided by manufacturer
    net = simplest_test_grid("async_doubly_fed")
    sc.calc_sc(net, ip=True)

    kappa = net.sgen.kappa.at[0]
    vn_kv = net.bus.vn_kv.at[0]
    max_ik_ka = net.sgen.max_ik_ka.at[0]
    z_wd = np.sqrt(2) * kappa * vn_kv / (np.sqrt(3) * max_ik_ka)
    c = 1.1
    ikss_ka = c * net.bus.vn_kv.at[0] / (np.sqrt(3) * z_wd)
    skss_mw = ikss_ka * vn_kv * np.sqrt(3)
    assert np.isclose(net.res_bus_sc.at[0, 'ikss_ka'], ikss_ka, rtol=0, atol=1e-12)
    assert np.isclose(net.res_bus_sc.at[0, 'ip_ka'], ikss_ka * kappa * np.sqrt(2), rtol=0, atol=1e-12)
    assert np.isclose(net.res_bus_sc.at[0, 'skss_mw'], skss_mw, rtol=0, atol=1e-9)

    # asyncronous generator (also with a step-up trafo)
    net = simplest_test_grid("async")
    sc.calc_sc(net, ip=True)

    vn_kv = net.bus.vn_kv.at[0]
    sn_mva = net.sgen.sn_mva.at[0]
    lrc_pu = net.sgen.lrc_pu.at[0]
    rx = net.sgen.rx.at[0]
    z_g = 1 / lrc_pu * vn_kv**2 / sn_mva
    c = 1.1
    ikss_ka = c * vn_kv / (np.sqrt(3) * z_g)
    kappa = 1.02 + 0.98 * np.exp(-3 * 0.1)
    skss_mw = ikss_ka * vn_kv * np.sqrt(3)
    assert np.isclose(net.res_bus_sc.at[0, 'ikss_ka'], ikss_ka, rtol=0, atol=1e-12)
    assert np.isclose(net.res_bus_sc.at[0, 'ip_ka'], ikss_ka * kappa * np.sqrt(2), rtol=0, atol=1e-12)
    assert np.isclose(net.res_bus_sc.at[0, 'skss_mw'], skss_mw, rtol=0, atol=1e-9)

    # now async with a step-up trafo
    net = simplest_test_grid("async", True)
    sc.calc_sc(net, ip=True)

    vn_kv = net.bus.vn_kv.at[1]
    sn_mva = net.sgen.sn_mva.at[0]
    lrc_pu = net.sgen.lrc_pu.at[0]
    rx = net.sgen.rx.at[0]
    z_g = 1 / lrc_pu * vn_kv ** 2 / sn_mva
    z_g_complex = (rx + 1j) * z_g / (np.sqrt(1 + rx ** 2))
    c = 1.1
    ikss_ka = c * vn_kv / (np.sqrt(3) * z_g)
    kappa = 1.02 + 0.98 * np.exp(-3 * rx)
    skss_mw = ikss_ka * 0.4 * np.sqrt(3)
    assert np.isclose(net.res_bus_sc.at[1, 'ikss_ka'], ikss_ka, rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[1, 'ip_ka'], ikss_ka * kappa * np.sqrt(2), rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[1, 'skss_mw'], skss_mw, rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[1, 'rk_ohm'], z_g_complex.real, rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[1, 'xk_ohm'], z_g_complex.imag, rtol=0, atol=1e-9)

    base_z_ohm = 20**2 / net.sn_mva
    z_thv = (net._ppc['branch'][0, BR_R] + 1j*net._ppc['branch'][0, BR_X]) * base_z_ohm
    t_r = 20 / 0.4
    z_block = t_r ** 2 * z_g_complex + z_thv
    ikss_block_ka = c * 20 / (np.sqrt(3) * abs(z_block))
    skss_block_mw = ikss_block_ka * 20 * np.sqrt(3)
    kappa_block = 1.02 + 0.98 * np.exp(-3 * z_block.real / z_block.imag)
    assert np.isclose(net.res_bus_sc.at[0, 'ikss_ka'], ikss_block_ka, rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[0, 'ip_ka'], ikss_block_ka * kappa_block * np.sqrt(2), rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[0, 'skss_mw'], skss_block_mw, rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[0, 'rk_ohm'], z_block.real, rtol=0, atol=1e-9)
    assert np.isclose(net.res_bus_sc.at[0, 'xk_ohm'], z_block.imag, rtol=0, atol=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

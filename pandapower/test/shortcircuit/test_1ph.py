# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pandapower.shortcircuit as sc
import pandapower.test
import numpy as np
import os
import pytest


def check_results(net, vc, result):
    res_ika = net.res_bus_sc[(net.bus.zone==vc) & (net.bus.in_service)].ikss_ka.values
    if not np.allclose(result, res_ika, rtol=0, atol=1e-6):
        raise ValueError("Incorrect results for vector group %s"%vc, res_ika, result)


def add_network(net, vector_group):
    b1 = pp.create_bus(net, 110, zone=vector_group, index=pp.get_free_id(net.bus))
    b2 = pp.create_bus(net, 20, zone=vector_group)
    pp.create_bus(net, 20, in_service=False)
    b3 = pp.create_bus(net, 20, zone=vector_group)
    b4 = pp.create_bus(net, 20, zone=vector_group)
    pp.create_bus(net, 20)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100, s_sc_min_mva=100, rx_min=0.35, rx_max=0.35)
    net.ext_grid["r0x0_max"] = 0.4
    net.ext_grid["x0x_max"] = 1.0

    net.ext_grid["r0x0_min"] = 0.4
    net.ext_grid["x0x_min"] = 1.0

    pp.create_std_type(net, {"r_ohm_per_km": 0.122, "x_ohm_per_km": 0.112, "c_nf_per_km": 304,
                         "max_i_ka": 0.421, "endtemp_degree": 70.0, "r0_ohm_per_km": 0.244,
                         "x0_ohm_per_km": 0.336, "c0_nf_per_km": 2000}, "unsymmetric_line_type")
    l1 = pp.create_line(net, b2, b3, length_km=10, std_type="unsymmetric_line_type",
                   index=pp.get_free_id(net.line)+1)
    l2 = pp.create_line(net, b3, b4, length_km=15, std_type="unsymmetric_line_type")
    pp.create_line(net, b3, b4, length_km=15, std_type="unsymmetric_line_type", in_service=False)

    transformer_type = {"i0_percent": 0.071, "pfe_kw": 29, "vkr_percent": 0.282,
            "sn_mva": 25, "vn_lv_kv": 20.0, "vn_hv_kv": 110.0, "vk_percent": 11.2,
            "shift_degree": 150, "vector_group": vector_group, "tap_side": "hv",
            "tap_neutral": 0, "tap_min": -9, "tap_max": 9, "tap_step_degree": 0,
            "tap_step_percent": 1.5, "tap_phase_shifter": False, "vk0_percent": 5,
            "vkr0_percent": 0.4, "mag0_percent": 10, "mag0_rx": 0.4,
            "si0_hv_partial": 0.9}
    pp.create_std_type(net, transformer_type, vector_group, "trafo")
    t1 = pp.create_transformer(net, b1, b2, std_type=vector_group, parallel=2,
                          index=pp.get_free_id(net.trafo)+1)
    pp.create_transformer(net, b1, b2, std_type=vector_group, in_service=False)
    pp.add_zero_impedance_parameters(net)
    return l1, l2, t1


def test_1ph_shortcircuit():
    # vector groups without "N" have no impact on the 1ph
    results = {
        "Yy": [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962],
        "Yyn": [0.52209347337, 2.5145986133, 1.6737892808, 1.1117955913],
        "Yd": [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962],
        "YNy": [0.6291931171, 0.74400073149, 0.74563682772, 0.81607276962],
        "YNyn": [0.62623661918, 2.9829679356, 1.8895041867, 1.2075537026],
        "YNd": [0.75701600162, 0.74400073149, 0.74563682772, 0.81607276962],
        "Dy": [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962],
        "Dyn": [0.52209347337, 3.5054043285, 2.1086590382, 1.2980120038],
        "Dd": [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962]
    }

    for vc, result in results.items():
        net = pp.create_empty_network(sn_mva=17)
        add_network(net, vc)
        try:
            sc.calc_sc(net, fault="1ph", case="max")
        except Exception as e:
            raise UserWarning(f"{str(e)}: Did not converge after adding transformer with vector group {vc}")
        check_results(net, vc, result)


def test_1ph_shortcircuit_3w():
    # vector groups without "N" have no impact on the 1ph
    # here we check both functions, with Y invertion and with LU factorization for individual buses
    # The cuirrents are taken from the calculation with commercial software for reference
    results = {
                "ddd":  [1.5193429, 0, 0],
                "ddy":  [1.5193429, 0, 0],
                "dyd":  [1.5193429, 0, 0],
                "dyy":  [1.5193429, 0, 0],
                "ydd":  [1.5193429, 0, 0],
                "ydy":  [1.5193429, 0, 0],
                "yyd":  [1.5193429, 0, 0],
                "yyy":  [1.5193429, 0, 0],
                "ynyd": [1.783257, 0, 0],
                "yndy": [1.79376470, 0, 0], # ok
                "yynd": [1.5193429, 3.339398, 0],
                "ydyn": [1.5193429, 0, 8.836452], # ok
                "ynynd": [1.783257, 3.499335, 0],
                "yndyn": [1.79376470, 0, 9.04238714], # ok
                "yndd": [1.843545, 0, 0],
                "ynyy": [1.5193429, 0, 0] # ok but why?
               }

    for vg, result in results.items():
        net = single_3w_trafo_grid(vg)
        sc.calc_sc(net, fault="1ph", case="max")
        assert np.allclose(net.res_bus_sc.ikss_ka.values, result, rtol=0, atol=1e-6)

        net2 = single_3w_trafo_grid(vg)
        for bus in net2.bus.index.values:
            sc.calc_sc(net2, fault="1ph", case="max", inverse_y=False, bus=bus)
            assert np.allclose(net.res_bus_sc.ikss_ka.at[bus], net2.res_bus_sc.ikss_ka.at[bus], rtol=0, atol=1e-9)


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_1ph_shortcircuit_min(inverse_y):
    results = {
                 "Yy":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"Yyn": [0.52209346201, 2.4135757259, 1.545054139, 0.99373917957]
                ,"Yd":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"YNy": [0.62316686505, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"YNyn":[0.620287259, 2.9155736491, 1.7561556936, 1.0807305212]
                ,"YNd": [0.75434229157, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"Dy":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"Dyn": [0.52209346201, 3.4393798093, 1.9535982949, 1.1558364456]
                ,"Dd":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
               }

    for vc, result in results.items():
        net = pp.create_empty_network(sn_mva=16)
        add_network(net, vc)
        try:
            sc.calc_sc(net, fault="1ph", case="min", inverse_y=inverse_y)
        except Exception as e:
            raise UserWarning(f"{str(e)}: Did not converge after adding transformer with vector group {vc}")
        check_results(net, vc, result)


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_iec60909_example_4(inverse_y):
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    sc.calc_sc(net, fault="1ph", inverse_y=inverse_y)
    assert np.isclose(net.res_bus_sc[net.bus.name == "Q"].ikss_ka.values[0], 10.05957231)
    assert np.isclose(net.res_bus_sc[net.bus.name == "T2LV"].ikss_ka.values[0], 34.467353142)
    assert np.isclose(net.res_bus_sc[net.bus.name == "F1"].ikss_ka.values[0], 35.53066312)
    assert np.isclose(net.res_bus_sc[net.bus.name == "F2"].ikss_ka.values[0], 34.89135137)
    assert np.isclose(net.res_bus_sc[net.bus.name == "F3"].ikss_ka.values[0], 5.0321033105)
    assert np.isclose(net.res_bus_sc[net.bus.name == "Cable/Line IC"].ikss_ka.values[0], 16.362586813)


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_iec60909_example_4_bus_selection(inverse_y):
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    sc.calc_sc(net, fault="1ph", inverse_y=inverse_y,
               bus=net.bus[net.bus.name.isin(("F1", "F2"))].index)
    assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name == "F1"].index[0],
                                        "ikss_ka"], 35.53066312)
    assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name == "F2"].index[0],
                                        "ikss_ka"], 34.89135137)


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_iec60909_example_4_bus_selection_br_res(inverse_y):
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    sc.calc_sc(net, fault="1ph", inverse_y=inverse_y,
               bus=net.bus[net.bus.name.isin(("F1", "F2"))].index,
               branch_results=True)
    sc.calc_sc(net, fault="1ph", inverse_y=inverse_y,
               bus=net.bus[net.bus.name.isin(("F1", "F2"))].index,
               branch_results=True, return_all_currents=True)
    assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name=="F1"].index[0],
                                        "ikss_ka"], 35.53066312)
    assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name=="F2"].index[0],
                                        "ikss_ka"], 34.89135137)


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_1ph_with_switches(inverse_y):
    net = pp.create_empty_network(sn_mva=67)
    vc = "Yy"
    l1, l2, _ = add_network(net, vc)
    sc.calc_sc(net, fault="1ph", case="max", inverse_y=inverse_y)
    pp.create_line(net, net.line.to_bus.at[l2], net.line.from_bus.at[l1], length_km=15,
                   std_type="unsymmetric_line_type", parallel=2.)
    pp.add_zero_impedance_parameters(net)
    pp.create_switch(net, bus=net.line.to_bus.at[l2], element=l2, et="l", closed=False)
    sc.calc_sc(net, fault="1ph", case="max")
    check_results(net, vc, [0.52209347338, 2.0620266652, 2.3255761263, 2.3066467489])


def single_3w_trafo_grid(vector_group, sn_mva=123):
    net = pp.create_empty_network(sn_mva=sn_mva)
    b1 = pp.create_bus(net, vn_kv=380., geodata=(1,1))
    b2 = pp.create_bus(net, vn_kv=110., geodata=(0,1))
    b3 = pp.create_bus(net, vn_kv=30., geodata=(1,0))
    pp.create_ext_grid(net, b1, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer3w_from_parameters(net,
                                            hv_bus=b1, mv_bus=b2, lv_bus=b3,
                                            vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
                                            sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
                                            pfe_kw=0, i0_percent=0,
                                            vk_hv_percent=21, vkr_hv_percent=.26,
                                            vk_mv_percent=7, vkr_mv_percent=.16,
                                            vk_lv_percent=10., vkr_lv_percent=.16,
                                            vk0_hv_percent=44.1, vkr0_hv_percent=0.26,
                                            vk0_mv_percent=6.2996, vkr0_mv_percent=0.03714,
                                            vk0_lv_percent=6.2996, vkr0_lv_percent=0.03714,
                                            vector_group=vector_group)
    return net


def iec_60909_4_small(n_t3=1, num_earth=1, with_gen=False):
    net = pp.create_empty_network(sn_mva=3)

    b1 = pp.create_bus(net, vn_kv=380.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110.)
    b5 = pp.create_bus(net, vn_kv=110.)
    b8 = pp.create_bus(net, vn_kv=30.)
    HG2 = pp.create_bus(net, vn_kv=10)

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1, x0x_max=3, r0x0_max=0.15,
                       s_sc_min_mva=38 * 380 * np.sqrt(3) / 10, rx_min=0.1, x0x_min=3, r0x0_min=0.15,)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1, x0x_max=3.3, r0x0_max=0.2,
                       s_sc_min_mva=16 * 110 * np.sqrt(3) / 10, rx_min=0.1, x0x_min=3.3, r0x0_min=0.2)

    if num_earth == 1:
        vector_group = ("YYNd", "YNYd")
    else:
        vector_group = ("YNYNd", "YNYNd")

    if n_t3==2:
        pp.create_transformer3w_from_parameters(net,
            hv_bus=b1, mv_bus=b2, lv_bus=b8,
            vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
            sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
            pfe_kw=0, i0_percent=0,
            vk_hv_percent=21, vkr_hv_percent=.26,
            vk_mv_percent=7, vkr_mv_percent=.16,
            vk_lv_percent=10., vkr_lv_percent=.16,
            vk0_hv_percent=44.1, vkr0_hv_percent=0.26,
            vk0_mv_percent=6.2996, vkr0_mv_percent=0.03714,
            vk0_lv_percent=6.2996, vkr0_lv_percent=0.03714,
            vector_group=vector_group[0])
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16,
        vk0_hv_percent=44.1, vkr0_hv_percent=0.26,
        vk0_mv_percent=6.2996, vkr0_mv_percent=0.03714,
        vk0_lv_percent=6.2996, vkr0_lv_percent=0.03714,
        vector_group=vector_group[1])

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.32, x0_ohm_per_km=1.26, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)
    pp.create_line_from_parameters(net, b2, b5, name="L3a",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)
    pp.create_line_from_parameters(net, b2, b5, name="L3b",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.096, x_ohm_per_km=0.388,
        r0_ohm_per_km=0.22, x0_ohm_per_km=1.1, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)

    if with_gen:
        t1 = pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
            pfe_kw=0, i0_percent=0, vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5,
            vk0_percent=12, vkr0_percent=0.5, mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5,
            shift_degree=5, vector_group="Yd", power_station_unit=True)
        pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                      xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100, pg_percent=7.5,
                      slack=True, power_station_trafo=t1)

    return net


def iec_60909_4_t1():
    net = pp.create_empty_network(sn_mva=26)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    t1 = pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                               pfe_kw=0, i0_percent=0,
                                               vn_hv_kv=115., vn_lv_kv=21, vk_percent=16, vkr_percent=0.5,
                                               pt_percent=12, oltc=True, vk0_percent=15.2,
                                               vkr0_percent=0.5, xn_ohm=22, vector_group="YNd",
                                               mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5,
                                               power_station_unit=True)
    pp.create_gen(net, 1, p_mw=0.85 * 150, vn_kv=21,
                  xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, sn_mva=150, pg_percent=0,
                  power_station_trafo=t1)
    return net


def vde_232():
    net = pp.create_empty_network(sn_mva=12)
    # hv buses
    pp.create_bus(net, 110, geodata=(0,0))
    pp.create_bus(net, 21, geodata=(1,0))

    pp.create_ext_grid(net, 0, s_sc_max_mva=13.61213 * 110 * np.sqrt(3), rx_max=0.20328,
                       x0x_max=3.47927, r0x0_max=3.03361*0.20328/3.47927)
    pp.create_transformer_from_parameters(net, 0, 1, 150, 115, 21, 0.5, 16,
                                          pfe_kw=0, i0_percent=0, tap_step_percent=1,
                                          tap_max=12, tap_min=-12, tap_neutral=0, tap_side='hv',
                                          vector_group="YNd",
                                          vk0_percent=np.sqrt(np.square(0.95*15.99219) + np.square(0.5)),
                                          vkr0_percent=0.5,
                                          mag0_percent=100, mag0_rx=0,
                                          si0_hv_partial=0.9,
                                          pt_percent=12, oltc=True,
                                          power_station_unit=True,
                                          xn_ohm=22)

    pp.create_gen(net, 1, 150, 1, 150, vn_kv=21, xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, power_station_trafo=0, pg_percent=5)

    # z_q
    u_nq = 110
    i_kqss = 13.61213
    z_q = 1.1 * u_nq / (np.sqrt(3) * i_kqss)
    rx_max = 0.20328
    x_q = z_q / np.sqrt(1+rx_max**2)
    x_0q = x_q * 3.47927
    r_0q = x_0q * 3.03361*0.20328/3.47927
    z_0q = r_0q + 1j*x_0q
    return net


def test_iec60909_example_4_one_trafo3w():
    net = iec_60909_4_small(n_t3=1)

    # r0 = 2.378330877
    # x0 = 17.335578502
    # r = 0.59621204768
    # x = 6.0598429694
    ikss_pf = [24.4009, 8.2481, 6.1728, 10.1851]

    sc.calc_sc(net, fault="1ph")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf), atol=1e-4)


def test_iec60909_example_4_two_trafo3w():
    net = iec_60909_4_small(n_t3=2)

    ikss_pf_2t3 = [24.5772, 14.7247, 8.1060, 15.2749]

    sc.calc_sc(net, fault="1ph")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf_2t3), atol=1e-4)


def test_iec60909_example_4_two_trafo3w_two_earth():
    net = iec_60909_4_small(n_t3=2, num_earth=2)

    # r0 = 2.378330877
    # x0 = 17.335578502
    # r = 0.59621204768
    # x = 6.0598429694
    ikss_pf_max = [26.0499, 20.9472, 9.1722, 18.7457]
    ikss_pf_min = [3.9622, 8.3914, 5.0248, 6.9413]

    sc.calc_sc(net, fault="1ph", case="max")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf_max), atol=1e-4)

    sc.calc_sc(net, fault="1ph", case="min")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf_min), atol=1e-4)


@pytest.mark.skip("1ph gen-close sc calculation still under develop")
def test_iec_60909_4_small_with_t2_1ph():
    net = iec_60909_4_small(n_t3=2, num_earth=1, with_gen=True)
    net.gen = net.gen.iloc[0:0, :]
    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_max = [24.57717, 16.96235, 11.6109, 18.07836]
    # ikss_min = [3.5001, 8.4362, 7.4743, 7.7707]
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_max), atol=1e-4)


@pytest.mark.skip("1ph gen-close sc calculation still under develop")
def test_iec_60909_4_small_with_gen_1ph_no_ps_detection():
    net = iec_60909_4_small(n_t3=2, num_earth=1, with_gen=True)
    net.gen.power_station_trafo=np.nan
    net.trafo.power_station_unit=np.nan
    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_max = [24.60896, 17.2703, 12.3771, 18.4723]
    # ikss_min = [3.5001, 8.4362, 7.4743, 7.7707]
    # ip_min = [8.6843, 21.6173, 18.0242, 19.4261]
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_max), atol=1e-4)


def test_iec_60909_4_small_with_gen_ps_unit_1ph():
    net = iec_60909_4_small(n_t3=2, num_earth=1, with_gen=True)

    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_max = [24.6109, 17.4363, 12.7497, 18.6883]
    # ikss_min = [3.5001, 8.4362, 7.4743, 7.7707]
    # ip_min = [8.6843, 21.6173, 18.0242, 19.4261]

    # TODO: This needs to be fixed!!
    # assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_max), atol=1e-4)


def test_vde_232_with_gen_ps_unit_1ph():
    # IEC 60909-4:2021, example from section 4.4.2
    # from pandapower.test.shortcircuit.test_1ph import *
    # from pandapower.pypower.idx_bus import *
    net = vde_232()

    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    assert np.isclose(net.res_bus_sc.at[0, 'ikss_ka'], 9.04979, rtol=0, atol=1e-4)
    assert np.isclose(net.res_bus_sc.at[0, 'rk0_ohm'], 2.09392, rtol=0, atol=1e-4)
    assert np.isclose(net.res_bus_sc.at[0, 'xk0_ohm'], 14.3989, rtol=0, atol=1e-4)


def test_t1_iec60909_4():
    net = iec_60909_4_t1()
    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    assert np.isclose(net.res_bus_sc.at[0, 'ikss_ka'], 1.587457, rtol=0, atol=1e-4)
    assert np.isclose(net.res_bus_sc.at[0, 'rk0_ohm'], 0.439059, rtol=0, atol=1e-4)
    assert np.isclose(net.res_bus_sc.at[0, 'xk0_ohm'], 79.340169, rtol=0, atol=1e-4)
    assert np.isclose(net.res_bus_sc.at[0, 'rk_ohm'], 0.498795, rtol=0, atol=1e-4)
    assert np.isclose(net.res_bus_sc.at[0, 'xk_ohm'], 26.336676, rtol=0, atol=1e-4)


def test_1ph_sn_mva_ext_grid():
    net1 = pp.create_empty_network(sn_mva=1)
    b1 = pp.create_bus(net1, 110)
    pp.create_ext_grid(net1, b1, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)


    net2 = pp.create_empty_network(sn_mva=17)
    b1 = pp.create_bus(net2, 110)
    pp.create_ext_grid(net2, b1, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    sc.calc_sc(net1, fault="1ph", case="max")
    sc.calc_sc(net2, fault="1ph", case="max")

    res = {'ikss_ka': 5.2486388108147795,
           'rk0_ohm': 1.3243945001694957,
           'xk0_ohm': 13.243945001694957,
           'rk1_ohm': 1.3243945001694957,
           'xk1_ohm': 13.243945001694957}

    for var in res.keys():
        assert np.allclose(net1.res_bus_sc[var], net2.res_bus_sc[var], atol=1e-6, rtol=0)
        assert np.allclose(net1.res_bus_sc[var], res[var], atol=1e-6, rtol=0)


def test_line():
    net = pp.create_empty_network(sn_mva=17)
    b1 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b1, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    b2 = pp.create_bus(net, 110)

    pp.create_line_from_parameters(net, b1, b2, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)
    sc.calc_sc(net, fault="1ph", case="max")

    assert np.allclose(net.res_bus_sc.ikss_ka, [5.248639, 4.968909], rtol=0, atol=1e-6)

    res = net.res_bus_sc.copy()
    sc.calc_sc(net, fault="1ph", case="max")
    assert np.allclose(net.res_bus_sc, res, rtol=0, atol=1e-6)

    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=1)
    assert np.allclose(net.res_line_sc.values,
                       [4.968909, 4.968909, -76.322582, 4.968909, 103.677418, 49.380121, 10.287525, 0., 0.,
                        0.159840, -64.554294, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.1, -120., 1.040081,
                        -122.717776, 0., 0., 0., 0., 0., 0., 0., 0., 1.1, 120., 1.173594, 118.620866],
                       rtol=0, atol=1e-5)


def test_2_lines():
    net = pp.create_empty_network(sn_mva=17)
    b1 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b1, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_line_from_parameters(net, b1, b2, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)
    pp.create_line_from_parameters(net, b2, b3, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)

    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=2)


def test_4_lines():
    net = pp.create_empty_network(sn_mva=17)
    pp.create_buses(net, 5, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_line_from_parameters(net, 0, 1, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)
    pp.create_line_from_parameters(net, 1, 2, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)
    pp.create_line_from_parameters(net, 1, 3, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)
    pp.create_line_from_parameters(net, 3, 4, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)

    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=4)


def test_trafo_temp():
    vc = "Dyn"
    # vc = "YNyn"
    net = pp.create_empty_network(sn_mva=1)
    pp.create_buses(net, 2, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_line_from_parameters(net, 0, 1, 1, 1, 0.5, 0., 10, r0_ohm_per_km=4, x0_ohm_per_km=0.25, c0_nf_per_km=0.)

    pp.create_transformer_from_parameters(net, 1, 2, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group=vc,
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5)

    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=2)


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_trafo_1ph(inverse_y):
    res_bus = {"YNd": {0: [6.380943, 0.399053, 6.214775, 1.324394, 13.243945, 1.324394, 13.243945],
                       1: [0, np.inf, np.inf, 0.056495, 0.844448, 0.056495, 0.844448]},
               "Dyn": {0: [5.248639, 1.324394, 13.243945, 1.324394, 13.243945, 1.324394, 13.243945],
                       1: [18.328769, 0.012713, 0.386279, 0.056495, 0.844448, 0.056495, 0.844448]}}

    res_trafo = {"YNd": {0: [1.132975, 93.535632, 0., 0., 0., 0., 0., 0., 0., 0., 0.208569, 1.650563, 1.132975,
                             93.535632, 0., 0., -67.674057, 26.725911, 0., 0., 1.01121, -108.014475, 0.9613, -96.225276,
                             1.132975, 93.535632, 0., 0., 69.155042, 18.271669, 0., 0., 0.994087, 108.335724, 0.955328,
                             96.264346],
                         1: [0., 0., 0., 0., 0., 0., 0., 0., 1.1, -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             1.1, -120., 1.905256, -150., 0., 0., 0., 0., 0., 0., 0., 0., 1.1, 120., 1.905256, 150.]
                         },
                 "Dyn": {0: [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.366667, 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.1,
                             -120., 0.970109, -100.893395, 0., 0., 0., 0., 0., 0., 0., 0., 1.1, 120., 0.970109,
                             100.893395],
                         1: [2.221669, -86.533547, 18.328769, 93.466453, 2.847334, 89.550826, 0., 0., 0.635006,
                             1.645305, 0., 0., 1.110835, 93.466453, 0., 0., -66.370695, 26.451227, 0., 0., 1.012757,
                             -108.262695, 1.00882, -107.694556, 1.110835, 93.466453, 0., 0., 67.794362, 18.324186, 0.,
                             0., 0.995459, 108.591552, 0.992704, 107.991556]
                         }
                 }

    def _check_result(vector_group, tol=(1e-6, 1e-6, 1e-6, 1e-6)):
        net.trafo.vector_group = vector_group
        fault_bus = 0
        sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False,
                   inverse_y=inverse_y)
        assert np.allclose(res_bus[vector_group][fault_bus], net.res_bus_sc.loc[fault_bus].values, rtol=0, atol=tol[0])
        assert np.allclose(res_trafo[vector_group][fault_bus], net.res_trafo_sc.loc[0].values, rtol=0, atol=tol[1])

        fault_bus = 1
        sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False,
                   inverse_y=inverse_y)
        assert np.allclose(res_bus[vector_group][fault_bus], net.res_bus_sc.loc[fault_bus].values, rtol=0, atol=tol[2])
        assert np.allclose(res_trafo[vector_group][fault_bus], net.res_trafo_sc.loc[0].values, rtol=0, atol=tol[3])

    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group="YNd",
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5)

    _check_result("YNd", tol=(1e-6, 5e-6, 1e-6, 1e-6))
    _check_result("Dyn", tol=(1e-6, 5e-6, 1e-6, 7e-6))

    # todo Yyn
    # todo YNyn


@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_trafo_1ph_c(inverse_y):
    res_bus = {"YNd": {0: [5.773829, 0.414233, 6.372184, 1.324104, 13.241768, 1.324104, 13.241768],
                       1: [0, np.inf, np.inf, 0.057101, 0.864122, 0.057101, 0.864122]},
               "Dyn": {0: [],
                       1: []}}

    res_trafo = {"YNd": {0: [1.002174, 93.490498, 0., 0., 0., 0., 0., 0., 0., 0., 0.1935, 1.60556, 1.002398, 93.529907,
                             0., 0., -54.406846, 21.863020, 0., 0., 0.921057, -108.362484, 0.874034, -96.352941,
                             1.002406, 93.451971, 0., 0., 55.627668, 15.149927, 0., 0., 0.905632, 108.686711, 0.868646,
                             96.392189],
                         1: [0.000787, -86.176339, 0., 0., 0.003334, 0.049885, 0., 0., 1., -0., 0., 0., 0.000787,
                             153.822675, 0., 0., 0.003333, 0.049885, 0., 0., 1., -120., 1.731912, -150.000162, 0.000787,
                             33.822748, 0., 0., 0.003333, 0.049884, 0., 0., 1., 120., 1.731912, 149.999837]
                         },
                 "Dyn": {0: [],
                         1: []
                         }
                 }

    def _check_result(vector_group, tol=(1e-6, 1e-6, 1e-6, 1e-6)):
        net.trafo.vector_group = vector_group
        pp.runpp(net)
        fault_bus = 0
        sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True,
                   inverse_y=inverse_y)
        assert np.allclose(res_bus[vector_group][fault_bus], net.res_bus_sc.loc[fault_bus].values, rtol=0, atol=tol[0])
        assert np.allclose(res_trafo[vector_group][fault_bus], net.res_trafo_sc.loc[0].values, rtol=0, atol=tol[1])

        fault_bus = 1
        sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True,
                   inverse_y=inverse_y)
        assert np.allclose(res_bus[vector_group][fault_bus], net.res_bus_sc.loc[fault_bus].values, rtol=0, atol=tol[2])
        assert np.allclose(res_trafo[vector_group][fault_bus], net.res_trafo_sc.loc[0].values, rtol=0, atol=tol[3])

    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group="YNd",
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5)

    _check_result("YNd", tol=(1e-6, 5e-6, 1e-5, 5e-3))
    non_degree = [i for i, c in enumerate(net.res_trafo_sc.columns) if "degree" not in c]
    assert np.allclose(np.array(res_trafo["YNd"][1])[non_degree], net.res_trafo_sc.loc[0].values[non_degree],
                       rtol=0, atol=1e-5)  # results for angle (degree) are not as accurate in this example

    # todo Dyn
    # todo Yyn
    # todo YNyn


def test_trafo_1ph_c_sgen():
    res_bus = {"YNd": {0: [5.773829, 0.414233, 6.372184, 1.324104, 13.241768],
                       1: [0, np.inf, np.inf, 0.057101, 0.864122]},
               "Dyn": {0: [],
                       1: []}}

    res_trafo = {"YNd": {0: [1.002174, 93.490498, 0., 0., 0., 0., 0., 0., 0., 0., 0.1935, 1.60556, 1.002398, 93.529907,
                             0., 0., -54.406846, 21.863020, 0., 0., 0.921057, -108.362484, 0.874034, -96.352941,
                             1.002406, 93.451971, 0., 0., 55.627668, 15.149927, 0., 0., 0.905632, 108.686711, 0.868646,
                             96.392189],
                         1: [0.000787, -86.176339, 0., 0., 0.003334, 0.049885, 0., 0., 1., -0., 0., 0., 0.000787,
                             153.822675, 0., 0., 0.003333, 0.049885, 0., 0., 1., -120., 1.731912, -150.000162, 0.000787,
                             33.822748, 0., 0., 0.003333, 0.049884, 0., 0., 1., 120., 1.731912, 149.999837]
                         },
                 "Dyn": {0: [],
                         1: []
                         }
                 }

    def _check_result(vector_group, tol=(1e-6, 1e-6, 1e-6, 1e-6)):
        net.trafo.vector_group = vector_group
        pp.runpp(net)
        fault_bus = 0
        sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True)
        assert np.allclose(res_bus[vector_group][fault_bus], net.res_bus_sc.loc[fault_bus].values, rtol=0, atol=tol[0])
        assert np.allclose(res_trafo[vector_group][fault_bus], net.res_trafo_sc.loc[0].values, rtol=0, atol=tol[1])

        fault_bus = 1
        sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True)
        assert np.allclose(res_bus[vector_group][fault_bus], net.res_bus_sc.loc[fault_bus].values, rtol=0, atol=tol[2])
        assert np.allclose(res_trafo[vector_group][fault_bus], net.res_trafo_sc.loc[0].values, rtol=0, atol=tol[3])

    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group="Dyn",
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5)
    #pp.create_sgen(net, 1, 2, 0, 1, k=0, generator_type="current_source", kappa=0)
    pp.create_sgen(net, 1, 20, 0, 100, k=0, generator_type="current_source", kappa=0, current_angle_degree=-32.869393)

    _check_result("YNd", tol=(1e-6, 5e-6, 1e-5, 5e-3))
    non_degree = [i for i, c in enumerate(net.res_trafo_sc.columns) if "degree" not in c]
    assert np.allclose(np.array(res_trafo["YNd"][1])[non_degree], net.res_trafo_sc.loc[0].values[non_degree],
                       rtol=0, atol=1e-5)  # results for angle (degree) are not as accurate in this example

    # todo Dyn
    # todo Yyn
    # todo YNyn


def test_isolated_sgen():
    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=20.)
    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)
    # pp.create_sgen(net, 1, 2, 0, 1, k=0, generator_type="current_source", kappa=0)
    pp.create_sgen(net, 0, 50, 0, 100, k=0, generator_type="current_source", kappa=0)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", bus=0, use_pre_fault_voltage=True)
    assert np.allclose(net.res_bus_sc.values,
                       [26.217212, 0.043782, 0.437816, 0.019738, 0.441312, 0.043782, 0.437816],
                       rtol=0, atol=1e-5)


def test_isolated_load():
    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=20.)
    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)
    # pp.create_sgen(net, 1, 2, 0, 1, k=0, generator_type="current_source", kappa=0)
    pp.create_load(net, 0, 50)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", bus=0, use_pre_fault_voltage=True)
    assert np.allclose(net.res_bus_sc.values,
                       [26.373383, 0.043782, 0.437816, 0.067045, 0.431784, 0.067045, 0.431784],
                       rtol=0, atol=5e-6)


def test_petersen_coil():
    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group="Dyn",
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5, xn_ohm=50)
    fault_bus = 1
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False)
    assert np.allclose(net.res_bus_sc.values,
                       [0.250568, 0.012714, 150.386279, 0.056495, 0.844448, 0.056495, 0.844448],
                       rtol=0, atol=1e-6)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True)
    assert np.allclose(net.res_bus_sc.values,
                       [0.227683, 0.013334, 150.405114, 0.057101, 0.864115, 0.057101, 0.864115],
                       rtol=0, atol=1e-6)

    net.bus.vn_kv = 115, 21
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False)
    assert np.allclose(net.res_bus_sc.values,
                       [0.262955, 0.012714, 150.386279, 0.060566, 0.885154, 0.060566, 0.885154],
                       rtol=0, atol=1e-6)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True)
    assert np.allclose(net.res_bus_sc.values,
                       [0.237905, 0.013334, 150.405114, 0.061169, 0.904801, 0.061169, 0.904801],
                       rtol=0, atol=1e-6)


def test_petersen_coil_compensation():
    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group="Dyn",
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5, xn_ohm=1200)
    pp.create_line_from_parameters(net, 1, 2, 1, 0.1, 0.2, 250, 1, r0_ohm_per_km=0.05, x0_ohm_per_km=0.1,
                                   c0_nf_per_km=300)
    fault_bus = 2
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False)
    #assert np.allclose(net.res_bus_sc.values, [0.006989, 0.10974, 5449.743428, 0.156495, 1.044448, 0.156495, 1.044448],
    #                   rtol=0, atol=1e-5)
    assert np.isclose(net.res_bus_sc.ikss_ka, 0.006989, atol=1e-6)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True)
    assert np.isclose(net.res_bus_sc.ikss_ka, 0.006354, atol=1e-6)
   # assert np.allclose(net.res_bus_sc.values,
    #                   [0.006354, 0.11116, 5449.786579, 0.157118, 1.064188, 0.157118, 1.064188],
     #                  rtol=0, atol=1e-6)

    net.trafo.xn_ohm = 100
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False)
    assert np.isclose(net.res_bus_sc.ikss_ka, 0.122395, atol=1e-6)
    assert np.allclose(net.res_bus_sc.values,
                       [0.122395, 0.064938, 309.241146, 0.156495, 1.044448, 0.156495, 1.044448],
                       rtol=0, atol=1e-5)


def test_fault_impedance():
    net = pp.create_empty_network(sn_mva=1)
    pp.create_bus(net, vn_kv=110.)
    pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)

    pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                          pfe_kw=10, i0_percent=0.1,
                                          vn_hv_kv=110., vn_lv_kv=20, vk_percent=16, vkr_percent=0.5,
                                          pt_percent=12, vk0_percent=15.2,
                                          vkr0_percent=0.5, vector_group="Dyn",
                                          mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5)
    fault_bus = 1
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False,
               x_fault_ohm=5, r_fault_ohm=1)
    assert np.allclose(net.res_bus_sc.values,
                       [2.195134, 1.012713, 5.386279, 1.056495, 5.844448, 1.056495, 5.844448],
                       rtol=0, atol=1e-6)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True,
               x_fault_ohm=5, r_fault_ohm=1)
    assert np.allclose(net.res_bus_sc.values,
                       [1.988823, 1.013333, 5.405114, 1.057101, 5.864115, 1.057101, 5.864115],
                       rtol=0, atol=1e-6)

    net.bus.vn_kv = 115, 21
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=False,
               x_fault_ohm=5, r_fault_ohm=1)
    assert np.allclose(net.res_bus_sc.values,
                       [2.294113, 1.012713, 5.386279, 1.060566, 5.885154, 1.060566, 5.885154],
                       rtol=0, atol=1e-6)

    pp.runpp(net)
    sc.calc_sc(net, fault="1ph", case="max", branch_results=True, bus=fault_bus, use_pre_fault_voltage=True,
               x_fault_ohm=5, r_fault_ohm=1)
    assert np.allclose(net.res_bus_sc.values,
                       [2.069538, 1.013333, 5.405114, 1.061169, 5.904801, 1.061169, 5.904801],
                       rtol=0, atol=1e-6)


def test_trafo():
    results = {
        "Yy": [5.248639, 0],
        "Yyn": [5.248639, 0.812581],
        "Yd": [5.248639, 0],
        "Dy": [5.248639, 0],
        "Dd": [5.248639, 0],
        "Dyn": [5.248639, 17.245191],
        "YNd": [6.324033, 0],
        "YNy": [5.265657, 0],
        "YNyn": [5.265737, 14.413729]}

    for vc in results.keys():
        net = pp.create_empty_network(sn_mva=1)
        pp.create_bus(net, vn_kv=110.)
        pp.create_bus(net, vn_kv=20.)

        pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                           rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                           rx_min=0.1, x0x_min=1, r0x0_min=0.1)

        t1 = pp.create_transformer_from_parameters(net, 0, 1, sn_mva=150,
                                                   pfe_kw=10, i0_percent=0.1,
                                                   vn_hv_kv=115., vn_lv_kv=21, vk_percent=16, vkr_percent=0.5,
                                                   pt_percent=12, vk0_percent=15.2,
                                                   vkr0_percent=0.5, vector_group=vc,
                                                   mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5)
        sc.calc_sc(net, fault="1ph", case="max")
        res = net.res_bus_sc.copy()

        net.sn_mva = 123
        sc.calc_sc(net, fault="1ph", case="max")
        assert np.allclose(net.res_bus_sc, res, rtol=0, atol=1e-6), f"failed for vector group {vc}"
        assert np.allclose(net.res_bus_sc.ikss_ka, results[vc], rtol=0, atol=1e-6), f"{vc}: inconsistent results"


def test_sc_1ph_impedance():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=1000, s_sc_min_mva=800,
                       rx_max=0.1, x0x_max=1, r0x0_max=0.1,
                       rx_min=0.1, x0x_min=1, r0x0_min=0.1)
    pp.create_impedance(net, 0, 1, rft_pu=0.2, xft_pu=0.4, sn_mva=50, rtf_pu=0.25, xtf_pu=0.5,
                        rft0_pu=0.1, xft0_pu=0.2, rtf0_pu=0.05, xtf0_pu=0.1)

    sc.calc_sc(net, fault="1ph")

    assert np.allclose(net.res_bus_sc.ikss_ka, [5.248639, 0.625166], rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus_sc.rk0_ohm, [1.324394, 12.762198], rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus_sc.xk0_ohm, [13.243945, 30.821973], rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus_sc.rk1_ohm, [1.3243945, 62.1554916], rtol=0, atol=1e-5)
    assert np.allclose(net.res_bus_sc.xk1_ohm, [13.2439445, 137.5549268], rtol=0, atol=1e-5)


def test_mv_oberrhein():
    net = pp.networks.mv_oberrhein()
    net.ext_grid["s_sc_max_mva"] = 100
    net.ext_grid["s_sc_min_mva"] = 100
    net.ext_grid["rx_max"] = 0.1  # R/X ratio
    net.ext_grid["rx_min"] = 0.1  # R/X ratio
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0  # 0-sequence/1-sequence ratio
    #
    # sgen
    #
    net.sgen['k'] = 1.0  # 1.3 is a better value. Contribution increase from an sgen in short-circuit conditions.
    # net.sgen['sn_mva'] = 2.
    # net.sgen['p_mw'] = 2.
    net.sgen['kappa'] = net.sgen['k']
    #
    # line
    #
    net.line["endtemp_degree"] = 20
    net.line["r0_ohm_per_km"] = 0.16
    net.line["x0_ohm_per_km"] = 0.12
    net.line["c0_nf_per_km"] = 280
    #
    # trafo
    #
    # D stands for "delta", Y for "Wye" (star), "n" for neutral (ground). Relevant for single-phase to ground only
    net.trafo["vector_group"] = 'Dyn'
    net.trafo["vk0_percent"] = net.trafo["vk_percent"]  # short circuit voltage, positive sequence or zero-sequence
    net.trafo["vkr0_percent"] = net.trafo["vkr_percent"]  # real part of zero sequence relative short-circuit voltage
    net.trafo["mag0_percent"] = 100  # ratio between magnetizing and short circuit impedance (zero sequence)
    net.trafo["mag0_rx"] = 0  # zero sequence magnetizing r/x ratio
    net.trafo["si0_hv_partial"] = 0.9  # zero sequence short circuit impedance distribution in hv side

    pp.runpp(net)
    sc.calc_sc(net, fault='1ph', case="max", bus=33, branch_results=True, use_pre_fault_voltage=True)


if __name__ == "__main__":
    pytest.main([__file__])

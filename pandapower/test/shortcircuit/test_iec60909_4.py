# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

@pytest.fixture
def iec_60909_4():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=380.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110.)
    b4 = pp.create_bus(net, vn_kv=110.)
    b5 = pp.create_bus(net, vn_kv=110.)
    b6 = pp.create_bus(net, vn_kv=10.)
    b7 = pp.create_bus(net, vn_kv=10.)
    b8 = pp.create_bus(net, vn_kv=30.)
    H = pp.create_bus(net, vn_kv=30.)
    HG1 = pp.create_bus(net, vn_kv=21.)
    HG2 = pp.create_bus(net, vn_kv=10.5)  # 10.5kV?
    T_T5 = pp.create_bus(net, vn_kv=10.5)
    T_T6 = pp.create_bus(net, vn_kv=10.5)

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1)

    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5, xdss_pu=0.16, rdss_pu=0.005, cos_phi=0.9, sn_mva=100)
    pp.create_gen(net, HG1, p_mw=0.85 * 150, vn_kv=21, xdss_pu=0.14, rdss_pu=0.002, cos_phi=0.85, sn_mva=150)
    pp.create_gen(net, b6, p_mw=0.8 * 10, vn_kv=10.5, xdss_pu=0.1, rdss_pu=0.018, cos_phi=0.8, sn_mva=10)

    pp.create_transformer_from_parameters(net, b4, HG1, sn_mva=150,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vn_hv_kv=115., vn_lv_kv=21, vk_percent=16, vkr_percent=0.5)
    pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0,
        vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5)

    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=H,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=10, vkr_mv_percent=.16,
        vk_lv_percent=7., vkr_lv_percent=.16)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=10, vkr_mv_percent=.16,
        vk_lv_percent=7., vkr_lv_percent=.16)

    pp.create_transformer3w_from_parameters(net,
        hv_bus=b5, mv_bus=b6, lv_bus=T_T5,
        vn_hv_kv=115., vn_mv_kv=10.5, vn_lv_kv=10.5,
        sn_hv_mva=31.5, sn_mv_mva=31.5, sn_lv_mva=31.5,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vk_hv_percent=12, vkr_hv_percent=.5,
        vk_mv_percent=12, vkr_mv_percent=.5,
        vk_lv_percent=12, vkr_lv_percent=.5)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b5, mv_bus=b6, lv_bus=T_T6,
        vn_hv_kv=115., vn_mv_kv=10.5, vn_lv_kv=10.5,
        sn_hv_mva=31.5, sn_mv_mva=31.5, sn_lv_mva=31.5,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vk_hv_percent=12, vkr_hv_percent=.5,
        vk_mv_percent=12, vkr_mv_percent=.5,
        vk_lv_percent=12, vkr_lv_percent=.5)

    # pp.create_transformer_from_parameters(net, b5, b6, sn_mva=31.5,
    #     pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
    #     vn_hv_kv=115., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5)
    # pp.create_transformer_from_parameters(net, b5, b6, sn_mva=31.5,
    #     pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
    #     vn_hv_kv=115., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5)

    # First try without motor in powerfactory
    # pp.create_motor(net, b7, p_mw=5.0, sn_mva=5.0 / 0.88,  k=5, rx=0.1)  # FIXME: R/X
    # pp.create_motor(net, b7, p_mw=2.0, sn_mva=2.0 / 0.89, k=5.2, rx=0.1)
    # pp.create_motor(net, b7, p_mw=2.0, sn_mva=2.0 / 0.89, k=5.2, rx=0.1)

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b3, b4, name="L2",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b2, b5, name="L3a",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b2, b5, name="L3b",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.096, x_ohm_per_km=0.388)
    pp.create_line_from_parameters(net, b5, b4, name="L5",
        c_nf_per_km=0, max_i_ka=0,
        length_km=15, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b6, b7, name="L6",
        c_nf_per_km=0, max_i_ka=0,
        length_km=1, r_ohm_per_km=0.12, x_ohm_per_km=0.39)

    return net

@pytest.fixture
def iec_60909_4_small():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=380.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110.)
    b5 = pp.create_bus(net, vn_kv=110.)
    b8 = pp.create_bus(net, vn_kv=30.)
    H = pp.create_bus(net, vn_kv=30.)
    HG1 = pp.create_bus(net, vn_kv=21.)
    HG2 = pp.create_bus(net, vn_kv=10)  # 10.5kV?
    T_T5 = pp.create_bus(net, vn_kv=10.5)
    T_T6 = pp.create_bus(net, vn_kv=10.5)

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1)

    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.16, rdss_pu=0.005, cos_phi=0.9, sn_mva=100)

    pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0,
        vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5)

    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=H,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16)

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39,)
    pp.create_line_from_parameters(net, b2, b5, name="L3a",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b2, b5, name="L3b",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.096, x_ohm_per_km=0.388)

    return net


def iec_60909_4_small_gen_only():
    net = pp.create_empty_network()

    b3 = pp.create_bus(net, vn_kv=110.)
    HG2 = pp.create_bus(net, vn_kv=10)  # 10.5kV?

    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.16, rdss_pu=0.00453, cos_phi=0.9, sn_mva=100, slack=True)

    pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0,
        vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5)
    return net


def test_iec_60909_4_3ph_small_without_gen(iec_60909_4_small):
    net = iec_60909_4_small
    # Deactivate all gens
    net.gen = net.gen.iloc[0:0, :]
    
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [40.3390,	28.4130, 14.2095, 28.7195, 13.4191]
    ip_pf = [99.7374, 72.6580, 32.1954, 72.1443, 36.5036]
    
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:5], np.array(ikss_pf), atol=1e-4)
    assert np.allclose(net.res_bus_sc.ip_ka.values[:5], np.array(ip_pf), atol=1e-4)

# WIP
@pytest.mark.skip("Gens are not supported yet")
def test_iec_60909_4_3ph_small_with_gen(iec_60909_4_small):
    net = iec_60909_4_small
    
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [40.4754, 29.8334, 16.1684, 30.3573]
    ip_pf = [100.1164, 76.1134, 37.3576, 76.2689]
    ib_pf = [40.4754, 29.7337, 15.9593, 30.2245]
    
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf), atol=1e-4)
    assert np.allclose(net.res_bus_sc.ip_ka.values[:4], np.array(ip_pf), atol=1e-4)

@pytest.mark.skip("Gens are not supported yet")
def test_iec_60909_4_3ph_small_gen_only(iec_60909_4_small_gen_only):
    net = iec_60909_4_small_gen_only
    
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [1.9755, 39.5042]
    ip_pf = [5.2316, 104.1085]
    ib_pf = [1.6071, 27.3470]

    assert np.allclose(net.res_bus_sc.ikss_ka[:2].values, np.array(ikss_pf), atol=1e-4)
    assert np.allclose(net.res_bus_sc.ip_ka[:2].values, np.array(ip_pf), atol=1e-4)

@pytest.mark.skip("Gens are not supported yet")
def test_iec_60909_4_3ph_without_motor(iec_60909_4):
    net = iec_60909_4
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_pf = [40.6347, 31.6635, 19.6231, 16.1956, 32.9971, 34.3559, 22.2762, 13.5726]
    ip_pf = [100.5427, 80.3509, 45.7157, 36.7855, 82,9406, 90.6143, 43.3826, 36,9103]
    
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:8], np.array(ikss_pf), atol=1e-4)

# def test_iec_60909_4_3ph(iec_60909_4):
#     net = iec_60909_4
#     sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

#     ikss = [40.6447, 31.7831, 19.6730, 16.2277, 33.1894, 37.5629, 25.5895, 13.5778]
#     ip = [100.5766, 80.8249, 45.8249, 36.8041, 83.6266, 99.1910, 51.3864, 36.9201]
#     ib = [40.645, 31.570, 19.388, 16.017, 32.795, 34.028, 23.212, 13.578]

#     assert abs(net.res_bus_sc.ikss_ka.at[0] - ikss[0]) < 1e-4

#     assert np.allclose(net.res_bus_sc.ikss_ka.values[:8], np.array(ikss), atol=1e-4)
#     assert np.allclose(net.res_bus_sc.ip.values[:8], np.array(ip), rtol=1e-4)

if __name__ == '__main__':
    pytest.main(["test_iec60909_4.py"])

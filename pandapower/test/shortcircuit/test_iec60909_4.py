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
    HG2 = pp.create_bus(net, vn_kv=10.)  # 10.5kV?

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1)

    pp.create_gen(net, HG2, 0.9 * 100e3, vn_kv=10.5, xdss=0.16, rdss=0.005, cos_phi=0.9, sn_kva=100e3)
    pp.create_gen(net, HG1, 0.85 * 150e3, vn_kv=21, xdss=0.14, rdss=0.002, cos_phi=0.85, sn_kva=150e3)
    pp.create_gen(net, b6, 0.8 * 10e3, vn_kv=10.5, xdss=0.1, rdss=0.018, cos_phi=0.8, sn_kva=10e3)

    pp.create_transformer_from_parameters(net, b4, HG1, sn_kva=150e3,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vn_hv_kv=115., vn_lv_kv=21, vsc_percent=16, vscr_percent=0.5)
    pp.create_transformer_from_parameters(net, b3, HG2, sn_kva=100e3,
        pfe_kw=0, i0_percent=0,
        vn_hv_kv=120., vn_lv_kv=10.5, vsc_percent=12, vscr_percent=0.5)

    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=H,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_kva=350e3, sn_mv_kva=350e3, sn_lv_kva=50e3,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vsc_hv_percent=21, vscr_hv_percent=.26,
        vsc_mv_percent=10, vscr_mv_percent=.16,
        vsc_lv_percent=7., vscr_lv_percent=.16)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_kva=350e3, sn_mv_kva=350e3, sn_lv_kva=50e3,
        pfe_kw=0, i0_percent=0,
        vsc_hv_percent=21, vscr_hv_percent=.26,
        vsc_mv_percent=10, vscr_mv_percent=.16,
        vsc_lv_percent=7., vscr_lv_percent=.16)

    pp.create_transformer_from_parameters(net, b5, b6, sn_kva=31.5e3, parallel=2,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vn_hv_kv=115., vn_lv_kv=10.5, vsc_percent=12, vscr_percent=0.5)

    pp.create_sgen(net, b7, p_kw=5e3, sn_kva=5e3 / 0.88, type="motor", k=5, rx=0.1)  # FIXME: R/X

    pp.create_sgen(net, b7, p_kw=2e3, sn_kva=2e3 / 0.89, type="motor", k=5.2, rx=0.1)
    pp.create_sgen(net, b7, p_kw=2e3, sn_kva=2e3 / 0.89, type="motor", k=5.2, rx=0.1)

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b3, b4, name="L2",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b2, b5, name="L3a+b", parallel=2,
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b5, b4, name="L5",
        c_nf_per_km=0, max_i_ka=0,
        length_km=15, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b6, b7, name="L6",
        c_nf_per_km=0, max_i_ka=0,
        length_km=1, r_ohm_per_km=0.12, x_ohm_per_km=0.39)

    return net

def test_iec_60909_4_3ph(iec_60909_4):
    net = iec_60909_4
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss = [40.6447, 31.7831, 19.6730, 16.2277, 33.1894, 37.5629, 25.5895, 13.5778]
    ip = [100.5766, 80.8249, 45.8249, 36.8041, 83.6266, 99.1910, 51.3864, 36.9201]
    ib = [40.645, 31.570, 19.388, 16.017, 32.795, 34.028, 23.212, 13.578]

    assert abs(net.res_bus_sc.ikss_ka.at[0] - ikss[0]) < 1e-4

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:8], np.array(ikss), rtol=1e-4)
    assert np.allclose(net.res_bus_sc.ip.values[:8], np.array(ip), rtol=1e-4)

if __name__ == '__main__':
    pytest.main(["test_iec60909_4.py"])

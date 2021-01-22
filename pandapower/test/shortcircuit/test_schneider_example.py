# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

def schneider_example():
    net = pp.create_empty_network()

    bg = pp.create_bus(net, vn_kv=20.)
    bt = pp.create_bus(net, vn_kv=220.)

    t = pp.create_transformer_from_parameters(net, bt, bg, sn_mva=250,
        pfe_kw=0, i0_percent=0,
        vn_hv_kv=240., vn_lv_kv=21., vk_percent=15., vkr_percent=0.208, pt_percent=0.0)

    pp.create_gen(net, bg, p_mw=0.9 * 100, vn_kv=21,
                  xdss_pu=0.17, rdss_ohm=0.0025, cos_phi=0.78, sn_mva=250, pg_percent=0.0,
                  slack=True, power_station_trafo=t)
    return net

def test_schneider_example():
    net = schneider_example()
    
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    assert np.allclose(net.res_bus_sc.loc[:, "ikss_ka"].values, [44.74, 2.08], atol=1e-2)
    # assert np.allclose(net.res_bus_sc.loc[:, "ip_ka"], [117.69, 5.61], atol=1e-2)
    # assert np.allclose(net.res_bus_sc.loc[:, "ib_ka"], [31.77, 1.77], atol=1e-2)

if __name__ == '__main__':
    pytest.main([__file__])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import create_empty_network, create_bus, create_ext_grid, \
    create_transformer_from_parameters
from pandapower.shortcircuit.calc_sc import calc_sc

S_SC = 250.
RX = 0.1
SN_T = 0.63


def _net(vk_percent, vkr_percent):
    net = create_empty_network()
    b_mv = create_bus(net, vn_kv=20.)
    b_lv = create_bus(net, vn_kv=0.4)
    create_ext_grid(net, b_mv, s_sc_max_mva=S_SC, s_sc_min_mva=S_SC, rx_max=RX, rx_min=RX,
                    x0x_max=1.0, r0x0_max=0.1, x0x_min=1.0, r0x0_min=0.1)
    create_transformer_from_parameters(net, b_mv, b_lv, sn_mva=SN_T, vn_hv_kv=20., vn_lv_kv=0.4,
                                       vk_percent=vk_percent, vkr_percent=vkr_percent,
                                       pfe_kw=0., i0_percent=0., shift_degree=0.,
                                       vector_group="Dyn", vk0_percent=vk_percent,
                                       vkr0_percent=vkr_percent, mag0_percent=100.,
                                       mag0_rx=0., si0_hv_partial=0.9)
    return net


def _ikss_lv_analytic(c, vk_percent, vkr_percent):
    zq = c * 20. ** 2 / S_SC
    xq = zq / np.sqrt(1 + RX ** 2)
    zq_complex = RX * xq + 1j * xq

    z_base_lv = 0.4 ** 2 / SN_T
    z_t = vk_percent / 100 * z_base_lv
    r_t = vkr_percent / 100 * z_base_lv
    x_t = np.sqrt(z_t ** 2 - r_t ** 2)
    k_t = 0.95 * 1.05 / (1 + 0.6 * x_t / z_base_lv)

    z_lv = zq_complex * (0.4 / 20.) ** 2 + k_t * (r_t + 1j * x_t)
    return c * 0.4 / (np.sqrt(3) * abs(z_lv))


@pytest.mark.parametrize("vk_percent, vkr_percent", [(1e-4, 1e-6), (4.0, 1.2)])
def test_zq_uses_c_max_of_fault_location(vk_percent, vkr_percent):
    # c_max is 1.05 at the 0.4 kV fault bus and 1.1 at the 20 kV ext_grid bus
    net = _net(vk_percent, vkr_percent)
    calc_sc(net, case="max", lv_tol_percent=6)
    assert abs(net.res_bus_sc.ikss_ka.at[1] / _ikss_lv_analytic(1.05, vk_percent, vkr_percent) - 1) < 1e-4


@pytest.mark.parametrize("lv_tol_percent, c_min_lv", [(6, 0.95), (10, 0.9)])
def test_zq_uses_c_min_of_fault_location(lv_tol_percent, c_min_lv):
    # c_min is 1.0 above 1 kV but 0.9 / 0.95 in LV, so the min case is affected at any tolerance
    net = _net(1e-4, 1e-6)
    calc_sc(net, case="min", lv_tol_percent=lv_tol_percent)
    zq = c_min_lv * 20. ** 2 / S_SC
    xq = zq / np.sqrt(1 + RX ** 2)
    z_lv = abs(RX * xq + 1j * xq) * (0.4 / 20.) ** 2
    assert abs(net.res_bus_sc.ikss_ka.at[1] / (c_min_lv * 0.4 / (np.sqrt(3) * z_lv)) - 1) < 1e-3


def test_zq_at_ext_grid_bus_unaffected():
    # the fault bus is the ext_grid bus, so its own c applies and nothing is rescaled
    net = _net(4.0, 1.2)
    calc_sc(net, case="max", lv_tol_percent=6)
    assert abs(net.res_bus_sc.ikss_ka.at[0] - S_SC / (np.sqrt(3) * 20.)) < 1e-6


@pytest.mark.parametrize("fault", ["3ph", "2ph", "1ph"])
def test_all_c_equal_gives_single_evaluation(fault):
    # c_max is 1.1 in every voltage level for lv_tol_percent=10, so the fault location cannot matter
    net_single = _net(4.0, 1.2)
    calc_sc(net_single, fault=fault, case="max", lv_tol_percent=10, bus=1)
    net_all = _net(4.0, 1.2)
    calc_sc(net_all, fault=fault, case="max", lv_tol_percent=10)
    assert abs(net_single.res_bus_sc.ikss_ka.at[1] - net_all.res_bus_sc.ikss_ka.at[1]) < 1e-9


@pytest.mark.parametrize("case", ["max", "min"])
@pytest.mark.parametrize("fault", ["3ph", "2ph", "1ph"])
def test_bus_subset_matches_full_calculation(fault, case):
    # results must not depend on which buses are requested together
    net_all = _net(4.0, 1.2)
    calc_sc(net_all, fault=fault, case=case, lv_tol_percent=6)
    for b in (0, 1):
        net_one = _net(4.0, 1.2)
        calc_sc(net_one, fault=fault, case=case, lv_tol_percent=6, bus=b)
        assert abs(net_one.res_bus_sc.ikss_ka.at[b] - net_all.res_bus_sc.ikss_ka.at[b]) < 1e-9


@pytest.mark.parametrize("inverse_y", [True, False])
def test_inverse_y_and_factorization_agree(inverse_y):
    net = _net(4.0, 1.2)
    calc_sc(net, case="max", lv_tol_percent=6, inverse_y=inverse_y)
    assert abs(net.res_bus_sc.ikss_ka.at[1] / _ikss_lv_analytic(1.05, 4.0, 1.2) - 1) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

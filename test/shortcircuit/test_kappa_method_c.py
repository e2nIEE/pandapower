# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import (
    create_bus,
    create_ext_grid,
    create_line_from_parameters,
    create_transformer3w_from_parameters,
)
from pandapower.network import pandapowerNet
from pandapower.shortcircuit.calc_sc import calc_sc


def single_fed_feeder(rx_grid):
    net = pandapowerNet(name="single_fed_feeder", sn_mva=1.0)
    grid_bus = create_bus(net, vn_kv=20.0)
    remote_bus = create_bus(net, vn_kv=20.0)
    create_ext_grid(net, grid_bus, s_sc_max_mva=300.0, rx_max=rx_grid)
    create_line_from_parameters(
        net, grid_bus, remote_bus, length_km=5.0, r_ohm_per_km=0.2, x_ohm_per_km=0.35, c_nf_per_km=0, max_i_ka=1
    )
    return net


@pytest.mark.parametrize("rx_grid", [0.0, 0.1])
@pytest.mark.parametrize("inverse_y", (True, False), ids=("Inverse Y", "LU factorization"))
def test_method_c_reproduces_r_over_x_of_a_single_fed_short_circuit(rx_grid, inverse_y):
    # For impedances in series, evaluating every X at fc and scaling R/X back by fc/f (IEC 60909-0:2016, 8.1.2 c))
    # gives exactly the R/X of the short-circuit impedance, so method C must equal the single-fed ip. A network feeder
    # with R/X = 0 is a purely reactive shunt (GS = 0) and was previously left at 50 Hz.
    net = single_fed_feeder(rx_grid)
    calc_sc(net, ip=True, kappa_method="C", topology="auto", inverse_y=inverse_y)
    ip_method_c_ka = net.res_bus_sc.ip_ka.values
    calc_sc(net, ip=True, topology="radial", inverse_y=inverse_y)
    assert np.allclose(ip_method_c_ka, net.res_bus_sc.ip_ka.values, atol=1e-9, rtol=0)


def test_method_c_kappa_is_capped_at_two():
    # Losses on one winding pair only give a negative star resistance in the three-winding transformer model; the
    # equivalent R/X can then turn negative, but kappa must not exceed 2
    net = pandapowerNet(name="three_winding_negative_star_resistance", sn_mva=1.0)
    hv_bus = create_bus(net, vn_kv=110.0)
    mv_bus = create_bus(net, vn_kv=20.0)
    lv_bus = create_bus(net, vn_kv=10.0)
    create_ext_grid(net, hv_bus, s_sc_max_mva=5000.0, rx_max=0.1)
    create_ext_grid(net, mv_bus, s_sc_max_mva=500.0, rx_max=0.1)
    create_transformer3w_from_parameters(
        net,
        hv_bus,
        mv_bus,
        lv_bus,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=40.0,
        sn_mv_mva=40.0,
        sn_lv_mva=20.0,
        vk_hv_percent=12.0,
        vk_mv_percent=6.0,
        vk_lv_percent=8.0,
        vkr_hv_percent=0.8,
        vkr_mv_percent=0.0,
        vkr_lv_percent=0.0,
        pfe_kw=0.0,
        i0_percent=0.0,
    )
    calc_sc(net, ip=True, kappa_method="C")
    kappa = net.res_bus_sc.ip_ka.values / (np.sqrt(2) * net.res_bus_sc.ikss_ka.values)
    assert np.all(kappa <= 2.0 + 1e-12)

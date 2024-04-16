# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


@pytest.fixture
def meshed_grid():
    net = pp.from_json(os.path.join(pp.pp_dir, "test", "shortcircuit", "sc_test_meshed_grid.json"))
    bid = pp.create_bus(net, vn_kv=10.)
    pp.create_switch(net, net.ext_grid.bus.iloc[0], bid, et="b")
    net.ext_grid.bus.iloc[0] = bid
    pp.create_bus(net, vn_kv=0.4, in_service=False)
    return net

def test_max_10_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent= 10., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [5.773503, 14.82619, 4.606440, 4.068637, 13.61509,
                        2.812111, 1.212288, 1.525655, 1.781087, 1.568337], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [14.256050, 33.751300, 6.759302, 6.359403, 26.49241,
                        4.726619, 2.015958, 2.538654, 2.576375, 2.608065], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [5.871191, 14.97527, 4.613454, 4.077662, 13.68449,
                        2.820525, 1.215770, 1.530048, 1.783442, 1.572843], atol=1e-5)


def test_max_6_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent = 6., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [5.773503, 14.75419, 4.437882, 4.068637, 13.53425,
                        2.701411, 1.159945, 1.460757, 1.705172, 1.501673], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [14.25605, 33.59996, 6.50406, 6.359403, 26.28476,
                        4.537759, 1.928734, 2.430331, 2.466185, 2.496901], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [5.871191, 14.90284, 4.44457, 4.077662, 13.60275,
                        2.709475, 1.163276, 1.464961, 1.707423, 1.505985], atol=1e-5)


def test_min_10_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent= 10., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [2.309401, 11.3267, 2.879343, 1.884323, 10.40083,
                        1.693922, 0.7107017, 0.9000445, 1.055881, 0.928488], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 26.01655, 4.166047, 3.124163, 20.04053,
                        2.813883, 1.179085, 1.493293, 1.523338, 1.540432], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 11.44622, 2.883161, 1.889675, 10.45195,
                        1.698768, 0.712725, 0.9026074, 1.057233, 0.9311316], atol=1e-5)


def test_min_6_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent = 6., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [2.309401, 11.75072, 2.895465, 1.884323, 10.77961,
                        1.700202, 0.7116519, 0.9016006, 1.0576, 0.9301236], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 27.00861, 4.18812, 3.124163, 20.72881,
                        2.824028, 1.180654, 1.495858, 1.525799, 1.543131], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 11.87518, 2.899291, 1.889675, 10.8322,
                        1.705064, 0.7136779, 0.9041679, 1.058954, 0.9327717], atol=1e-5)

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

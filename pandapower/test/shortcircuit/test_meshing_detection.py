# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os

import numpy as np
import pytest

from pandapower import pp_dir
from pandapower.create import create_bus, create_switch
from pandapower.file_io import from_json
from pandapower.shortcircuit.calc_sc import calc_sc


@pytest.fixture
def meshed_grid():
    net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "sc_test_meshed_grid.json"))
    bid = create_bus(net, vn_kv=10.)
    create_switch(net, net.ext_grid.bus.iloc[0], bid, et="b")
    net.ext_grid.loc[net.ext_grid.index[0], "bus"] = bid
    create_bus(net, vn_kv=0.4, in_service=False)
    return net


def test_max_10_meshed_grid(meshed_grid):
    net = meshed_grid
    calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=10., kappa_method="B")
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
    calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=6., kappa_method="B")
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
    calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent=10., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [2.309401, 10.73055529, 2.72779866, 1.884323, 9.8534192,
                        1.60476785, 0.67329634, 0.8526737, 1.00030861, 0.87962022], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 24.64726031, 3.94678188, 3.124163, 18.98576367,
                        2.66578429, 1.11702772, 1.41469836, 1.44316232, 1.45935648], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 10.84378991, 2.73141581, 1.889675, 9.90184777,
                        1.60935911, 0.67521318, 0.85510173, 1.00158882, 0.88212465], atol=1e-5)


def test_min_6_meshed_grid(meshed_grid):
    net = meshed_grid
    calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent=6., kappa_method="B")
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

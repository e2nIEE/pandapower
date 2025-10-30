# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
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
                       [2.309401, 11.409498, 2.882583, 1.884323, 10.571978,
                        1.696813, 0.711139, 0.900761, 1.056227, 0.929242], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 26.210173, 4.170478, 3.124163, 20.351853,
                        2.818552, 1.179808, 1.494475, 1.523833, 1.541675], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 11.529984, 2.886403, 1.889675, 10.623765,
                        1.701667, 0.713164, 0.903326, 1.057579, 0.931887], atol=1e-5)


def test_min_6_meshed_grid(meshed_grid):
    net = meshed_grid
    calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent=6., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [2.309401, 11.409498, 2.882583, 1.884323, 10.571978,
                        1.696813, 0.711139, 0.900761, 1.056227, 0.929242], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 26.210173, 4.170478, 3.124163, 20.351853,
                        2.818552, 1.179808, 1.494475, 1.523833, 1.541675], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 11.529984, 2.886403, 1.889675, 10.623765,
                        1.701667, 0.713164, 0.903326, 1.057579, 0.931887], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

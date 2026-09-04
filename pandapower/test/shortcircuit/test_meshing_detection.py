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
                       [5.773503, 14.826194, 4.442226, 4.068637, 13.593368,
                        2.702972, 1.160182, 1.461142, 1.705690, 1.502069], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [14.256045, 33.751297, 6.509470, 6.359402, 26.376542,
                        4.540027, 1.929106, 2.430930, 2.466889, 2.497524], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [5.871191, 14.975270, 4.448912, 4.077662, 13.661941,
                        2.711038, 1.163513, 1.465347, 1.707941, 1.506382], atol=1e-5)


def test_min_10_meshed_grid(meshed_grid):
    net = meshed_grid
    calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent=10., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [2.309401, 11.038191, 2.738712, 1.884323, 10.206614,
                        1.610008, 0.674064, 0.853936, 1.001438, 0.880943], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 25.320227, 3.961612, 3.124163, 19.571930,
                        2.674199, 1.118293, 1.416777, 1.444777, 1.461538], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 11.153828, 2.742333, 1.889675, 10.255897,
                        1.614613, 0.675983, 0.856368, 1.002719, 0.883452], atol=1e-5)


def test_min_6_meshed_grid(meshed_grid):
    net = meshed_grid
    calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent=6., kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10],
                       [2.309401, 11.529194, 2.886724, 1.884323, 10.671890,
                        1.698134, 0.711326, 0.901069, 1.056650, 0.929563], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ip_ka.values[:10],
                       [5.702418, 26.466020, 4.176088, 3.124163, 20.504414,
                        2.820659, 1.180114, 1.494981, 1.524438, 1.542205], atol=1e-5)

    assert np.allclose(net.res_bus_sc.ith_ka.values[:10],
                       [2.348476, 11.650463, 2.890545, 1.889675, 10.723793,
                        1.702991, 0.713351, 0.903635, 1.058003, 0.932210], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

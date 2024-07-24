# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np
import logging

import pandapower as pp
from pandapower.control import Characteristic, SplineCharacteristic, TapDependentImpedance, \
    trafo_characteristics_diagnostic


def test_tap_dependent_impedance_control():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    pp.create_ext_grid(net, b1)
    pp.create_transformer_from_parameters(net, b1, b2, 40, 110, 21, 0.5, 12.3, 25, 0.11, 0, 'hv', 10, 20, 0, 1.8, 180, 10)

    characteristic_vk = Characteristic.from_points(net, ((0, 13.5), (10, 12.3), (20, 11.1)))
    characteristic_vkr = Characteristic.from_points(net, ((0, 0.52), (10, 0.5), (20, 0.53)))
    TapDependentImpedance(net, 0, characteristic_vk.index, output_variable='vk_percent', restore=False)
    TapDependentImpedance(net, 0, characteristic_vkr.index, output_variable='vkr_percent', restore=False)

    pp.runpp(net, run_control=True)
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5

    net.trafo.tap_pos = 0
    pp.runpp(net, run_control=True)
    assert net.trafo.vk_percent.at[0] == 13.5
    assert net.trafo.vkr_percent.at[0] == 0.52

    net.trafo.tap_pos = 20
    pp.runpp(net, run_control=True)
    assert net.trafo.vk_percent.at[0] == 11.1
    assert net.trafo.vkr_percent.at[0] == 0.53


def test_tap_dependent_impedance_restore():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b2, 20)
    pp.create_transformer_from_parameters(net, b1, b2, 40, 110, 21, 0.5, 12.3, 25, 0.11, 0, 'hv', 10, 20, 0, 1.8, 180, 10)

    characteristic_vk = Characteristic.from_points(net, ((0, 13.5), (10, 12.3), (20, 11.1)))
    characteristic_vkr = Characteristic.from_points(net, ((0, 0.52), (10, 0.5), (20, 0.53)))
    TapDependentImpedance(net, 0, characteristic_vk.index, output_variable='vk_percent', restore=True)
    TapDependentImpedance(net, 0, characteristic_vkr.index, output_variable='vkr_percent', restore=True)

    pp.runpp(net, run_control=True)
    # remember the losses for the neutral position
    pl_mw_neutral = net.res_trafo.pl_mw.at[0]
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5

    net.trafo.tap_pos = 0
    pp.runpp(net, run_control=True)
    # check if the impedance has been restored
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5
    # check if the losses are different than at the neutral position -> the tap-dependent impedance has been conbsidered
    assert abs(net.res_trafo.pl_mw.at[0] - pl_mw_neutral) > 0.015

    net.trafo.tap_pos = 20
    pp.runpp(net, run_control=True)
    # check if the impedance has been restored
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5
    # check if the losses are different than at the neutral position -> the tap-dependent impedance has been conbsidered
    assert abs(net.res_trafo.pl_mw.at[0] - pl_mw_neutral) > 0.002


def test_characteristic():
    net = pp.create_empty_network()
    x_points = [0, 1, 2]
    y_points = [3, 4, 5]
    c = Characteristic(net, x_points, y_points)

    assert np.array_equal(c(x_points), y_points)
    # bounds are fixed:
    assert c(-1) == 3
    assert c(3) == 5

    assert c.satisfies(0.5, 3.5, 1e-9)
    assert not c.satisfies(0.5, 4, 1e-9)

    assert c.diff(1, 5) == 1

    # testing alternative constructors:
    c1 = Characteristic.from_points(net, ((0, 3), (1, 4), (2, 5)))
    assert np.array_equal(c1(x_points), y_points)

    c2 = Characteristic.from_gradient(net, 3, 1, 3, 5)
    assert np.array_equal(c2(x_points), y_points)

    x_spline = [0, 1, 2]
    y_spline = [0, 1, 4]
    c3 = SplineCharacteristic(net, x_points, y_spline)

    assert np.array_equal(c3(x_spline), y_spline)
    assert c3(1.5) == 2.25
    assert c3(3) == 9

    c4 = SplineCharacteristic(net, x_points, y_spline, fill_value=(y_spline[0], y_spline[-1]))
    assert c4(2) == 4


def test_characteristic_diagnostic():
    net = pp.create_empty_network()
    vn_kv = 20
    b1 = pp.create_bus(net, vn_kv=vn_kv)
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    b2 = pp.create_bus(net, vn_kv=vn_kv)
    l1 = pp.create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                        c_nf_per_km=300, max_i_ka=.2, df=.8)
    cb = pp.create_bus(net, vn_kv=0.4)
    pp.create_load(net, cb, 0.2, 0.05)
    pp.create_transformer(net, hv_bus=b2, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)

    pp.control.create_trafo_characteristics(net, 'trafo', 0, 'vk_percent',
                                            [-2, -1, 0, 1, 2], [5, 5.2, 6, 6.8, 7])  # single mode
    pp.control.create_trafo_characteristics(net, 'trafo', [0], 'vkr_percent',
                                            [[-2, -1, 0, 1, 2]], [[1.3, 1.4, 1.44, 1.5, 1.6]])  # multiple indices

    # let's make some invalid configurations
    net.trafo.at[0, "vk_percent"] += 1e-6
    # missing any characteristic
    pp.create_transformer(net, hv_bus=net.trafo.at[0, 'hv_bus'],
                          lv_bus=net.trafo.at[0, 'lv_bus'], std_type="0.25 MVA 20/0.4 kV", tap_pos=2,
                          tap_dependent_impedance=True)
    b2 = net.trafo.at[0, "hv_bus"]
    cb = pp.create_bus(net, vn_kv=0.4)
    pp.create_load(net, cb, 0.2, 0.05)
    pp.create_transformer(net, hv_bus=b2, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)
    # missing columns for characteristics

    cbm = pp.create_bus(net, vn_kv=0.9)
    pp.create_load(net, cbm, 0.1, 0.03)
    pp.create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=cbm, lv_bus=cb,
                                            vn_hv_kv=20., vn_mv_kv=0.9, vn_lv_kv=0.45, sn_hv_mva=0.6,
                                            sn_mv_mva=0.5, sn_lv_mva=0.4, vk_hv_percent=1.,
                                            vk_mv_percent=1., vk_lv_percent=1., vkr_hv_percent=0.3,
                                            vkr_mv_percent=0.3, vkr_lv_percent=0.3, pfe_kw=0.2,
                                            i0_percent=0.3, tap_neutral=0., tap_pos=2,
                                            tap_step_percent=1., tap_min=-2, tap_max=2)

    net.trafo3w['tap_dependent_impedance'] = True

    import io
    s = io.StringIO()
    h = logging.StreamHandler(stream=s)
    logger = pp.control.util.diagnostic.logger
    logger.addHandler(h)

    old_level = logger.level
    logger.setLevel("INFO")
    trafo_characteristics_diagnostic(net)
    logger.setLevel(old_level)

    msg = s.getvalue()

    assert "trafo: found 2 transformer(s) with tap-dependent impedance" in msg
    assert "Power flow calculation will raise an error" in msg
    assert "vk_percent_characteristic is missing for some transformers" in msg
    assert "vk_percent_characteristic is missing for some transformers" in msg
    assert "vkr_percent_characteristic is missing for some transformers" in msg
    assert "The characteristic value of 6.0 at the neutral tap position 0 does not match the value 6.000001" in msg
    assert "trafo3w: found 1 transformer(s) with tap-dependent impedance" in msg
    assert "No columns defined for transformer tap characteristics in trafo3w" in msg
    assert "vk_hv_percent_characteristic is missing" in msg
    assert "vkr_hv_percent_characteristic is missing" in msg
    assert "vk_mv_percent_characteristic is missing" in msg
    assert "vkr_mv_percent_characteristic is missing" in msg
    assert "vk_lv_percent_characteristic is missing" in msg
    assert "vkr_lv_percent_characteristic is missing" in msg

    pp.control.util.diagnostic.logger.removeHandler(h)
    del h
    del s


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
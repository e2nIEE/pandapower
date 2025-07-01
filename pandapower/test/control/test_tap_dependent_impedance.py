# -*- coding: utf-8 -*-
# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandas as pd
import pytest
import numpy as np

from pandapower.control import Characteristic, SplineCharacteristic, TapDependentImpedance, \
    trafo_characteristic_table_diagnostic
from pandapower.control.util.diagnostic import shunt_characteristic_table_diagnostic
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_transformer_from_parameters, \
    create_load, create_line_from_parameters, create_transformer, create_shunt
from pandapower.run import runpp


def test_tap_dependent_impedance_control():
    net = create_empty_network()
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    create_ext_grid(net, b1)
    create_transformer_from_parameters(net, b1, b2, 40, 110, 21, 0.5, 12.3, 25, 0.11, 0, 'hv', 10, 20, 0, 1.8, 180, 10,
                                       tap_changer_type="Ratio")

    characteristic_vk = Characteristic.from_points(net, ((0, 13.5), (10, 12.3), (20, 11.1)))
    characteristic_vkr = Characteristic.from_points(net, ((0, 0.52), (10, 0.5), (20, 0.53)))
    TapDependentImpedance(net, 0, characteristic_vk.index, output_variable='vk_percent', restore=False)
    TapDependentImpedance(net, 0, characteristic_vkr.index, output_variable='vkr_percent', restore=False)

    runpp(net, run_control=True)
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5

    net.trafo.tap_pos = 0
    runpp(net, run_control=True)
    assert net.trafo.vk_percent.at[0] == 13.5
    assert net.trafo.vkr_percent.at[0] == 0.52

    net.trafo.tap_pos = 20
    runpp(net, run_control=True)
    assert net.trafo.vk_percent.at[0] == 11.1
    assert net.trafo.vkr_percent.at[0] == 0.53


def test_tap_dependent_impedance_restore():
    net = create_empty_network()
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    create_ext_grid(net, b1)
    create_load(net, b2, 20)
    create_transformer_from_parameters(net, b1, b2, 40, 110, 21, 0.5, 12.3, 25, 0.11, 0, 'hv', 10, 20, 0, 1.8, 180, 10,
                                       tap_changer_type="Ratio")

    characteristic_vk = Characteristic.from_points(net, ((0, 13.5), (10, 12.3), (20, 11.1)))
    characteristic_vkr = Characteristic.from_points(net, ((0, 0.52), (10, 0.5), (20, 0.53)))
    TapDependentImpedance(net, 0, characteristic_vk.index, output_variable='vk_percent', restore=True)
    TapDependentImpedance(net, 0, characteristic_vkr.index, output_variable='vkr_percent', restore=True)

    runpp(net, run_control=True)
    # remember the losses for the neutral position
    pl_mw_neutral = net.res_trafo.pl_mw.at[0]
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5

    net.trafo.tap_pos = 0
    runpp(net, run_control=True)
    # check if the impedance has been restored
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5
    # check if the losses are different from the neutral position -> the tap-dependent impedance has been considered
    assert abs(net.res_trafo.pl_mw.at[0] - pl_mw_neutral) > 0.015

    net.trafo.tap_pos = 20
    runpp(net, run_control=True)
    # check if the impedance has been restored
    assert net.trafo.vk_percent.at[0] == 12.3
    assert net.trafo.vkr_percent.at[0] == 0.5
    # check if the losses are different from the neutral position -> the tap-dependent impedance has been considered
    assert abs(net.res_trafo.pl_mw.at[0] - pl_mw_neutral) > 0.002


def test_characteristic():
    net = create_empty_network()
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


def test_trafo_characteristic_table_diagnostic():
    net = create_empty_network()
    vn_kv = 20
    b1 = create_bus(net, vn_kv=vn_kv)
    create_ext_grid(net, b1, vm_pu=1.01)
    b2 = create_bus(net, vn_kv=vn_kv)
    create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                        c_nf_per_km=300, max_i_ka=.2, df=.8)
    cb = create_bus(net, vn_kv=0.4)
    create_load(net, cb, 0.2, 0.05)
    create_transformer(net, hv_bus=b2, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)

    # initially no trafo_characteristic_table is available
    assert trafo_characteristic_table_diagnostic(net) is False

    # add trafo_characteristic_table
    net["trafo_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_percent': [5, 5.2, 6, 6.8, 7],
         'vkr_percent': [1.3, 1.4, 1.44, 1.5, 1.6], 'vk_hv_percent': np.nan, 'vkr_hv_percent': np.nan,
         'vk_mv_percent': np.nan, 'vkr_mv_percent': np.nan, 'vk_lv_percent': np.nan, 'vkr_lv_percent': np.nan})
    # populate id_characteristic_table parameter
    net.trafo['id_characteristic_table'].at[0] = 0
    net.trafo['tap_dependency_table'].at[0] = False
    with pytest.warns(UserWarning):
        trafo_characteristic_table_diagnostic(net)
    # populate tap_dependency_table parameter
    net.trafo['tap_dependency_table'].at[0] = True
    assert trafo_characteristic_table_diagnostic(net) is True

    # add trafo_characteristic_table with missing parameter values
    net["trafo_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_percent': [5, 5.2, 6, 6.8, 7],
         'vkr_percent': np.nan, 'vk_hv_percent': np.nan, 'vkr_hv_percent': np.nan,
         'vk_mv_percent': np.nan, 'vkr_mv_percent': np.nan, 'vk_lv_percent': np.nan, 'vkr_lv_percent': np.nan})
    with pytest.warns(UserWarning):
        trafo_characteristic_table_diagnostic(net)

    # let's make some invalid configurations
    net.trafo.at[0, "tap_dependency_table"] = 0
    with pytest.warns(UserWarning):
        trafo_characteristic_table_diagnostic(net)
    net.trafo.at[0, "tap_dependency_table"] = True
    net.trafo.at[0, "id_characteristic_table"] = int(2)
    with pytest.warns(UserWarning):
        trafo_characteristic_table_diagnostic(net)


def test_shunt_characteristic_table_diagnostic():
    net = create_empty_network()
    vn_kv = 20
    b1 = create_bus(net, vn_kv=vn_kv)
    create_shunt(net, bus=b1, q_mvar=-50, p_mw=0, step=1, max_step=5)
    create_ext_grid(net, b1, vm_pu=1.01)
    b2 = create_bus(net, vn_kv=vn_kv)
    create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    cb = create_bus(net, vn_kv=0.4)
    create_load(net, cb, 0.2, 0.05)
    create_transformer(net, hv_bus=b2, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)

    # initially no shunt_characteristic_table is available
    assert shunt_characteristic_table_diagnostic(net) is False

    # add shunt_characteristic_table
    net["shunt_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [1, 2, 3, 4, 5], 'q_mvar': [-25, -55, -75, -120, -125],
         'p_mw': [1, 1.5, 3, 4.5, 5]})
    # populate id_characteristic_table parameter
    net.shunt['id_characteristic_table'].at[0] = 0
    net.shunt['step_dependency_table'].at[0] = False
    with pytest.warns(UserWarning):
        shunt_characteristic_table_diagnostic(net)
    # populate step_dependency_table parameter
    net.shunt['step_dependency_table'].at[0] = True
    assert shunt_characteristic_table_diagnostic(net) is True

    # add shunt_characteristic_table with missing parameter values
    net["shunt_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [1, 2, 3, 4, 5], 'q_mvar': [-25, -55, -75, -120, -125],
         'p_mw': np.nan})
    with pytest.warns(UserWarning):
        shunt_characteristic_table_diagnostic(net)

    # let's make some invalid configurations
    net.shunt.at[0, "step_dependency_table"] = 0
    with pytest.warns(UserWarning):
        shunt_characteristic_table_diagnostic(net)
    net.shunt.at[0, "step_dependency_table"] = True
    net.shunt.at[0, "id_characteristic_table"] = int(2)
    with pytest.warns(UserWarning):
        shunt_characteristic_table_diagnostic(net)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

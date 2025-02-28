# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import pandapower as pp
from pandapower.control import Characteristic, TapDependentImpedance


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


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
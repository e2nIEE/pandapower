# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import pytest
from math import pi
from pandapower.create import create_empty_network, create_bus, create_sgen, create_load


def test_calculate_unique_angles():
    from pandapower.plotting.simple_plot import calculate_unique_angles

    net = create_empty_network()
    b = create_bus(net, 50, "bus")
    b1 = create_bus(net, 50, "bus")

    create_sgen(net, b, 5)
    one = calculate_unique_angles(net)

    assert one == {
        0: {"sgen": {"none": 0.0}}
    }

    create_sgen(net, b, 5)
    two = calculate_unique_angles(net)
    assert two == {
        0: {"sgen": {"none": 0.0}}
    }

    create_sgen(net, b, 5, type="PV")
    three = calculate_unique_angles(net)
    assert three == {
        0: {"sgen": {"PV": 0.0, "none": pi}}
    }

    create_load(net, b1, 5)
    create_sgen(net, b1, 5, type="custom")
    create_sgen(net, b1, 5, type="WP")
    create_sgen(net, b1, 5, type="PV")
    four = calculate_unique_angles(net)
    assert four == {
        0: {"sgen": {"PV": 0.0, "none": pi}},
        1: {"sgen": {"PV": 0.0, "WP": pi / 2, "custom": pi}, "load": pi / 2 * 3}
    }


@pytest.mark.xfail(raises=NotImplementedError)
def test_simple_plot():
    raise NotImplementedError

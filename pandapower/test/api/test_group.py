import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks

from pandapower.test.toolbox import assert_net_equal


def group_example_simple():
    net = pp.networks.example_simple()
    pp.group.add_elements(net, "test", "bus")
    pp.group.add_elements(net, "test", "line", [1, 2])
    pp.group.add_elements(net, "hello", "line", [2, 3], element_index_column="name")
    pp.group.add_elements(net, "test", "trafo", [0])
    pp.group.add_elements(net, "test", "switch", [0, 1, 2, 3])
    pp.group.add_elements(net, "hello", "switch", [2, 3, 4, 5])

    return net


def test_group_set_value():
    pass


def test_group_set_in_service():
    net = group_example_simple()
    pp.group.set_in_service(net, "test", False)

    assert np.all(~net.bus.in_service)

    assert np.all(~net.line.loc[[1, 2]].in_service)
    assert np.all(net.line.loc[np.setdiff1d(net.line.index, [1, 2])].in_service)

    assert np.all(~net.trafo.loc[[0]].in_service)
    assert np.all(net.trafo.loc[np.setdiff1d(net.trafo.index, [0])].in_service)

    assert "in_service" not in net.switch.columns


def test_add_elements():
    net = group_example_simple()
    group_tab = net.group.loc[net.group.name == "test"]

    assert np.array_equal(group_tab.loc[group_tab.element == "bus", "element_index"], net.bus.index)
    assert np.array_equal(group_tab.loc[group_tab.element == "line", "element_index"], [1, 2])
    assert np.array_equal(group_tab.loc[group_tab.element == "trafo", "element_index"], [0])
    assert np.array_equal(group_tab.loc[group_tab.element == "switch", "element_index"], [0, 1, 2, 3])


def test_json_io():
    net = group_example_simple()
    net_io = pp.from_json_string(pp.to_json(net))
    assert_net_equal(net, net_io)

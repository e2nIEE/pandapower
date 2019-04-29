import pytest
from copy import deepcopy
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.plotting import create_generic_coordinates
from pandapower.converter import replace_branch_switches, create_branch_switches

try:
    import pplog as logging
except ImportError:
    import logging

try:
    import igraph
    igraph_installed = True
except ImportError:
    igraph_installed = False

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def _net_to_test():
    # create test grid
    net = pp.create_empty_network()
    # buses
    hv_bus = pp.create_bus(net, 110, index=8, name="HV Bus 0")
    lv_buses = pp.create_buses(net, 6, 20, index=[2, 0, 5, 4, 3, 1],
                               name=["MV Bus %i" % i for i in range(6)])
    # trafo
    t0 = pp.create_transformer(net, hv_bus, lv_buses[1], "40 MVA 110/20 kV", name="T0")
    t1 = pp.create_transformer(net, hv_bus, lv_buses[4], "40 MVA 110/20 kV", name="T1")
    # lines
    l0 = pp.create_line(net, lv_buses[1], lv_buses[2], 1.11, "94-AL1/15-ST1A 20.0", index=3,
                        name="L0")
    l1 = pp.create_line(net, lv_buses[2], lv_buses[3], 1.11, "94-AL1/15-ST1A 20.0", index=7,
                        name="L1")
    l2 = pp.create_line(net, lv_buses[3], lv_buses[0], 2, "94-AL1/15-ST1A 20.0", name="L2")
    l3 = pp.create_line(net, lv_buses[1], lv_buses[3], 2.4, "94-AL1/15-ST1A 20.0", name="L3")
    # bus-bus switch
    pp.create_switch(net, lv_buses[5], lv_buses[2], "b", index=3)
    # trafo switches
    pp.create_switch(net, hv_bus, t0, "t", name="dfdfg", index=4)
    pp.create_switch(net, lv_buses[1], t0, "t", False, name="dfhgjdf")
    pp.create_switch(net, lv_buses[4], t1, "t", False, name="dfhgj", index=2)
    # line switches
    pp.create_switch(net, lv_buses[0], l2, "l", False, name="klar")
    pp.create_switch(net, lv_buses[3], l3, "l", False, name="klar", index=8)
    pp.create_switch(net, lv_buses[1], l0, "l")
    pp.create_switch(net, lv_buses[2], l0, "l")
    pp.create_switch(net, lv_buses[3], l1, "l")

    create_generic_coordinates(net)

    return net


@pytest.mark.skipif(igraph_installed==False, reason="requires igraph")
def test_branch_switch_changes():
    """ Tests replace_branch_switches() and create_branch_switches(). """
    net_orig = _net_to_test()

    net1 = deepcopy(net_orig)
    replace_branch_switches(net1)

    assert net_orig.switch.shape == net1.switch.shape
    assert (net_orig.switch.bus == net1.switch.bus).all()
    assert net1.bus.shape[0] == net_orig.bus.shape[0] + net_orig.switch.shape[0] - sum(
        net_orig.switch.et == "b")
    assert pd.Series(net1.bus.index[net1.bus.type == "auxiliary"]).isin(net1.switch.element).all()
    assert pd.Series(net1.bus.index[net1.bus.type == "auxiliary"]).isin(
        set(net1.line.from_bus) | set(net1.line.to_bus) | set(net1.trafo.hv_bus) |
        set(net1.trafo.lv_bus)).all()

    net2 = deepcopy(net1)
    create_branch_switches(net2)

    assert pp.nets_equal(net_orig, net2)


if __name__ == "__main__":
    if 0:
        pytest.main(["test_pp_net_manipulation.py", "-xs"])
    else:
#        test_branch_switch_changes()
        pass

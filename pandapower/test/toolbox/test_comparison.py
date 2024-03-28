# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import pytest

import pandapower as pp
import pandapower.toolbox
import pandapower.toolbox.comparison as tbc
import pandapower.networks as nw


def test_nets_equal():
    tbc.logger.setLevel(40)
    original = nw.create_cigre_network_lv()
    net = copy.deepcopy(original)

    # should be equal
    assert pandapower.toolbox.nets_equal(original, net)
    assert pandapower.toolbox.nets_equal(net, original)

    # detecting additional element
    pp.create_bus(net, vn_kv=.4)
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting removed element
    net["bus"] = net["bus"].drop(net.bus.index[0])
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting alternated value
    net["load"]["p_mw"][net["load"].index[0]] += 0.1
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting added column
    net["load"]["new_col"] = 0.1
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # not detecting alternated value if difference is beyond tolerance
    net["load"]["p_mw"][net["load"].index[0]] += 0.0001
    assert pandapower.toolbox.nets_equal(original, net, atol=0.1)
    assert pandapower.toolbox.nets_equal(net, original, atol=0.1)

    # check controllers
    original.trafo.tap_side = original.trafo.tap_side.fillna("hv")
    net1 = original.deepcopy()
    net2 = original.deepcopy()
    pp.control.ContinuousTapControl(net1, 0, 1.0)
    pp.control.ContinuousTapControl(net2, 0, 1.0)
    c1 = net1.controller.at[0, "object"]
    c2 = net2.controller.at[0, "object"]
    assert c1 == c2
    assert c1 is not c2
    assert pandapower.toolbox.nets_equal(net1, net2)
    c1.vm_set_pu = 1.01
    assert c1 != c2
    assert pandapower.toolbox.nets_equal(net1, net2, exclude_elms=["controller"])
    assert not pandapower.toolbox.nets_equal(net1, net2)


if __name__ == '__main__':
    pytest.main([__file__, "-x"])

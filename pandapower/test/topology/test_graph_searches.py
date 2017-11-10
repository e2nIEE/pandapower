# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapower as pp
import pytest
import pandapower.topology as top


@pytest.fixture
def feeder_network():
    net = pp.create_empty_network()
    current_bus = pp.create_bus(net, vn_kv=20.)
    pp.create_ext_grid(net, current_bus)
    for length in [12, 6, 8]:
        new_bus = pp.create_bus(net, vn_kv=20.)
        pp.create_line(net, current_bus, new_bus, length_km=length,
                       std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
        current_bus = new_bus
    pp.create_line(net, current_bus, 0, length_km=5, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    return net


def test_determine_stubs(feeder_network):
    net = feeder_network
    sec_bus = pp.create_bus(net, vn_kv=20.)
    sec_line = pp.create_line(net, 3, sec_bus, length_km=3, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    top.determine_stubs(net)
    assert not np.any(net.bus.on_stub.loc[set(net.bus.index) - {sec_bus}].values)
    assert not np.any(net.line.is_stub.loc[set(net.line.index) - {sec_line}].values)
    assert net.bus.on_stub.at[sec_bus] == True
    assert net.line.is_stub.at[sec_line] == True


def test_distance(feeder_network):
    net = feeder_network
    dist = top.calc_distance_to_bus(net, 0)
    assert np.allclose(dist.values, [0, 12, 13, 5])

    dist = top.calc_distance_to_bus(net, 0, notravbuses={3})
    assert np.allclose(dist.values, [0, 12, 18, 5])

    pp.create_switch(net, bus=3, element=2, et="l", closed=False)
    dist = top.calc_distance_to_bus(net, 0)
    assert np.allclose(dist.values, [0, 12, 18, 5])


def test_unsupplied_buses():
    # IS ext_grid --- open switch --- OOS bus --- open switch --- IS bus
    net = pp.create_empty_network()

    bus_sl = pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, bus_sl)

    bus0 = pp.create_bus(net, 0.4, in_service=False)
    pp.create_switch(net, bus_sl, bus0, 'b', False)

    bus1 = pp.create_bus(net, 0.4, in_service=True)
    pp.create_switch(net, bus0, bus1, 'b', False)

    ub = top.unsupplied_buses(net)
    assert ub == {2}

    # OOS ext_grid --- closed switch --- IS bus
    net = pp.create_empty_network()

    bus_sl = pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, bus_sl, in_service=False)

    bus0 = pp.create_bus(net, 0.4, in_service=True)
    pp.create_switch(net, bus_sl, bus0, 'b', True)

    ub = top.unsupplied_buses(net)
    assert ub == {0, 1}


def test_graph_characteristics(feeder_network):
    # adapt network
    net = feeder_network
    bus0 = pp.create_bus(net, vn_kv=20.0)
    bus1 = pp.create_bus(net, vn_kv=20.0)
    bus2 = pp.create_bus(net, vn_kv=20.0)
    bus3 = pp.create_bus(net, vn_kv=20.0)
    bus4 = pp.create_bus(net, vn_kv=20.0)
    bus5 = pp.create_bus(net, vn_kv=20.0)
    bus6 = pp.create_bus(net, vn_kv=20.0)
    new_connections = [(3, bus0), (bus0, bus1), (bus0, bus2), (1, bus3), (2, bus4), (bus3, bus4),
                       (bus4, bus5), (bus4, bus6), (bus5, bus6)]
    for fb, tb in new_connections:
        pp.create_line(net, fb, tb, length_km=1.0, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")

    # get characteristics
    mg = top.create_nxgraph(net, respect_switches=False)
    characteristics = ["bridges", "articulation_points", "connected", "stub_buses",
                       "required_bridges", "notn1_areas"]
    char_dict = top.find_graph_characteristics(mg, net.ext_grid.bus, characteristics)
    bridges = char_dict["bridges"]
    articulation_points = char_dict["articulation_points"]
    connected = char_dict["connected"]
    stub_buses = char_dict["stub_buses"]
    required_bridges = char_dict["required_bridges"]
    notn1_areas = char_dict["notn1_areas"]
    assert bridges == {(3, 4), (4, 5), (4, 6)}
    assert articulation_points == {8, 3, 4}
    assert connected == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    assert stub_buses == {4, 5, 6}
    assert required_bridges == {4: [(3, 4)], 5: [(3, 4), (4, 5)], 6: [(3, 4), (4, 6)]}
    assert notn1_areas == {8: {9, 10}, 3: {4, 5, 6}}


if __name__ == '__main__':
    pytest.main(["test_graph_searches.py"])

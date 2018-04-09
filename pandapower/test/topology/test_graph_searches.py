# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


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
    assert net.bus.on_stub.at[sec_bus]
    assert net.line.is_stub.at[sec_line]


def test_distance(feeder_network):
    net = feeder_network
    dist = top.calc_distance_to_bus(net, 0)
    assert np.allclose(dist.values, [0, 12, 13, 5])

    dist = top.calc_distance_to_bus(net, 0, notravbuses={3})
    assert np.allclose(dist.values, [0, 12, 18, 5])

    pp.create_switch(net, bus=3, element=2, et="l", closed=False)
    dist = top.calc_distance_to_bus(net, 0)
    assert np.allclose(dist.values, [0, 12, 18, 5])


def test_unsupplied_buses_with_in_service():
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


def test_unsupplied_buses_with_switches():
    net = pp.create_empty_network()
    pp.create_buses(net, 8, 20)
    pp.create_buses(net, 5, 0.4)
    pp.create_ext_grid(net, 0)
    pp.create_line(net, 0, 1, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 0, "l", closed=True)
    pp.create_switch(net, 1, 0, "l", closed=False)
    pp.create_line(net, 0, 2, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 1, "l", closed=False)
    pp.create_switch(net, 2, 1, "l", closed=True)
    pp.create_line(net, 0, 3, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 2, "l", closed=False)
    pp.create_switch(net, 3, 2, "l", closed=False)
    pp.create_line(net, 0, 4, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 3, "l", closed=True)
    pp.create_switch(net, 4, 3, "l", closed=True)
    pp.create_line(net, 0, 5, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")

    pp.create_switch(net, 0, 6, "b", closed=True)
    pp.create_switch(net, 0, 7, "b", closed=False)

    pp.create_transformer(net, 0, 8, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 0, "t", closed=True)
    pp.create_switch(net, 8, 0, "t", closed=False)
    pp.create_transformer(net, 0, 9, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 1, "t", closed=False)
    pp.create_switch(net, 9, 1, "t", closed=True)
    pp.create_transformer(net, 0, 10, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 2, "t", closed=False)
    pp.create_switch(net, 10, 2, "t", closed=False)
    pp.create_transformer(net, 0, 11, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 3, "t", closed=True)
    pp.create_switch(net, 11, 3, "t", closed=True)
    pp.create_transformer(net, 0, 12, "0.63 MVA 20/0.4 kV")

    pp.create_buses(net, 2, 20)
    pp.create_impedance(net, 0, 13, 1, 1, 10)
    pp.create_impedance(net, 0, 14, 1, 1, 10, in_service=False)

    ub = top.unsupplied_buses(net)
    assert ub == {1, 2, 3, 7, 8, 9, 10, 14}
    ub = top.unsupplied_buses(net, respect_switches=False)
    assert ub == {14}


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
    pass
#    pytest.main(["test_graph_searches.py"])

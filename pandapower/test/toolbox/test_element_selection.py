# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import pandas as pd
import pytest

import pandapower as pp
import pandapower.toolbox
from pandapower import networks as nw


def test_get_element_indices():
    net = nw.example_multivoltage()
    idx1 = pp.get_element_indices(net, "bus", ["Bus HV%i" % i for i in range(1, 4)])
    idx2 = pp.get_element_indices(net, ["bus", "line"], "HV", exact_match=False)
    idx3 = pp.get_element_indices(net, ["bus", "line"], ["Bus HV3", "MV Line6"])
    assert [32, 33, 34] == idx1
    assert ([32, 33, 34, 35] == idx2[0]).all()
    assert ([0, 1, 2, 3, 4, 5] == idx2[1]).all()
    assert [34, 11] == idx3


def test_next_bus():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=110)
    bus1 = pp.create_bus(net, vn_kv=20)
    bus2 = pp.create_bus(net, vn_kv=10)
    bus3 = pp.create_bus(net, vn_kv=0.4)
    bus4 = pp.create_bus(net, vn_kv=0.4)
    bus5 = pp.create_bus(net, vn_kv=20)

    trafo0 = pp.create_transformer3w(net, hv_bus=bus0, mv_bus=bus1, lv_bus=bus2, name='trafo0',
                                     std_type='63/25/38 MVA 110/20/10 kV')
    trafo1 = pp.create_transformer(net, hv_bus=bus2, lv_bus=bus3, std_type='0.4 MVA 10/0.4 kV')

    line1 = pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=20.1,
                           std_type='24-AL1/4-ST1A 0.4', name='line1')

    # switch0=pp.create_switch(net, bus = bus0, element = trafo0, et = 't3') #~~~~~ not implementable now
    switch1 = pp.create_switch(net, bus=bus1, element=bus5, et='b')
    switch2 = pp.create_switch(net, bus=bus2, element=trafo1, et='t')
    switch3 = pp.create_switch(net, bus=bus3, element=line1, et='l')

    # assert pp.next_bus(net,bus0,trafo0,et='trafo3w')==bus1                         # not implemented in existing toolbox
    # assert pp.next_bus(net,bus0,trafo0,et='trafo3w',choice_for_trafo3w='lv')==bus2 # not implemented in existing toolbox
    assert pp.next_bus(net, bus1, switch1, et='switch') == bus5  # Switch with bus2bus connection
    # assert not pp.next_bus(net,bus2,switch2,et='switch')==bus3  # Switch with bus2trafo connection:- gives trasformer id instead of bus id
    assert pp.next_bus(net, bus2, trafo1, et='trafo') == bus3
    # assert pp.next_bus(net,bus3,switch3,et='switch') ==bus4  # Switch with bus2line connection :- gives line id instead of bus id
    assert pp.next_bus(net, bus3, line1, et='line') == bus4


def test_get_connected_lines_at_bus():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, 0.4)
    bus1 = pp.create_bus(net, 0.4)

    line0 = pp.create_line(net, bus0, bus1, length_km=1., std_type="NAYY 4x50 SE")
    line1 = pp.create_line(net, bus0, bus1, length_km=1., std_type="NAYY 4x50 SE")
    line2 = pp.create_line(net, bus0, bus1, in_service=False, length_km=1., std_type="NAYY 4x50 SE")
    line3 = pp.create_line(net, bus0, bus1, length_km=1., std_type="NAYY 4x50 SE")

    pp.create_switch(net, bus0, line0, "l")
    pp.create_switch(net, bus0, line1, "l", closed=False)
    pp.create_switch(net, bus0, line2, "l")

    lines = pp.get_connected_elements(net, "line", bus0, respect_switches=False,
                                      respect_in_service=False)

    assert set(lines) == {line0, line1, line2, line3}

    lines = pp.get_connected_elements(net, "line", bus0, respect_switches=True,
                                      respect_in_service=False)
    assert set(lines) == {line0, line2, line3}

    lines = pp.get_connected_elements(net, "line", bus0, respect_switches=True,
                                      respect_in_service=True)
    assert set(lines) == {line0, line3}

    lines = pp.get_connected_elements(net, "line", bus0, respect_switches=False,
                                      respect_in_service=True)
    assert set(lines) == {line0, line1, line3}


def test_get_connected_buses():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=110)
    bus1 = pp.create_bus(net, vn_kv=20)
    bus2 = pp.create_bus(net, vn_kv=10)
    bus3 = pp.create_bus(net, vn_kv=0.4)
    bus4 = pp.create_bus(net, vn_kv=0.4)
    bus5 = pp.create_bus(net, vn_kv=20)

    trafo0 = pp.create_transformer3w(net, hv_bus=bus0, mv_bus=bus1, lv_bus=bus2,
                                     std_type='63/25/38 MVA 110/20/10 kV')
    trafo1 = pp.create_transformer(net, hv_bus=bus2, lv_bus=bus3, std_type='0.4 MVA 10/0.4 kV')
    line1 = pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=20.1,
                           std_type='24-AL1/4-ST1A 0.4')

    switch0a = pp.create_switch(net, bus=bus0, element=trafo0, et='t3')
    switch0b = pp.create_switch(net, bus=bus2, element=trafo0, et='t3')
    switch1 = pp.create_switch(net, bus=bus1, element=bus5, et='b')
    switch2 = pp.create_switch(net, bus=bus2, element=trafo1, et='t')
    switch3 = pp.create_switch(net, bus=bus3, element=line1, et='l')

    assert list(pp.get_connected_buses(net, [bus0])) == [bus1, bus2]
    assert list(pp.get_connected_buses(net, [bus1])) == [bus0, bus2, bus5]
    assert list(pp.get_connected_buses(net, [bus2])) == [bus0, bus1, bus3]
    assert list(pp.get_connected_buses(net, [bus3])) == [bus2, bus4]
    assert list(pp.get_connected_buses(net, [bus4])) == [bus3]
    assert list(pp.get_connected_buses(net, [bus5])) == [bus1]
    assert list(pp.get_connected_buses(net, [bus0, bus1])) == [bus2, bus5]
    assert list(pp.get_connected_buses(net, [bus2, bus3])) == [bus0, bus1, bus4]

    net.switch.loc[[switch0b, switch1, switch2, switch3], 'closed'] = False
    assert list(pp.get_connected_buses(net, [bus0])) == [bus1]
    assert list(pp.get_connected_buses(net, [bus1])) == [bus0]
    assert list(pp.get_connected_buses(net, [bus3])) == []
    assert list(pp.get_connected_buses(net, [bus4])) == []


def test_get_connected_buses_at_switches():
    net = pp.networks.example_multivoltage()
    switches = [net.switch.index[net.switch.et == et][0] for et in "blt"]
    expected = set(net.switch.loc[switches[0], ["bus", "element"]])
    expected |= set(net.switch.loc[switches, "bus"])
    expected |= set(net.line.loc[net.switch.at[switches[1], "element"], ["from_bus", "to_bus"]])
    expected |= set(net.trafo.loc[net.switch.at[switches[1], "element"], ["hv_bus", "lv_bus"]])
    assert not bool(len(expected - pp.toolbox.get_connected_buses_at_switches(net, switches)))


def test_get_false_links():
    net = pp.create_empty_network()
    pp.create_buses(net, 6, 10, index=[0, 1, 3, 4, 6, 7])

    # --- gens
    pp.create_gens(net, [0, 1, 3], 5)
    # manipulate to not existing
    net.gen.at[1, "bus"] = 999

    # --- sgens
    pp.create_sgens(net, [0, 1, 3], 5)

    # --- lines
    for fbus, tbus in zip([0, 1, 4, 6, 7], [1, 4, 6, 7, 3]):
        pp.create_line(net, fbus, tbus, 2., "NA2XS2Y 1x185 RM/25 6/10 kV")
    # manipulate to not existing
    net.line.at[1, "from_bus"] = 2
    net.line.at[4, "to_bus"] = 999

    # --- measurements
    pp.create_measurement(net, "v", "bus", 1.01, 5, 1)
    pp.create_measurement(net, "i", "line", 0.41, 1, 0, side="from")
    pp.create_measurement(net, "i", "line", 0.41, 1, 2, side="from")
    pp.create_measurement(net, "v", "bus", 1.01, 5, 6)
    pp.create_measurement(net, "i", "line", 0.41, 1, 1, side="from")
    # manipulate to not existing
    net.measurement.at[1, "element"] = 999
    net.measurement.at[3, "element"] = 999

    # --- poly_cost
    pp.create_poly_cost(net, 0, "gen", 5)
    pp.create_poly_cost(net, 1, "gen", 5)
    pp.create_poly_cost(net, 0, "sgen", 5)
    pp.create_poly_cost(net, 1, "sgen", 5)
    # manipulate to not existing
    net.poly_cost.at[1, "element"] = 999
    net.poly_cost.at[2, "element"] = 999

    expected = {"gen": {1},
                "line": {1, 4},
                "measurement": {1, 3},
                "poly_cost": {1, 2}}
    determined = pp.false_elm_links_loop(net)
    assert {elm: set(idx) for elm, idx in determined.items()} == expected


def test_element_bus_tuples():
    ebts = pandapower.toolbox.element_bus_tuples()
    assert isinstance(ebts, list)
    assert len(ebts) >= 20
    item = next(iter(ebts))
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert len({"line", "gen"} & {elm for (elm, buses) in ebts}) == 2
    assert {buses for (elm, buses) in ebts} == {"bus", "to_bus", "from_bus", 'hv_bus', 'mv_bus',
                                                'lv_bus'}
    assert len(pandapower.toolbox.element_bus_tuples(bus_elements=False, res_elements=True)) > \
           1.5 * len(
        pandapower.toolbox.element_bus_tuples(bus_elements=False, res_elements=False)) > 0


def test_pp_elements():
    elms = pandapower.toolbox.pp_elements()
    assert isinstance(elms, set)
    assert "bus" in elms
    assert "measurement" in elms
    assert "sgen" in elms
    assert len(pandapower.toolbox.pp_elements(bus=False, other_elements=False, bus_elements=True,
                                                                branch_elements=False)) == \
           len(pandapower.toolbox.element_bus_tuples(bus_elements=True, branch_elements=False))


def test_branch_element_bus_dict():
    bebd = pandapower.toolbox.branch_element_bus_dict()
    assert isinstance(bebd, dict)
    assert len(bebd) >= 5
    assert bebd["trafo"] == ["hv_bus", "lv_bus"]
    bebd = pandapower.toolbox.branch_element_bus_dict(include_switch=True)
    assert "bus" in bebd["switch"]


def test_count_elements():
    case9_counts = {"bus": 9, "line": 9, "ext_grid": 1, "gen": 2, "load": 3}
    net = nw.case9()
    received = pandapower.toolbox.count_elements(net)
    assert isinstance(received, pd.Series)
    assert received.to_dict() == case9_counts
    assert pandapower.toolbox.count_elements(net, bus=False).to_dict() == {
        et: num for et, num in case9_counts.items() if et not in ["bus"]}
    assert pandapower.toolbox.count_elements(net, bus=False, branch_elements=False).to_dict() == {
        et: num for et, num in case9_counts.items() if et not in ["bus", "line"]}
    received = pandapower.toolbox.count_elements(net, return_empties=True)
    assert len(received.index) == len(pandapower.toolbox.pp_elements())
    assert set(received.index) == pandapower.toolbox.pp_elements()


if __name__ == '__main__':
    # pytest.main([__file__, "-x"])
    test_get_connected_buses_at_switches()
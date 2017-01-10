# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import copy
import pandapower as pp
import pandapower.toolbox as tb
import pandapower.networks as nw

def test_equal_nets():
    tb.logger.setLevel(40)
    original = nw.create_cigre_network_lv()
    net = copy.deepcopy(original)

    # should be equal
    assert tb.equal_nets(original, net)
    assert tb.equal_nets(net, original)

    # detecting additional element
    pp.create_bus(net, vn_kv=.4)
    assert not tb.equal_nets(original, net)
    assert not tb.equal_nets(net, original)
    net = copy.deepcopy(original)

    # detecting removed element
    net["bus"].drop(net.bus.index[0], inplace=True)
    assert not tb.equal_nets(original, net)
    assert not tb.equal_nets(net, original)
    net = copy.deepcopy(original)

    # detecting alternated value
    net["load"]["p_kw"][net["load"].index[0]] += 0.1
    assert not tb.equal_nets(original, net)
    assert not tb.equal_nets(net, original)
    net = copy.deepcopy(original)

    # detecting added column
    net["load"]["new_col"] = 0.1
    assert not tb.equal_nets(original, net)
    assert not tb.equal_nets(net, original)
    net = copy.deepcopy(original)

    # not detecting alternated value if difference is beyond tolerance
    net["load"]["p_kw"][net["load"].index[0]] += 0.0001
    assert tb.equal_nets(original, net, tol=0.1)
    assert tb.equal_nets(net, original, tol=0.1)



def test_continuos_bus_numbering():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, 0.4,  index=12)
    pp.create_load(net, bus0, p_kw=0.)
    pp.create_load(net, bus0, p_kw=0.)
    pp.create_load(net, bus0, p_kw=0.)
    pp.create_load(net, bus0, p_kw=0.)

    bus0 = pp.create_bus(net, 0.4, index=42)
    pp.create_sgen(net, bus0, p_kw=0.)
    pp.create_sgen(net, bus0, p_kw=0.)
    pp.create_sgen(net, bus0, p_kw=0.)

    bus0 = pp.create_bus(net, 0.4, index=543)
    pp.create_shunt(net, bus0, 2, 1)
    pp.create_shunt(net, bus0, 2, 1)
    pp.create_shunt(net, bus0, 2, 1)

    bus0 = pp.create_bus(net, 0.4,  index=5675)
    pp.create_ward(net, bus0, 2, 1, 1, 2,)
    pp.create_ward(net, bus0, 2, 1, 1, 2,)
    pp.create_ward(net, bus0, 2, 1, 1, 2,)

    tb.create_continuous_bus_index(net)

    l = net.bus.index
    assert all(l[i] <= l[i+1] for i in range(len(l)-1))  # is ordered
    assert all(l[i]+1 == l[i+1] for i in range(len(l)-1))  # is consecutive
    assert l[0] == 0  # starts at zero

    used_buses = []
    for element in net.keys():
        try:
            used_buses + net[element].bus.values
        except:
            try:
                used_buses + net[element].from_bus.values
                used_buses + net[element].to_bus.values
            except:
                try:
                    used_buses + net[element].hv_bus.values
                    used_buses + net[element].lv_bus.values
                except:
                    continue

    # assert that no buses were used except the ones in net.bus
    assert set(list(used_buses)) - set(list(net.bus.index.values)) == set()

def test_scaling_by_type():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, 0.4)
    pp.create_load(net, bus0, p_kw=0., type="Household")
    pp.create_sgen(net, bus0, p_kw=0., type="PV")

    tb.set_scaling_by_type(net, {"Household": 42., "PV": 12})

    assert net.load.at[0, "scaling"] == 42
    assert net.sgen.at[0, "scaling"] == 12

def test_drop_inactive_elements():
    net = pp.create_empty_network()

    service = 0

    bus0 = pp.create_bus(net, vn_kv=.4, in_service=service)
    pp.create_ext_grid(net, bus0, in_service=service)

    bus1 = pp.create_bus(net, vn_kv=.4, in_service=service)
    pp.create_transformer(net, bus0, bus1, in_service=service,
                          std_type= '63 MVA 110/20 kV')

    bus2 = pp.create_bus(net, vn_kv=.4, in_service=service)
    pp.create_line(net, bus1, bus2, length_km=1, in_service=service,
                   std_type='149-AL1/24-ST1A 10.0')

    pp.create_load(net, bus2, p_kw=0., in_service=service)
    pp.create_sgen(net, bus2, p_kw=0., in_service=service)

    # drop them
    tb.drop_inactive_elements(net)

    sum_of_elements = 0
    for element in net.keys():
        # skip this one since we expect items here
        if element == "std_types":
            continue

        try:
            sum_of_elements += len(net[element])
        except TypeError:
            # _ppc is initialized with None and clashes when checking
            continue

    assert sum_of_elements == 0


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
    
    lines = tb.get_connected_elements(net, "line", bus0, respect_switches=False, respect_in_service=False)

    assert set(lines) == set([line0, line1, line2, line3])

    lines = tb.get_connected_elements(net, "line", bus0, respect_switches=True,
                                          respect_in_service=False)
    assert set(lines) == set([line0, line2, line3])

    lines = tb.get_connected_elements(net, "line", bus0, respect_switches=True,
                                          respect_in_service=True)
    assert set(lines) == set([line0, line3])

    lines = tb.get_connected_elements(net, "line", bus0, respect_switches=False,
                                          respect_in_service=True)
    assert set(lines) == set([line0, line1, line3])


if __name__ == "__main__":
    pytest.main(["test_toolbox.py", "-xs"])

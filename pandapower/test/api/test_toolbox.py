# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import copy
import pandapower as pp
import pandapower.toolbox as tb
import pandapower.networks as nw

def test_nets_equal():
    tb.logger.setLevel(40)
    original = nw.create_cigre_network_lv()
    net = copy.deepcopy(original)

    # should be equal
    assert tb.nets_equal(original, net)
    assert tb.nets_equal(net, original)

    # detecting additional element
    pp.create_bus(net, vn_kv=.4)
    assert not tb.nets_equal(original, net)
    assert not tb.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting removed element
    net["bus"].drop(net.bus.index[0], inplace=True)
    assert not tb.nets_equal(original, net)
    assert not tb.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting alternated value
    net["load"]["p_kw"][net["load"].index[0]] += 0.1
    assert not tb.nets_equal(original, net)
    assert not tb.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting added column
    net["load"]["new_col"] = 0.1
    assert not tb.nets_equal(original, net)
    assert not tb.nets_equal(net, original)
    net = copy.deepcopy(original)

    # not detecting alternated value if difference is beyond tolerance
    net["load"]["p_kw"][net["load"].index[0]] += 0.0001
    assert tb.nets_equal(original, net, tol=0.1)
    assert tb.nets_equal(net, original, tol=0.1)



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

    tb.set_scaling_by_type(net, {"Household": 0, "PV": 0})

    assert net.load.at[0, "scaling"] == 0
    assert net.sgen.at[0, "scaling"] == 0


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
        if element == "std_types" or element == "_pd2ppc_lookups" or element == "_ppc2pd_lookups":
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

    


######################################################


def test_overloaded_lines():
        
    net = pp.create_empty_network()
    
    bus0 = pp.create_bus(net, vn_kv=.4)
    bus1 = pp.create_bus(net, vn_kv=.4)
    
    ext_grid0 = pp.create_ext_grid(net, bus0, vm_pu=4)
    
    line0 = pp.create_line(net, bus0, bus1, length_km=1, std_type="NAYY 4x50 SE")
    line1 = pp.create_line(net, bus0, bus1, length_km=1, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    line2 = pp.create_line(net, bus0, bus1, length_km=1, std_type="15-AL1/3-ST1A 0.4")
    line3 = pp.create_line(net, bus0, bus1, length_km=10, std_type="149-AL1/24-ST1A 10.0")
    
    pp.runpp(net)
    
    # test the overloaded lines by default value of max_load=100
    overloaded_lines = tb.overloaded_lines(net,max_load=100)
      
    assert set(overloaded_lines) == set([line0, line1, line2])
    
    # test the overloaded lines by a self defined value of max_load=50
    overloaded_lines = tb.overloaded_lines(net,max_load=50)
    
    assert set(overloaded_lines) == set([line0, line1, line2, line3])  



def test_violated_buses():
    
    net = nw.create_cigre_network_lv()
    
    pp.runpp(net)
    
    # set the range of vm.pu
    min_vm_pu = 0.92
    max_vm_pu = 1.1
    
    # print out the list of violated_bus's index
    violated_bus = tb.violated_buses(net, min_vm_pu, max_vm_pu)
    
    # check if the elements in the violated_bus list is beyond range
    for i in violated_bus:   
        
        if net["res_bus"]["vm_pu"].loc[i] > max_vm_pu or net["res_bus"]["vm_pu"].loc[i] < min_vm_pu:
            continue
        else:
            return ("bus index %d is not violated" % i)
        
    assert set(violated_bus) == set(net["bus"].index[[16,35,36,40]])



def test_add_zones_to_elements():
    net = nw.create_cigre_network_mv()

    # add zones to lines and switchs
    tb.add_zones_to_elements(net, elements=["line","switch"])

    #create 2 arrays which include "zone" in lines and switchs
    zone_line = net["line"]["zone"].values
    zone_switch = net["switch"]["zone"].values

    assert any([x=="CIGRE_MV" for x in zone_line])
    assert any([x=="CIGRE_MV" for x in zone_switch])




def test_fuse_buses():
    
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=1, name="b1")
    b2 = pp.create_bus(net, vn_kv=1.5, name="b2")
    
    line1 = pp.create_line(net, b2, b1, length_km=1, std_type="NAYY 4x50 SE")
    
    sw1 = pp.create_switch(net, b2, line1, et="l")
    sw2 = pp.create_switch(net, b1, b2, et="b")
    
    load1 = pp.create_load(net,b1,p_kw=6)
    load2 = pp.create_load(net,b2,p_kw=5)
    
        
    tb.fuse_buses(net, b1, b2, drop=True)
    
    # assertion: elements connected to b2 are given to b1 instead
    assert net["line"]["from_bus"].loc[0] == b1
    assert net["switch"]["bus"].loc[0] == b1
    assert net["load"]["bus"].loc[1] == b1

    

  




  
if __name__ == "__main__":
    pytest.main(["test_toolbox.py", "-xs"])

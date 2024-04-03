# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import pandas as pd
from pandas._testing import assert_series_equal
import pytest

import pandapower as pp
import pandapower.networks as nw
import pandapower.toolbox


def test_drop_inactive_elements():
    for service in (False, True):
        net = pp.create_empty_network()
        bus_sl = pp.create_bus(net, vn_kv=.4, in_service=service)
        pp.create_ext_grid(net, bus_sl, in_service=service)
        bus0 = pp.create_bus(net, vn_kv=.4, in_service=service)
        pp.create_switch(net, bus_sl, bus0, 'b', not service)
        bus1 = pp.create_bus(net, vn_kv=.4, in_service=service)
        pp.create_transformer(net, bus0, bus1, in_service=service,
                              std_type='63 MVA 110/20 kV')
        bus2 = pp.create_bus(net, vn_kv=.4, in_service=service)
        pp.create_line(net, bus1, bus2, length_km=1, in_service=service,
                       std_type='149-AL1/24-ST1A 10.0')
        pp.create_load(net, bus2, p_mw=0., in_service=service)
        pp.create_sgen(net, bus2, p_mw=0., in_service=service)
        bus3 = pp.create_bus(net, vn_kv=.4, in_service=service)
        bus4 = pp.create_bus(net, vn_kv=.4, in_service=service)
        pp.create_transformer3w_from_parameters(net, bus2, bus3, bus4, 0.4, 0.4, 0.4, 100, 50, 50,
                                                3, 3, 3, 1, 1, 1, 5, 1)
        # drop them
        pp.drop_inactive_elements(net)

        sum_of_elements = 0
        for element, table in net.items():
            # skip this one since we expect items here
            if element.startswith("_") or not isinstance(table, pd.DataFrame):
                continue
            try:
                if service and (element == 'ext_grid' or (element == 'bus' and len(net.bus) == 1)):
                    # if service==True, the 1 ext_grid and its bus are not dropped
                    continue
                if len(table) > 0:
                    sum_of_elements += len(table)
                    # print(element)
            except TypeError:
                # _ppc is initialized with None and clashes when checking
                continue

        assert sum_of_elements == 0
        if service:
            assert len(net.ext_grid) == 1
            assert len(net.bus) == 1
            assert bus_sl in net.bus.index.values

    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=.4, in_service=True)
    pp.create_ext_grid(net, bus0, in_service=True)
    bus1 = pp.create_bus(net, vn_kv=.4, in_service=False)
    pp.create_line(net, bus0, bus1, length_km=1, in_service=False,
                   std_type='149-AL1/24-ST1A 10.0')
    gen0 = pp.create_gen(net, bus=bus1, p_mw=0.001)

    pp.drop_inactive_elements(net)

    assert gen0 not in net.gen.index


def test_merge_indices():
    net1 = nw.create_cigre_network_mv()
    pp.create_pwl_cost(net1, 0, "load", [[0, 20, 1], [20, 30, 2]])
    pp.create_pwl_cost(net1, 2, "load", [[0, 20, 0.5], [20, 30, 2]])
    pp.reindex_buses(net1, {3: 29})
    assert 29 in net1.bus.index.values
    assert 29 in net1.load.bus.values

    net2 = nw.create_cigre_network_mv(with_der="pv_wind")
    pp.create_pwl_cost(net2, 1, "load", [[0, 20, 1], [20, 30, 2]], index=5)
    pp.create_pwl_cost(net2, 2, "sgen", [[0, 20, 0.5], [20, 30, 2]])

    net = pp.merge_nets(net1, net2, net2_reindex_log_level="debug")

    # check
    for et in pandapower.toolbox.pp_elements(cost_tables=True):
        assert net[et].shape[0] == net1[et].shape[0] + net2[et].shape[0]
    assert net.bus.index.tolist() == net1.bus.index.tolist() + [
        i+29+1 if i < 3 else i+29 if i > 3 else 3 for i in net2.bus.index]
    assert net.load.index.tolist() == list(range(net.load.shape[0]))
    assert net.load.bus.tolist() == net1.load.bus.tolist() + [
        i+29+1 if i < 3 else i+29 if i > 3 else 3 for i in net2.load.bus]
    assert net.pwl_cost.index.tolist() == [0, 1, 5, 6]
    assert net.pwl_cost.element.tolist() == [0, 2, 19, 2]
    assert net.pwl_cost.et.tolist() == ["load"]*3 + ["sgen"]


def test_merge_and_split_nets():
    net1 = nw.mv_oberrhein()
    pp.create_poly_cost(net1, 2, "sgen", 8)
    pp.create_poly_cost(net1, 0, "sgen", 9)
    # TODO there are some geodata values in oberrhein without corresponding lines
    net1.line_geodata.drop(set(net1.line_geodata.index) - set(net1.line.index), inplace=True)
    n1 = len(net1.bus)
    pp.runpp(net1)
    net2 = nw.create_cigre_network_mv(with_der="pv_wind")
    pp.create_poly_cost(net2, 3, "sgen", 10)
    pp.create_poly_cost(net2, 0, "sgen", 11)
    pp.runpp(net2)

    net1_before = copy.deepcopy(net1)
    net2_before = copy.deepcopy(net2)
    net = pp.merge_nets(net1, net2, net2_reindex_log_level="debug")
    pp.runpp(net)

    # check that merge_nets() doesn't change inputs (but result tables)
    pp.test.assert_net_equal(net1, net1_before, check_without_results=True)
    pp.test.assert_net_equal(net2, net2_before, check_without_results=True)

    # check that results of merge_nets() fit
    assert np.allclose(net.res_bus.vm_pu.iloc[:n1].values, net1.res_bus.vm_pu.values)
    assert np.allclose(net.res_bus.vm_pu.iloc[n1:].values, net2.res_bus.vm_pu.values)

    # check content of merge_nets() output
    assert np.array_equal(
        pd.concat([net1.sgen.name.loc[net1.poly_cost.element],
                   net2.sgen.name.loc[net2.poly_cost.element]]).values,
        net.sgen.name.loc[net.poly_cost.element].values)

    # check that results stay the same after net split
    net3 = pp.select_subnet(net, net.bus.index[:n1], include_results=True)
    assert pandapower.toolbox.dataframes_equal(net3.res_bus[["vm_pu"]], net1.res_bus[["vm_pu"]])

    net4 = pp.select_subnet(net, net.bus.index[n1:], include_results=True)
    assert np.allclose(net4.res_bus.vm_pu.values, net2.res_bus.vm_pu.values)


def test_merge_asymmetric():
    """Test that merging nets properly handles bus IDs for asymmetric elements
    """
    net1 = nw.ieee_european_lv_asymmetric()
    net2 = nw.ieee_european_lv_asymmetric()
    n_load_busses = len(net1.asymmetric_load.bus.unique())
    n_sgen_busses = len(net1.asymmetric_sgen.bus.unique())

    net1_before = copy.deepcopy(net1)
    net2_before = copy.deepcopy(net2)
    net = pp.merge_nets(net1, net2, net2_reindex_log_level="debug")

    pp.test.assert_net_equal(net1, net1_before, check_without_results=True)
    pp.test.assert_net_equal(net2, net2_before, check_without_results=True)
    assert len(net.asymmetric_load.bus.unique()) == 2 * n_load_busses
    assert len(net.asymmetric_sgen.bus.unique()) == 2 * n_sgen_busses


def test_merge_with_groups():
    """Test that group data are correctly considered by merge_nets()
    """
    net1 = nw.create_cigre_network_mv()
    net2 = nw.create_cigre_network_mv()
    for elm in ["bus", "load", "line"]:
        net2[elm].name = "new " + net2[elm].name
    pp.create_group(net1, "bus", [[0, 2]], name="group of net1")
    pp.create_group(net2, ["bus", "load"], [[1], [0, 3]], name="group1 of net2")
    pp.create_group(net2, ["line"], [[1, 3]], name="group2 of net2", index=4)

    net = pp.merge_nets(net1, net2, net2_reindex_log_level="debug")

    # check that all group lines are available
    assert net.group.shape[0] == net1.group.shape[0] + net2.group.shape[0]

    # check (adapted) index
    assert set(net.group.index) == {0, 5, 4}

    # check that net2 groups link to the same elements as later in net.group (checking by element names)
    assert net2.bus.name.loc[pp.group_element_index(net2, 0, "bus")].tolist() == \
        net.bus.name.loc[pp.group_element_index(net, 5, "bus")].tolist()
    assert net2.load.name.loc[pp.group_element_index(net2, 0, "load")].tolist() == \
        net.load.name.loc[pp.group_element_index(net, 5, "load")].tolist()
    assert net2.trafo.name.loc[pp.group_element_index(net2, 4, "trafo")].tolist() == \
        net.trafo.name.loc[pp.group_element_index(net, 4, "trafo")].tolist()

    # check that net2 groups link to the same elements as later in net.group (checking by element index)
    assert list(pp.group_element_index(net, 0, "bus")) == [0, 2]
    assert list(pp.group_element_index(net, 5, "bus")) == [net1.bus.shape[0]+1]
    assert list(pp.group_element_index(net, 5, "load")) == list(np.array([0, 3], dtype=np.int64) + \
        net1.load.shape[0])
    assert list(pp.group_element_index(net, 4, "line")) == list(np.array([1, 3], dtype=np.int64) + \
        net1.line.shape[0])


def test_select_subnet():
    # This network has switches of type 'l' and 't'
    net = nw.create_cigre_network_mv()

    # Do nothing
    same_net = pp.select_subnet(net, net.bus.index)
    assert pandapower.toolbox.dataframes_equal(net.bus, same_net.bus)
    assert pandapower.toolbox.dataframes_equal(net.switch, same_net.switch)
    assert pandapower.toolbox.dataframes_equal(net.trafo, same_net.trafo)
    assert pandapower.toolbox.dataframes_equal(net.line, same_net.line)
    assert pandapower.toolbox.dataframes_equal(net.load, same_net.load)
    assert pandapower.toolbox.dataframes_equal(net.ext_grid, same_net.ext_grid)
    same_net2 = pp.select_subnet(net, net.bus.index, include_results=True,
                                 keep_everything_else=True)
    assert pandapower.toolbox.nets_equal(net, same_net2)

    # Remove everything
    empty = pp.select_subnet(net, set())
    assert len(empty.bus) == 0
    assert len(empty.line) == 0
    assert len(empty.load) == 0
    assert len(empty.trafo) == 0
    assert len(empty.switch) == 0
    assert len(empty.ext_grid) == 0

    # Should keep all trafo ('t') switches when buses are included
    hv_buses = set(net.trafo.hv_bus)
    lv_buses = set(net.trafo.lv_bus)
    trafo_switch_buses = set(net.switch[net.switch.et == 't'].bus)
    subnet = pp.select_subnet(net, hv_buses | lv_buses | trafo_switch_buses)
    assert net.switch[net.switch.et == 't'].index.isin(subnet.switch.index).all()

    # Should keep all line ('l') switches when buses are included
    from_bus = set(net.line.from_bus)
    to_bus = set(net.line.to_bus)
    line_switch_buses = set(net.switch[net.switch.et == 'l'].bus)
    subnet = pp.select_subnet(net, from_bus | to_bus | line_switch_buses)
    assert net.switch[net.switch.et == 'l'].index.isin(subnet.switch.index).all()
    ls = net.switch.loc[net.switch.et == "l"]
    subnet = pp.select_subnet(net, list(ls.bus.values)[::2], include_switch_buses=True)
    assert net.switch[net.switch.et == 'l'].index.isin(subnet.switch.index).all()
    assert net.switch[net.switch.et == 'l'].bus.isin(subnet.bus.index).all()

    # This network has switches of type 'b'
    net2 = nw.create_cigre_network_lv()

    # Should keep all bus-to-bus ('b') switches when buses are included
    buses = set(net2.switch[net2.switch.et == 'b'].bus)
    elements = set(net2.switch[net2.switch.et == 'b'].element)
    subnet = pp.select_subnet(net2, buses | elements)
    assert net2.switch[net2.switch.et == 'b'].index.isin(subnet.switch.index).all()


def test_add_zones_to_elements():
    net = nw.create_cigre_network_mv()

    # add zones to lines and switchs
    pp.add_zones_to_elements(net, elements=["line", "switch"])

    # create 2 arrays which include "zone" in lines and switches
    zone_line = net["line"]["zone"].values
    zone_switch = net["switch"]["zone"].values

    assert "CIGRE_MV" in zone_line
    assert "CIGRE_MV" in zone_switch


def test_drop_inner_branches():
    def check_elm_number(net1, net2, excerpt_elms=None):
        excerpt_elms = set() if excerpt_elms is None else set(excerpt_elms)
        for elm in set(pandapower.toolbox.pp_elements()) - excerpt_elms:
            assert net1[elm].shape[0] == net2[elm].shape[0]

    net = nw.example_simple()
    new_bus = pp.create_bus(net, 10)
    pp.create_transformer3w(net, 2, 3, new_bus, "63/25/38 MVA 110/20/10 kV")

    net1 = copy.deepcopy(net)
    pp.drop_inner_branches(net1, [2, 3], branch_elements=["line"])
    check_elm_number(net1, net)
    pp.drop_inner_branches(net1, [0, 1], branch_elements=["line"])
    check_elm_number(net1, net, ["line"])
    assert all(net.line.index.difference({0}) == net1.line.index)

    net2 = copy.deepcopy(net)
    pp.drop_inner_branches(net2, [2, 3, 4, 5])
    assert all(net.line.index.difference({1}) == net2.line.index)
    assert all(net.trafo.index.difference({0}) == net2.trafo.index)
    assert all(net.switch.index.difference({1, 2, 3}) == net2.switch.index)
    check_elm_number(net2, net, ["line", "switch", "trafo"])


def test_fuse_buses():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=1, name="b1")
    b2 = pp.create_bus(net, vn_kv=1.5, name="b2")
    b3 = pp.create_bus(net, vn_kv=2, name="b2")

    line1 = pp.create_line(net, b2, b1, length_km=1, std_type="NAYY 4x50 SE")
    line2 = pp.create_line(net, b2, b3, length_km=1, std_type="NAYY 4x50 SE")

    sw1 = pp.create_switch(net, b2, line1, et="l")
    sw2 = pp.create_switch(net, b1, b2, et="b")

    pp.create_load(net, b1, p_mw=0.006)
    pp.create_load(net, b2, p_mw=0.005)
    pp.create_load(net, b3, p_mw=0.005)

    pp.create_measurement(net, "v", "bus", 1.2, 0.03, b2)

    # --- drop = True
    net1 = copy.deepcopy(net)
    pp.fuse_buses(net1, b1, b2, drop=True)

    # assertion: elements connected to b2 are given to b1 instead
    assert line1 not in net1.line.index
    assert line2 in net1.line.index
    assert sw1 not in net1.switch.index
    assert sw2 not in net1.switch.index
    assert list(net1["load"]["bus"].values) == [b1, b1, b3]
    assert net1["measurement"]["element"].at[0] == b1
    # assertion: b2 not in net.bus table if drop=True
    assert b2 not in net1.bus.index
    assert b3 in net1.bus.index

    # --- drop = False
    net2 = copy.deepcopy(net)
    pp.fuse_buses(net2, b1, b2, drop=False)

    # assertion: elements connected to b2 are given to b1 instead
    assert net2["line"]["from_bus"].at[0] == b1
    assert line2 in net2.line.index
    assert net2["switch"]["bus"].at[0] == b1
    assert net2["load"]["bus"].tolist() == [b1, b1, b3]
    assert net2["measurement"]["element"].at[0] == b1
    # assertion: b2 remains in net.bus table
    assert b2 in net2.bus.index
    assert b3 in net2.bus.index


def test_close_switch_at_line_with_two_open_switches():
    net = pp.create_empty_network()

    bus1 = pp.create_bus(net, vn_kv=.4)
    bus2 = pp.create_bus(net, vn_kv=.4)
    bus3 = pp.create_bus(net, vn_kv=.4)

    line1 = pp.create_line(net, bus2, bus3, length_km=1., std_type="NAYY 4x50 SE")
    line2 = pp.create_line(net, bus2, bus3, length_km=1., std_type="NAYY 4x50 SE")
    pp.create_line(net, bus2, bus3, length_km=1., std_type="NAYY 4x50 SE")  # line3

    pp.create_switch(net, bus1, bus2, et="b", closed=True)  # sw0

    pp.create_switch(net, bus2, line1, et="l", closed=False)  # sw1
    pp.create_switch(net, bus3, line1, et="l", closed=False)  # sw2

    pp.create_switch(net, bus2, line2, et="l", closed=True)  # sw3
    pp.create_switch(net, bus3, line2, et="l", closed=False)  # sw4

    pp.create_switch(net, bus3, line2, et="l", closed=True)  # sw5
    pp.create_switch(net, bus3, line2, et="l", closed=True)  # sw6

    pp.close_switch_at_line_with_two_open_switches(net)

    # assertion: sw2 closed
    assert net.switch.closed.loc[1]


def test_create_replacement_switch_for_branch():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=0.4)
    bus1 = pp.create_bus(net, vn_kv=0.4)
    bus2 = pp.create_bus(net, vn_kv=0.4)
    bus3 = pp.create_bus(net, vn_kv=0.4)

    pp.create_ext_grid(net, bus0, vm_pu=0.4)

    line0 = pp.create_line(net, bus0, bus1, length_km=1, std_type="NAYY 4x50 SE")
    line1 = pp.create_line(net, bus2, bus3, length_km=1, std_type="NAYY 4x50 SE")
    impedance0 = pp.create_impedance(net, bus1, bus2, 0.01, 0.01, sn_mva=100)
    impedance1 = pp.create_impedance(net, bus1, bus2, 0.01, 0.01, sn_mva=100)

    pp.create_load(net, bus2, 0.001)

    pp.runpp(net)

    # look that the switch is created properly
    pp.create_replacement_switch_for_branch(net, 'line', line0)
    pp.create_replacement_switch_for_branch(net, 'impedance', impedance0)
    net.line.at[line0, "in_service"] = False
    net.impedance.at[impedance0, "in_service"] = False

    assert 'REPLACEMENT_line_0' in net.switch.name.values
    assert 'REPLACEMENT_impedance_0' in net.switch.name.values
    assert net.switch.closed.at[0]
    assert net.switch.closed.at[1]
    pp.runpp(net)

    # look that the switch is created with the correct closed status
    net.line.at[line1, "in_service"] = False
    net.impedance.at[impedance1, "in_service"] = False
    pp.create_replacement_switch_for_branch(net, 'line', line1)
    pp.create_replacement_switch_for_branch(net, 'impedance', impedance1)

    assert 'REPLACEMENT_line_1' in net.switch.name.values
    assert 'REPLACEMENT_impedance_1' in net.switch.name.values
    assert ~net.switch.closed.at[2]
    assert ~net.switch.closed.at[3]


@pytest.fixture
def net():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=0.4)
    bus1 = pp.create_bus(net, vn_kv=0.4)
    bus2 = pp.create_bus(net, vn_kv=0.4)
    bus3 = pp.create_bus(net, vn_kv=0.4)
    bus4 = pp.create_bus(net, vn_kv=0.4)
    bus5 = pp.create_bus(net, vn_kv=0.4)
    bus6 = pp.create_bus(net, vn_kv=0.4)

    pp.create_ext_grid(net, bus0, vm_pu=0.4)

    pp.create_line(net, bus0, bus1, length_km=0, std_type="NAYY 4x50 SE")  # line0
    pp.create_line_from_parameters(net, bus2, bus3, length_km=1, r_ohm_per_km=0, x_ohm_per_km=0.1,
                                   c_nf_per_km=0, max_i_ka=1)  # line1
    pp.create_line_from_parameters(net, bus3, bus4, length_km=1, r_ohm_per_km=0, x_ohm_per_km=0,
                                   c_nf_per_km=0, max_i_ka=1)  # line2

    pp.create_impedance(net, bus1, bus2, 0.01, 0.01, sn_mva=100)  # impedance0
    pp.create_impedance(net, bus4, bus5, 0, 0, sn_mva=100)  # impedance1
    pp.create_impedance(net, bus5, bus6, 0, 0, rtf_pu=0.1, sn_mva=100)  # impedance2
    return net


def test_for_line_with_zero_length(net):
    pp.replace_zero_branches_with_switches(net, elements=('line',), zero_impedance=False)
    assert 'REPLACEMENT_line_0' in net.switch.name.values
    assert ~net.line.in_service.at[0]
    assert 'REPLACEMENT_line_2' not in net.switch.name.values


def test_drop(net):
    pp.replace_zero_branches_with_switches(net, elements=('line', 'impedance'), drop_affected=True)
    assert len(net.line) == 1
    assert len(net.impedance) == 2


def test_in_service_only(net):
    pp.replace_zero_branches_with_switches(net, elements=('line',))
    assert len(net.switch.loc[net.switch.name == 'REPLACEMENT_line_0']) == 1
    pp.replace_zero_branches_with_switches(net, elements=('line',), in_service_only=False)
    assert len(net.switch.loc[net.switch.name == 'REPLACEMENT_line_0']) == 2
    assert ~net.switch.closed.at[2]


def test_line_with_zero_impediance(net):
    # test for line with zero impedance
    pp.replace_zero_branches_with_switches(net, elements=('line',), zero_length=False)
    assert 'REPLACEMENT_line_1' not in net.switch.name.values
    assert 'REPLACEMENT_line_2' in net.switch.name.values


def test_impedance(net):
    pp.replace_zero_branches_with_switches(net, elements=('impedance',), zero_length=False,
                                           zero_impedance=True, in_service_only=True)
    assert 'REPLACEMENT_impedance_0' not in net.switch.name.values
    assert 'REPLACEMENT_impedance_1' in net.switch.name.values
    assert 'REPLACEMENT_impedance_2' not in net.switch.name.values


def test_all(net):
    pp.replace_zero_branches_with_switches(net, elements=('impedance', 'line'), zero_length=True,
                                           zero_impedance=True, in_service_only=True)
    assert 'REPLACEMENT_impedance_1' in net.switch.name.values
    assert 'REPLACEMENT_line_0' in net.switch.name.values
    assert 'REPLACEMENT_line_2' in net.switch.name.values
    assert ~net.line.in_service.at[0]
    assert net.line.in_service.at[1]
    assert ~net.line.in_service.at[2]
    assert 'REPLACEMENT_impedance_0' not in net.switch.name.values
    assert 'REPLACEMENT_impedance_2' not in net.switch.name.values
    assert 'REPLACEMENT_line_1' not in net.switch.name.values
    assert net.impedance.in_service.at[0]
    assert ~net.impedance.in_service.at[1]
    assert net.impedance.in_service.at[2]


def test_drop_elements_at_buses():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=110)
    bus1 = pp.create_bus(net, vn_kv=20)
    bus2 = pp.create_bus(net, vn_kv=10)
    bus3 = pp.create_bus(net, vn_kv=0.4)
    bus4 = pp.create_bus(net, vn_kv=0.4)
    bus5 = pp.create_bus(net, vn_kv=20)

    pp.create_ext_grid(net, 0)

    trafo0 = pp.create_transformer3w(net, hv_bus=bus0, mv_bus=bus1, lv_bus=bus2, name='trafo0',
                                     std_type='63/25/38 MVA 110/20/10 kV')
    trafo1 = pp.create_transformer(net, hv_bus=bus2, lv_bus=bus3, std_type='0.4 MVA 10/0.4 kV')

    line1 = pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=20.1,
                           std_type='24-AL1/4-ST1A 0.4', name='line1')
    pp.create_sgen(net, 1, 0)

    switch0a = pp.create_switch(net, bus=bus0, element=trafo0, et='t3')
    switch0b = pp.create_switch(net, bus=bus1, element=trafo0, et='t3')
    switch0c = pp.create_switch(net, bus=bus2, element=trafo0, et='t3')
    switch1 = pp.create_switch(net, bus=bus1, element=bus5, et='b')
    switch2a = pp.create_switch(net, bus=bus2, element=trafo1, et='t')
    switch2b = pp.create_switch(net, bus=bus3, element=trafo1, et='t')
    switch3a = pp.create_switch(net, bus=bus3, element=line1, et='l')
    switch3b = pp.create_switch(net, bus=bus4, element=line1, et='l')
    # bus id needs to be entered as iterable, not done in the function

    for b in net.bus.index.values:
        net1 = net.deepcopy()
        cd = pp.get_connected_elements_dict(net1, b, connected_buses=False)
        swt3w = set(net1.switch.loc[net1.switch.element.isin(cd.get('trafo3w', [1000])) &
                                    (net1.switch.et == 't3')].index)
        swt = set(net1.switch.loc[net1.switch.element.isin(cd.get('trafo', [1000])) &
                                  (net1.switch.et == 't')].index)
        swl = set(net1.switch.loc[net1.switch.element.isin(cd.get('line', [1000])) &
                                  (net1.switch.et == 'l')].index)
        sw = swt3w | swt | swl
        pp.drop_elements_at_buses(net1, [b])
        assert b not in net1.switch.bus.values
        assert b not in net1.switch.query("et=='b'").element.values
        assert sw.isdisjoint(set(net1.switch.index))
        for elm, id in cd.items():
            assert len(net1[elm].loc[net1[elm].index.isin(id)]) == 0


def test_impedance_line_replacement():
    # create test net
    net1 = pp.create_empty_network(sn_mva=1.1)
    pp.create_buses(net1, 2, 10)
    pp.create_ext_grid(net1, 0)
    pp.create_impedance(net1, 0, 1, 0.1, 0.1, 8.7e-3)
    pp.create_load(net1, 1, 7e-3, 2e-3)

    # validate loadflow results
    pp.runpp(net1)

    net2 = copy.deepcopy(net1)
    pp.replace_impedance_by_line(net2)

    pp.runpp(net2)

    assert pandapower.toolbox.nets_equal(net1, net2, exclude_elms={"line", "impedance"})
    cols = ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "pl_mw", "ql_mvar", "i_from_ka",
            "i_to_ka"]
    assert np.allclose(net1.res_impedance[cols].values, net2.res_line[cols].values)

    net3 = copy.deepcopy(net2)
    pp.replace_line_by_impedance(net3)

    pp.runpp(net3)

    assert pandapower.toolbox.nets_equal(net2, net3, exclude_elms={"line", "impedance"})
    assert np.allclose(net3.res_impedance[cols].values, net2.res_line[cols].values)


def test_replace_ext_grid_gen():
    for i in range(2):
        net = nw.example_simple()
        net.ext_grid["uuid"] = "test"
        pp.runpp(net, calculate_voltage_angles="auto")
        assert list(net.res_ext_grid.index.values) == [0]
        pp.create_group(net, ["line", "ext_grid"], [[0], [0]])

        # replace_ext_grid_by_gen
        if i == 0:
            pp.replace_ext_grid_by_gen(net, 0, gen_indices=[4], add_cols_to_keep=["uuid"])
        elif i == 1:
            pp.replace_ext_grid_by_gen(net, [0], gen_indices=[4], cols_to_keep=["uuid", "max_p_mw"])
        assert not net.ext_grid.shape[0]
        assert not net.res_ext_grid.shape[0]
        assert np.allclose(net.gen.vm_pu.values, [1.03, 1.02])
        assert net.res_gen.p_mw.dropna().shape[0] == 2
        assert np.allclose(net.gen.index.values, [0, 4])
        assert net.gen.loc[4, "uuid"] == "test"
        assert net.group.element_type.tolist() == ["line", "gen"]
        assert net.group.element.iat[1] == [4]

        # replace_gen_by_ext_grid
        if i == 0:
            pp.replace_gen_by_ext_grid(net)
        elif i == 1:
            pp.replace_gen_by_ext_grid(net, [0, 4], ext_grid_indices=[2, 3])
            assert np.allclose(net.ext_grid.index.values, [2, 3])
        assert not net.gen.shape[0]
        assert not net.res_gen.shape[0]
        assert net.ext_grid.va_degree.dropna().shape[0] == 2
        assert any(np.isclose(net.ext_grid.va_degree.values, 0))
        assert net.res_ext_grid.p_mw.dropna().shape[0] == 2


def test_replace_gen_sgen():
    for i in range(2):
        net = nw.case9()
        vm_set = [1.03, 1.02]
        net.gen["vm_pu"] = vm_set
        net.gen["slack_weight"] = 1
        pp.runpp(net)
        assert list(net.res_gen.index.values) == [0, 1]

        # replace_gen_by_sgen
        if i == 0:
            pp.replace_gen_by_sgen(net)
        elif i == 1:
            pp.replace_gen_by_sgen(net, [0, 1], sgen_indices=[4, 1], cols_to_keep=[
                "max_p_mw"], add_cols_to_keep=["slack_weight"])  # min_p_mw is not in cols_to_keep
            assert np.allclose(net.sgen.index.values, [4, 1])
            assert np.allclose(net.sgen.slack_weight.values, 1)
            assert "max_p_mw" in net.sgen.columns
            assert "min_p_mw" not in net.sgen.columns
        assert not net.gen.shape[0]
        assert not net.res_gen.shape[0]
        assert not np.allclose(net.sgen.q_mvar.values, 0)
        assert net.res_gen.shape[0] == 0
        pp.runpp(net)
        assert np.allclose(net.res_bus.loc[net.sgen.bus, "vm_pu"].values, vm_set)

        # replace_sgen_by_gen
        net2 = copy.deepcopy(net)
        if i == 0:
            pp.replace_sgen_by_gen(net2, [1])
        elif i == 1:
            pp.replace_sgen_by_gen(net2, 1, gen_indices=[2], add_cols_to_keep=["slack_weight"])
            assert np.allclose(net2.gen.index.values, [2])
            assert np.allclose(net2.gen.slack_weight.values, 1)
        assert net2.gen.shape[0] == 1
        assert net2.res_gen.shape[0] == 1
        assert net2.gen.shape[0] == 1
        assert net2.res_gen.shape[0] == 1

        if i == 0:
            pp.replace_sgen_by_gen(net, 1)
            assert pandapower.toolbox.nets_equal(net, net2)


def test_replace_pq_elmtype():
    def check_elm_shape(net, elm_shape: dict):
        for elm, no in elm_shape.items():
            assert net[elm].shape[0] == no

    net = pp.create_empty_network()
    pp.create_buses(net, 3, 20)
    pp.create_ext_grid(net, 0)
    for to_bus in [1, 2]:
        pp.create_line(net, 0, to_bus, 0.6, 'NA2XS2Y 1x95 RM/25 12/20 kV')
    names = ["load 1", "load 2"]
    types = ["house", "commercial"]
    pp.create_loads(net, [1, 2], 0.8, 0.1, sn_mva=1, min_p_mw=0.5, max_p_mw=1.0, controllable=True,
                    name=names, scaling=[0.8, 1], type=types)
    pp.create_poly_cost(net, 0, "load", 7)
    pp.create_poly_cost(net, 1, "load", 3)
    pp.runpp(net)
    net.load["controllable"] = net.load["controllable"].astype(bool)
    net_orig = copy.deepcopy(net)

    # --- test unset old_indices, cols_to_keep and add_cols_to_keep
    pp.replace_pq_elmtype(net, "load", "sgen", new_indices=[2, 7], cols_to_keep=["type"],
                          add_cols_to_keep=["scaling"])  # cols_to_keep is not
    # default but ["type"] -> min/max p_mw get lost
    check_elm_shape(net, {"load": 0, "sgen": 2})
    assert list(net.sgen.index) == [2, 7]
    assert list(net.sgen.type.values) == types
    assert list(net.sgen.name.values) == names
    assert net.sgen.controllable.astype(bool).all()
    assert "min_p_mw" not in net.sgen.columns
    pp.runpp(net)
    assert pandapower.toolbox.dataframes_equal(net_orig.res_bus, net.res_bus)

    # --- test set old_indices and add_cols_to_keep for different element types
    net = copy.deepcopy(net_orig)
    add_cols_to_keep = ["scaling", "type", "sn_mva"]
    pp.replace_pq_elmtype(net, "load", "sgen", old_indices=1, add_cols_to_keep=add_cols_to_keep)
    check_elm_shape(net, {"load": 1, "sgen": 1})
    pp.runpp(net)
    assert pandapower.toolbox.dataframes_equal(net_orig.res_bus, net.res_bus)
    assert net.sgen.max_p_mw.at[0] == - 0.5
    assert net.sgen.min_p_mw.at[0] == - 1.0

    pp.replace_pq_elmtype(net, "sgen", "storage", old_indices=0, add_cols_to_keep=add_cols_to_keep)
    check_elm_shape(net, {"load": 1, "storage": 1})
    pp.runpp(net)
    assert pandapower.toolbox.dataframes_equal(net_orig.res_bus, net.res_bus)

    pp.replace_pq_elmtype(net, "storage", "load", add_cols_to_keep=add_cols_to_keep)
    pp.runpp(net)
    check_elm_shape(net, {"storage": 0, "sgen": 0})
    net.poly_cost.element = net.poly_cost.element.astype(net_orig.poly_cost.dtypes["element"])
    assert pandapower.toolbox.nets_equal(net_orig, net, exclude_elms={"sgen", "storage"})


def test_get_connected_elements_dict():
    net = nw.example_simple()
    conn = pp.get_connected_elements_dict(net, [0])
    assert conn == {"line": [0], 'ext_grid': [0], 'bus': [1]}
    conn = pp.get_connected_elements_dict(net, [3, 4])
    assert conn == {'line': [1, 3], 'switch': [1, 2, 7], 'trafo': [0], 'bus': [2, 5, 6]}


def test_get_connected_elements_empty_in_service():
    # would cause an error with respect_in_service=True for the case of:
    #  - empty element tables
    #  - element tables without in_service column (e.g. measurement)
    #  - element_table was unbound for the element table measurement
    #  see #1592
    net = nw.example_simple()
    net.bus.at[6, "in_service"] = False
    conn = pp.get_connected_elements_dict(net, [0], respect_switches=False, respect_in_service=True)
    assert conn == {"line": [0], 'ext_grid': [0], 'bus': [1]}
    conn = pp.get_connected_elements_dict(net, [3, 4], respect_switches=False, respect_in_service=True)
    assert conn == {'line': [1, 3], 'switch': [1, 2, 7], 'trafo': [0], 'bus': [2, 5]}


def test_replace_ward_by_internal_elements():
    net = nw.example_simple()
    pp.create_ward(net, 1, 10, 5, -20, -10, name="ward_1")
    pp.create_ward(net, 5, 6, 8, 10, 5, name="ward_2")
    pp.create_ward(net, 6, -1, 9, 11, 6, name="ward_3", in_service=False)
    pp.create_group_from_dict(net, {"ward": [0]}, name="test group")
    pp.runpp(net)
    net_org = copy.deepcopy(net)
    pp.replace_ward_by_internal_elements(net)
    for elm in ["load", "shunt"]:
        assert net[elm].shape[0] == 4
    res_load_created, res_shunt_created = copy.deepcopy(net.res_load), copy.deepcopy(net.res_shunt)
    pp.runpp(net)
    assert np.allclose(net_org.res_ext_grid.p_mw, net.res_ext_grid.p_mw)
    assert np.allclose(net_org.res_ext_grid.q_mvar, net.res_ext_grid.q_mvar)
    assert np.allclose(res_load_created, net.res_load)
    assert np.allclose(res_shunt_created, net.res_shunt)

    new_ets = pd.Index(["load", "shunt"])
    assert pp.count_group_elements(net_org, 0).to_dict() == {"ward": 1}
    assert pp.count_group_elements(net, 0).to_dict() == {et: 1 for et in new_ets}
    elm_change = pandapower.toolbox.count_elements(net, return_empties=True) - pandapower.toolbox.count_elements(
        net_org, return_empties=True)
    assert set(elm_change.loc[new_ets]) == {3}
    assert elm_change.at["ward"] == -3
    assert set(elm_change.loc[elm_change.index.difference(new_ets).difference(pd.Index([
        "ward"]))]) == {0}

    net = nw.example_simple()
    pp.create_ward(net, 1, 10, 5, -20, -10, name="ward_1")
    pp.create_ward(net, 5, 6, 8, 10, 5, name="ward_2")
    pp.create_ward(net, 6, -1, 9, 11, 6, name="ward_3", in_service=False)
    pp.runpp(net)
    net_org = copy.deepcopy(net)
    pp.replace_ward_by_internal_elements(net, [1])
    for elm in ["load", "shunt"]:
        assert net[elm].shape[0] == 2
    res_load_created, res_shunt_created = copy.deepcopy(net.res_load), copy.deepcopy(net.res_shunt)
    pp.runpp(net)
    assert np.allclose(net_org.res_ext_grid.p_mw, net.res_ext_grid.p_mw)
    assert np.allclose(net_org.res_ext_grid.q_mvar, net.res_ext_grid.q_mvar)
    assert np.allclose(res_load_created, net.res_load)
    assert np.allclose(res_shunt_created, net.res_shunt)


def test_replace_xward_by_internal_elements():
    net = nw.example_simple()
    pp.create_xward(net, 1, 10, 5, -20, -10, 0.1, 0.55, 1.02, name="xward_1")
    pp.create_xward(net, 5, 6, 8, 10, 5, 0.009, 0.678, 1.03, name="xward_2")
    pp.create_xward(net, 6, 6, 8, 10, 5, 0.009, 0.678, 1.03, in_service=False, name="xward_3")
    pp.create_group_from_dict(net, {"xward": [0]}, name="test group")
    pp.runpp(net)
    net_org = copy.deepcopy(net)
    pp.replace_xward_by_internal_elements(net)
    pp.runpp(net)
    assert abs(max(net_org.res_ext_grid.p_mw - net.res_ext_grid.p_mw)) < 1e-10
    assert abs(max(net_org.res_ext_grid.q_mvar - net.res_ext_grid.q_mvar)) < 1e-10

    new_ets = pd.Index(["load", "shunt", "gen", "impedance", "bus"])
    assert pp.count_group_elements(net_org, 0).to_dict() == {"xward": 1}
    assert pp.count_group_elements(net, 0).to_dict() == {et: 1 for et in new_ets}
    elm_change = pandapower.toolbox.count_elements(net, return_empties=True) - pandapower.toolbox.count_elements(
        net_org, return_empties=True)
    assert set(elm_change.loc[new_ets]) == {3}
    assert elm_change.at["xward"] == -3
    assert set(elm_change.loc[elm_change.index.difference(new_ets).difference(pd.Index([
        "xward"]))]) == {0}

    net = nw.example_simple()
    pp.create_xward(net, 1, 10, 5, -20, -10, 0.1, 0.55, 1.02, name="xward_1")
    pp.create_xward(net, 5, 6, 8, 10, 5, 0.009, 0.678, 1.03, name="xward_2")
    pp.create_xward(net, 6, 6, 8, 10, 5, 0.009, 0.678, 1.03, in_service=False, name="xward_3")
    pp.runpp(net)
    net_org = copy.deepcopy(net)
    pp.replace_xward_by_internal_elements(net, [0, 1])
    pp.runpp(net)
    assert abs(max(net_org.res_ext_grid.p_mw - net.res_ext_grid.p_mw)) < 1e-10
    assert abs(max(net_org.res_ext_grid.q_mvar - net.res_ext_grid.q_mvar)) < 1e-10


def test_repl_to_line():
    net = nw.simple_four_bus_system()
    idx = 0
    std_type = "NAYY 4x150 SE"
    new_idx = pp.repl_to_line(net, idx, std_type, in_service=True)
    pp.runpp(net)

    vm1 = net.res_bus.vm_pu.values
    va1 = net.res_bus.va_degree.values

    net.line.at[new_idx, "in_service"] = False
    pp.change_std_type(net, idx, std_type)
    pp.runpp(net)

    vm0 = net.res_bus.vm_pu.values
    va0 = net.res_bus.va_degree.values

    assert np.allclose(vm1, vm0)
    assert np.allclose(va1, va0)


def test_repl_to_line_with_switch():
    """
    Same test as above, but this time in comparison to actual replacement
    """
    net = nw.example_multivoltage()
    pp.runpp(net)

    for testindex in net.line.index:
        if net.line.in_service.loc[testindex]:
            line = net.line.loc[testindex]
            fbus = line.from_bus
            tbus = line.to_bus
            len = line.length_km

            if "184-AL1/30-ST1A" in net.line.std_type.loc[testindex]:
                std = "243-AL1/39-ST1A 110.0"
            elif "NA2XS2Y" in net.line.std_type.loc[testindex]:
                std = "NA2XS2Y 1x240 RM/25 6/10 kV"
            elif "NAYY" in net.line.std_type.loc[testindex]:
                std = "NAYY 4x150 SE"
            elif " 15-AL1/3-ST1A" in net.line.std_type.loc[testindex]:
                std = "24-AL1/4-ST1A 0.4"

            # create an oos line at the same buses
            REPL = pp.create_line(net, from_bus=fbus, to_bus=tbus, length_km=len, std_type=std)

            for bus in fbus, tbus:
                if bus in net.switch[~net.switch.closed & (net.switch.element == testindex)].bus.values:
                    pp.create_switch(net, bus=bus, element=REPL, closed=False, et="l", type="LBS")

            # calculate runpp with REPL
            net.line.in_service[testindex] = False
            net.line.in_service[REPL] = True
            pp.runpp(net)

            fbus_repl = net.res_bus.loc[fbus]
            tbus_repl = net.res_bus.loc[tbus]

            ploss_repl = (net.res_line.loc[REPL].p_from_mw - net.res_line.loc[REPL].p_to_mw)
            qloss_repl = (net.res_line.loc[REPL].q_from_mvar - net.res_line.loc[REPL].q_to_mvar)

            # get ne line impedances
            new_idx = pp.repl_to_line(net, testindex, std, in_service=True)
            # activate new idx line
            net.line.in_service[REPL] = False
            net.line.in_service[testindex] = True
            net.line.in_service[new_idx] = True
            pp.runpp(net)
            # compare lf results
            fbus_ne = net.res_bus.loc[fbus]
            tbus_ne = net.res_bus.loc[tbus]
            ploss_ne = (net.res_line.loc[testindex].p_from_mw -
                        net.res_line.loc[testindex].p_to_mw) + \
                       (net.res_line.loc[new_idx].p_from_mw - net.res_line.loc[new_idx].p_to_mw)
            qloss_ne = (net.res_line.loc[testindex].q_from_mvar -
                        net.res_line.loc[testindex].q_to_mvar) + \
                       (net.res_line.loc[new_idx].q_from_mvar - net.res_line.loc[new_idx].q_to_mvar)

            assert_series_equal(fbus_repl, fbus_ne, atol=1e-2)
            assert_series_equal(tbus_repl, tbus_ne)
            assert np.isclose(ploss_repl, ploss_ne, atol=1e-5)
            assert np.isclose(qloss_repl, qloss_ne)

            # and reset to unreinforced state again
            net.line.in_service[testindex] = True
            net.line.in_service[new_idx] = False
            net.line.in_service[REPL] = False


def test_merge_parallel_line():
    net = nw.example_multivoltage()
    pp.runpp(net)
    assert net.line.parallel.at[5] == 2

    line = net.line.loc[5]
    fbus = line.from_bus
    tbus = line.to_bus

    fbus_0 = net.res_bus.loc[fbus]
    tbus_0 = net.res_bus.loc[tbus]
    ploss_0 = (net.res_line.loc[5].p_from_mw - net.res_line.loc[5].p_to_mw)
    qloss_0 = (net.res_line.loc[5].q_from_mvar - net.res_line.loc[5].q_to_mvar)

    net = pp.merge_parallel_line(net, 5)

    assert net.line.parallel.at[5] == 1
    pp.runpp(net)
    fbus_1 = net.res_bus.loc[fbus]
    tbus_1 = net.res_bus.loc[tbus]
    ploss_1 = (net.res_line.loc[5].p_from_mw - net.res_line.loc[5].p_to_mw)
    qloss_1 = (net.res_line.loc[5].q_from_mvar - net.res_line.loc[5].q_to_mvar)

    assert_series_equal(fbus_0, fbus_1)
    assert_series_equal(tbus_0, tbus_1)
    assert np.isclose(ploss_0, ploss_1, atol=1e-5)
    assert np.isclose(qloss_0, qloss_1)


def test_merge_same_bus_generation_plants():
    gen_elms = ["ext_grid", "gen", "sgen"]

    # --- test with case9
    net = nw.case9()
    buses = np.hstack([net[elm].bus.values for elm in gen_elms])
    has_dupls = len(buses) > len(set(buses))

    something_merged = pp.merge_same_bus_generation_plants(net)

    assert has_dupls == something_merged

    # --- test with case24_ieee_rts
    net = nw.case24_ieee_rts()

    # manipulate net for different functionality checks
    # 1) q_mvar should be summed which is only possible if no gen or ext_grid has the same bus
    net.gen = net.gen.drop(net.gen.index[net.gen.bus == 22])
    net.sgen["q_mvar"] = np.arange(net.sgen.shape[0])
    # 2) remove limit columns or values to check whether merge_same_bus_generation_plants() can
    # handle that
    del net.sgen["max_q_mvar"]
    net.sgen.at[1, "min_p_mw"] = np.nan

    # prepare expatation values
    dupl_buses = [0, 1, 6, 12, 14, 21, 22]
    n_plants = sum([net[elm].bus.isin(dupl_buses).sum() for elm in gen_elms])
    assert n_plants > len(dupl_buses)  # check that in net are plants with same buses
    expected_no_of_plants = sum([net[elm].shape[0] for elm in gen_elms]) - n_plants + \
                            len(dupl_buses)

    # run function
    something_merged = pp.merge_same_bus_generation_plants(net)

    # check results
    assert something_merged
    buses = np.hstack([net[elm].bus.values for elm in gen_elms])
    assert len(buses) == len(set(buses))  # no dupl buses in gen plant dfs
    n_plants = sum([net[elm].shape[0] for elm in gen_elms])
    assert n_plants == expected_no_of_plants
    assert np.isclose(net.ext_grid.p_disp_mw.at[0], 95.1 * 2)  # correct value sum (p_disp)
    assert np.isclose(net.gen.p_mw.at[0], 10 * 2 + 76 * 2)  # correct value sum (p_mw)
    assert np.isclose(net.gen.min_p_mw.at[0], 16 * 2 + 15.2)  # correct value sum (min_p_mw) (
    # 1x 15.2 has been removed above)
    assert np.isclose(net.gen.max_p_mw.at[0], 20 * 2 + 76 * 2)  # correct value sum (max_p_mw)
    assert np.isclose(net.gen.min_q_mvar.at[8], -10 - 16 * 5)  # correct value sum (min_q_mvar)
    assert np.isclose(net.gen.max_q_mvar.at[8], 16)  # correct value sum (max_q_mvar) (
    # the sgen max_q_mvar column has been removed above)
    idx_sgen22 = net.sgen.index[net.sgen.bus == 22]
    assert len(idx_sgen22) == 1
    assert np.isclose(net.sgen.q_mvar.at[idx_sgen22[0]], 20 + 21)  # correct value sum (q_mvar)


def test_set_isolated_areas_out_of_service():
    net = nw.case9()
    pp.create_switch(net, 6, 5, "l", False)
    pp.create_switch(net, 8, 7, "l", False)

    pp.toolbox.set_isolated_areas_out_of_service(net)

    isolated_buses = [7, 1]
    isolated_lines = [5, 7, 6]

    assert not np.any(net.bus.loc[isolated_buses, 'in_service'])
    assert np.all(net.bus.loc[np.setdiff1d(net.bus.index, isolated_buses), 'in_service'])

    assert not np.any(net.line.loc[isolated_lines, 'in_service'])
    assert np.all(net.line.loc[np.setdiff1d(net.line.index, isolated_lines), 'in_service'])


if __name__ == '__main__':
    pytest.main([__file__, "-x"])

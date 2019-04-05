# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.networks as nw
import pandapower.toolbox as tb


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
    net["load"]["p_mw"][net["load"].index[0]] += 0.1
    assert not tb.nets_equal(original, net)
    assert not tb.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting added column
    net["load"]["new_col"] = 0.1
    assert not tb.nets_equal(original, net)
    assert not tb.nets_equal(net, original)
    net = copy.deepcopy(original)

    # not detecting alternated value if difference is beyond tolerance
    net["load"]["p_mw"][net["load"].index[0]] += 0.0001
    assert tb.nets_equal(original, net, tol=0.1)
    assert tb.nets_equal(net, original, tol=0.1)


def test_add_column_from_node_to_elements():
    net = nw.create_cigre_network_mv("pv_wind")
    net.bus["subnet"] = ["subnet_%i" % i for i in range(net.bus.shape[0])]
    net.sgen["subnet"] = "already_given"
    net.switch["subnet"] = None
    net_orig = copy.deepcopy(net)

    branch_bus = ["from_bus", "lv_bus"]
    pp.add_column_from_node_to_elements(net, "subnet", False, branch_bus=branch_bus)

    def check_subnet_correctness(net, elements, branch_bus):
        for elm in elements:
            if "bus" in net[elm].columns:
                assert all(pp.compare_arrays(net[elm]["subnet"].values,
                                             np.array(["subnet_%i" % bus for bus in net[elm].bus])))
            elif branch_bus[0] in net[elm].columns:
                assert all(pp.compare_arrays(net[elm]["subnet"].values, np.array([
                        "subnet_%i" % bus for bus in net[elm][branch_bus[0]]])))
            elif branch_bus[1] in net[elm].columns:
                assert all(pp.compare_arrays(net[elm]["subnet"].values, np.array([
                        "subnet_%i" % bus for bus in net[elm][branch_bus[1]]])))

    check_subnet_correctness(net, pp.pp_elements(bus=False)-{"sgen"}, branch_bus)

    pp.add_column_from_node_to_elements(net_orig, "subnet", True, branch_bus=branch_bus)
    check_subnet_correctness(net_orig, pp.pp_elements(bus=False), branch_bus)


def test_add_column_from_element_to_elements():
    net = nw.create_cigre_network_mv()
    pp.create_measurement(net, "i", "trafo", 5, 3, 0, side="hv")
    pp.create_measurement(net, "i", "line", 5, 3, 0, side="to")
    pp.create_measurement(net, "p", "bus", 5, 3, 2)
    assert net.measurement.name.isnull().all()
    assert ~net.switch.name.isnull().all()
    orig_switch_names = copy.deepcopy(net.switch.name.values)
    expected_measurement_names = np.array([
        net.trafo.name.loc[0], net.line.name.loc[0], net.bus.name.loc[2]])
    expected_switch_names = np.append(
        net.line.name.loc[net.switch.element.loc[net.switch.et == "l"]].values,
        net.trafo.name.loc[net.switch.element.loc[net.switch.et == "t"]].values)

    pp.add_column_from_element_to_elements(net, "name", False)
    assert all(pp.compare_arrays(net.measurement.name.values, expected_measurement_names))
    assert all(pp.compare_arrays(net.switch.name.values, orig_switch_names))

    del net.measurement["name"]
    pp.add_column_from_element_to_elements(net, "name", True)
    assert all(pp.compare_arrays(net.measurement.name.values, expected_measurement_names))
    assert all(pp.compare_arrays(net.switch.name.values, expected_switch_names))


def test_continuos_bus_numbering():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, 0.4, index=12)
    pp.create_load(net, bus0, p_mw=0.)
    pp.create_load(net, bus0, p_mw=0.)
    pp.create_load(net, bus0, p_mw=0.)
    pp.create_load(net, bus0, p_mw=0.)

    bus0 = pp.create_bus(net, 0.4, index=42)
    pp.create_sgen(net, bus0, p_mw=0.)
    pp.create_sgen(net, bus0, p_mw=0.)
    pp.create_sgen(net, bus0, p_mw=0.)

    bus0 = pp.create_bus(net, 0.4, index=543)
    pp.create_shunt(net, bus0, 2, 1)
    pp.create_shunt(net, bus0, 2, 1)
    pp.create_shunt(net, bus0, 2, 1)

    bus0 = pp.create_bus(net, 0.4, index=5675)
    pp.create_ward(net, bus0, 2, 1, 1, 2, )
    pp.create_ward(net, bus0, 2, 1, 1, 2, )
    pp.create_ward(net, bus0, 2, 1, 1, 2, )

    tb.create_continuous_bus_index(net)

    l = net.bus.index
    assert all(l[i] <= l[i + 1] for i in range(len(l) - 1))  # is ordered
    assert all(l[i] + 1 == l[i + 1] for i in range(len(l) - 1))  # is consecutive
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
    pp.create_load(net, bus0, p_mw=0., type="Household")
    pp.create_sgen(net, bus0, p_mw=0., type="PV")

    tb.set_scaling_by_type(net, {"Household": 42., "PV": 12})

    assert net.load.at[0, "scaling"] == 42
    assert net.sgen.at[0, "scaling"] == 12

    tb.set_scaling_by_type(net, {"Household": 0, "PV": 0})

    assert net.load.at[0, "scaling"] == 0
    assert net.sgen.at[0, "scaling"] == 0


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
        tb.drop_inactive_elements(net)

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
                    print(element)
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

    tb.drop_inactive_elements(net)

    assert gen0 not in net.gen.index


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

    lines = tb.get_connected_elements(net, "line", bus0, respect_switches=False,
                                      respect_in_service=False)

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


def test_merge_and_split_nets():
    net1 = nw.mv_oberrhein()
    # TODO there are some geodata values in oberrhein without corresponding lines
    net1.line_geodata.drop(set(net1.line_geodata.index) - set(net1.line.index), inplace=True)
    n1 = len(net1.bus)
    pp.runpp(net1)
    net2 = nw.create_cigre_network_mv()
    pp.runpp(net2)
    net = pp.merge_nets(net1, net2)
    pp.runpp(net)
    assert np.allclose(net.res_bus.vm_pu.iloc[:n1].values, net1.res_bus.vm_pu.values)
    assert np.allclose(net.res_bus.vm_pu.iloc[n1:].values, net2.res_bus.vm_pu.values)

    net3 = pp.select_subnet(net, net.bus.index[:n1], include_results=True)
    assert pp.dataframes_equal(net3.res_bus[["vm_pu"]], net1.res_bus[["vm_pu"]])

    net4 = pp.select_subnet(net, net.bus.index[n1:], include_results=True)
    assert np.allclose(net4.res_bus.vm_pu.values, net2.res_bus.vm_pu.values)


def test_overloaded_lines():
    net = pp.create_empty_network()

    bus0 = pp.create_bus(net, vn_kv=.4)
    bus1 = pp.create_bus(net, vn_kv=.4)

    pp.create_ext_grid(net, bus0)

    line0 = pp.create_line(net, bus0, bus1, length_km=1, std_type="NAYY 4x50 SE")
    line1 = pp.create_line(net, bus0, bus1, length_km=1, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    line2 = pp.create_line(net, bus0, bus1, length_km=1, std_type="15-AL1/3-ST1A 0.4")
    line3 = pp.create_line(net, bus0, bus1, length_km=10, std_type="149-AL1/24-ST1A 10.0")

    pp.create_load(net, bus1, p_mw=0.2, q_mvar=0.05)

    pp.runpp(net)
    # test the overloaded lines by default value of max_load=100
    overloaded_lines = tb.overloaded_lines(net, max_load=100)

    assert set(overloaded_lines) == set([line0, line1])

    # test the overloaded lines by a self defined value of max_load=50
    overloaded_lines = tb.overloaded_lines(net, max_load=50)

    assert set(overloaded_lines) == set([line0, line1, line2])


def test_violated_buses():
    net = nw.create_cigre_network_lv()

    pp.runpp(net)

    # set the range of vm.pu
    min_vm_pu = 0.92
    max_vm_pu = 1.1

    # print out the list of violated_bus's index
    violated_bus = tb.violated_buses(net, min_vm_pu, max_vm_pu)

    assert set(violated_bus) == set(net["bus"].index[[16, 35, 36, 40]])


def test_add_zones_to_elements():
    net = nw.create_cigre_network_mv()

    # add zones to lines and switchs
    tb.add_zones_to_elements(net, elements=["line", "switch"])

    # create 2 arrays which include "zone" in lines and switchs
    zone_line = net["line"]["zone"].values
    zone_switch = net["switch"]["zone"].values

    assert "CIGRE_MV" in net["line"]["zone"].values
    assert "CIGRE_MV" in net["switch"]["zone"].values


def test_fuse_buses():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=1, name="b1")
    b2 = pp.create_bus(net, vn_kv=1.5, name="b2")

    line1 = pp.create_line(net, b2, b1, length_km=1, std_type="NAYY 4x50 SE")

    sw1 = pp.create_switch(net, b2, line1, et="l")
    sw2 = pp.create_switch(net, b1, b2, et="b")

    load1 = pp.create_load(net, b1, p_mw=0.006)
    load2 = pp.create_load(net, b2, p_mw=0.005)

    tb.fuse_buses(net, b1, b2, drop=True)

    # assertion: elements connected to b2 are given to b1 instead
    assert net["line"]["from_bus"].loc[0] == b1
    assert net["switch"]["bus"].loc[0] == b1
    assert net["load"]["bus"].loc[1] == b1
    # assertion: b2 not in net.bus table if drop=True
    assert b2 not in net.bus.index


def test_close_switch_at_line_with_two_open_switches():
    net = pp.create_empty_network()

    bus1 = pp.create_bus(net, vn_kv=.4)
    bus2 = pp.create_bus(net, vn_kv=.4)
    bus3 = pp.create_bus(net, vn_kv=.4)

    line1 = pp.create_line(net, bus2, bus3, length_km=1., std_type="NAYY 4x50 SE")
    line2 = pp.create_line(net, bus2, bus3, length_km=1., std_type="NAYY 4x50 SE")
    line3 = pp.create_line(net, bus2, bus3, length_km=1., std_type="NAYY 4x50 SE")

    sw1 = pp.create_switch(net, bus1, bus2, et="b", closed=True)

    sw2 = pp.create_switch(net, bus2, line1, et="l", closed=False)
    sw3 = pp.create_switch(net, bus3, line1, et="l", closed=False)

    sw4 = pp.create_switch(net, bus2, line2, et="l", closed=True)
    sw5 = pp.create_switch(net, bus3, line2, et="l", closed=False)

    sw6 = pp.create_switch(net, bus3, line2, et="l", closed=True)
    sw7 = pp.create_switch(net, bus3, line2, et="l", closed=True)

    tb.close_switch_at_line_with_two_open_switches(net)

    # assertion: sw2 closed
    assert net.switch.closed.loc[1]


def test_pq_from_cosphi():
    p, q = pp.pq_from_cosphi(1/0.95, 0.95, "ind", "load")
    assert np.isclose(p, 1)
    assert np.isclose(q, 0.3286841051788632)

    s = np.array([1, 1, 1])
    cosphi = np.array([1, 0.5, 0])
    pmode = np.array(["load", "load", "load"])
    qmode = np.array(["ind", "ind", "ind"])
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    excpected_values = (np.array([1, 0.5, 0]), np.array([0, 0.8660254037844386, 1]))
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    pmode = "gen"
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, -excpected_values[1])

    qmode = "cap"
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    try:
        pp.pq_from_cosphi(1, 0.95, "ohm", "gen")
        bool_ = False
    except ValueError:
        bool_ = True
    assert bool_

    p, q = pp.pq_from_cosphi(0, 0.8, "cap", "gen")
    assert np.isclose(p, 0)
    assert np.isclose(q, 0)


def test_cosphi_from_pq():
    cosphi, s, qmode, pmode = pp.cosphi_from_pq(1, 0.4)
    assert np.isclose(cosphi, 0.9284766908852593)
    assert np.isclose(s, 1.077032961426901)
    assert qmode == 'ind'
    assert pmode == 'load'

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1])
    q = np.array([1, -1, 0, 0.5, -0.5, 1, -1, 0, 1, -1, 0])
    cosphi, s, qmode, pmode = pp.cosphi_from_pq(p, q)
    assert np.allclose(cosphi[[0, 1, 8, 9]], 2**0.5/2)
    assert np.allclose(cosphi[[3, 4]], 0.89442719)
    assert np.allclose(cosphi[[2, 10]], 1)
    assert pd.Series(cosphi[[5, 6, 7]]).isnull().all()
    assert np.allclose(s, (p ** 2 + q ** 2) ** 0.5)
    assert all(pmode == np.array(["load"]*5+["undef"]*3+["gen"]*3))
    ind_cap_ohm = ["ind", "cap", "ohm"]
    assert all(qmode == np.array(ind_cap_ohm+["ind", "cap"]+ind_cap_ohm*2))


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
    tb.create_replacement_switch_for_branch(net, 'line', line0)
    tb.create_replacement_switch_for_branch(net, 'impedance', impedance0)
    net.line.in_service.at[line0] = False
    net.impedance.in_service.at[impedance0] = False

    assert 'REPLACEMENT_line_0' in net.switch.name.values
    assert 'REPLACEMENT_impedance_0' in net.switch.name.values
    assert net.switch.closed.at[0]
    assert net.switch.closed.at[1]
    pp.runpp(net)

    # look that the switch is created with the correct closed status
    net.line.in_service.at[line1] = False
    net.impedance.in_service.at[impedance1] = False
    tb.create_replacement_switch_for_branch(net, 'line', line1)
    tb.create_replacement_switch_for_branch(net, 'impedance', impedance1)

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

    line0 = pp.create_line(net, bus0, bus1, length_km=0, std_type="NAYY 4x50 SE")
    line1 = pp.create_line_from_parameters(net, bus2, bus3, length_km=1, r_ohm_per_km=0,
                                           x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    line2 = pp.create_line_from_parameters(net, bus3, bus4, length_km=1, r_ohm_per_km=0,
                                           x_ohm_per_km=0, c_nf_per_km=0, max_i_ka=1)

    impedance0 = pp.create_impedance(net, bus1, bus2, 0.01, 0.01, sn_mva=100)
    impedance1 = pp.create_impedance(net, bus4, bus5, 0, 0, sn_mva=100)
    impedance2 = pp.create_impedance(net, bus5, bus6, 0, 0, rtf_pu=0.1, sn_mva=100)
    return net


def test_for_line_with_zero_length(net):
    tb.replace_zero_branches_with_switches(net, elements=('line',), zero_impedance=False)
    assert 'REPLACEMENT_line_0' in net.switch.name.values
    assert ~net.line.in_service.at[0]
    assert 'REPLACEMENT_line_2' not in net.switch.name.values


def test_drop(net):
    tb.replace_zero_branches_with_switches(net, elements=('line', 'impedance'), drop_affected=True)
    assert len(net.line) == 1
    assert len(net.impedance) == 2


def test_in_service_only(net):
    tb.replace_zero_branches_with_switches(net, elements=('line',))
    assert len(net.switch.loc[net.switch.name == 'REPLACEMENT_line_0']) == 1
    tb.replace_zero_branches_with_switches(net, elements=('line',), in_service_only=False)
    assert len(net.switch.loc[net.switch.name == 'REPLACEMENT_line_0']) == 2
    assert ~net.switch.closed.at[2]


def test_line_with_zero_impediance(net):
    # test for line with zero impedance
    tb.replace_zero_branches_with_switches(net, elements=('line',), zero_length=False)
    assert 'REPLACEMENT_line_1' not in net.switch.name.values
    assert 'REPLACEMENT_line_2' in net.switch.name.values


def test_impedance(net):
    tb.replace_zero_branches_with_switches(net, elements=('impedance',), zero_length=False,
                                           zero_impedance=True, in_service_only=True)
    assert 'REPLACEMENT_impedance_0' not in net.switch.name.values
    assert 'REPLACEMENT_impedance_1' in net.switch.name.values
    assert 'REPLACEMENT_impedance_2' not in net.switch.name.values


def test_all(net):
    tb.replace_zero_branches_with_switches(net, elements=('impedance', 'line'), zero_length=True,
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

    line1 = pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=20.1, std_type='24-AL1/4-ST1A 0.4', name='line1')

    # switch0=pp.create_switch(net, bus = bus0, element = trafo0, et = 't3') #~~~~~ not implementable now
    switch1 = pp.create_switch(net, bus=bus1, element=bus5, et='b')
    switch2 = pp.create_switch(net, bus=bus2, element=trafo1, et='t')
    switch3 = pp.create_switch(net, bus=bus3, element=line1, et='l')

    # assert tb.next_bus(net,bus0,trafo0,et='trafo3w')==bus1                         # not implemented in existing toolbox
    # assert tb.next_bus(net,bus0,trafo0,et='trafo3w',choice_for_trafo3w='lv')==bus2 # not implemented in existing toolbox
    assert tb.next_bus(net, bus1, switch1, et='switch') == bus5  # Switch with bus2bus connection
    # assert not tb.next_bus(net,bus2,switch2,et='switch')==bus3  # Switch with bus2trafo connection:- gives trasformer id instead of bus id
    assert tb.next_bus(net, bus2, trafo1, et='trafo') == bus3
    # assert tb.next_bus(net,bus3,switch3,et='switch') ==bus4  # Switch with bus2line connection :- gives line id instead of bus id
    assert tb.next_bus(net, bus3, line1, et='line') == bus4


def test_get_connected_buses():
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

    line1 = pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=20.1, std_type='24-AL1/4-ST1A 0.4', name='line1')

    # switch0=pp.create_switch(net, bus = bus0, element = trafo0, et = 't3') #~~~~~ not implementable
    switch1 = pp.create_switch(net, bus=bus1, element=bus5, et='b')
    switch2 = pp.create_switch(net, bus=bus2, element=trafo1, et='t')
    switch3 = pp.create_switch(net, bus=bus3, element=line1, et='l')

    assert list(tb.get_connected_buses(net, [bus0])) == [bus1, bus2]  # trafo3w has not been implemented in the function
    assert list(tb.get_connected_buses(net, [bus1])) == [bus0, bus2,
                                                         bus5]  # trafo3w has not been implemented in the function
    assert list(tb.get_connected_buses(net, [bus2])) == [bus0, bus1,
                                                         bus3]  # trafo3w has not been implemented in the function
    assert list(tb.get_connected_buses(net, [bus3])) == [bus2, bus4]
    assert list(tb.get_connected_buses(net, [bus4])) == [bus3]
    assert list(tb.get_connected_buses(net, [bus5])) == [bus1]

    assert list(tb.get_connected_buses(net, [bus0, bus1])) == [bus2, bus5]
    assert list(tb.get_connected_buses(net, [bus2, bus3])) == [bus0, bus1, bus4]


def test_drop_elements_at_buses():
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

    line1 = pp.create_line(net, from_bus=bus3, to_bus=bus4, length_km=20.1, std_type='24-AL1/4-ST1A 0.4', name='line1')

    # switch0=pp.create_switch(net, bus = bus0, element = trafo0, et = 't3') #~~~~~ not implementable now
    switch1 = pp.create_switch(net, bus=bus1, element=bus5, et='b')
    switch2 = pp.create_switch(net, bus=bus2, element=trafo1, et='t')
    switch3 = pp.create_switch(net, bus=bus3, element=line1, et='l')
    # bus id needs to be entered as iterable, not done in the function
    tb.drop_elements_at_buses(net, [bus5])
    assert len(net.switch) == 2
    assert len(net.trafo) == 1
    assert len(net.trafo3w) == 1
    assert len(net.line) == 1
    tb.drop_elements_at_buses(net, [bus4])
    assert len(net.switch) == 1
    assert len(net.line) == 0
    assert len(net.trafo) == 1
    assert len(net.trafo3w) == 1
    tb.drop_elements_at_buses(net, [bus3])
    assert len(net.switch) == 0
    assert len(net.line) == 0
    assert len(net.trafo) == 0
    assert len(net.trafo3w) == 1
    tb.drop_elements_at_buses(net, [bus2])
    assert len(net.switch) == 0
    assert len(net.line) == 0
    assert len(net.trafo) == 0
    assert len(net.trafo3w) == 0
    tb.drop_elements_at_buses(net, [bus1])
    assert len(net.switch) == 0
    assert len(net.line) == 0
    assert len(net.trafo) == 0
    assert len(net.trafo3w) == 0


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

    assert pp.nets_equal(net1, net2, exclude_elms={"line", "impedance"})  # Todo: exclude_elms
    cols = ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "pl_mw", "ql_mvar", "i_from_ka",
            "i_to_ka"]
    assert np.allclose(net1.res_impedance[cols].values, net2.res_line[cols].values)

    net3 = copy.deepcopy(net2)
    pp.replace_line_by_impedance(net3)

    pp.runpp(net3)

    assert pp.nets_equal(net2, net3, exclude_elms={"line", "impedance"})
    assert np.allclose(net3.res_impedance[cols].values, net2.res_line[cols].values)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

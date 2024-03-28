import pytest
import numpy as np
import pandapower as pp
import pandapower.networks
import pandapower.grid_equivalents
import os
import pandas as pd
from random import sample

import pandapower.toolbox
from pandapower.control import ConstControl
from pandapower import pp_dir
from pandapower.timeseries import DFData
from pandapower.grid_equivalents.auxiliary import replace_motor_by_load
from pandapower.grid_equivalents.ward_generation import \
    create_passive_external_net_for_ward_admittance
from pandapower.grid_equivalents.auxiliary import _runpp_except_voltage_angles

try:
    from misc.groups import Group
    group_imported = True
except ImportError:
    group_imported = False


def create_test_net():
    net = pp.create_empty_network()
    # buses
    pp.create_buses(net, 7, 20, zone=[0, 0, 1, 1, 1, 0, 0], name=["bus %i" % i for i in range(7)],
                    min_vm_pu=np.append(np.arange(.9, 0.94, .01), [np.nan, np.nan, np.nan]))

    # ext_grid
    idx = pp.create_ext_grid(net, 0, 1.0, 0.0)
    pp.create_poly_cost(net, idx, "ext_grid", 10)

    # sgens
    idx = pp.create_sgen(net, 6, 1.2)
    pp.create_poly_cost(net, idx, "sgen", 12)
    idx = pp.create_sgen(net, 5, 1.5, index=5)
    pp.create_poly_cost(net, idx, "sgen", 14)

    # loads
    for load_bus in [0, 2, 3, 4, 6]:
        pp.create_load(net, load_bus, 1.2)

    # lines
    for i in range(6):
        pp.create_line(net, i, i+1, 1.1, 'NA2XS2Y 1x185 RM/25 12/20 kV')
    pp.create_line(net, 0, 6, 3.2, 'NA2XS2Y 1x185 RM/25 12/20 kV')

    # runpp and return
    pp.runpp(net, calculate_voltage_angles=True)
    return net


def run_basic_usecases(eq_type, net=None):

    if net is None:
        net = create_test_net()

    # UC1: get internal buses [0, 1, 5, 6] and equivalent connected to buses [1, 5]
    eq_net1 = pp.grid_equivalents.get_equivalent(
        net, eq_type, boundary_buses=[1, 5], internal_buses=[0, 6])
    pp.runpp(eq_net1, calculate_voltage_angles=True)

    # UC2: don't get the internal buses [0, 1, 5, 6] but the boundary buses [2, 4] and the
    # equivalent
    eq_net2 = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses=[2, 4],
                                 internal_buses=[0, 1, 5, 6], return_internal=False)

    # UC3: the input is only the subnet including the external buses [2, 3, 4] and the
    # boundary buses [1, 5] -> expected return are the boundary buses and the equivalent
    subnet = pp.select_subnet(net, list(range(1, 6)), include_results=True)
    subnet_rest = pp.select_subnet(net, [0, 1, 5, 6], include_results=True)
    eq_net3a = pp.grid_equivalents.get_equivalent(
        subnet, eq_type, boundary_buses=[1, 5], internal_buses=None)

    # UC3b tests whether this also works for 'internal_buses' as empty list
    eq_net3b = pp.grid_equivalents.get_equivalent(
        subnet, eq_type, boundary_buses=[1, 5], internal_buses=[])
    eq_net3a.sgen = eq_net3a.sgen.drop(columns=["origin_id"])
    eq_net3b.sgen = eq_net3b.sgen.drop(columns=["origin_id"])

    assert set(eq_net3a["group"].index) == set(eq_net3b["group"].index)
    assert pandapower.toolbox.nets_equal(eq_net3a, eq_net3b, exclude_elms=["group"])

    elm_lists = pp.group_element_lists(eq_net3b, eq_net3b.group.index[0])
    idx2comp = pp.create_group(eq_net3a, elm_lists[0], elm_lists[1], reference_columns=elm_lists[2])
    assert pp.compare_group_elements(eq_net3a, eq_net3a.group.index[0], idx2comp)

    # UC3: merge eq_net3 with subnet_rest
    eq_net3 = pp.grid_equivalents.merge_internal_net_and_equivalent_external_net(
        eq_net3a, subnet_rest)
    pp.runpp(eq_net3, calculate_voltage_angles=True)
    assert pandapower.toolbox.nets_equal(net, create_test_net())
    return eq_net1, eq_net2, eq_net3


def check_elements_amount(net, elms_dict, check_all_pp_elements=True):
    if check_all_pp_elements:
        elms_dict.update({elm: 0 for elm in pandapower.toolbox.pp_elements() if elm not in elms_dict.keys()})
    for key, val in elms_dict.items():
        if not net[key].shape[0] == val:
            raise ValueError("The net has %i %ss but %i are expected." % (
                    net[key].shape[0], key, int(val)))


def check_res_bus(net_orig, net_eq):
    """
    Checks all voltage results of all buses whose names occure in net_orig and net_eq.
    """
    orig_bus_names = net_orig.bus.name.astype(str)
    orig_bus_names_are_in_net_eq = orig_bus_names.isin(net_eq.bus.name)
    net_eq_bus_names = list(net_eq.bus.name.astype(str))
    idx_eq = [net_eq_bus_names.index(orig_bus_name) for orig_bus_name in orig_bus_names[
        orig_bus_names_are_in_net_eq]]
    assert np.allclose(
        net_eq.res_bus.loc[net_eq.bus.index[idx_eq], ["vm_pu", "va_degree"]].values,
        net_orig.res_bus.loc[orig_bus_names_are_in_net_eq, ["vm_pu", "va_degree"]].values)


def check_results_without_order(eq_net2, eq_net3):
    for elm in pandapower.toolbox.pp_elements():
        res_table = "res_"+elm
        if res_table in eq_net2.keys() and eq_net2[res_table].shape[0]:
            idxs3 = list(eq_net3[res_table].index)
            for idx2 in eq_net2[res_table].index:
                same_values_found = False
                for idx3 in idxs3:
                    if np.allclose(eq_net2[res_table].loc[idx2].values,
                                   eq_net2[res_table].loc[idx2].values):
                        same_values_found = True
                        idxs3.remove(idx3)
                        break
                if not same_values_found:
                    raise AssertionError("In eq_net3[%s], no line is found with values close to" +
                                         str(eq_net2[res_table].loc[idx2]))


def test_cost_consideration():
    """
    Checks whether the cost data are considered by REI equivalents (with (s)gen_seperate).
    """
    # input
    net = create_test_net()
    idx = pp.create_sgen(net, 1, 1.3, index=2)
    pp.create_poly_cost(net, idx, "sgen", 2.3, index=4)
    pp.runpp(net)
    assert all(net.sgen.index.values == np.array([0, 5, 2]))
    assert all(net.poly_cost.element == np.array([0, 0, 5, 2]))

    for cost_type in ["poly_cost", "pwl_cost"]:

        if cost_type == "pwl_cost":
            for poly in net.poly_cost.itertuples():
                net.poly_cost = net.poly_cost.drop(poly.Index)
                pp.create_pwl_cost(net, poly.element, poly.et, [[0, 20, 1]], index=poly.Index)

        # eq generation
        boundary_buses = [0, 2]
        internal_buses = [1]
        eq_net1 = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses, internal_buses)
        eq_net2 = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses, internal_buses,
                                     return_internal=False)

        # check elements
        check_elements_amount(eq_net1, {"bus": 6, "load": 3, "sgen": 3, "shunt": 5, "ext_grid": 1,
                                        "line": 2, "impedance": 7, cost_type: 4},
                              check_all_pp_elements=True)
        check_elements_amount(eq_net2, {"bus": 5, "load": 3, "sgen": 2, "shunt": 5, "ext_grid": 1,
                                        "impedance": 7, cost_type: 3},
                              check_all_pp_elements=True)
        assert all(eq_net1.sgen.index.values == np.array([2, 3, 4]))  # simple create_sgen()
        # without index=... expected
        assert all(eq_net2.sgen.index.values == np.array([3, 4]))

        # --- check poly cost
        # eq_net1
        assert np.all(net[cost_type].loc[net[cost_type].et == "ext_grid"].values ==
                      eq_net1[cost_type].loc[eq_net1[cost_type].et == "ext_grid"])
        for i in range(3):
            idx_net = net.sgen.sort_values("p_mw").index[i]
            idx_eq_net = eq_net1.sgen.sort_values("p_mw").index[i]
            assert np.all(net[cost_type].loc[(net[cost_type].element == idx_net) &
                                             (net[cost_type].et == "sgen")].drop(
                          columns=["element"]).values ==
                          eq_net1[cost_type].loc[(eq_net1[cost_type].element == idx_eq_net) &
                                                 (eq_net1[cost_type].et == "sgen")].drop(
                          columns=["element"]).values)

        # eq_net2
        assert np.all(net[cost_type].loc[net[cost_type].et == "ext_grid"].values ==
                      eq_net2[cost_type].loc[eq_net2[cost_type].et == "ext_grid"])
        for i in range(2):
            idx_net = net.sgen.loc[~net.sgen.bus.isin(boundary_buses+internal_buses)].sort_values(
                "p_mw").index[i]
            idx_eq_net = eq_net2.sgen.sort_values("p_mw").index[i]
            assert np.all(net[cost_type].loc[(net[cost_type].element == idx_net) &
                                             (net[cost_type].et == "sgen")].drop(
                          columns=["element"]).values ==
                          eq_net2[cost_type].loc[(eq_net2[cost_type].element == idx_eq_net) &
                                                 (eq_net2[cost_type].et == "sgen")].drop(
                          columns=["element"]).values)


def test_basic_usecases():
    """
    This test checks basic use cases of network equivalents for resulting elements amount and the
    validity of net.res_bus.
    """
    net = create_test_net()
    eq_types = ["rei", "ward", "xward"]
    for eq_type in eq_types:
        net1, net2, net3 = run_basic_usecases(eq_type)

        if eq_type == "rei":
            check_elements_amount(net1, {"bus": 5, "load": 3, "sgen": 2, "shunt": 3, "ext_grid": 1,
                                         "line": 3, "impedance": 3}, check_all_pp_elements=True)
            check_res_bus(net, net1)
            assert np.allclose(net1.bus.min_vm_pu.values,
                               np.array([0.9, 0.91, np.nan, np.nan, 0.93]), equal_nan=True)
            check_elements_amount(net2, {"bus": 3, "load": 3, "sgen": 0, "shunt": 3, "ext_grid": 0,
                                         "line": 0, "impedance": 2}, check_all_pp_elements=True)
            check_res_bus(net, net2)
            assert np.allclose(net2.bus.min_vm_pu.values,
                               net.bus.min_vm_pu.loc[[2, 4, 3]].values, equal_nan=True)
            check_elements_amount(net3, {"bus": 5, "load": 3, "sgen": 2, "shunt": 3, "ext_grid": 1,
                                         "line": 3, "impedance": 3}, check_all_pp_elements=True)
            check_res_bus(net, net3)
            assert np.allclose(net1.bus.min_vm_pu.values,
                               np.array([0.9, 0.91, np.nan, np.nan, 0.93]), equal_nan=True)

        elif "ward" in eq_type:
            check_elements_amount(net1, {"bus": 4, "load": 2, "sgen": 2, "ext_grid": 1, "line": 3,
                                         eq_type: 2, "impedance": 1}, check_all_pp_elements=True)
            check_res_bus(net, net1)
            check_elements_amount(net2, {"bus": 2, "load": 2, "sgen": 0, "ext_grid": 0, "line": 0,
                                         eq_type: 2, "impedance": 1}, check_all_pp_elements=True)
            check_res_bus(net, net2)
            check_elements_amount(net3, {"bus": 4, "load": 2, "sgen": 2, "ext_grid": 1, "line": 3,
                                         eq_type: 2, "impedance": 1}, check_all_pp_elements=True)
            check_res_bus(net, net3)


def test_case9_with_slack_generator_in_external_net():
    net = pp.networks.case9()
    idx = pp.replace_ext_grid_by_gen(net)
    net.gen.loc[idx, "slack"] = True
    net.bus_geodata = net.bus_geodata.drop(net.bus_geodata.index)
    pp.runpp(net)

    # since the only slack is in the external_buses, we expect get_equivalent() to move the slack
    # bus into the boundary for use case 1 and 2. In use case 3 the slack generator stays in the
    # external net because no information is given that in the internal net a slack is missing

    boundary_buses = {5, 7}
    internal_buses = {1, 2, 6}
    external_buses = {0, 3, 4, 8}
    slack_bus = {0}

    eq_types = ["rei", "ward", "xward"]

    for eq_type in eq_types:

        # ---
        # UC1
        eq_net1 = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses, internal_buses)
        eq_net1b = pp.grid_equivalents.get_equivalent(net, eq_type, list(boundary_buses), list(internal_buses))
        eq_net1.gen = eq_net1.gen.drop(columns=["origin_id"])
        eq_net1b.gen = eq_net1b.gen.drop(columns=["origin_id"])
        assert pandapower.toolbox.nets_equal(eq_net1, eq_net1b)
        assert net.bus.name.loc[list(boundary_buses | internal_buses | slack_bus)].isin(
            eq_net1.bus.name).all()
        assert eq_net1.gen.slack.sum() == 1
        assert eq_net1.gen.slack.loc[eq_net1.gen.bus.isin(slack_bus)].all()
        if eq_type == "rei":
            check_elements_amount(eq_net1, {"bus": 7, "load": 2, "gen": 3, "shunt": 4, "line": 4,
                                            "impedance": 6}, check_all_pp_elements=True)
            pp.runpp(eq_net1)
            check_res_bus(net, eq_net1)
        elif "ward" in eq_type:
            check_elements_amount(eq_net1, {"bus": 6, "load": 1, "gen": 3, eq_type: 3,
                                            "line": 4, "impedance": 3}, check_all_pp_elements=True)
            check_res_bus(net, eq_net1)
        # ---
        # UC2: return_internal=False
        eq_net2 = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses, internal_buses,
                                     return_internal=False)
        assert net.bus.name.loc[list(boundary_buses | slack_bus)].isin(eq_net2.bus.name).all()
        assert eq_net2.gen.slack.all()
        if eq_type == "rei":
            check_elements_amount(eq_net2, {"bus": 4, "load": 1, "gen": 1, "shunt": 4,
                                            "impedance": 6}, check_all_pp_elements=True)
            check_res_bus(net, eq_net1)
        elif "ward" in eq_type:
            check_elements_amount(eq_net2, {"bus": 3, "gen": 1, eq_type: 3,
                                            "impedance": 3}, check_all_pp_elements=True)
            check_res_bus(net, eq_net1)
        # ---
        # UC3: input is only boundary and external net
        ib_net = pp.select_subnet(net, internal_buses | boundary_buses, include_results=True)
        be_net = pp.select_subnet(net, boundary_buses | external_buses, include_results=True)
        eq_net3 = pp.grid_equivalents.get_equivalent(be_net, eq_type, boundary_buses, internal_buses=[])
        assert not net.bus.name.loc[list(external_buses)].isin(eq_net3.bus.name).all()
        if eq_type == "rei":
            assert eq_net3.gen.slack.all()
            """vorher war assert not eq_net3.gen.slack.all()
               ich habe hier kurz angepasst. Nach equivalent das gen-slack ist beibehalten,
               mit gen.slack=True
            """
        else:
            assert len(eq_net3.gen) == 0
        if eq_type == "rei":
            check_elements_amount(eq_net3, {"bus": 4, "load": 1, "gen": 1, "shunt": 4,
                                            "impedance": 6}, check_all_pp_elements=True)
            # merge eq_net with internal net to get a power flow runable net to check the results
            eq_net3.gen.slack = True
            eq_net4 = pp.grid_equivalents.merge_internal_net_and_equivalent_external_net(
                eq_net3, ib_net)
            pp.runpp(eq_net4)
            check_res_bus(net, eq_net4)
        elif "ward" in eq_type:
            check_elements_amount(eq_net3, {"bus": 2, eq_type: 2, "impedance": 1},
                                  check_all_pp_elements=True)
            check_res_bus(net, eq_net3)


def test_adopt_columns_to_separated_eq_elms():

    # --- gen_separate
    net = pp.networks.case9()
    pp.replace_ext_grid_by_gen(net, slack=True)
    net.gen.index = [1, 2, 0]
    net.poly_cost["element"] = net.gen.index.values
    net.gen.sort_index(inplace=True)
    net.gen["origin_id"] = ["gen_"+str(i) for i in range(net.gen.shape[0])]

    eq_net = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses={4, 8}, internal_buses={0, 3},
                                                gen_separate=True)
    columns_to_check = ["p_mw", "vm_pu", "sn_mva", "scaling", "controllable", "origin_id",
                        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    assert pandapower.toolbox.dataframes_equal(net.gen[columns_to_check], eq_net.gen[columns_to_check])
    assert (net.gen.origin_id.loc[net.poly_cost.element].values ==
            eq_net.gen.origin_id.loc[eq_net.poly_cost.element].values).all()

    # --- sgen_separate0
    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    net.sgen["origin_id"] = ["sgen_%i" % i for i in range(net.sgen.shape[0])]

    eq_net = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses={4, 8}, internal_buses={0, 3},
                                sgen_separate=True)
    columns_to_check = ["p_mw", "q_mvar", "scaling", "sn_mva", "controllable", "origin_id",
                        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    assert pandapower.toolbox.dataframes_equal(net.sgen[columns_to_check], eq_net.sgen[columns_to_check])


def test_equivalent_groups():
    net = pp.networks.example_multivoltage()
    # net.sn_mva = 100
    for elm in pandapower.toolbox.pp_elements():
        if net[elm].shape[0] and not net[elm].name.duplicated().any():
            net[elm]["origin_id"] = net[elm].name

    net.trafo["shift_degree"] = 0
    net.trafo3w["shift_mv_degree"] = 0
    net.trafo3w["shift_lv_degree"] = 0

    ext1 = {42, 43, 44}
    bb1 = {37, 41}
    int1 = set(net.bus.index) - ext1 - bb1
    net_eq1 = pp.grid_equivalents.get_equivalent(net, "rei", bb1, int1,
                                                 reference_column="origin_id")
    assert len(set(net_eq1.group.index)) == 1
    gr1_idx = net_eq1.group.index[0]
    for elm, no in [("bus", 3), ("load", 1), ("sgen", 2)]:
        assert len(pp.group_row(net_eq1, gr1_idx, elm).at["element"]) == no
    assert len(pp.group_row(net_eq1, gr1_idx, "impedance").at["element"]) == net_eq1.impedance.shape[0] - 1
    assert len(pp.group_row(net_eq1, gr1_idx, "shunt").at["element"]) == net_eq1.shunt.shape[0] - 1
    pp.set_group_reference_column(net_eq1, gr1_idx, "origin_id")

    bb2 = {37}
    int2 = set(net_eq1.bus.index[net_eq1.bus.vn_kv > 11]) | set(pp.group_element_index(
        net_eq1, gr1_idx,"bus"))

    # test 2nd rei
    for sgen_separate in [True, False]:
        print("sgen_separate is " + str(sgen_separate))
        # test fails with lightsim2grid, for unknown reason
        net_eq2 = pp.grid_equivalents.get_equivalent(
            net_eq1, "rei", bb2, int2, sgen_separate=sgen_separate, reference_column="origin_id", lightsim2grid=False)
        gr2_idx = net_eq2.group.index[-1]
        assert len(set(net_eq2.group.index)) == 2
        assert len(set(pp.count_group_elements(net_eq2, gr2_idx).index) ^ {
            "bus", "load", "sgen", "impedance", "shunt"}) == 0
        no_sg = 6 if sgen_separate else 1  # number of expected sgens
        no_l = 1  # number of expected loads
        no_b = no_sg + no_l # number of expected buses
        # print(pp.count_group_elements(net_eq2, gr2_idx))
        for elm, no in [("bus", no_b), ("load", no_l), ("sgen", no_sg)]:
            assert len(pp.group_row(net_eq2, gr2_idx, elm).at["element"]) == no
        assert len(pp.group_row(net_eq2, gr2_idx, "impedance").at["element"]) > 0.5 * (no_b-1)**2  # the
        # number of impedances is lower than no_b**2 since imp < 1e-8 were dropped

    # test 2nd xward
    net_eq2 = pp.grid_equivalents.get_equivalent(
        net_eq1, "xward", bb2, int2, reference_column="origin_id")
    gr2_idx = net_eq2.group.index[-1]
    assert len(set(net_eq2.group.index)) == 2
    assert len(set(pp.count_group_elements(net_eq2, gr2_idx).index) ^ {"xward"}) == 0
    for elm, no in [("xward", 1)]:
            assert len(pp.group_row(net_eq2, gr2_idx, elm).at["element"]) == no


def test_shifter_degree():
    net = pp.networks.example_multivoltage()
    net.trafo.at[0, "shift_degree"] = 30
    net.trafo.at[1, "shift_degree"] = -60
    net.trafo3w.at[0, "shift_mv_degree"] = 90
    net.trafo3w.at[0, "shift_lv_degree"] = 150
    pp.runpp(net, calculate_voltage_angles=True)

    boundary_buses = list([net.trafo.hv_bus.values[1]]) + list(net.trafo.lv_bus.values) + \
        list(net.trafo3w.hv_bus.values) + list(net.trafo3w.lv_bus.values)
    i = net.ext_grid.bus.values[0]

    for eq_type in  ["rei"]:
        for b in boundary_buses:
            net_rei = pp.grid_equivalents.get_equivalent(net, eq_type, [b], [i],
                                      calculate_voltage_angles=True,
                                      sgen_separate=False)
            all_i_buses = net_rei.bus_lookups["origin_all_internal_buses"]
            vm_error = max(abs(net_rei.res_bus.vm_pu[all_i_buses].values -
                                net.res_bus.vm_pu[all_i_buses].values))
            va_error = max(abs(net_rei.res_bus.va_degree[all_i_buses].values -
                                net.res_bus.va_degree[all_i_buses].values))
            assert vm_error < 1e-3
            assert va_error < 0.5


def test_retain_original_internal_indices():
    net = pp.networks.case30()
    pp.replace_gen_by_sgen(net)
    sgen_idxs = sample(list(range(100)), len(net.sgen))
    line_idxs = sample(list(range(100)), len(net.line))
    bus_idxs = sample(list(range(100)), len(net.bus))
    bus_lookup = dict(zip(net.bus.index.tolist(), bus_idxs))
    net.sgen.index = sgen_idxs
    net.line.index = line_idxs
    pp.reindex_buses(net, bus_lookup)
    first3buses = net.bus.index.tolist()[0:3]
    assert not np.array_equal(first3buses, list(range(3)))
    pp.runpp(net)
    eq_type = "rei"
    boundary_buses = [bus_lookup[b] for b in [3, 9, 22]]
    internal_buses = [bus_lookup[0]]

    net_eq = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses, internal_buses,
                                                calculate_voltage_angles=True,
                                                retain_original_internal_indices=True)

    assert net_eq.sgen.index.tolist()[:3] == sgen_idxs[:3]
    assert set(net_eq.line.index.tolist()) - set(line_idxs) == set()
    assert set(net_eq.bus.index.tolist()[:-2]) - set(bus_idxs) == set()
    assert np.array_equal(first3buses, net_eq.bus.index.tolist()[0:3])


def test_switch_sgens():
    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    pp.create_bus(net, 345)
    pp.create_switch(net, 9, 1, "b")
    pp.create_sgen(net, 9, 10, 10)
    pp.runpp(net)
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0])
    assert max(net.res_bus.vm_pu[[0, 3, 4, 8]].values - net_eq.res_bus.vm_pu[[0, 3, 4, 8]].values) < 1e-6
    assert max(net.res_bus.va_degree[[0, 3, 4, 8]].values - net_eq.res_bus.va_degree[[0, 3, 4, 8]].values) < 1e-6


def test_characteristic():
    net = pp.networks.example_multivoltage()
    pp.control.create_trafo_characteristics(net, "trafo", [1], 'vk_percent',
                                            [[-2,-1,0,1,2]], [[2,3,4,5,6]])
    pp.runpp(net)
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [41], [0])
    assert len(net_eq.characteristic) == 1


def test_controller():
    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    pp.create_load(net, 5, 10, 10)
    pp.create_sgen(net, 3, 1, 1)

    net.sgen.loc[:, "type"] = "wind"
    net.load.loc[:, "type"] = "residential"
    net.sgen.name = ["sgen0", "sgen1", "sgen3"]
    net.load.name = ["load0", "load1", "load2", "load3"]

    # load time series
    json_path = os.path.join(pp_dir, "test", "opf", "cigre_timeseries_15min.json")
    time_series = pd.read_json(json_path)
    time_series.sort_index(inplace=True)
    sgen_p = net["sgen"].loc[:, "p_mw"].values
    load_p = net["load"].loc[:, "p_mw"].values
    sgen_ts = pd.DataFrame(index=time_series.index.tolist(), columns=net.sgen.index.tolist())
    load_ts = pd.DataFrame(index=time_series.index.tolist(), columns=net.load.index.tolist())
    for t in range(96):
        load_ts.loc[t] = load_p * time_series.at[t, "residential"]
        sgen_ts.loc[t] = sgen_p * time_series.at[t, "wind"]

    # create control
    ConstControl(net, element="load", variable="p_mw",
                 element_index=net.load.index.tolist(), profile_name=net.load.index.tolist(),
                 data_source=DFData(load_ts))
    ConstControl(net, element="sgen", variable="p_mw",
                 element_index=net.sgen.index.tolist(), profile_name=net.sgen.index.tolist(),
                 data_source=DFData(sgen_ts))

    pp.runpp(net)

    # getting equivalent
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0])

    assert net_eq.controller.object[0].__dict__["element_index"] == [0, 2]
    assert net_eq.controller.object[0].__dict__["matching_params"]["element_index"] == [0, 2]
    for i in net.controller.index:
        assert set(net_eq.controller.object[i].__dict__["element_index"]) - \
            set(net.controller.object[i].__dict__["element_index"]) == set([])
        assert set(net_eq.controller.object[i].__dict__["profile_name"]) - \
            set(net.controller.object[i].__dict__["profile_name"]) == set([])

    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0],
                                                retain_original_internal_indices=True)
    assert net_eq.controller.object[0].__dict__["element_index"] == [0, 2]
    assert net_eq.controller.object[0].__dict__["matching_params"]["element_index"] == [0, 2]

    # test individual controller:
    net.controller = net.controller.drop(net.controller.index)
    for li in net.load.index:
        ConstControl(net, element='load', variable='p_mw', element_index=[li],
                     data_source=DFData(load_ts), profile_name=[li])
    assert len(net.controller) == 4
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0],
                                                retain_original_internal_indices=True)
    assert net_eq.controller.index.tolist() == [0, 2]


def test_motor():
    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    pp.create_motor(net, 5, 12, 0.9, scaling=0.8, loading_percent=89, efficiency_percent=90)
    pp.create_motor(net, 7, 18, 0.9, scaling=0.9, loading_percent=88, efficiency_percent=95, in_service=False)
    pp.create_motor(net, 6, 10, 0.6, scaling=0.4, loading_percent=98, efficiency_percent=88)
    pp.create_motor(net, 3, 3, 0.6, scaling=0.4, loading_percent=89, efficiency_percent=99)
    pp.create_motor(net, 4, 6, 0.96, scaling=0.4, loading_percent=78, efficiency_percent=90)
    pp.runpp(net)
    values1 = net.res_bus.vm_pu.values.copy()

    for eq in ["rei", "ward", "xward"]:
        net_eq = pp.grid_equivalents.get_equivalent(net, eq, [4, 8], [0],
                                                    retain_original_internal_indices=True,
                                                    show_computing_time=True)

        assert max(net_eq.res_bus.vm_pu[[0,3,4,8]].values - net.res_bus.vm_pu[[0,3,4,8]].values) < 1e-8
        assert net_eq.motor.bus.values.tolist() == [3, 4]

    replace_motor_by_load(net, net.bus.index.tolist())
    assert len(net.motor) == 0
    assert len(net.res_motor) == 0
    assert len(net.load) == 8
    assert len(net.res_load) == 8
    assert net.res_load.loc[4].values.tolist() == [0, 0]
    pp.runpp(net)
    values2 = net.res_bus.vm_pu.values.copy()
    assert max(values1 - values2) < 1e-10


def test_sgen_bswitch():
    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    pp.create_sgen(net, 1, 10)
    pp.create_sgen(net, 1, 5, in_service=False)
    pp.runpp(net)
    net.sgen.name = ["aa", "bb", "cc", "dd"]
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0],
                                                    retain_original_internal_indices=True)
    assert net_eq.sgen.name[0] == 'aa//cc//dd-sgen_separate_rei_1'
    assert net_eq.sgen.p_mw[0] == 173

    net = pp.networks.case9()
    pp.replace_gen_by_sgen(net)
    pp.create_bus(net, 345)
    pp.create_bus(net, 345)
    pp.create_sgen(net, 9, 10)
    pp.create_sgen(net, 10, 5, in_service=False)
    pp.create_switch(net, 1, 9, "b")
    pp.create_switch(net, 1, 10, "b")
    net.sgen.name = ["aa", "bb", "cc", "dd"]
    pp.runpp(net)
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0],
                                                retain_original_internal_indices=True)

    assert net_eq.sgen.name[0] == 'aa//cc-sgen_separate_rei_1'
    assert net_eq.sgen.p_mw[0] == 173

    # add some columns for test
    net.bus["voltLvl"]=1
    net.sgen["col_mixed"] = ["1", 2, None, True]
    net.sgen["col_same_str"] = ["str_test", "str_test", "str_test", "str_test"]
    net.sgen["col_different_str"] = ["str_1", "str_2", "str_3", "str_4"]
    net.sgen["bool"] = [False, True, False, False]
    net.sgen["voltLvl"] = [1, 1, 1, 1]
    net_eq = pp.grid_equivalents.get_equivalent(net, "rei", [4, 8], [0])
    assert net_eq.sgen["col_mixed"][0] == "mixed data type"
    assert net_eq.sgen["col_same_str"][0] == "str_test"
    assert net_eq.sgen["col_different_str"][0] == "str_3//str_1"
    assert net_eq.sgen["col_different_str"][1] == "str_2"
    assert net_eq.sgen["bool"][0] == False
    assert net_eq.sgen["bool"][1] == True
    assert net_eq.sgen["voltLvl"].values.tolist() == [1, 1]


def test_ward_admittance():
    net = pp.networks.case9()
    pp.runpp(net)
    res_bus = net.res_bus.copy()
    create_passive_external_net_for_ward_admittance(net, [1, 2, 5, 6, 7],
                                                    [4,8], True,
                                                    _runpp_except_voltage_angles)
    assert len(net.shunt)==3
    assert np.allclose(net.res_bus.vm_pu.values, res_bus.vm_pu.values)


if __name__ == "__main__":
    pytest.main(['-x', __file__])

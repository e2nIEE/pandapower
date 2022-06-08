import pytest
import numpy as np
import pandapower as pp

import pandapower.networks
import pandapower.grid_equivalents

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
    eq_net1 = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses=[1, 5], internal_buses=[0, 6])
    pp.runpp(eq_net1, calculate_voltage_angles=True)

    # UC2: don't get the internal buses [0, 1, 5, 6] but the boundary buses [2, 4] and the
    # equivalent
    eq_net2 = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses=[2, 4],
                                 internal_buses=[0, 1, 5, 6], return_internal=False)

    # UC3: the input is only the subnet including the external buses [2, 3, 4] and the
    # boundary buses [1, 5] -> expected return are the boundary buses and the equivalent
    subnet = pp.select_subnet(net, list(range(1, 6)), include_results=True)
    subnet_rest = pp.select_subnet(net, [0, 1, 5, 6], include_results=True)
    eq_net3a = pp.grid_equivalents.get_equivalent(subnet, eq_type, boundary_buses=[1, 5], internal_buses=None)

    # UC3b tests whether this also works for 'internal_buses' as empty list
    eq_net3b = pp.grid_equivalents.get_equivalent(subnet, eq_type, boundary_buses=[1, 5], internal_buses=[])
    eq_net3a.sgen.drop(columns=["origin_id"], inplace=True)
    eq_net3b.sgen.drop(columns=["origin_id"], inplace=True)
    if group_imported:
        assert set(eq_net3a["group"].index) == set(eq_net3b["group"].index)
        assert eq_net3a.group.object.at[0].compare_elms_dict(eq_net3a.group.object.at[0].elms_dict)
    assert pp.nets_equal(eq_net3a, eq_net3b, exclude_elms=["group"])

    # UC3: merge eq_net3 with subnet_rest
    eq_net3 = pp.grid_equivalents.merge_internal_net_and_equivalent_external_net(eq_net3a, subnet_rest, eq_type)
    pp.runpp(eq_net3, calculate_voltage_angles=True)
    assert pp.nets_equal(net, create_test_net())
    return eq_net1, eq_net2, eq_net3


def check_elements_amount(net, elms_dict, check_all_pp_elements=True):
    if check_all_pp_elements:
        elms_dict.update({elm: 0 for elm in pp.pp_elements() if elm not in elms_dict.keys()})
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
    for elm in pp.pp_elements():
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
                net.poly_cost.drop(poly.Index, inplace=True)
                pp.create_pwl_cost(net, poly.element, poly.et, [[0, 20, 1]], index=poly.Index)

        # eq generation
        boundary_buses = [0, 2]
        internal_buses = [1]
        eq_net1 = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses, internal_buses)
        eq_net2 = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses, internal_buses,
                                     return_internal=False)

        # check elements
        check_elements_amount(eq_net1, {"bus": 6, "load": 3, "sgen": 3, "shunt": 5, "ext_grid": 1,
                                        "line": 2, "impedance": 10, cost_type: 4},
                              check_all_pp_elements=True)
        check_elements_amount(eq_net2, {"bus": 5, "load": 3, "sgen": 2, "shunt": 5, "ext_grid": 1,
                                        "impedance": 10, cost_type: 3},
                              check_all_pp_elements=True)
        assert all(eq_net1.sgen.index.values == np.array([0, 1, 2]))  # simple create_sgen()
        # without index=... expected
        assert all(eq_net2.sgen.index.values == np.array([0, 1]))

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
                                         "line": 0, "impedance": 3}, check_all_pp_elements=True)
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
    net.sn_mva = 1.
    idx = pp.replace_ext_grid_by_gen(net)
    net.gen.slack.loc[idx] = True
    net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
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
        eq_net1.gen.drop(columns=["origin_id"], inplace=True)
        eq_net1b.gen.drop(columns=["origin_id"], inplace=True)
        assert pp.nets_equal(eq_net1, eq_net1b)
        assert net.bus.name.loc[boundary_buses | internal_buses | slack_bus].isin(
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
        assert net.bus.name.loc[boundary_buses | slack_bus].isin(eq_net2.bus.name).all()
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
        assert not net.bus.name.loc[external_buses].isin(eq_net3.bus.name).all()
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
            eq_net4 = pp.grid_equivalents.merge_internal_net_and_equivalent_external_net(eq_net3, ib_net, eq_type)
            pp.runpp(eq_net4)
            check_res_bus(net, eq_net4)
        elif "ward" in eq_type:
            check_elements_amount(eq_net3, {"bus": 2, eq_type: 2, "impedance": 1},
                                  check_all_pp_elements=True)
            check_res_bus(net, eq_net3)


def test_adopt_columns_to_separated_eq_elms():

    # --- gen_separate
    net = pp.networks.case9()
    net.sn_mva = 1.
    pp.replace_ext_grid_by_gen(net, slack=True)
    net.gen.index = [1, 2, 0]
    net.poly_cost["element"] = net.gen.index.values
    net.gen.sort_index(inplace=True)
    net.gen["origin_id"] = ["gen_"+str(i) for i in range(net.gen.shape[0])]

    eq_net = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses={4, 8}, internal_buses={0, 3},
                                gen_separate=True)
    columns_to_check = ["p_mw", "vm_pu", "sn_mva", "scaling", "controllable", "origin_id",
                        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    assert pp.dataframes_equal(net.gen[columns_to_check], eq_net.gen[columns_to_check])
    assert (net.gen.origin_id.loc[net.poly_cost.element].values ==
            eq_net.gen.origin_id.loc[eq_net.poly_cost.element].values).all()

    # --- sgen_separate0
    net = pp.networks.case9()
    net.sn_mva = 1.
    pp.replace_gen_by_sgen(net)
    net.sgen["origin_id"] = ["sgen_%i" % i for i in range(net.sgen.shape[0])]

    eq_net = pp.grid_equivalents.get_equivalent(net, "rei", boundary_buses={4, 8}, internal_buses={0, 3},
                                sgen_separate=True)
    columns_to_check = ["p_mw", "q_mvar", "scaling", "sn_mva", "controllable", "origin_id",
                        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    assert pp.dataframes_equal(net.sgen[columns_to_check], eq_net.sgen[columns_to_check])


@pytest.mark.skipif(not group_imported, reason="Group is not installed")
def test_equivalent_groups():
    net = pp.networks.example_multivoltage()
    for elm in pp.pp_elements():
        if net[elm].shape[0] and not net[elm].name.duplicated().any():
            net[elm]["origin_id"] = net[elm].name

    net.trafo["shift_degree"] = 0
    net.trafo3w["shift_mv_degree"] = 0
    net.trafo3w["shift_lv_degree"] = 0

    ext1 = {42, 43, 44}
    bb1 = {37, 41}
    int1 = set(net.bus.index) - ext1 - bb1
    net_eq1 = pp.grid_equivalents.get_equivalent(net, "rei", bb1, int1, elm_col="origin_id")
    assert net_eq1.group.shape[0] == 1
    for elm, no in [("bus", 3), ("load", 1), ("sgen", 2)]:
        assert len(net_eq1.group.object.at[0].elms_dict[elm]) == no
    assert len(net_eq1.group.object.at[0].elms_dict["impedance"]) == net_eq1.impedance.shape[0] - 1
    assert len(net_eq1.group.object.at[0].elms_dict["shunt"]) == net_eq1.shunt.shape[0] - 1
    net_eq1.group.object.at[0].set_elm_col(net_eq1, "origin_id")

    bb2 = {37}
    int2 = set(net_eq1.bus.index[net_eq1.bus.vn_kv > 11]) | set(net_eq1.group.object.at[
        0].get_idx(net_eq1, "bus"))

    # test 2nd rei
    for sgen_separate in [True, False]:
        print("sgen_separate is " + str(sgen_separate))
        net_eq2 = pp.grid_equivalents.get_equivalent(net_eq1, "rei", bb2, int2, sgen_separate=sgen_separate,
                                     elm_col="origin_id")
        assert net_eq2.group.shape[0] == 2
        assert len(set(net_eq2.group.object.at[1].elms_dict.keys()) ^ {
            "bus", "load", "sgen", "impedance", "shunt"}) == 0
        no_sg = 6 if sgen_separate else 1  # number of expected sgens
        no_l = 1  # number of expected loads
        no_b = no_sg + no_l # number of expected buses
        for elm, no in [("bus", no_b), ("load", no_l), ("sgen", no_sg), ("shunt", no_b+1)]:
            assert len(net_eq2.group.object.at[1].elms_dict[elm]) == no
        assert len(net_eq2.group.object.at[1].elms_dict["impedance"]) > 0.5 * no_b**2  # the number
        # of impedances is lower than no_b**2 since imp < 1e-8 were dropped

    # test 2nd xward
    net_eq2 = pp.grid_equivalents.get_equivalent(net_eq1, "xward", bb2, int2, elm_col="origin_id")
    assert net_eq2.group.shape[0] == 2
    assert len(set(net_eq2.group.object.at[1].elms_dict.keys()) ^ {"xward"}) == 0
    for elm, no in [("xward", 1)]:
        assert len(net_eq2.group.object.at[1].elms_dict[elm]) == no


def test_shifter_degree():
    net = pp.networks.example_multivoltage()
    net.trafo.shift_degree[0] = 30
    net.trafo.shift_degree[1] = -60
    net.trafo3w.shift_mv_degree[0] = 90
    net.trafo3w.shift_lv_degree[0] = 150
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


if __name__ == "__main__":
    pytest.main(['-x', __file__])
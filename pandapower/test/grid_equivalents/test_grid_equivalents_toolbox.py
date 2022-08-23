import pytest
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandapower.grid_equivalents as geq


def check_elements_amount(net, elms_dict, check_all_pp_elements=True):
    if check_all_pp_elements:
        elms_dict.update({elm: 0 for elm in pp.pp_elements() if elm not in elms_dict.keys()})
    for key, val in elms_dict.items():
        if not net[key].shape[0] == val:
            raise ValueError("The net has %i %ss but %i are expected." % (
                    net[key].shape[0], key, int(val)))


def test_set_bus_zone_by_boundary_branches_and_split_grid_by_bus_zone_with_boundary_branches():
    # --- test set_bus_zone_by_boundary_branches()
    net = pn.case9()
    bb_in = {"line": {2, 7}}
    geq.set_bus_zone_by_boundary_branches(net, bb_in)
    assert all(net.bus.zone.values == np.array([0, 1, 1, 0, 0, 1, 1, 1, 0]))
    pp.runpp(net)

    # --- a simple test of split_grid_by_bus_zone_with_boundary_branches()
    boundary_buses, boundary_branches, nets_i, nets_ib, _, _, _ = \
        geq.split_grid_by_bus_zone_with_boundary_branches(net)

    assert set(nets_i.keys()) == {0, 1}
    assert set(nets_i[0].bus.index) == {0, 3, 4, 8}
    assert set(nets_i[1].bus.index) == {1, 2, 5, 6, 7}

    assert set(nets_ib[0].bus.index) == {0, 3, 4, 5, 7, 8}
    assert set(nets_ib[1].bus.index) == {1, 2, 4, 5, 6, 7, 8}

    expected = {"all": {4, 5, 7, 8},
                0: {"all": {4, 5, 7, 8},
                    "internal": {4, 8},
                    "external": {5, 7},
                    1: {5, 7}},
                1: {"all": {4, 5, 7, 8},
                    "internal": {5, 7},
                    "external": {4, 8},
                    0: {4, 8}}}
    assert boundary_buses == expected

    assert boundary_branches["all"] == bb_in

    # --- test against set_bus_zone_by_boundary_branches()
    bb_in = {"line": {2, 4, 7}}
    geq.set_bus_zone_by_boundary_branches(net, bb_in)
    assert all(net.bus.zone.values == np.array([0, 1, 2, 0, 0, 2, 1, 1, 0]))


def test_append_set_to_dict():
    keys = [2, 3, 5]
    dict1 = {}
    dict2 = {2: {},
             7: "hkj"}
    dict3 = {2: {"hjk": 6,
                 3: dict()}}
    dict4 = {2: {"hjk": 6,
                 3: {5: set([8, 2])}
                 }
             }
    dict5 = {2: {"hjk": 6,
                 3: {5: {2, 8}
                     }
                 }
             }

    val = {1, 2, 3}

    geq.append_set_to_dict(dict1, val, keys)
    geq.append_set_to_dict(dict2, val, keys)
    geq.append_set_to_dict(dict3, val, keys)
    geq.append_set_to_dict(dict4, val, keys)
    geq.append_set_to_dict(dict5, val, keys)

    assert dict1 == {2: {3: {5: {1, 2, 3}}}}
    assert dict2 == {2: {3: {5: {1, 2, 3}}}, 7: "hkj"}
    assert dict3 == {2: {3: {5: {1, 2, 3}}, "hjk": 6}}
    assert dict4 == {2: {3: {5: {1, 2, 3, 8}}, "hjk": 6}}
    assert dict5 == dict4


def test_split_grid_by_bus_zone_with_boundary_branches():
    net = pn.case9()
    pp.runpp(net)
    net.bus["zone"] = ["a", "b", "c", "a", "a", "c", "c", "b", "b"]
    boundary_buses, boundary_branches, nets_i, nets_ib, nets_ib0, nets_ib_eq_load, nets_b = \
        geq.split_grid_by_bus_zone_with_boundary_branches(net)

    assert len(boundary_buses.keys()) == 4
    for key in ["a", "b", "c"]:
        assert not len({"internal", "external"} - set(boundary_buses[key].keys()))
    assert set(boundary_buses["a"]["internal"]) == {3, 4}
    assert set(boundary_buses["a"]["external"]) == {5, 8}

    assert boundary_branches == {"all": {"line": {2, 5, 8}},
                                 "a": {"line": {2, 8}},
                                 "b": {"line": {5, 8}},
                                 "c": {"line": {2, 5}}}

    assert set(nets_i.keys()) == set(nets_ib.keys()) == set(nets_ib0.keys()) == \
        set(nets_ib_eq_load.keys()) == set(nets_b.keys()) == {"a", "b", "c"}

    # nets_i
    assert set(nets_i["a"].bus.index) == {0, 3, 4}
    check_elements_amount(nets_i["a"], {"bus": 3, "load": 1, "ext_grid": 1, "line": 2})
    assert set(nets_i["b"].bus.index) == {1, 7, 8}
    check_elements_amount(nets_i["b"], {"bus": 3, "load": 1, "gen": 1, "line": 2})
    assert set(nets_i["c"].bus.index) == {2, 5, 6}
    check_elements_amount(nets_i["c"], {"bus": 3, "load": 1, "gen": 1, "line": 2})

    # nets_ib
    assert set(nets_ib["a"].bus.index) == {0, 3, 4, 5, 8}
    check_elements_amount(nets_ib["a"], {"bus": 5, "load": 2, "ext_grid": 1, "line": 4})
    assert set(nets_ib["a"].load.index) == {0, 2}
    assert set(nets_ib["b"].bus.index) == {1, 7, 8, 3, 6}
    check_elements_amount(nets_ib["b"], {"bus": 5, "load": 2, "gen": 1, "line": 4})
    assert set(nets_ib["b"].load.index) == {1, 2}
    assert set(nets_ib["c"].bus.index) == {2, 5, 6, 4, 7}
    check_elements_amount(nets_ib["c"], {"bus": 5, "load": 2, "gen": 1, "line": 4})
    assert set(nets_ib["c"].load.index) == {0, 1}

    # nets_ib0
    assert set(nets_ib0["a"].bus.index) == {0, 3, 4, 5, 8}
    check_elements_amount(nets_ib0["a"], {"bus": 5, "load": 1, "ext_grid": 1, "line": 4})
    assert set(nets_ib0["a"].load.index) == {0}
    assert set(nets_ib0["b"].bus.index) == {1, 7, 8, 3, 6}
    check_elements_amount(nets_ib0["b"], {"bus": 5, "load": 1, "gen": 1, "line": 4})
    assert set(nets_ib0["b"].load.index) == {2}
    assert set(nets_ib0["c"].bus.index) == {2, 5, 6, 4, 7}
    check_elements_amount(nets_ib0["c"], {"bus": 5, "load": 1, "gen": 1, "line": 4})
    assert set(nets_ib0["c"].load.index) == {1}

    # nets_ib_eq_load
    assert set(nets_ib_eq_load["a"].bus.index) == {0, 3, 4, 5, 8}
    check_elements_amount(nets_ib_eq_load["a"], {"bus": 5, "load": 3, "ext_grid": 1, "line": 4})
    assert set(nets_ib_eq_load["a"].load.index) == {0, 3, 4}
    assert set(nets_ib_eq_load["b"].bus.index) == {1, 7, 8, 3, 6}
    check_elements_amount(nets_ib_eq_load["b"], {"bus": 5, "load": 3, "gen": 1, "line": 4})
    assert set(nets_ib_eq_load["b"].load.index) == {2, 5, 6}
    assert set(nets_ib_eq_load["c"].bus.index) == {2, 5, 6, 4, 7}
    check_elements_amount(nets_ib_eq_load["c"], {"bus": 5, "load": 3, "gen": 1, "line": 4})
    assert set(nets_ib_eq_load["c"].load.index) == {1, 7, 8}

    # nets_b
    assert set(nets_b["a"].bus.index) == {5, 8}
    check_elements_amount(nets_b["a"], {"bus": 2, "load": 1, "line": 2})
    assert set(nets_b["a"].load.index) == {2}
    assert set(nets_b["b"].bus.index) == {3, 6}
    check_elements_amount(nets_b["b"], {"bus": 2, "load": 1, "line": 2})
    assert set(nets_b["b"].load.index) == {1}
    assert set(nets_b["c"].bus.index) == {4, 7}
    check_elements_amount(nets_b["c"], {"bus": 2, "load": 1, "line": 2})
    assert set(nets_b["c"].load.index) == {0}


def test_split_grid_by_bus_zone_with_boundary_branches2():
    # --- create test net
    net1 = pn.case9()
    net1.bus["zone"] = "a"
    net2 = pn.case9()
    net2.bus["zone"] = "b"
    net2.ext_grid["p_disp_mw"] = 71.9547
    pp.replace_ext_grid_by_gen(net2)
    net = pp.merge_nets(net1, net2, merge_results=False, validate=False)
    new_bus = pp.create_bus(net, 345, zone="b")
    new_line = pp.create_line_from_parameters(
        net, net.bus.index[net.bus.name == 9][0], net.bus.index[net.bus.name == 9][1], 1,
        0, 65, 0, 0.41)
    new_imp = pp.create_impedance(net, net.bus.index[net.bus.name == 5][0],
                                  net.bus.index[net.bus.name == 5][1], 0, 0.06, 250)
    new_sw = pp.create_switch(net, net.bus.index[net.bus.name == 7][0],
                              net.bus.index[net.bus.name == 7][1], "b")
    new_tr = pp.create_transformer_from_parameters(
        net, net.bus.index[net.bus.name == 8][0], net.bus.index[net.bus.name == 8][1], 250,
        345, 345, 0, 10, 50, 0)
    new_tr3 = pp.create_transformer3w_from_parameters(
        net, net.bus.index[net.bus.name == 3][0], new_bus, net.bus.index[net.bus.name == 3][1],
        345, 345, 345, 250, 250, 250, 10, 10, 10, 0, 0, 0, 50, 0)
    pp.runpp(net)

    # --- run function
    boundary_buses, boundary_branches, _, _, _, _, _ = \
        geq.split_grid_by_bus_zone_with_boundary_branches(net)

    # --- check form of boundary_buses
    assert sorted(boundary_buses.keys()) == ["a", "all", "b"]
    for key in ["a", "b"]:
        assert not len({"internal", "external"} - set(boundary_buses[key].keys()))

    # --- check boundary_buses content
    assert boundary_buses["a"]["internal"] == set(net.bus.index[net.bus.name.isin(
        [9, 5, 7, 8, 3]) & (net.bus.zone == "a")])
    assert boundary_buses["a"]["external"] == set(net.bus.index[net.bus.name.isin(
        [9, 5, 7, 8, 3]) & (net.bus.zone == "b")]) | {new_bus}
    trafo3w_buses = set(net.trafo3w[["hv_bus", "mv_bus", "lv_bus"]].values.flatten())
    assert boundary_buses["b"]["internal"] - trafo3w_buses == \
        boundary_buses["a"]["external"] - trafo3w_buses
    assert boundary_buses["b"]["external"] - trafo3w_buses == \
        boundary_buses["a"]["internal"] - trafo3w_buses

    # --- check boundary_branches content
    bbr = {"line": {new_line}, "impedance": {new_imp}, "trafo": {new_tr}, "trafo3w": {new_tr3},
           "switch": {new_sw}}
    assert boundary_branches["a"] == bbr
    assert boundary_branches["b"] == bbr
    assert boundary_branches["all"] == bbr


if __name__ == "__main__":
    if 0:
        pytest.main(['-x', "test_grid_equivalents_toolbox.py"])
    else:
        test_set_bus_zone_by_boundary_branches_and_split_grid_by_bus_zone_with_boundary_branches()
        test_append_set_to_dict()
        test_split_grid_by_bus_zone_with_boundary_branches()
        test_split_grid_by_bus_zone_with_boundary_branches2()
        pass

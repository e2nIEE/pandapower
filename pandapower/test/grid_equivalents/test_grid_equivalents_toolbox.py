import numpy as np
import pytest

from pandapower.create import create_bus, create_line_from_parameters, create_transformer3w_from_parameters, \
    create_impedance, create_switch, create_transformer_from_parameters
from pandapower.grid_equivalents.toolbox import set_bus_zone_by_boundary_branches, \
    get_boundaries_by_bus_zone_with_boundary_branches, append_set_to_dict
from pandapower.networks.power_system_test_cases import case9
from pandapower.run import runpp
from pandapower.toolbox.grid_modification import replace_ext_grid_by_gen, merge_nets


def boundary_testnet(which):
    if which == "case9_27":
        net = case9()
        expected_bbr = {"line": {2, 7}}
        expected_bb = {
            "all": {4, 5, 7, 8},
            0: {"all": {4, 5, 7, 8},
                "internal": {4, 8},
                "external": {5, 7},
                1: {5, 7}},
            1: {"all": {4, 5, 7, 8},
                "internal": {5, 7},
                "external": {4, 8},
                0: {4, 8}}}
    elif which == "case9_abc":
        net = case9()
        net.bus["zone"] = ["a", "b", "c", "a", "a", "c", "c", "b", "b"]
        expected_bbr = {"all": {"line": {2, 5, 8}},
                        "a": {"line": {2, 8}},
                        "b": {"line": {5, 8}},
                        "c": {"line": {2, 5}}}
        expected_bb = {
            "a": {"internal": {3, 4},
                  "external": {5, 8}}}
    elif which == "case9_ab_merged":
        net1 = case9()
        net1.bus["zone"] = "a"
        net2 = case9()
        net2.bus["zone"] = "b"
        net2.ext_grid["p_disp_mw"] = 71.9547
        replace_ext_grid_by_gen(net2)
        net = merge_nets(net1, net2, merge_results=False, validate=False,
                         net2_reindex_log_level=None)
        new_bus = create_bus(net, 345, zone="b")

        # expected_bbr
        expected_bbr = dict()
        expected_bbr["line"] = {create_line_from_parameters(
            net, net.bus.index[net.bus.name == 9][0], net.bus.index[net.bus.name == 9][1], 1,
            0, 65, 0, 0.41)}
        expected_bbr["impedance"] = {create_impedance(net, net.bus.index[net.bus.name == 5][0],
                                                      net.bus.index[net.bus.name == 5][1], 0, 0.06, 250)}
        expected_bbr["switch"] = {create_switch(net, net.bus.index[net.bus.name == 7][0],
                                                net.bus.index[net.bus.name == 7][1], "b")}
        expected_bbr["trafo"] = {create_transformer_from_parameters(
            net, net.bus.index[net.bus.name == 8][0], net.bus.index[net.bus.name == 8][1], 250,
            345, 345, 0, 10, 50, 0)}
        expected_bbr["trafo3w"] = {create_transformer3w_from_parameters(
            net, net.bus.index[net.bus.name == 3][0], new_bus, net.bus.index[net.bus.name == 3][1],
            345, 345, 345, 250, 250, 250, 10, 10, 10, 0, 0, 0, 50, 0)}

        # expected_bb
        expected_bb = {key: dict() for key in ["a", "b"]}
        expected_bb["a"]["internal"] = set(net.bus.index[net.bus.name.isin(
            [9, 5, 7, 8, 3]) & (net.bus.zone == "a")])
        expected_bb["a"]["external"] = set(net.bus.index[net.bus.name.isin(
            [9, 5, 7, 8, 3]) & (net.bus.zone == "b")]) | {new_bus}
        expected_bb["b"]["internal"] = expected_bb["a"]["external"] - {18}
        expected_bb["b"]["external"] = expected_bb["a"]["internal"] | {18}

    runpp(net)
    return net, expected_bb, expected_bbr


def test_set_bus_zone_by_boundary_branches_and_get_boundaries_by_bus_zone_with_boundary_branches1():
    net, expected_bb, expected_bbr = boundary_testnet("case9_27")
    set_bus_zone_by_boundary_branches(net, expected_bbr)
    assert all(net.bus.zone.values == np.array([0, 1, 1, 0, 0, 1, 1, 1, 0]))

    boundary_buses, boundary_branches = \
        get_boundaries_by_bus_zone_with_boundary_branches(net)

    assert boundary_buses == expected_bb
    assert boundary_branches["all"] == expected_bbr

    # --- test against set_bus_zone_by_boundary_branches()
    bb_in = {"line": {2, 4, 7}}
    set_bus_zone_by_boundary_branches(net, bb_in)
    assert all(net.bus.zone.values == np.array([0, 1, 2, 0, 0, 2, 1, 1, 0]))


def test_set_bus_zone_by_boundary_branches_and_get_boundaries_by_bus_zone_with_boundary_branches2():
    net, expected_bb, expected_bbr = boundary_testnet("case9_abc")
    boundary_buses, boundary_branches = \
        get_boundaries_by_bus_zone_with_boundary_branches(net)

    assert len(boundary_buses.keys()) == 4
    for key in ["a", "b", "c"]:
        assert not len({"internal", "external"} - set(boundary_buses[key].keys()))
    assert set(boundary_buses["a"]["internal"]) == expected_bb["a"]["internal"]
    assert set(boundary_buses["a"]["external"]) == expected_bb["a"]["external"]
    assert boundary_branches == expected_bbr


def test_set_bus_zone_by_boundary_branches_and_get_boundaries_by_bus_zone_with_boundary_branches3():
    net, expected_bb, expected_bbr = boundary_testnet("case9_ab_merged")
    boundary_buses, boundary_branches = \
        get_boundaries_by_bus_zone_with_boundary_branches(net)

    # --- check form of boundary_buses
    assert sorted(boundary_buses.keys()) == ["a", "all", "b"]
    for key in ["a", "b"]:
        assert not len({"internal", "external"} - set(boundary_buses[key].keys()))

    # --- check boundary_buses content
    trafo3w_buses = set(net.trafo3w[["hv_bus", "mv_bus", "lv_bus"]].values.flatten())
    in_ext = ["internal", "external"]
    for in_ext1, in_ext2 in zip(in_ext, in_ext[::-1]):
        assert boundary_buses["a"][in_ext1] == expected_bb["a"][in_ext1]
        assert boundary_buses["b"][in_ext1] - trafo3w_buses == \
               boundary_buses["a"][in_ext2] - trafo3w_buses

    # --- check boundary_branches content
    assert boundary_branches["a"] == expected_bbr
    assert boundary_branches["b"] == expected_bbr
    assert boundary_branches["all"] == expected_bbr


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

    append_set_to_dict(dict1, val, keys)
    append_set_to_dict(dict2, val, keys)
    append_set_to_dict(dict3, val, keys)
    append_set_to_dict(dict4, val, keys)
    append_set_to_dict(dict5, val, keys)

    assert dict1 == {2: {3: {5: {1, 2, 3}}}}
    assert dict2 == {2: {3: {5: {1, 2, 3}}}, 7: "hkj"}
    assert dict3 == {2: {3: {5: {1, 2, 3}}, "hjk": 6}}
    assert dict4 == {2: {3: {5: {1, 2, 3, 8}}, "hjk": 6}}
    assert dict5 == dict4


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.toolbox
import pandapower.networks as nw


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


def test_res_power_columns():
    assert pandapower.toolbox.res_power_columns("gen") == ["p_mw", "q_mvar"]
    assert pandapower.toolbox.res_power_columns("line") == pandapower.toolbox.res_power_columns("line", side="from") == \
           pandapower.toolbox.res_power_columns("line", side=0) == ["p_from_mw", "q_from_mvar"]
    assert pandapower.toolbox.res_power_columns("line", side="all") == [
        "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
    assert pandapower.toolbox.res_power_columns("trafo3w", side="all") == [
        "p_hv_mw", "q_hv_mvar", "p_mv_mw", "q_mv_mvar", "p_lv_mw", "q_lv_mvar"]


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


def test_signing_system_value():
    assert pp.signing_system_value("sgen") == -1
    assert pp.signing_system_value("load") == 1
    for bus_elm in pandapower.toolbox.pp_elements(bus=False, branch_elements=False, other_elements=False):
        assert pp.signing_system_value(bus_elm) in [1, -1]
    try:
        pp.signing_system_value("sdfjio")
        assert False
    except ValueError:
        pass


def test_pq_from_cosphi():
    p, q = pp.pq_from_cosphi(1 / 0.95, 0.95, "underexcited", "load")
    assert np.isclose(p, 1)
    assert np.isclose(q, 0.3286841051788632)

    s = np.array([1, 1, 1])
    cosphi = np.array([1, 0.5, 0])
    pmode = np.array(["load", "load", "load"])
    qmode = np.array(["underexcited", "underexcited", "underexcited"])
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    excpected_values = (np.array([1, 0.5, 0]), np.array([0, 0.8660254037844386, 1]))
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    pmode = "gen"
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, -excpected_values[1])

    qmode = "overexcited"
    p, q = pp.pq_from_cosphi(s, cosphi, qmode, pmode)
    assert np.allclose(p, excpected_values[0])
    assert np.allclose(q, excpected_values[1])

    with pytest.raises(ValueError):
        pp.pq_from_cosphi(1, 0.95, "ohm", "gen")

    p, q = pp.pq_from_cosphi(0, 0.8, "overexcited", "gen")
    assert np.isclose(p, 0)
    assert np.isclose(q, 0)


def test_cosphi_from_pq():
    cosphi, s, qmode, pmode = pp.cosphi_from_pq(1, 0.4)
    assert np.isclose(cosphi, 0.9284766908852593)
    assert np.isclose(s, 1.077032961426901)
    assert qmode == 'underexcited'
    assert pmode == 'load'

    p = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1])
    q = np.array([1, -1, 0, 0.5, -0.5, 1, -1, 0, 1, -1, 0])
    cosphi, s, qmode, pmode = pp.cosphi_from_pq(p, q)
    assert np.allclose(cosphi[[0, 1, 8, 9]], 2 ** 0.5 / 2)
    assert np.allclose(cosphi[[3, 4]], 0.89442719)
    assert np.allclose(cosphi[[2, 10]], 1)
    assert pd.Series(cosphi[[5, 6, 7]]).isnull().all()
    assert np.allclose(s, (p ** 2 + q ** 2) ** 0.5)
    assert all(pmode == np.array(["load"] * 5 + ["undef"] * 3 + ["gen"] * 3))
    ind_cap_ind = ["underexcited", "overexcited", "underexcited"]
    assert all(qmode == np.array(ind_cap_ind + ["underexcited", "overexcited"] + ind_cap_ind * 2))


def test_nets_equal():
    tbgi.logger.setLevel(40)
    original = nw.create_cigre_network_lv()
    net = copy.deepcopy(original)

    # should be equal
    assert pandapower.toolbox.nets_equal(original, net)
    assert pandapower.toolbox.nets_equal(net, original)

    # detecting additional element
    pp.create_bus(net, vn_kv=.4)
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting removed element
    net["bus"].drop(net.bus.index[0], inplace=True)
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting alternated value
    net["load"]["p_mw"][net["load"].index[0]] += 0.1
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # detecting added column
    net["load"]["new_col"] = 0.1
    assert not pandapower.toolbox.nets_equal(original, net)
    assert not pandapower.toolbox.nets_equal(net, original)
    net = copy.deepcopy(original)

    # not detecting alternated value if difference is beyond tolerance
    net["load"]["p_mw"][net["load"].index[0]] += 0.0001
    assert pandapower.toolbox.nets_equal(original, net, atol=0.1)
    assert pandapower.toolbox.nets_equal(net, original, atol=0.1)

    # check controllers
    original.trafo.tap_side.fillna("hv", inplace=True)
    net1 = original.deepcopy()
    net2 = original.deepcopy()
    pp.control.ContinuousTapControl(net1, 0, 1.0)
    pp.control.ContinuousTapControl(net2, 0, 1.0)
    c1 = net1.controller.at[0, "object"]
    c2 = net2.controller.at[0, "object"]
    assert c1 == c2
    assert c1 is not c2
    assert pandapower.toolbox.nets_equal(net1, net2)
    c1.vm_set_pu = 1.01
    assert c1 != c2
    assert pandapower.toolbox.nets_equal(net1, net2, exclude_elms=["controller"])
    assert not pandapower.toolbox.nets_equal(net1, net2)


if __name__ == '__main__':
    pytest.main([__file__, "-x"])

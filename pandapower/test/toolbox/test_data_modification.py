# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower.create import create_measurement, create_empty_network, create_bus, create_load, create_sgen, \
    create_shunt, create_ward
from pandapower.networks.cigre_networks import create_cigre_network_mv
from pandapower.networks.create_examples import example_simple, example_multivoltage
from pandapower.run import runpp
from pandapower.test.helper_functions import assert_net_equal
from pandapower.toolbox.comparison import compare_arrays
from pandapower.toolbox.data_modification import reindex_elements, reindex_buses, add_column_from_node_to_elements, \
    add_column_from_element_to_elements, create_continuous_bus_index, create_continuous_elements_index, \
    set_scaling_by_type
from pandapower.toolbox.element_selection import pp_elements


def test_add_column_from_node_to_elements():
    net = create_cigre_network_mv("pv_wind")
    net.bus["subnet"] = ["subnet_%i" % i for i in range(net.bus.shape[0])]
    net.sgen["subnet"] = "already_given"
    net.switch["subnet"] = None
    net_orig = copy.deepcopy(net)

    branch_bus = ["from_bus", "lv_bus"]
    add_column_from_node_to_elements(net, "subnet", False, branch_bus=branch_bus)

    def check_subnet_correctness(ntw, elements, branch_bus_el):
        for elm in elements:
            if "bus" in ntw[elm].columns:
                assert all(compare_arrays(ntw[elm]["subnet"].values,
                                          np.array(["subnet_%i" % bus for bus in ntw[elm].bus])))
            elif branch_bus_el[0] in ntw[elm].columns:
                assert all(compare_arrays(ntw[elm]["subnet"].values, np.array([
                    "subnet_%i" % bus for bus in ntw[elm][branch_bus_el[0]]])))
            elif branch_bus_el[1] in ntw[elm].columns:
                assert all(compare_arrays(ntw[elm]["subnet"].values, np.array([
                    "subnet_%i" % bus for bus in ntw[elm][branch_bus_el[1]]])))

    check_subnet_correctness(net, pp_elements(bus=False) - {"sgen"}, branch_bus)

    add_column_from_node_to_elements(net_orig, "subnet", True, branch_bus=branch_bus)
    check_subnet_correctness(net_orig, pp_elements(bus=False), branch_bus)


def test_add_column_from_element_to_elements():
    net = create_cigre_network_mv()
    create_measurement(net, "i", "trafo", 5, 3, 0, side="hv")
    create_measurement(net, "i", "line", 5, 3, 0, side="to")
    create_measurement(net, "p", "bus", 5, 3, 2)
    assert net.measurement.name.isnull().all()
    assert ~net.switch.name.isnull().all()
    orig_switch_names = copy.deepcopy(net.switch.name.values)
    expected_measurement_names = np.array([
        net.trafo.name.loc[0], net.line.name.loc[0], net.bus.name.loc[2]])
    expected_switch_names = np.append(
        net.line.name.loc[net.switch.element.loc[net.switch.et == "l"]].values,
        net.trafo.name.loc[net.switch.element.loc[net.switch.et == "t"]].values)

    add_column_from_element_to_elements(net, "name", False)
    assert all(compare_arrays(net.measurement.name.values, expected_measurement_names))
    assert all(compare_arrays(net.switch.name.values, orig_switch_names))

    del net.measurement["name"]
    add_column_from_element_to_elements(net, "name", True)
    assert all(compare_arrays(net.measurement.name.values, expected_measurement_names))
    assert all(compare_arrays(net.switch.name.values, expected_switch_names))


def test_reindex_buses():
    net_orig = example_simple()
    net = example_simple()

    to_add = 5
    new_bus_idxs = np.array(list(net.bus.index)) + to_add
    bus_lookup = dict(zip(net["bus"].index.values, new_bus_idxs))
    # a more complexe bus_lookup of course should also work, but this one is easy to check
    reindex_buses(net, bus_lookup)

    for elm in net.keys():
        if isinstance(net[elm], pd.DataFrame) and net[elm].shape[0]:
            cols = pd.Series(net[elm].columns)
            bus_cols = cols.loc[cols.str.contains("bus")]
            if len(bus_cols):
                for bus_col in bus_cols:
                    assert all(net[elm][bus_col] == net_orig[elm][bus_col] + to_add)
            if elm == "bus":
                assert all(np.array(list(net[elm].index)) == np.array(list(
                    net_orig[elm].index)) + to_add)


def test_continuos_bus_numbering():
    net = create_empty_network()

    bus0 = create_bus(net, 0.4, index=12)
    create_load(net, bus0, p_mw=0.)
    create_load(net, bus0, p_mw=0.)
    create_load(net, bus0, p_mw=0.)
    create_load(net, bus0, p_mw=0.)

    bus0 = create_bus(net, 0.4, index=42)
    create_sgen(net, bus0, p_mw=0.)
    create_sgen(net, bus0, p_mw=0.)
    create_sgen(net, bus0, p_mw=0.)

    bus0 = create_bus(net, 0.4, index=543)
    create_shunt(net, bus0, 2, 1)
    create_shunt(net, bus0, 2, 1)
    create_shunt(net, bus0, 2, 1)

    bus0 = create_bus(net, 0.4, index=5675)
    create_ward(net, bus0, 2, 1, 1, 2)
    create_ward(net, bus0, 2, 1, 1, 2)
    create_ward(net, bus0, 2, 1, 1, 2)

    create_continuous_bus_index(net)

    buses = net.bus.index
    assert all(buses[i] <= buses[i + 1] for i in range(len(buses) - 1))  # is ordered
    assert all(buses[i] + 1 == buses[i + 1] for i in range(len(buses) - 1))  # is consecutive
    assert buses[0] == 0  # starts at zero

    used_buses = []
    for element in net.keys():
        try:
            used_buses.extend(net[element].bus.values)
        except AttributeError:
            try:
                used_buses.extend(net[element].from_bus.values)
                used_buses.extend(net[element].to_bus.values)
            except AttributeError:
                try:
                    used_buses.extend(net[element].hv_bus.values)
                    used_buses.extend(net[element].lv_bus.values)
                except AttributeError:
                    continue

    # assert that no buses were used except the ones in net.bus
    assert set(list(used_buses)) - set(list(net.bus.index.values)) == set()


def test_reindex_elements():
    net = example_simple()

    new_sw_idx = [569, 763, 502, 258, 169, 259, 348, 522]
    reindex_elements(net, "switch", new_sw_idx)
    assert np.allclose(net.switch.index.values, new_sw_idx)

    net2 = copy.deepcopy(net)

    previous_idx = new_sw_idx[:3]
    new_sw_idx = [2, 3, 4]
    reindex_elements(net, "switch", new_sw_idx, previous_idx)
    assert np.allclose(net.switch.index.values[:3], new_sw_idx)

    # using lookup
    reindex_elements(net2, "switch", lookup=dict(zip(previous_idx, new_sw_idx)))
    assert_net_equal(net, net2)

    reindex_elements(net, "line", [77, 22], [2, 0])
    assert np.allclose(net.line.index.values, [22, 1, 77, 3])
    assert np.allclose(net.switch.element.iloc[[4, 5]], [77, 77])

    old_idx = copy.deepcopy(net.bus.index.values)
    reindex_elements(net, "bus", old_idx + 2)
    assert np.allclose(net.bus.index.values, old_idx + 2)

    reindex_elements(net, "bus", [400, 600], [4, 6])
    assert 400 in net.bus.index
    assert 600 in net.bus.index


def test_continuous_element_numbering():
    from pandapower.estimation.util import add_virtual_meas_from_loadflow
    net = example_multivoltage()

    # Add noises to index with some large number
    net.line = net.line.rename(index={4: 280})
    net.trafo = net.trafo.rename(index={0: 300})
    net.trafo = net.trafo.rename(index={1: 400})
    net.trafo3w = net.trafo3w.rename(index={0: 540})

    net.switch.loc[(net.switch.et == "l") & (net.switch.element == 4), "element"] = 280
    net.switch.loc[(net.switch.et == "t") & (net.switch.element == 0), "element"] = 300
    net.switch.loc[(net.switch.et == "t") & (net.switch.element == 1), "element"] = 400
    runpp(net)
    add_virtual_meas_from_loadflow(net)
    assert net.measurement["element"].max() == 540

    create_continuous_elements_index(net)
    assert net.line.index.max() == net.line.shape[0] - 1
    assert net.trafo.index.max() == net.trafo.shape[0] - 1
    assert net.trafo3w.index.max() == net.trafo3w.shape[0] - 1
    assert net.measurement["element"].max() == net.bus.shape[0] - 1


def test_scaling_by_type():
    net = create_empty_network()

    bus0 = create_bus(net, 0.4)
    create_load(net, bus0, p_mw=0., type="Household")
    create_sgen(net, bus0, p_mw=0., type="PV")

    set_scaling_by_type(net, {"Household": 42., "PV": 12})

    assert net.load.at[0, "scaling"] == 42
    assert net.sgen.at[0, "scaling"] == 12

    set_scaling_by_type(net, {"Household": 0, "PV": 0})

    assert net.load.at[0, "scaling"] == 0
    assert net.sgen.at[0, "scaling"] == 0


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

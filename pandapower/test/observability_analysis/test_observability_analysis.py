import unittest
from copy import deepcopy

import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal

import pandapower as pp
from pandapower.observability_analysis.observability_analysis import run_observability_analysis_for_ppnet


class Test6BusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 7):  # Buses are numbered from 1 to 6
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6)
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[1], vm_pu=1.03)

        cls.net = net

    def test_observability_case6_A1(self):
        net = deepcopy(self.net)

        # Add measurements to make the network observable
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=4, side="from")  # Line active power

        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # There is only one island, but it lacks a voltage measurement, making all elements in the island unobservable (-1).
        self.assertEqual(len(connected_components), 1)
        unobservable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][unobservable_buses], -1 * np.ones(len(unobservable_buses)))
        unobservable_lines = list(net.line.index)
        assert_array_equal(net._observability_lookup['line'][unobservable_lines], -1 * np.ones(len(unobservable_lines)))

        # An island must contain at least one voltage measurement to be considered observable.
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=1)
        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Now all buses are observable because there is only one island, and this island contains a voltage measurement.
        self.assertEqual(len(connected_components), 1)
        observable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][observable_buses], np.zeros(len(unobservable_buses)))
        # Line with index 2 is unobservable because it lacks both a flow measurement
        # and at least one injection measurement at its terminal nodes.
        line_obs_result = [0, 0, -1, 0, 0, 0]
        lines = [0, 1, 2, 3, 4, 5]
        assert_array_equal(net._observability_lookup['line'][lines], line_obs_result)

    def test_observability_case6_A2(self):
        net = deepcopy(self.net)

        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=4, side="from")  # Line active power

        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Note: The indices 0, 1, 2, 3, ... correspond to positions in net.bus (not the net.bus index itself).
        expected_connected_components = [
            {0, 1, 2},
            {3, 4},
            {5}
        ]

        # There is 3 islands, but they lacks a voltage measurement, making all elements in the all islands unobservable (-1).
        self.assertEqual(connected_components, expected_connected_components)
        unobservable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][unobservable_buses], -1 * np.ones(len(unobservable_buses)))
        unobservable_lines = list(net.line.index)
        assert_array_equal(net._observability_lookup['line'][unobservable_lines], -1 * np.ones(len(unobservable_lines)))

        # Add voltage measurements to the first and second islands to make the islands observable.
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=1)
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=4)
        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Now first and second islands are observable, but third island unobservable.
        self.assertEqual(len(connected_components), 3)
        buses = [1, 2, 3, 4, 5, 6]
        bus_obs_result = [0, 0, 0, 1, 1, -1]
        assert_array_equal(net._observability_lookup['bus'][buses], bus_obs_result)
        lines = [0, 1, 2, 3, 4, 5]
        line_obs_result = [0, 0, -1, -1, 1, -1]
        assert_array_equal(net._observability_lookup['line'][lines], line_obs_result)


class Test14BusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 15):  # Buses are numbered from 1 to 14
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 5), (1, 2), (2, 5), (2, 4), (2, 3), (3, 4), (5, 4), (5, 6), (4, 7), (4, 9),
            (7, 8), (7, 9),
            (9, 14), (9, 11), (11, 10), (6, 10), (6, 12), (6, 13), (12, 13), (13, 14),
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add example load at a bus
        pp.create_load(net, buses[8], p_mw=5, q_mvar=2)

        # Add example generator at a bus
        pp.create_gen(net, buses[1], p_mw=10, vm_pu=1.02)

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[6], vm_pu=1.03)

        cls.net = net

    def test_observability_case14_A1(self):
        net = deepcopy(self.net)
        # Add measurements to make the network observable
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=13)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=12)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=11)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=9)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=5)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=2)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=8, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=10, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=11, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=9, side="from")  # Line active power

        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # There is only one island, but it lacks a voltage measurement, making all elements in the island unobservable (-1).
        self.assertEqual(len(connected_components), 1)
        unobservable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][unobservable_buses], -1 * np.ones(len(unobservable_buses)))
        unobservable_lines = list(net.line.index)
        assert_array_equal(net._observability_lookup['line'][unobservable_lines], -1 * np.ones(len(unobservable_lines)))

        # An island must contain at least one voltage measurement to be considered observable.
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=1)
        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Now all buses are observable because there is only one island, and this island contains a voltage measurement.
        self.assertEqual(len(connected_components), 1)
        buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][buses], np.zeros(len(buses)))
        lines = list(net.bus.index)
        assert_array_equal(net._observability_lookup['line'][lines], np.zeros(len(lines)))

    def test_observability_case14_A2(self):
        net = deepcopy(self.net)
        # Add measurements to make the network observable
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=13)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=12)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=11)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=9)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=4)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=2)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=1)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=8, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=10, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=11, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=9, side="from")  # Line active power

        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Note: The indices 0, 1, 2, 3, ... correspond to positions in net.bus (not the net.bus index itself).
        expected_connected_components = [
            {0, 1, 2, 3, 4, 6, 7, 8},
            {5}, {9}, {10}, {11}, {12}, {13}
        ]

        # There is only 7 islands, but it lacks a voltage measurement, making all elements in the islands unobservable (-1).
        self.assertEqual(connected_components, expected_connected_components)
        unobservable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][unobservable_buses], -1 * np.ones(len(unobservable_buses)))
        unobservable_lines = list(net.line.index)
        assert_array_equal(net._observability_lookup['line'][unobservable_lines], -1 * np.ones(len(unobservable_lines)))

        # Add voltage measurements to the first island to make the island observable.
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=1)
        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Now first island is observable, the remaining islands remain unobservable.
        self.assertEqual(len(connected_components), 7)
        buses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        bus_obs_result = [0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1]
        assert_array_equal(net._observability_lookup['bus'][buses], bus_obs_result)
        lines = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        line_obs_result = [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]
        assert_array_equal(net._observability_lookup['line'][lines], line_obs_result)


class TestMonticelliBusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 9):  # Buses are numbered from 1 to 8
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 3), (2, 4), (3, 7), (3, 5), (4, 7), (4, 6), (6, 8), (5, 8), (7, 8)
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[1], vm_pu=1.03)

        cls.net = net

    def test_observability(self):
        net = deepcopy(self.net)
        # Add measurements to make the network observable
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=0, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=5, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=3, side="from")  # Line active power

        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Note: The indices 0, 1, 2, 3, ... correspond to positions in net.bus (not the net.bus index itself).
        expected_connected_components = [
            {0, 2, 4, 6},
            {1, 3, 5},
            {7}

        ]
        # There is 3 islands, but they lacks a voltage measurement, making all elements in the islands unobservable (-1).
        self.assertEqual(connected_components, expected_connected_components)
        unobservable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][unobservable_buses], -1 * np.ones(len(unobservable_buses)))
        unobservable_lines = list(net.line.index)
        assert_array_equal(net._observability_lookup['line'][unobservable_lines], -1 * np.ones(len(unobservable_lines)))

        # Add voltage measurements to the first and second islands to make the islands observable.
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=1)
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=4)
        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Now first and second islands are observable, but third island unobservable.
        self.assertEqual(len(connected_components), 3)
        buses = [1, 2, 3, 4, 5, 6, 7, 8]
        bus_obs_result = [0, 1, 0, 1, 0, 1, 0, -1]
        assert_array_equal(net._observability_lookup['bus'][buses], bus_obs_result)
        lines = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        line_obs_result = [0, 1, 0, 0, -1, 1, -1, -1, -1]
        assert_array_equal(net._observability_lookup['line'][lines], line_obs_result)


class TestAburBusSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create an empty Pandapower network
        net = pp.create_empty_network()

        # Add buses
        buses = {}
        for i in range(1, 7):  # Buses are numbered from 1 to 6
            buses[i] = pp.create_bus(net, vn_kv=110, index=i)  # Assume 110 kV voltage level for simplicity

        # Add lines based on the connections in the diagram
        lines = [
            (1, 2), (1, 6), (2, 3), (2, 6), (2, 5), (3, 4), (4, 5), (5, 6)
        ]

        for line in lines:
            pp.create_line_from_parameters(
                net, from_bus=buses[line[0]], to_bus=buses[line[1]],
                length_km=1.0,  # Assume 1 km for all lines
                r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10, max_i_ka=1
            )

        # Add external grid connection at bus 6
        pp.create_ext_grid(net, buses[1], vm_pu=1.03)

        cls.net = net

    def test_observability(self):

        net = deepcopy(self.net)
        # Add measurements to make the network observable
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=3)  # Active power injection
        pp.create_measurement(net, "p", "bus", 0.0, 0.01, element=6)  # Active power injection

        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=5, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=1, side="from")  # Line active power
        pp.create_measurement(net, "p", "line", 1.0, 0.01, element=2, side="from")  # Line active power

        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Note: The indices 0, 1, 2, 3, ... correspond to positions in net.bus (not the net.bus index itself).
        expected_connected_components = [
            {1, 2, 3},
            {0, 5},
            {4}
        ]

        # There is 3 islands, but they lacks a voltage measurement, making all elements in the islands unobservable (-1).
        self.assertEqual(connected_components, expected_connected_components)
        unobservable_buses = list(net.bus.index)
        assert_array_equal(net._observability_lookup['bus'][unobservable_buses], -1 * np.ones(len(unobservable_buses)))
        unobservable_lines = list(net.line.index)
        assert_array_equal(net._observability_lookup['line'][unobservable_lines], -1 * np.ones(len(unobservable_lines)))

        # Add voltage measurements to the first and second islands to make the islands observable.
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=2)
        pp.create_measurement(net, "v", "bus", 1.0, 0.01, element=1)
        graph = run_observability_analysis_for_ppnet(net)
        connected_components = list(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Now first and second islands are observable, but third island unobservable.
        self.assertEqual(len(connected_components), 3)
        buses = [1, 2, 3, 4, 5, 6]
        bus_obs_result = [ 0, 1, 1, 1, -1, 0]
        assert_array_equal(net._observability_lookup['bus'][buses], bus_obs_result)
        lines = [0, 1, 2, 3, 4, 5, 6, 7]
        line_obs_result = [-1, 0, 1, -1, -1, 1, -1, -1]
        assert_array_equal(net._observability_lookup['line'][lines], line_obs_result)


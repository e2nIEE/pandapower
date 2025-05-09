# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
from pandapower.estimation.ppc_conversion import ExtendedPPCI

import pandapower as pp
from pandapower.pypower.idx_brch import branch_cols
from pandapower.topology.create_graph import add_edges

from pandapower.pypower.idx_bus import bus_cols
from pandapower.estimation.idx_bus import P
from pandapower.estimation.idx_brch import (P_FROM, P_TO)

INDEX = 0
F_BUS = 1
T_BUS = 2


def get_elements_without_measurements(eppci: ExtendedPPCI):
    """
    Function to identify branches without measurements and without injection at connected buses.
    """

    elements_to_drop = []
    ppci = eppci.data
    for idx, branch in enumerate(ppci['branch']):
        bus_from, bus_to = int(branch[0]), int(branch[1])
        bus_from_has_p_injection = ~np.isnan(ppci["bus"][bus_from][bus_cols + P])
        bus_to_has_p_injection = ~np.isnan(ppci["bus"][bus_to][bus_cols + P])
        has_p_from_flow = ~np.isnan(branch[branch_cols + P_FROM])
        has_p_to_flow = ~np.isnan(branch[branch_cols + P_TO])
        if not any([bus_from_has_p_injection, bus_to_has_p_injection, has_p_from_flow, has_p_to_flow]):
            elements_to_drop.append(idx)

    return elements_to_drop


def init_par(tab: np.ndarray):
    """
    Initializes branch parameters for further processing.

    Parameters:
        tab (np.ndarray): Array containing branch data.

    Returns:
        tuple:
            - indices (np.ndarray): Array of shape (n, 3) with branch indices.
            - parameters (np.ndarray): Array of shape (n, 1) with default parameter values (set to 1.0).
            - in_service (list[bool]): List indicating whether each branch is in service (default: True for all).
    """
    n = tab.shape[0]  # Number of branches
    indices = np.zeros((n, 3), dtype=np.int64)  # Initialize indices array
    indices[:, INDEX] = tab[:, -1]  # Populate the last column of indices with branch data

    parameters = np.ones((n, 1), dtype=float)  # Default parameter values set to 1.0
    in_service = [True] * n  # All branches assumed to be in service

    return indices, parameters, in_service



def create_graph_from_eppci(eppci: ExtendedPPCI) -> nx.MultiGraph:
    """
       Creates a MultiGraph representation of the network from the given ExtendedPPCI.

       Parameters:
           eppci (ExtendedPPCI): The extended power flow data structure.

       Returns:
           nx.MultiGraph: A graph representing the network, with buses as nodes and branches as edges.
    """

    mg = nx.MultiGraph()
    branch = eppci.data["branch"]

    # Initialize branch parameters
    indices, parameter, in_service = init_par(branch)
    indices[:, F_BUS] = branch[:, 0]
    indices[:, T_BUS] = branch[:, 1]

    # Add edges to the graph
    add_edges(mg, indices, parameter, in_service, None, "line", False, 'pu')

    # Add missing buses as isolated nodes
    all_buses = set(range(eppci.data["bus"].shape[0]))
    existing_buses = set(mg.nodes)
    missing_buses = all_buses - existing_buses
    if missing_buses:
        mg.add_nodes_from(missing_buses)

    return mg


def print_connected_components(mg, net: pp.pandapowerNet):
    """
        Prints the connected components of a network graph and validates that all buses are included.

        Parameters:
            mg (nx.Graph): The network graph.
            net (pp.pandapowerNet): The pandapower network.
    """
    # Map ExtendedPPCI bus indices to pandapower net bus indices
    eppci_bus_to_ppnet_map = defaultdict(list)
    for i, v in enumerate(net._pd2ppc_lookups["bus"]):
        if v != -1:
            eppci_bus_to_ppnet_map[v].append(i)

    connected_components = list(nx.connected_components(mg))
    max_bus_index = max(net.bus.index)
    print("\nResult: ")
    all_busses_nested = []
    counter = 0
    for component in connected_components:
        bus_idx = [[j for j in eppci_bus_to_ppnet_map[i] if j <= max_bus_index] for i in component]
        all_busses_nested += bus_idx
        if any(bus_idx):
            bus_idx = [i for i in bus_idx if i]
            print(f"Component {counter}: Len : {len(bus_idx)}  Bus: {bus_idx}")
            counter += 1

    all_busses = list(chain.from_iterable(all_busses_nested))
    if len(all_busses) != len(net.bus):
        raise Exception("Error: The result does not include all buses!")

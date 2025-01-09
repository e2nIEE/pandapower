# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np

import pandapower as pp
from pandapower.estimation.idx_bus import VM
from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.pypower.idx_bus import bus_cols
from pandapower.results_branch import _get_trafo3w_lookups


def _map_branches_to_lines(net: pp.pandapowerNet, component_id, component_branches):
    """
    Assign line branches of a component to the observability structure in the network.
    """

    if not "line" in net._pd2ppc_lookups["branch"]:
        return

    line_index_from, line_index_to = net._pd2ppc_lookups["branch"]["line"]
    lines_eppci_idx = filter(lambda idx: line_index_from <= idx < line_index_to, component_branches)

    # Extract line network indices
    lines_net_idx = net.line.iloc[lines_eppci_idx]

    # Update observability lookup with component ID
    net._observability_lookup["line"][list(lines_net_idx.index)] = component_id


def _map_branches_to_trafo(net: pp.pandapowerNet, component_id, component_branches):
    """
       Assign transformer branches of a component to the observability structure in the network.
    """

    if not "trafo" in net._pd2ppc_lookups["branch"]:
        return

    trafo_index_from, trafo_index_to = net._pd2ppc_lookups["branch"]["trafo"]
    trafo_eppci_idx = [i - trafo_index_from for i in component_branches if trafo_index_from <= i < trafo_index_to]

    # Extract transformer network indices
    trafo_net = net.trafo.iloc[trafo_eppci_idx]

    # Update observability lookup with component ID
    net._observability_lookup["trafo"][list(trafo_net.index)] = component_id


def _map_branches_to_trafo3w(net: pp.pandapowerNet, component_id, component_branches):
    """
       Assign 3-winding transformer branches of a component to the observability structure in the network.
    """

    if not "trafo3w" in net._pd2ppc_lookups["branch"]:
        return

    trafo3w_index_from, trafo3w_index_hv, trafo3w_index_mv, trafo3w_index_lv = _get_trafo3w_lookups(net)

    trafo3w_hv = [
        i - trafo3w_index_from
        for i in component_branches
        if trafo3w_index_from <= i < trafo3w_index_hv
    ]
    trafo3w_mv = [
        i - trafo3w_index_hv
        for i in component_branches
        if trafo3w_index_hv <= i < trafo3w_index_mv
    ]
    trafo3w_lv = [
        i - trafo3w_index_mv
        for i in component_branches
        if trafo3w_index_mv <= i < trafo3w_index_lv
    ]

    # Combine all indices and remove duplicates
    all_trafo3w_idx = list(set(trafo3w_lv + trafo3w_mv + trafo3w_hv))

    # Extract transformer 3-winding network indices
    trafo3w_hv_net = net.trafo3w.iloc[all_trafo3w_idx]

    # Update observability lookup with component ID
    net._observability_lookup["trafo3w"][list(trafo3w_hv_net.index)] = component_id


def create_lookup(element):
    return np.full(max(element.index) + 1, -1, dtype=int) if not element.empty else np.array([], dtype=int)


def _init_observability_lookup(net):
    net._observability_lookup = {
        "line": create_lookup(net.line),
        "trafo": create_lookup(net.trafo),
        "trafo3w": create_lookup(net.trafo3w),
        "bus": create_lookup(net.bus),
    }


def _map_buses_to_components(eppci: ExtendedPPCI, net: pp.pandapowerNet):
    """
        Maps buses in the ExtendedPPCI structure to their corresponding observable islands
        and updates the observability lookup in the pandapower network
    """

    # Map ExtendedPPCI bus indices to corresponding pandapower net bus indices
    eppci_bus_to_ppnet_map = defaultdict(list)
    for pp_net_index, eppci_index in enumerate(net._pd2ppc_lookups["bus"]):
        if eppci_index != -1:
            eppci_bus_to_ppnet_map[eppci_index].append(pp_net_index)

    # Group buses by their observable island IDs
    sub_components = defaultdict(list)
    for ind, value in enumerate(eppci.data['bus'][:, -1]):
        sub_components[value].append(ind)

    # Update observability lookup for each bus group
    max_bus_index = max(net.bus.index)
    for component_id, node_buses in sub_components.items():
        bus_idx = [[j for j in eppci_bus_to_ppnet_map[i] if j <= max_bus_index] for i in node_buses]
        chain_bus_idx = list(chain.from_iterable(bus_idx))
        if any(chain_bus_idx):
            net._observability_lookup["bus"][chain_bus_idx] = component_id


def _map_branches_to_components(eppci: ExtendedPPCI, net: pp.pandapowerNet):
    """
        Maps branches in the eppci structure to their respective components and integrates this mapping
        into the pandapower network model.
    """

    # Group branch indices by component ID
    sub_components = defaultdict(list)
    for ind, value in enumerate(eppci.data['branch'][:, -1]):
        sub_components[value].append(ind)

    # Add branches to the network model based on their component ID
    for component_id, node_buses in sub_components.items():
        _map_branches_to_lines(net, component_id, node_buses)
        _map_branches_to_trafo(net, component_id, node_buses)
        _map_branches_to_trafo3w(net, component_id, node_buses)


def add_connected_components_to_net(eppci: ExtendedPPCI, net: pp.pandapowerNet):
    """
    Pass information of observability back to pandapower from eppci to net._observability_lookup
    """

    _init_observability_lookup(net)
    _map_buses_to_components(eppci, net)
    _map_branches_to_components(eppci, net)


def get_components_with_voltage_measurements(graph: nx.MultiGraph, original_eppci: ExtendedPPCI):
    result = []
    subgraphs = [graph.subgraph(component) for component in nx.connected_components(graph)]
    for ind, component in enumerate(subgraphs):
        # Check if there is a voltage measurement for an observable island
        if np.any(~np.isnan(original_eppci.data['bus'][component.nodes][:, bus_cols + VM])):
            result.append(component)
    return result


def add_connected_components_to_eppci(graph: nx.MultiGraph, original_eppci: ExtendedPPCI):
    """
       Assign buses and branches in the ExtendedPPCI to their corresponding observable islands
       based on connected components in the graph.
    """

    # Find all subgraphs
    subgraphs = get_components_with_voltage_measurements(graph, original_eppci)

    # Add additional column to 'bus'
    all_bus_idx = np.full(original_eppci.data['bus'].shape[0], -1, dtype=int)
    original_eppci.data['bus'] = np.hstack((original_eppci.data['bus'], all_bus_idx.reshape(-1, 1)))

    # Assign each bus to its corresponding observable island
    for island_id, component in enumerate(subgraphs):
        original_eppci.data['bus'][:, -1][list(component.nodes)] = island_id

    # Add additional column to 'branch'
    all_branch_idx = np.full(original_eppci.data['branch'].shape[0], -1, dtype=int)
    original_eppci.data['branch'] = np.hstack((original_eppci.data['branch'], all_branch_idx.reshape(-1, 1)))

    # Assign each branch to its corresponding observable island
    for island_id, component in enumerate(subgraphs):
        component_branches = [edge_data[2][1] for edge_data in component.edges]
        if any(component_branches):
            original_eppci.data['branch'][:, -1][component_branches] = island_id

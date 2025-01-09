# -*- coding: utf-8 -*-

import pandapower as pp
from pandapower.estimation.ppc_conversion import ExtendedPPCI, pp2eppci
from pandapower.observability_analysis.algorithm.analyzer import ObservabilityAnalyzer
from pandapower.observability_analysis.results import add_connected_components_to_eppci, add_connected_components_to_net


def run_observability_analysis_for_eppci(eppci: ExtendedPPCI):
    """
    Runs the observability analysis on the given ExtendedPPCI object and updates it with
    the results.

    Parameters:
        eppci (ExtendedPPCI): The extended power flow data structure to analyze.

    Returns:
        nx.MultiGraph: The graph representing observability relationships in the network.
    """
    # Initialize the observability analyzer with the given eppci
    analyzer = ObservabilityAnalyzer(eppci)

    # Execute the observability analysis and obtain the resulting graph
    graph = analyzer.run_observability_analysis()

    # Process the graph to add observable islands to the eppci data
    add_connected_components_to_eppci(graph, eppci)

    return graph

def run_observability_analysis_for_ppnet(
        net: pp.pandapowerNet,
        v_start=None,
        delta_start=None,
        calculate_voltage_angles=True,
        zero_injection=None,
        algorithm='wls'
):
    """
    Runs observability analysis for a given pandapower network and updates it with
    the results in net._observability_lookup.

    Parameters:
        net (pp.pandapowerNet): The pandapower network to analyze.
        v_start (Optional[np.ndarray]): Initial voltage magnitudes (optional).
        delta_start (Optional[np.ndarray]): Initial voltage angles (optional).
        calculate_voltage_angles (bool): Flag to indicate if voltage angles should be calculated.
        zero_injection (Optional[bool]): Option to include zero injection buses in the analysis.
        algorithm (str): The state estimation algorithm to use (default: 'wls').

    Returns:
        nx.MultiGraph: The graph representing observability relationships in the network.
    """
    # Convert pandapower network (ppnet) to extended power flow data structure (eppci)
    _, _, eppci = pp2eppci(
        net,
        v_start=v_start,
        delta_start=delta_start,
        calculate_voltage_angles=calculate_voltage_angles,
        zero_injection=zero_injection,
        algorithm=algorithm
    )

    # Perform observability analysis on the eppci data
    graph = run_observability_analysis_for_eppci(eppci)

    # Map the results back to the pandapower network
    add_connected_components_to_net(eppci, net)

    return graph

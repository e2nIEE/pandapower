# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import logging

import networkx as nx
import numpy as np
from scipy.linalg import lu

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.idx_brch import P_FROM, P_TO
from pandapower.estimation.idx_bus import P, P_STD
from pandapower.observability_analysis.algorithm.network_analysis_core import NetworkAnalysisCore
from pandapower.observability_analysis.network_utils import get_elements_without_measurements, create_graph_from_eppci
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols

logger = logging.getLogger(__name__)


class ObservabilityAnalyzer(NetworkAnalysisCore):

    def _delete_branch(self, lines: np.ndarray):
        """
        Deletes specified branches from the network and updates related data structures.

        Parameters:
            lines (np.ndarray): Array of branch indices to be deleted.
        """

        ppci = self.eppci.data
        N = ppci["branch"].shape[0]

        # Delete the specified branches from the branch matrix
        ppci["branch"] = np.delete(ppci["branch"], lines, axis=0)

        # Update non-NaN measurement selector
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
        p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
        meas_mask = np.hstack([p_bus_not_nan, p_line_f_not_nan, p_line_t_not_nan])
        self.eppci.non_nan_meas_selector = np.flatnonzero(meas_mask)

        # Update internal Ybus matrices if they exist
        if "Ybus" in ppci["internal"] and ppci["internal"]["Ybus"].size:
            rows_to_keep = sorted(set(range(N)) - set(lines))  # Ensure the order is maintained
            ppci["internal"]["Yf"] = ppci["internal"]["Yf"][rows_to_keep, :]
            ppci["internal"]["Yt"] = ppci["internal"]["Yt"][rows_to_keep, :]

    def _delete_p_measurement(self, bus_positions: np.ndarray):
        """
        Delete active power (P) measurements for specified buses and update the non-NaN measurement selector.

        Parameters:
            bus_positions (np.ndarray): Array of bus positions for which P measurements should be deleted.
        """

        ppci = self.eppci.data

        # Set P measurements and their standard deviations to NaN for the specified buses
        ppci["bus"][bus_positions, bus_cols + P] = np.NaN
        ppci["bus"][bus_positions, bus_cols + P_STD] = np.NaN

        # Create masks to identify non-NaN P measurements for buses, lines from, and lines to
        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
        p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
        p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])

        # Combine the masks to form the overall measurement mask
        meas_mask = np.hstack([p_bus_not_nan, p_line_f_not_nan, p_line_t_not_nan])

        # Update the non-NaN measurement selector
        self.eppci.non_nan_meas_selector = np.flatnonzero(meas_mask)

    def _get_branches_idx_without_power_flow(self, tolerance: float, branch_power_flow: np.ndarray):
        """
           Identify branches without significant power flow and determine buses to delete.

           Parameters:
               tolerance (float): The threshold below which a branch is considered to have no power flow.
               branch_power_flow (np.ndarray): Array of branch power flows.

           Returns:
               Tuple[np.ndarray, np.ndarray]:
                   - Indices of branches without significant power flow.
                   - Buses connected to these branches that should be removed.
        """

        # Create a boolean mask for branches with power flow below the threshold
        branch_mask_without_power_flow = [True if abs(i) > tolerance else False for i in branch_power_flow]
        branches_idx_without_power_flow = np.flatnonzero(branch_mask_without_power_flow)

        # Extract branch data for branches without significant power flow
        branch_without_power_flow = self.eppci.data['branch'][branches_idx_without_power_flow]
        logger.info(f"Number of branches without power flow {len(branch_without_power_flow)}. Branches: {branches_idx_without_power_flow}")

        # Identify buses connected to these branches
        buses_with_p_to_delete = np.unique(np.concatenate((branch_without_power_flow[:, 0], branch_without_power_flow[:, 1])))

        return branches_idx_without_power_flow, buses_with_p_to_delete

    def _drop_power_injections(self, buses_with_p_to_delete: np.ndarray):
        self._delete_p_measurement(buses_with_p_to_delete.astype(np.int64))
        logger.info(f"Number of power injections to delete {len(buses_with_p_to_delete)}. At buses {buses_with_p_to_delete} ")

    def run_observability_analysis(self, max_iter=50, tolerance=1e-10) -> nx.MultiGraph:
        """
           Perform observability analysis on the power system network.

           Parameters:
               max_iter (int): Maximum number of iterations before stopping.
               tolerance (float): Threshold for identifying zero pivots and stopping conditions.

           Returns:
               nx.MultiGraph: A network graph representing the observable system.

           Raises:
               Exception: If the maximum number of iterations is reached without convergence.
        """

        # Step 1: Initialization
        self._clean_not_p_measurements()
        N = int(self.eppci.data['bus'].shape[0])
        self._set_delta_v_bus_selector()
        self.eppci.E[:N - 1] = 0
        self.eppci.E[N:] = 1
        self._reset_network_values()

        all_branches_idx = np.arange(self.eppci.data['branch'].shape[0])
        self.eppci.data['branch'] = np.hstack((self.eppci.data['branch'], all_branches_idx.reshape(-1, 1)))

        current_iteration = 1
        while current_iteration <= max_iter:
            logger.info(f"Iteration: {current_iteration}")

            # Step 2: Identify and delete elements without measurements
            elements_to_drop = get_elements_without_measurements(self.eppci)
            logger.info(f"Number of branches without measurements to delete = {len(elements_to_drop)}. Branches {elements_to_drop}")
            self._delete_branch(elements_to_drop)

            # Step 3: Construct Jacobian matrix and gain matrix
            sem = BaseAlgebra(self.eppci)
            jacobian = sem.create_hx_jacobian(self.eppci.E)
            gain_matrix = np.dot(jacobian.T, jacobian)

            # Step 4: LU decomposition and zero pivot validation
            P, L, U = lu(gain_matrix)
            zero_pivots = np.where(np.abs(np.diag(U)) < tolerance)[0]
            stop_iterations = self._validate_zero_pivots(zero_pivots, N)
            if stop_iterations is True:
                break
            jacobian_with_pseudo_meas = self._add_pseudo_meas_to_jacobian(jacobian, zero_pivots)

            # Step 5: Solve for DC estimator equation
            solution = self._solve_dc_estimator_equation(jacobian_with_pseudo_meas, zero_pivots)

            # Step 6: Calculate branch power flow
            branch_power_flow = self._calculate_branch_power_flow(solution)

            # Step 7: Identify and delete branches without power flow
            branch_idx_without_power_flow, buses_with_p_to_delete = self._get_branches_idx_without_power_flow(tolerance, branch_power_flow)
            if not branch_idx_without_power_flow.any():
                logger.info("No branches without power flow. Stop iterations. ")
                break
            self._delete_branch(branch_idx_without_power_flow)

            # Step 8: Remove power injections for identified buses
            self._drop_power_injections(buses_with_p_to_delete)

            # Step 9 - increase counter and go to step 2
            current_iteration += 1

        if current_iteration == max_iter:
            raise Exception("Maximum number of iterations reached. Algorithm did not converge.")

        graph = create_graph_from_eppci(self.eppci)
        return graph

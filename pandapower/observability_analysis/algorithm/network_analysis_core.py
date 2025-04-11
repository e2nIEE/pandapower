# -*- coding: utf-8 -*-

# This code was written by Matsiushonak Siarhei and Zografos Dimitrios.
# Contributions made on 2025.

import logging
from copy import deepcopy

import numpy as np
from scipy.sparse.linalg import spsolve

from pandapower.estimation.idx_brch import Q_FROM, Q_TO, Q_FROM_STD, Q_TO_STD, IA_FROM, IA_FROM_STD, IA_TO, IA_TO_STD, \
    IM_FROM, IM_FROM_STD, IM_TO, IM_TO_STD, P_FROM, P_FROM_STD, P_TO, P_TO_STD
from pandapower.estimation.idx_bus import Q, Q_STD, VM, VM_STD, VA, VA_STD, P, P_STD
from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B, BR_G, SHIFT, TAP
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols, GS, BS
from scipy.sparse import csr_matrix, vstack

logger = logging.getLogger(__name__)


class NetworkAnalysisCore:
    def __init__(self, eppci: ExtendedPPCI):
        self.eppci = deepcopy(eppci)

    def _clean_p_without_q_measurements(self):
        ppci = self.eppci.data
        # Create bus mask where Q is NaN
        q_bus_nan_mask = np.isnan(ppci["bus"][:, bus_cols + Q])

        # Set P and P_STD to NaN only where Q is NaN
        ppci["bus"][q_bus_nan_mask, bus_cols + P] = np.nan
        ppci["bus"][q_bus_nan_mask, bus_cols + P_STD] = np.nan

        # Create from branch mask where Q is NaN
        q_branch_from_nan_mask = np.isnan(ppci["branch"][:, branch_cols + Q_FROM])
        ppci["branch"][q_branch_from_nan_mask, branch_cols + P_FROM] = np.nan
        ppci["branch"][q_branch_from_nan_mask, branch_cols + P_FROM_STD] = np.nan

        # Create to branch mask where Q is NaN
        q_branch_to_nan_mask = np.isnan(ppci["branch"][:, branch_cols + Q_TO])
        ppci["branch"][q_branch_to_nan_mask, branch_cols + P_TO] = np.nan
        ppci["branch"][q_branch_to_nan_mask, branch_cols + P_TO_STD] = np.nan



    def _clean_not_p_measurements(self):
        ppci = self.eppci.data

        ppci["bus"][:, bus_cols + VM] = np.nan
        ppci["bus"][:, bus_cols + VM_STD] = np.nan

        ppci["bus"][:, bus_cols + Q] = np.nan
        ppci["bus"][:, bus_cols + Q_STD] = np.nan

        ppci["bus"][:, bus_cols + VA] = np.nan
        ppci["bus"][:, bus_cols + VA_STD] = np.nan

        ppci["branch"][:, branch_cols + Q_FROM] = np.nan
        ppci["branch"][:, branch_cols + Q_FROM_STD] = np.nan

        ppci["branch"][:, branch_cols + Q_TO] = np.nan
        ppci["branch"][:, branch_cols + Q_TO_STD] = np.nan

        ppci["branch"][:, branch_cols + IA_FROM] = np.nan
        ppci["branch"][:, branch_cols + IA_FROM_STD] = np.nan

        ppci["branch"][:, branch_cols + IA_TO] = np.nan
        ppci["branch"][:, branch_cols + IA_TO_STD] = np.nan

        ppci["branch"][:, branch_cols + IM_FROM] = np.nan
        ppci["branch"][:, branch_cols + IM_FROM_STD] = np.nan

        ppci["branch"][:, branch_cols + IM_TO] = np.nan
        ppci["branch"][:, branch_cols + IM_TO_STD] = np.nan

        self.eppci._initialize_meas()

    def _reset_network_values(self):
        """
           Resets network parameters in the eppci data structure to default values.
           Ensures consistent initialization and avoids numerical artifacts in the Jacobian.
        """

        self.eppci.data['branch'][:, BR_R] = np.zeros(len(self.eppci.data['branch'][:, BR_R]))
        self.eppci.data['branch'][:, BR_X] = np.ones(len(self.eppci.data['branch'][:, BR_X]))

        self.eppci.data['branch'][:, BR_B] = np.zeros(len(self.eppci.data['branch'][:, BR_B]))
        self.eppci.data['branch'][:, BR_G] = np.zeros(len(self.eppci.data['branch'][:, BR_G]))

        self.eppci.data['bus'][:, GS] = np.zeros(len(self.eppci.data['bus'][:, GS]))
        self.eppci.data['bus'][:, BS] = np.zeros(len(self.eppci.data['bus'][:, BS]))

        self.eppci.data['branch'][:, TAP] = np.ones(len(self.eppci.data['branch'][:, TAP]))
        self.eppci.data['branch'][:, SHIFT] = np.zeros(len(self.eppci.data['branch'][:, SHIFT]))

    def _set_delta_v_bus_selector(self):
        """
        Set selector to work only with dP/dθ part of the Jacobian.
        """

        self.eppci.delta_v_bus_selector = np.arange(len(self.eppci.data['bus']))

    def _calculate_branch_power_flow(self, theta_vector: np.ndarray) -> np.ndarray:
        """
        Calculates the power flow across branches based on bus voltage angles.

        Args:
            theta_vector (np.ndarray): Vector of bus voltage angles (in radians).

        Returns:
            np.ndarray: Array of angle differences (theta_diff) between connected buses.
        """

        # Extract 'from' and 'to' bus indices from the branch data
        from_buses = self.eppci.data['branch'][:, 0].real.astype(np.int64)
        to_buses = self.eppci.data['branch'][:, 1].real.astype(np.int64)

        # Calculate the voltage angle differences between 'from' and 'to' buses
        theta_diff = theta_vector[from_buses] - theta_vector[to_buses]

        return theta_diff

    def _validate_solution(self, A: np.ndarray, x: np.ndarray, b: np.ndarray) -> None:
        """
           Checks for nan values in the solution vector x, and if valid, computes and prints the squared residual.

           Parameters:
               A (np.ndarray): The coefficient matrix.
               x (np.ndarray): The solution vector.
               b (np.ndarray): The right-hand side vector.

           Raises:
               ValueError: If x contains nan values.
           """

        if np.any(np.isnan(x)):
            raise Exception("Equation solving failed")

        # Compute the residual
        residual = np.dot(A, x) - b
        squared_residual = np.sum(residual ** 2)
        logger.info(f"Residual for theta vector at step 5: {squared_residual}")

    def _solve_dc_estimator_equation(
            self, jacobian_with_pseudo_meas: np.ndarray, zero_pivots: np.ndarray
    ) -> np.ndarray:
        """
        Solves the DC estimator equation using pseudo-measurements.

        Args:
            jacobian_with_pseudo_meas (np.ndarray): Jacobian matrix augmented with pseudo-measurements.
            zero_pivots (np.ndarray): Indices of zero pivots to handle.

        Returns:
            np.ndarray: Solution vector of the DC estimation equation.
        """

        # Create the measurement vector z
        z = np.zeros(jacobian_with_pseudo_meas.shape[0])
        z[-len(zero_pivots):] = np.arange(len(zero_pivots))
        # jacobian_with_pseudo_meas = csr_matrix(jacobian_with_pseudo_meas)

        # Calculate the product of Jacobian transpose and z
        h_w_z = jacobian_with_pseudo_meas.T @ z  # W is identity, so it's skipped

        # Compute the gain matrix
        gain_matrix_with_pseudo_meas = jacobian_with_pseudo_meas.T @ jacobian_with_pseudo_meas

        # Solve the equation using sparse solver
        solution = spsolve(gain_matrix_with_pseudo_meas, h_w_z)

        return solution

    def _validate_zero_pivots(self, zero_pivots: np.ndarray, N: int) -> bool:
        """
        Validates zero pivots to determine if iterations should stop.

        Args:
            zero_pivots (np.ndarray): Array of zero pivot indices.
            N (int): Total number of elements in the system.

        Returns:
            bool: True if iterations should stop, False otherwise.
        """

        if zero_pivots.size == 0:
            logger.info("No zero pivots. Stop iterations.")
            return True

        if zero_pivots.size == 1 and zero_pivots[0] == N - 1:
            logger.info("Only one zero pivot at the last element. Stop iterations.")
            return True

        logger.info(f"Zero pivots detected: {zero_pivots}")
        return False

    def _add_pseudo_meas_to_jacobian(self, jacobian: np.ndarray, zero_pivots: np.ndarray) -> np.ndarray:
        """
        Adds pseudo-measurements to the Jacobian matrix for zero pivots.

        Args:
            jacobian (np.ndarray): The original Jacobian matrix.
            zero_pivots (np.ndarray): Indices of zero pivots to add pseudo-measurements for.

        Returns:
            np.ndarray: The updated Jacobian matrix with pseudo-measurements added.
        """

        # Create new rows with pseudo measurements
        new_jacobian_rows = csr_matrix(
            (np.ones(len(zero_pivots)), (np.arange(len(zero_pivots)), zero_pivots)),
            shape=(len(zero_pivots), jacobian.shape[1])
        )
        # Stack the new rows with the original matrix
        jacobian_with_pseudo_meas = vstack([jacobian, new_jacobian_rows])
        logger.info(f"Introduced {len(zero_pivots)} pseudo measurements")

        return jacobian_with_pseudo_meas

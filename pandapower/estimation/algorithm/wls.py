# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.pypower.idx_bus import BUS_TYPE, VA, VM, bus_cols
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.ppc_conversion import _build_measurement_vectors, ExtendedPPCI

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra,\
    BaseAlgebraZeroInjConstraints

__all__ = ["WLSAlgorithm", "WLSZeroInjectionConstraintsAlgorithm"]


class WLSAlgorithm:
    def __init__(self, tolerance, maximum_iterations, logger):
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        self.logger = logger
        self.successful = False
        self.iterations = None

        # Parameters for estimate
        self.e_ppci = None
        self.v = None
        self.delta = None
        self.E = None
        self.pp_meas_indices = None

        # Parameters for Bad data detection
        self.R_inv = None
        self.Gm = None
        self.r = None
        self.H = None
        self.hx = None

    def check_observability(self, ppci, z):
        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * ppci["bus"].shape[0] - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * ppci["bus"].shape[0] - 1))
            raise UserWarning("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * ppci["bus"].shape[0] - 1))

    def check_result(self, current_error, cur_it):
        # print output for results
        if current_error <= self.tolerance:
            self.successful = True
            self.logger.debug("WLS State Estimation successful ({:d} iterations)".format(cur_it))
        else:
            self.successful = False
            self.logger.debug("WLS State Estimation not successful ({:d}/{:d} iterations)".format(cur_it,
                                                                                                  self.max_iterations))

    def initialize(self, ppci):
        # check whether or not is the grid observed
        e_ppci = ExtendedPPCI(ppci)
        self.pp_meas_indices = e_ppci.pp_meas_indices
        self.check_observability(e_ppci, e_ppci.z)

        # initialize voltage vector
        self.v = e_ppci.v_init.copy()
        self.delta = e_ppci.delta_init.copy()  # convert to rad
        self.E = np.r_[self.delta[e_ppci.non_slack_bus_mask], self.v]
        self.e_ppci = e_ppci
        return self.e_ppci

    def update_v(self, E=None):
        E = self.E if E is None else E
        self.v = E[self.e_ppci.num_non_slack_bus:]
        self.delta[self.e_ppci.non_slack_buses] = E[:self.e_ppci.num_non_slack_bus]

    def estimate(self, ppci, **kwargs):
        e_ppci = self.initialize(ppci)

        # matrix calculation object
        sem = BaseAlgebra(e_ppci)

        current_error, cur_it = 100., 0
        # invert covariance matrix
        r_inv = csr_matrix(np.diagflat(1/self.e_ppci.r_cov) ** 2)

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(self.E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(self.E))

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))

                # Update E!! Important!!
                self.E += d_E.ravel()

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        # update V/delta
        self.update_v()
        V = self.v * np.exp(1j * self.delta)
        if self.successful:
            # store variables required for chi^2 and r_N_max test:
            self.R_inv = r_inv.toarray()
            self.Gm = G_m.toarray()
            self.r = r.toarray()
            self.H = H.toarray()
            # create h(x) for the current iteration
            self.hx = sem.create_hx(self.E)
        return V


class WLSZeroInjectionConstraintsAlgorithm(WLSAlgorithm):
    def estimate(self, ppci, **kwargs):
        # state vector built from delta, |V| and zero injections
        # Find pq bus with zero p,q and shunt admittance
        zero_injection_bus = np.argwhere(ppci["bus"][:, bus_cols+ZERO_INJ_FLAG] == True).ravel()
        ppci["bus"][zero_injection_bus, [bus_cols+P, bus_cols+P_STD, bus_cols+Q, bus_cols+Q_STD]] = np.NaN
        # Withn pq buses with zero injection identify those who have also no p or q measurement
        p_zero_injections = zero_injection_bus
        q_zero_injections = zero_injection_bus
        new_states = np.zeros(len(p_zero_injections) + len(q_zero_injections))

        e_ppci = self.initialize(ppci)
        num_bus = e_ppci.bus.shape[0]

        # update the E matrix
        E_ext = np.r_[self.E, new_states]

        # matrix calculation object
        sem = BaseAlgebraZeroInjConstraints(e_ppci)

        current_error, cur_it = 100., 0
        r_inv = csr_matrix((np.diagflat(1/e_ppci.r_cov) ** 2))
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                c_x = sem.create_cx(self.E, p_zero_injections, q_zero_injections)

                # residual r
                r = csr_matrix(sem.create_rx(self.E)).T
                c_rxh = csr_matrix(c_x).T

                # jacobian matrix H
                H_temp = sem.create_hx_jacobian(self.E)
                C_temp = sem.create_cx_jacobian(self.E, p_zero_injections, q_zero_injections)
                H, C = csr_matrix(H_temp), csr_matrix(C_temp)

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # building a new gain matrix for new constraints.
                A_1 = vstack([G_m, C])
                c_ax = hstack([C, np.zeros((C.shape[0], C.shape[0]))])
                c_xT = c_ax.T
                M_tx = csr_matrix(hstack((A_1, c_xT)))  # again adding to the new gain matrix
                rhs = H.T * (r_inv * r)  # original right hand side
                C_rhs = vstack((rhs, -c_rxh))  # creating the righ hand side with new constraints

                # state vector difference d_E
                d_E_ext = spsolve(M_tx, C_rhs)
                E_ext += d_E_ext.ravel()
                self.E = E_ext[:self.E.shape[0]]

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E_ext[:len(e_ppci.non_slack_buses) + num_bus]))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        # update V/delta
        self.update_v()
        return self.v * np.exp(1j * self.delta)

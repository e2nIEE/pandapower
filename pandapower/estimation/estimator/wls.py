# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.idx_bus import BUS_TYPE, VA, VM, bus_cols
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.ppc_conversions import _build_measurement_vectors

from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraZeroInjectionConstraints
#from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra
#from pandapower.estimation.estimator.wls_matrix_ops_old import WLSAlgebraZeroInjectionConstraints


class WLSEstimator:
    def __init__(self, net, tolerance, maximum_iterations, logger):
        self.net = net
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        self.logger = logger
        self.successful = False
        self.iterations = None

        self.R_inv = None
        self.Gm = None
        self.r = None
        self.H = None
        self.Ht = None
        self.hx = None
        self.V = None
        self.delta = None

        self.pp_meas_indices = None

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

    def wls_preprocessing(self, ppci):
        # calculate relevant vectors from ppci measurements
        z, self.pp_meas_indices, r_cov, non_nan_meas_mask = _build_measurement_vectors(ppci)
        # invert covariance matrix
        r_inv = csr_matrix((np.diagflat(1/r_cov) ** 2))

        # check whether or not is the grid observed
        self.check_observability(ppci, z)

        # number of nodes
        non_slack_buses = np.argwhere(ppci["bus"][:, BUS_TYPE] != 3).ravel()
        non_slack_bus_mask = (ppci["bus"][:, BUS_TYPE] != 3)

        # set the starting values for all active buses
        v_m = ppci["bus"][:, VM]
        delta = ppci["bus"][:, VA] * np.pi / 180  # convert to rad
        delta_masked = delta[non_slack_bus_mask]
        E = np.r_[delta_masked, v_m]
        return non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask

    def estimate(self, ppci):
        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask = self.wls_preprocessing(ppci)

        # matrix calculation object
        sem = WLSAlgebra(ppci, non_nan_meas_mask)

        current_error, cur_it = 100., 0
        G_m, r, H, h_x = None, None, None, None

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # create h(x) for the current iteration
                h_x = sem.create_hx(v_m, delta)

                # residual r
                r = csr_matrix(z - h_x).T
                
                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(v_m, delta))

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))
                E += d_E.ravel()

                # update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = E[len(non_slack_buses):]

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
#                print(cur_it, current_error)
                self.logger.debug("Current error: {:.7f}".format(current_error))

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        V = v_m * np.exp(1j * delta)
        if self.successful:
            # store variables required for chi^2 and r_N_max test:
            self.R_inv = r_inv.toarray()
            self.Gm = G_m.toarray()
            self.r = r.toarray()
            self.H = H.toarray()
            self.Ht = self.H.T
            self.hx = h_x
            self.V = v_m
            self.delta = delta
        return V


class WLSEstimatorZeroInjectionConstraints(WLSEstimator):
    def estimate(self, ppci):
        # state vector built from delta, |V| and zero injections
        # Find pq bus with zero p,q and shunt admittance
        zero_injection_bus = np.argwhere(ppci["bus"][:, bus_cols+ZERO_INJ_FLAG] == True).ravel()
        ppci["bus"][zero_injection_bus, [bus_cols+P, bus_cols+P_STD, bus_cols+Q, bus_cols+Q_STD]] = np.NaN
        # Withn pq buses with zero injection identify those who have also no p or q measurement
        p_zero_injections = zero_injection_bus
        q_zero_injections = zero_injection_bus
        new_states = np.zeros(len(p_zero_injections) + len(q_zero_injections))
        
        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask = self.wls_preprocessing(ppci)
        num_bus = ppci["bus"].shape[0]


        # update the E matrix
        E = np.r_[E, new_states]

        # matrix calculation object
        sem = WLSAlgebraZeroInjectionConstraints(ppci, non_nan_meas_mask)

        current_error, cur_it = 100., 0
        G_m, r, H, h_x = None, None, None, None

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # create h(x) for the current iteration
                h_x = sem.create_hx(v_m, delta)
                c_x = sem.create_cx(v_m, delta, p_zero_injections, q_zero_injections)

                # residual r
                r = csr_matrix(z - h_x).T
                c_rxh = csr_matrix(c_x).T

                # jacobian matrix H
                H_temp = sem.create_hx_jacobian(v_m, delta)
                C_temp = sem.create_cx_jacobian(v_m, delta, p_zero_injections, q_zero_injections)
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
                d_E = spsolve(M_tx, C_rhs)
                E += d_E.ravel()

                # update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = np.squeeze(E[len(non_slack_buses):len(non_slack_buses) + num_bus])

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E[:len(non_slack_buses) + num_bus]))
                self.logger.debug("Current error: {:.7f}".format(current_error))

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return v_m * np.exp(1j * delta)

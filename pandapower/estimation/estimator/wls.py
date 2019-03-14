# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.pypower.idx_bus import BUS_TYPE, VA, VM, bus_cols
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.ppc_conversions import _build_measurement_vectors
from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraZeroInjectionConstraints


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

    def wls_preprocessing(self, ppci):
        # calculate relevant vectors from ppci measurements
        z, self.pp_meas_indices, r_cov = _build_measurement_vectors(ppci)

        # number of nodes
        n_active = len(np.where(ppci["bus"][:, 1] != 4)[0])
        slack_buses = np.where(ppci["bus"][:, 1] == 3)[0]

        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * n_active - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * n_active - 1))
            raise UserWarning("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * n_active - 1))

        # set the starting values for all active buses
        v_m = ppci["bus"][:, 7]
        delta = ppci["bus"][:, 8] * np.pi / 180  # convert to rad
        delta_masked = np.ma.array(delta, mask=False)
        delta_masked.mask[slack_buses] = True
        non_slack_buses = np.arange(len(delta))[~delta_masked.mask]

        # invert covariance matrix
        r_inv = csr_matrix(np.linalg.inv(np.diagflat(r_cov) ** 2))
        return slack_buses, non_slack_buses,  n_active, r_inv, v_m, delta_masked, delta, z

    def check_result(self, current_error, cur_it):
        # print output for results
        if current_error <= self.tolerance:
            self.successful = True
            self.logger.debug("WLS State Estimation successful ({:d} iterations)".format(cur_it))
        else:
            self.successful = False
            self.logger.debug("WLS State Estimation not successful ({:d}/{:d} iterations)".format(cur_it,
                                                                                                  self.max_iterations))

    def estimate(self, ppci):
        slack_buses, non_slack_buses, n_active, r_inv, v_m, delta_masked, delta, z = self.wls_preprocessing(ppci)

        # state vector
        E = np.concatenate((delta_masked.compressed(), v_m))
        # matrix calculation object
        sem = WLSAlgebra(ppci, slack_buses, non_slack_buses)

        current_error = 100.
        cur_it = 0
        G_m, r, H, h_x = None, None, None, None

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # create h(x) for the current iteration
                h_x = sem.create_hx(v_m, delta)

                # residual r
                r = csr_matrix(z - h_x).T

                # jacobian matrix H
                H = csr_matrix(sem.create_jacobian(v_m, delta))

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))
                E += d_E

                # update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = np.squeeze(E[len(non_slack_buses):])

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
        return delta, v_m


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

        slack_buses, non_slack_buses, n_active, r_inv, v_m, delta_masked, delta, z = self.wls_preprocessing(ppci)

        E = np.concatenate((delta_masked.compressed(), v_m, new_states))
        # matrix calculation object
        sem = WLSAlgebraZeroInjectionConstraints(ppci, slack_buses, non_slack_buses)

        current_error = 100.
        cur_it = 0
        G_m, r, H, h_x = None, None, None, None

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # create h(x) for the current iteration
                h_x, c_x = sem.create_hx_cx(v_m, delta, p_zero_injections, q_zero_injections)

                # residual r
                r = csr_matrix(z - h_x).T
                c_rxh = csr_matrix(c_x).T

                # jacobian matrix H
                H_temp, C_temp = sem.create_jacobian(v_m, delta, p_zero_injections, q_zero_injections)
                H = csr_matrix(H_temp)
                C = csr_matrix(C_temp)

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
                E += d_E

                # update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = np.squeeze(E[len(non_slack_buses):len(non_slack_buses) + n_active])

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E[:len(non_slack_buses) + n_active]))
                self.logger.debug("Current error: {:.7f}".format(current_error))

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return delta, v_m
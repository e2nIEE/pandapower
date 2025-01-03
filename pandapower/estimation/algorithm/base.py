# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve, norm, inv

from pandapower.estimation.algorithm.estimator import BaseEstimatorIRWLS, get_estimator
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra, \
    BaseAlgebraZeroInjConstraints
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.pypower.idx_bus import bus_cols

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)

__all__ = ["WLSAlgorithm", "WLSZeroInjectionConstraintsAlgorithm", "IRWLSAlgorithm"]


class BaseAlgorithm:
    def __init__(self, tolerance, maximum_iterations, logger=std_logger):
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        self.logger = logger
        self.successful = False
        self.iterations = None

        # Parameters for estimate
        self.eppci = None
        self.pp_meas_indices = None

    def check_observability(self, eppci: ExtendedPPCI, z):
        # Check if observability criterion is fulfilled and the state estimation is possible
        num_slacks = sum(~eppci.non_slack_bus_mask)
        if len(z) < 2 * eppci["bus"].shape[0] - num_slacks:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * eppci["bus"].shape[0] - 1))
            raise UserWarning("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * eppci["bus"].shape[0] - 1))

    def check_result(self, current_error, cur_it):
        # print output for results
        if current_error <= self.tolerance:
            self.successful = True
            self.logger.debug("State Estimation successful ({:d} iterations)".format(cur_it))
        else:
            self.successful = False
            self.logger.debug("State Estimation not successful ({:d}/{:d} iterations)".format(cur_it,
                                                                                              self.max_iterations))

    def initialize(self, eppci: ExtendedPPCI):
        # Check observability
        self.eppci = eppci
        self.pp_meas_indices = eppci.pp_meas_indices
        self.check_observability(eppci, eppci.z)

    def estimate(self, eppci: ExtendedPPCI, **kwargs):
        # Must be implemented individually!!
        pass


class WLSAlgorithm(BaseAlgorithm):
    def __init__(self, tolerance, maximum_iterations, logger=std_logger):
        super(WLSAlgorithm, self).__init__(tolerance, maximum_iterations, logger)

        # Parameters for Bad data detection
        self.R_inv = None
        self.Gm = None
        self.r = None
        self.H = None
        self.hx = None
        self.iterations = None
        self.obj_func = None
        logging.basicConfig(level=logging.DEBUG)

    def estimate(self, eppci: ExtendedPPCI, **kwargs):
        self.initialize(eppci)
        # matrix calculation object
        sem = BaseAlgebra(eppci)

        current_error, cur_it = 100., 0
        # invert covariance matrix
        eppci.r_cov[eppci.r_cov<(10**(-5))] = 10**(-5)
        r_inv = csr_matrix(np.diagflat(1 / eppci.r_cov ** 2))
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            # self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                # remove current magnitude measurements at the first iteration 
                # because with flat start they have null derivative
                if cur_it == 0 and eppci.any_i_meas:
                    idx = eppci.idx_non_imeas
                    r_inv = r_inv[idx,:][:,idx]
                    r = r[idx,:]
                    H = H[idx,:]

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)
                norm_G = norm(G_m, np.inf)
                norm_invG = norm(inv(G_m), np.inf)
                cond = norm_G*norm_invG
                if cond > 10**18:
                    self.logger.warning("WARNING: Gain matrix is ill-conditioned: {:.2E}".format(cond))

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))

                # Scaling of Delta_X to avoid divergence due o ill-conditioning and 
                # operating conditions far from starting state variables
                current_error = np.max(np.abs(d_E))
                if current_error > 0.35:
                    d_E = d_E*0.35/current_error

                # Update E with d_E
                E += d_E.ravel()
                eppci.update_E(E)

                # log data 
                # obj_func = (r.T*r_inv*r)[0,0]
                # self.logger.debug("Current delta_x: {:.7f}".format(current_error))
                # self.logger.debug("Current objective function value: {:.1f}".format(obj_func))

                # Restore full weighting matrix with current measurements
                if cur_it == 0 and eppci.any_i_meas:
                    r_inv = csr_matrix(np.diagflat(1 / eppci.r_cov ** 2))

                # prepare next iteration
                cur_it += 1

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        self.iterations = cur_it
        # self.obj_func = obj_func
        if self.successful:
            # store variables required for chi^2 and r_N_max test:
            self.R_inv = r_inv.toarray()
            self.Gm = G_m.toarray()
            self.r = r.toarray()
            self.H = H.toarray()
            # create h(x) for the current iteration
            self.hx = sem.create_hx(eppci.E)
        return eppci


class WLSZeroInjectionConstraintsAlgorithm(BaseAlgorithm):
    def estimate(self, eppci: ExtendedPPCI, **kwargs):
        # state vector built from delta, |V| and zero injections
        # Find pq bus with zero p,q and shunt admittance
        if not np.any(eppci["bus"][:, bus_cols + ZERO_INJ_FLAG]):
            raise UserWarning("Network has no bus with zero injections! Please use WLS instead!")
        zero_injection_bus = np.argwhere(eppci["bus"][:, bus_cols + ZERO_INJ_FLAG]).ravel()
        eppci["bus"][np.ix_(zero_injection_bus, [bus_cols + P, bus_cols + P_STD, bus_cols + Q, bus_cols + Q_STD])] = np.nan
        # Withn pq buses with zero injection identify those who have also no p or q measurement
        p_zero_injections = zero_injection_bus
        q_zero_injections = zero_injection_bus
        new_states = np.zeros(len(p_zero_injections) + len(q_zero_injections))

        num_bus = eppci["bus"].shape[0]

        # matrix calculation object
        sem = BaseAlgebraZeroInjConstraints(eppci)

        current_error, cur_it = 100., 0
        r_inv = csr_matrix((np.diagflat(1 / eppci.r_cov) ** 2))
        E = eppci.E
        # update the E matrix
        E_ext = np.r_[eppci.E, new_states]

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                c_x = sem.create_cx(E, p_zero_injections, q_zero_injections)

                # residual r
                r = csr_matrix(sem.create_rx(E)).T
                c_rxh = csr_matrix(c_x).T

                # jacobian matrix H
                H_temp = sem.create_hx_jacobian(E)
                C_temp = sem.create_cx_jacobian(E, p_zero_injections, q_zero_injections)
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

                # state vector difference d_E and update E
                d_E_ext = spsolve(M_tx, C_rhs)
                E_ext += d_E_ext.ravel()
                E = E_ext[:E.shape[0]]
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E_ext[:len(eppci.non_slack_buses) + num_bus]))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return eppci


class IRWLSAlgorithm(BaseAlgorithm):
    def estimate(self, eppci: ExtendedPPCI, estimator="wls", **kwargs):
        self.initialize(eppci)

        # matrix calculation object
        sem = get_estimator(BaseEstimatorIRWLS, estimator)(eppci, **kwargs)

        current_error, cur_it = 100., 0
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                # gain matrix G_m
                # G_m = H^t * Phi * H
                phi = csr_matrix(sem.create_phi(E))
                G_m = H.T * (phi * H)

                # state vector difference d_E and update E
                d_E = spsolve(G_m, H.T * (phi * r))
                E += d_E.ravel()
                eppci.update_E(E)

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
        return eppci


class AFWLSAlgorithm(BaseAlgorithm):
    def __init__(self, tolerance, maximum_iterations, logger=std_logger):
        super(AFWLSAlgorithm, self).__init__(tolerance, maximum_iterations, logger)

        # Parameters for Bad data detection
        self.R_inv = None
        self.Gm = None
        self.r = None
        self.H = None
        self.hx = None
        self.iterations = None
        self.obj_func = None
        logging.basicConfig(level=logging.DEBUG)

    def estimate(self, eppci: ExtendedPPCI, **kwargs):
        self.initialize(eppci)
        # matrix calculation object
        sem = BaseAlgebra(eppci)

        current_error, cur_it = 100., 0
        # invert covariance matrix
        eppci.r_cov[eppci.r_cov<(10**(-5))] = 10**(-5)
        r_inv = csr_matrix(np.diagflat(1 / eppci.r_cov ** 2))
        E = eppci.E
        num_clusters = len(self.eppci["clusters"])
        while current_error > self.tolerance and cur_it < self.max_iterations:
            # self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                if cur_it == 0 and eppci.any_i_meas:
                    idx = eppci.idx_non_imeas
                    r_inv = r_inv[idx,:][:,idx]
                    r = r[idx,:]
                    H = H[idx,:]

                # gain matrix G_m
                G_m = H.T * (r_inv * H)
                norm_G = norm(G_m, np.inf)
                norm_invG = norm(inv(G_m), np.inf)
                cond = norm_G*norm_invG
                if cond > 10**18:
                    self.logger.warning("WARNING: Gain matrix is ill-conditioned: {:.2E}".format(cond))

                # state vector difference d_E
                d_E = spsolve(G_m, H.T * (r_inv * r))

                # Update E with d_E
                E += d_E.ravel()

                # log data 
                current_error = np.max(np.abs(d_E))
                # obj_func = (r.T*r_inv*r)[0,0]
                # self.logger.debug("Current delta_x: {:.7f}".format(current_error))
                # self.logger.debug("Current objective function value: {:.1f}".format(obj_func))

                # Restore full weighting matrix
                if cur_it == 0 and eppci.any_i_meas:
                    r_inv = csr_matrix(np.diagflat(1 / eppci.r_cov ** 2))

                # prepare next iteration
                cur_it += 1

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        self.iterations = cur_it
        # self.obj_func = obj_func
        if self.successful:
            # store variables required for chi^2 and r_N_max test:
            self.R_inv = r_inv.toarray()
            self.Gm = G_m.toarray()
            self.r = r.toarray()
            self.H = H.toarray()
            # split voltage and allocation factor variables
            E1 = E[:-num_clusters]
            E2 = E[-num_clusters:]
            eppci.update_E(E1)
            eppci.clusters = E2
        return eppci
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve, inv
from scipy.optimize import minimize

from pandapower.idx_bus import BUS_TYPE, VA, VM
from pandapower.estimation.ppc_conversions import _build_measurement_vectors

from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebraOptimization
from pandapower.estimation.estimator.wls import WLSEstimator
from pandapower.estimation.estimator.robust_matrix_ops import QCRobustAlgebraOptimization, SHGMAlgebra

#class QCEstimator(WLSEstimator):
#    def estimate(self, ppci, **hyperparameter):
#        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask = self.wls_preprocessing(ppci)
#        sigma = r_cov
#
#        # matrix calculation object
##        sem = WLSAlgebraOptimization(ppci, non_nan_meas_mask, z, sigma)
#        sem = QCRobustAlgebraOptimization(ppci, non_nan_meas_mask, z, sigma)
#        sem.initialize(a = 3)
#
##        current_error, cur_it = 100., 0
##        G_m, r, H, h_x = None, None, None, None
#        
##        res = minimize(sem.object_func, E, jac=sem.create_jac, tol=1e-8, method='Newton-CG')
#        res = minimize(sem.object_func, E, jac=sem.create_jac, tol=1e-8, method='Nelder-Mead')
##        print(res)
#        E = res.x
#        delta[non_slack_buses] = E[:len(non_slack_buses)]
#        v_m = E[len(non_slack_buses):]
#        V = v_m * np.exp(1j * delta)
#        return V
    
class SHGMEstimator(WLSEstimator):
    pass

#class QCEstimator(WLSEstimator):
#    def estimate(self, ppci, **hyperparameter):
#        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask = self.wls_preprocessing(ppci)
#
#        # matrix calculation object
#        sem = QCRobustAlgebra(ppci, non_nan_meas_mask)
#        sem.initialize(**hyperparameter)
#        sigma = r_cov
#
#        current_error, cur_it = 100., 0
#        G_m, r, H, h_x = None, None, None, None
#
#        while current_error > self.tolerance and cur_it < self.max_iterations:
#            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
#            try:
#                # create h(x) for the current iteration
#                h_x = sem.create_hx(v_m, delta)
#
#                # residual r
#                r = csr_matrix(z - h_x).T
#
#                # jacobian matrix H
#                H = csr_matrix(sem.create_hx_jacobian(v_m, delta))
#
#                # gain matrix G_m
#                # G_m = H^t * R^-1 * H
#                inv_ypsilon_jac_diag = inv(csr_matrix(np.diagflat(sem.create_ypsilon_jacobian(r, sigma))))
#                G_m = H.T * (inv_ypsilon_jac_diag * H)
#
#                # state vector difference d_E
#                # d_E = G_m^-1 * (H' * R^-1 * r)
#                d_E = spsolve(G_m, H.T * sem.create_ypsilon(r, sigma))
#                E += d_E.ravel()
#
#                # update V/delta
#                delta[non_slack_buses] = E[:len(non_slack_buses)]
#                v_m = E[len(non_slack_buses):]
#
#                # prepare next iteration
#                cur_it += 1
#                current_error = np.max(np.abs(d_E))
##                print(cur_it, current_error)
#                self.logger.debug("Current error: {:.7f}".format(current_error))
#
#            except np.linalg.linalg.LinAlgError:
#                self.logger.error("A problem appeared while using the linear algebra methods."
#                                  "Check and change the measurement set.")
#                return False
#
#        # check if the estimation is successfull
#        self.check_result(current_error, cur_it)
#        V = v_m * np.exp(1j * delta)
#        return V
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.estimation.ppc_conversions import _build_measurement_vectors
from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraZeroInjectionConstraints
#from pandapower.estimation.estimator.wls import WLSEstimator
from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_TYPE

try:
    from pandapower.pf.makeYbus import makeYbus
except ImportError:
    from pandapower.pf.makeYbus_pypower import makeYbus


#class MAlgebra(WLSAlgebra):
#    def ypsilon(self):
#        pass
#    
#    def delta_ypsilon(self):
#        pass
#
#class MEstimator(WLSEstimator):
#    def estimate(self, ppci):
#        slack_buses, non_slack_buses, n_active, r_inv, v_m, delta_masked, delta, z = self.wls_preprocessing(ppci)
#
#        # state vector
#        E = np.concatenate((delta_masked.compressed(), v_m))
#        # matrix calculation object
#        sem = WLSAlgebra(ppci, slack_buses, non_slack_buses)
#
#        current_error = 100.
#        cur_it = 0
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
#                H = csr_matrix(sem.create_jacobian(v_m, delta))
#
#                # gain matrix G_m
#                # G_m = H^t * R^-1 * H
#                G_m = H.T * (r_inv * H)
#
#                # state vector difference d_E
#                # d_E = G_m^-1 * (H' * R^-1 * r)
#                d_E = spsolve(G_m, H.T * (r_inv * r))
#                E += d_E
#
#                # update V/delta
#                delta[non_slack_buses] = E[:len(non_slack_buses)]
#                v_m = np.squeeze(E[len(non_slack_buses):])
#
#                # prepare next iteration
#                cur_it += 1
#                current_error = np.max(np.abs(d_E))
#                self.logger.debug("Current error: {:.7f}".format(current_error))
#
#            except np.linalg.linalg.LinAlgError:
#                self.logger.error("A problem appeared while using the linear algebra methods."
#                                  "Check and change the measurement set.")
#                return False
#
#        # check if the estimation is successfull
#        self.check_result(current_error, cur_it)
#        return delta, v_m
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.pypower.idx_bus import BUS_TYPE, VA, VM, bus_cols
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.algorithm.matrix_irwls import WLSEstimatorIRWLS, SHGMEstimatorIRWLS
from pandapower.estimation.algorithm.wls import WLSAlgorithm

ESTIMATOR_MAPPING = {'wls': WLSEstimatorIRWLS,
                     'shgm': SHGMEstimatorIRWLS}


class IRWLSAlgorithm(WLSAlgorithm):
    def estimate(self, ppci, estimator="shgm", **kwargs):
        e_ppci = self.initialize(ppci)

        # matrix calculation object
        wls_sem = WLSEstimatorIRWLS(e_ppci, **kwargs)
        sem = ESTIMATOR_MAPPING[estimator.lower()](e_ppci, **kwargs)

        current_error, cur_it = 100., 0
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(self.E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(self.E))

                # gain matrix G_m
                # G_m = H^t * Phi * H
                if current_error < 1e-5:
                    phi = csr_matrix(sem.create_phi(self.E))
                else:
                    phi = csr_matrix(wls_sem.create_phi(self.E))
                G_m = H.T * (phi * H)

                # state vector difference d_E
                d_E = spsolve(G_m, H.T * (phi * r))
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
        return self.v * np.exp(1j * self.delta)
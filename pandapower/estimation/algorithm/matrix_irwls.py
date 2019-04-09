# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from numba import jit
from scipy.stats import chisquare

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.ppc_conversion import ExtendedPPCI


class BaseEstimatorIRWLS(BaseAlgebra):
    def __init__(self, e_ppci: ExtendedPPCI, **hyperparameters):
        # Initilize BaseAlgebra object for calculation of relevant matrix
        super(BaseEstimatorIRWLS, self).__init__(e_ppci)
        self.sigma = self.e_ppci.r_cov

    def create_phi(self, E):
        # Must be implemented!
        pass


class WLSEstimatorIRWLS(BaseEstimatorIRWLS):
    def create_phi(self, E):
        # Standard WLS does not update this matrix
        return np.diagflat(1/self.sigma**2)


class SHGMEstimatorIRWLS(BaseEstimatorIRWLS):
    def __init__(self, e_ppci: ExtendedPPCI, **hyperparameters):
        super(SHGMEstimatorIRWLS, self).__init__(e_ppci, **hyperparameters)
        assert 'a' in hyperparameters
        self.a = hyperparameters.get('a')

    def create_phi(self, E):
        r = self.create_rx(E)
        w = self.weight(E)
        phi = 1/(self.sigma**2)
        phi[np.abs(r/(self.sigma * w))>self.a] = (a * w) * np.sign(r) / (r * self.sigma) 
        return np.diagflat(phi)

    def weight(self, E):
        H = self.create_hx(E)
        v = np.sum(np.nonzero(H))
        p = 0.975
        chi2 = None
        ps = self._ps(H)

        return np.min(np.c_[(chi2/ps)**2, np.ones(ps.shape)])

    def _ps(self, H):
        omega = H @ H.T

        x = np.zeros(omega.shape[0]-1)
        y = np.zeros(omega.shape[0])
        sm = np.zeros(omega.shape[0])
        lomed_ix = (omega.shape[0]) // 2
        lomed_ix_x = (omega.shape[0]-1) // 2 

        @jit
        def closure(omega, x, y, sm, lomed_ix, lomed_ix_x, ps):
            for k in range(omega.shape[0]):
                for i in range(omega.shape[0]):
                    for j in range(omega.shape[0]):
                        if j != i:
                            x_ix = j if j < i else j-1
                            x[x_ix] = np.abs(omega[i, k]+omega[j, k])
                    y[i] = np.sort(x)[lomed_ix_x]
                sm[k] = np.sort(y)[lomed_ix] * 1.1926

            for i in range(omega.shape[0]):
                ps[i] = np.max(np.abs(omega[i, :]/sm))
            return ps
        
        ps = closure(omega, x, y, sm, lomed_ix, lomed_ix_x, ps=np.zeros(omega.shape[0]))
        return ps
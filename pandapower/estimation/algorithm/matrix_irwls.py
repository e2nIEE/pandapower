# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from numba import jit
from scipy.stats import chi2

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
        condition_mask = np.abs(r/(self.sigma * w))>self.a
        phi[condition_mask] = ((self.a * w) * np.sign(r) / (r * self.sigma))[condition_mask] 
        return np.diagflat(phi)


    def weight(self, E):
        H = self.create_hx_jacobian(E)
        v = np.sum(H != 0, axis=1)
        chi2_res = chi2.ppf(0.975, v)
        ps = self._ps(H)
        return np.min(np.c_[(chi2_res/ps)**2, np.ones(ps.shape)], axis=1)

    def _ps(self, H):
        omega = H @ H.T

        x = np.zeros(omega.shape[0]-1)
        y = np.zeros(omega.shape[0])
        sm = np.zeros(omega.shape[0])
        lomed_ix = (omega.shape[0]) // 2
        lomed_ix_x = (omega.shape[0]-1) // 2
        
        @jit
        def closure(omega, x, y, sm, lomed_ix, lomed_ix_x,sm_not_null_mask, ps):
            for k in range(omega.shape[0]):
                for i in range(omega.shape[0]):
                    for j in range(omega.shape[0]):
                        if j != i:
                            x_ix = j if j < i else j-1
                            x[x_ix] = np.abs(omega[i, k]+omega[j, k])
                    y[i] = np.sort(x)[lomed_ix_x]
                sm[k] = np.sort(y)[lomed_ix] * 1.1926
            
            sm_not_null_mask = (sm != 0)
            for i in range(omega.shape[0]):
#                ps[i] = np.max(np.abs(omega[i, :][sm_not_null_mask]/sm[sm_not_null_mask]))
                ps[i] = np.max(omega[i, :][sm_not_null_mask]/sm[sm_not_null_mask])
            return ps
        
        ps = closure(omega, x, y, sm, lomed_ix, lomed_ix_x, 
                     sm_not_null_mask=np.zeros(omega.shape[0]),
                     ps=np.zeros(omega.shape[0]))
        return ps
    

class QLEstimatorIRWLS(BaseEstimatorIRWLS):
    def __init__(self, e_ppci: ExtendedPPCI, **hyperparameters):
        super(QLEstimatorIRWLS, self).__init__(e_ppci, **hyperparameters)
        assert 'a' in hyperparameters
        self.a = hyperparameters.get('a')

    def create_phi(self, E):
        r = self.create_rx(E)

        phi = 2/(self.sigma**2)
        condition_mask = np.abs(r/(self.sigma))>self.a
        phi[condition_mask] = (2 * self.a * self.sigma * np.sign(r)) [condition_mask]
        return np.diagflat(phi)
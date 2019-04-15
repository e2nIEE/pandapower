# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from numba import jit
from scipy.stats import chi2

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.ppc_conversion import ExtendedPPCI

from statistics import median_low, median_high


class BaseEstimatorIRWLS(BaseAlgebra):
    def __init__(self, eppci: ExtendedPPCI, **hyperparameters):
        # Initilize BaseAlgebra object for calculation of relevant matrix
        super(BaseEstimatorIRWLS, self).__init__(eppci)
        self.sigma = self.eppci.r_cov

    def create_phi(self, E):
        # Must be implemented!
        pass


class WLSEstimatorIRWLS(BaseEstimatorIRWLS):
    def create_phi(self, E):
        # Standard WLS does not update this matrix
        return np.diagflat(1/self.sigma**2)


class SHGMEstimatorIRWLS(BaseEstimatorIRWLS):
    # Still need test!
    def __init__(self, eppci: ExtendedPPCI, **hyperparameters):
        super(SHGMEstimatorIRWLS, self).__init__(eppci, **hyperparameters)
        assert 'a' in hyperparameters
        self.a = hyperparameters.get('a')

    def create_phi(self, E):
        r = self.create_rx(E)
#        chi2_res, w = self.weight(E)
        w = 1
        phi = 1/(self.sigma**2)
        condition_mask = np.abs(r/(self.sigma * w))>self.a
        phi[condition_mask] = (1/(self.sigma**2) * np.abs(self.a * self.sigma * w / r))[condition_mask] 
        return np.diagflat(phi)

    def weight(self, E):
        H = self.create_hx_jacobian(E)
        v = np.sum(H != 0, axis=1)
        chi2_res = chi2.ppf(0.975, v)
        ps = self._ps(H) * 1.005
        return chi2_res, np.min(np.c_[(chi2_res/ps)**2, np.ones(ps.shape)], axis=1)
    
    def _ps(self, H):
        omega = H @ H.T

        x = np.zeros(omega.shape[0]-1)
        y = np.zeros(omega.shape[0])
        sm = np.zeros(omega.shape[0])

        @jit
        def closure(omega, x, y, sm, ps):
            x_shape = x.shape[0]
            y_shape = y.shape[0]
            for k in range(omega.shape[0]):
                for i in range(omega.shape[0]):
                    for j in range(omega.shape[0]):
                        if j != i:
                            x_ix = j if j < i else j-1
                            x[x_ix] = np.abs(omega[i, k]+omega[j, k])
                    count_0 = np.sum(x==0)
                    y[i] = np.sort(x)[count_0 + (x_shape - count_0 + 1)//2 - 1]
                sm[k] = np.sort(y)[(y_shape + 1)//2 - 1] * 1.1926

            for i in range(omega.shape[0]):
                ps[i] = np.max(np.abs(omega[i, :])/sm)
            return ps

        ps = closure(omega, x, y, sm,
                     ps=np.zeros(omega.shape[0]))
        return ps


class QLEstimatorIRWLS(BaseEstimatorIRWLS):
    # Still need test!
    def __init__(self, eppci: ExtendedPPCI, **hyperparameters):
        super(QLEstimatorIRWLS, self).__init__(eppci, **hyperparameters)
        assert 'a' in hyperparameters
        self.a = hyperparameters.get('a')

    def create_phi(self, E):
        r = self.create_rx(E)
        phi = 1/(self.sigma**2)
        condition_mask = np.abs(r/self.sigma)>self.a
        phi[condition_mask] = (1/(self.sigma**2) * np.abs(self.a * self.sigma / r))[condition_mask] 
        return np.diagflat(phi)


 
class QCEstimatorIRWLS(BaseEstimatorIRWLS):
    # Still need test!
    def __init__(self, eppci: ExtendedPPCI, **hyperparameters):
        super(QCEstimatorIRWLS, self).__init__(eppci, **hyperparameters)
        assert 'a' in hyperparameters
        self.a = hyperparameters.get('a')

    def create_phi(self, E):
        r = self.create_rx(E)

        phi = 2/(self.sigma**2)
        condition_mask = np.abs(r/self.sigma)>self.a
        phi[condition_mask] = 0
        return np.diagflat(phi)
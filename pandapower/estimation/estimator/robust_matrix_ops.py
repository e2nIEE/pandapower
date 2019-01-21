# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:56:17 2019

@author: e2n035
"""

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack

from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraOptimization
#from pandapower.estimation.estimator.wls import WLSEstimator
from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_TYPE


class RobustAlgebra(WLSAlgebra):
    def initialize(self, a):
        self.a = a

    def create_rhox(self, ):
        pass

    def create_ypsilon(self, ):
        pass

    def create_ypsilon_jacobian(self, ):
        pass


class QCRobustAlgebra(RobustAlgebra):   
    def create_rhox(self, r, sigma):
        rhox = (r**2)/(sigma**2)
        
        otherwise_mask = (np.abs(r/sigma) > self.a)
        rhox[otherwise_mask] = (self.a)**2/(sigma**2)
        return rhox
    
    def create_ypsilon(self, r, sigma):
        ypsilon = (2*r)/(sigma**2)
        
        otherwise_mask = (np.abs(r/sigma) > self.a)
        ypsilon[otherwise_mask] = 0
        return ypsilon
    
    def create_ypsilon_jacobian(self, r, sigma):
        ypsilon = (2*r)/(sigma**2)
        
        otherwise_mask = (np.abs(r/sigma) > self.a)
        ypsilon[otherwise_mask] = 0
        return ypsilon


class QCRobustAlgebraOptimization(WLSAlgebraOptimization, RobustAlgebra):
    def object_func(self, E):
        return np.sum(self.cost_function(E))
    
    def cost_function(self, E):
        rx = self.create_rx(E)
        rhox = (rx**2)/(self.sigma**2)
        
        otherwise_mask = (np.abs(rx/self.sigma) > self.a)
        rhox[otherwise_mask] = (self.a)**2/(self.sigma[otherwise_mask]**2)
        return rhox
     
    

        

        
    
    


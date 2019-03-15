# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:56:17 2019

@author: e2n035
"""

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.stats import chi2

from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraOptimization
#from pandapower.estimation.estimator.wls import WLSEstimator
from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_TYPE


class RobustAlgebra(WLSAlgebra):
    def initialize(self, a):
        self.a = a

    def create_rhox(self, ):
        pass

    def create_upsilon(self, ):
        pass

    def create_upsilon_jacobian(self, ):
        pass
    

class SHGMAlgebra(RobustAlgebra):
    def create_rhox(self, r, sigma, omega):
        pass
    
    
    def create_upsilon(self, r, sigma, w):
        upsilon = r / (sigma**2)
        
        otherwise_mask = np.abs(r/(sigma * w)) > self.a
        upsilon[otherwise_mask] = ((self.a * w / sigma) * np.sign(r))[otherwise_mask]
        return upsilon
    
    def create_phi(self, r, sigma, w):
        return self.create_upsilon(r, sigma, w) / r
    
    def create_w(self, r, sigma):
        i = None
        H = self.create_hx_jacobian(v, delta)
        v = np.count_nonzero(H[i])
        chi2 = chi2.ppf(p, v)
        
        pass
    


#class QCRobustAlgebra(RobustAlgebra):   
#    def create_rhox(self, r, sigma):
#        rhox = (r**2)/(sigma**2)
#        
#        otherwise_mask = (np.abs(r/sigma) > self.a)
#        rhox[otherwise_mask] = (self.a)**2/(sigma**2)
#        return rhox
#    
#    def create_upsilon(self, r, sigma):
#        upsilon = (2*r)/(sigma**2)
#        
#        otherwise_mask = (np.abs(r/sigma) > self.a)
#        upsilon[otherwise_mask] = 0
#        return upsilon
#    
#    def create_upsilon_jacobian(self, r, sigma):
#        upsilon = (2*r)/(sigma**2)
#        
#        otherwise_mask = (np.abs(r/sigma) > self.a)
#        upsilon[otherwise_mask] = 0
#        return upsilon
#
#
#class QCRobustAlgebraOptimization(WLSAlgebraOptimization, RobustAlgebra):
#    def object_func(self, E):
#        return np.sum(self.cost_function(E))
#    
#    def cost_function(self, E):
#        rx = self.create_rx(E)
#        rhox = (rx**2)/(self.sigma**2)
#        
#        otherwise_mask = (np.abs(rx/self.sigma) > self.a)
#        rhox[otherwise_mask] = (self.a)**2/(self.sigma[otherwise_mask]**2)
#        return rhox
     
        
if __name__ == '__main__':
    from scipy.spatial.distance import mahalanobis
    from scipy.linalg import inv
    
#    H = np.array([110, -100, 0,
#                  100, -100, 0,
#                  10, 0, 0,
#                  0, 10, 0,
#                  0, -10, 20,
#                  0, -10, 10,
#                  0, 0, 10,
#                  -10, -10, -10,
#                  -10, 0, 0]).reshape((9,3))
    H = np.array([10, 1, -1, 0, 0, 11, -1, -10, 0, 0, -1, 1, -10, -1]).reshape((2, 7)).T
#    H_msk = H <= 0
#    K = H @ inv(H.T @ H) @ H.T
#    n = np.trace(K)
#    K_mask = K > 2 * 3 / 9
    
#    H = np.array([10, 1, -1, 0, 0, 11, -1, -10, 0, 0, -1, 1, -10, -1]).reshape((2, 7)).T
    
#    H = H @ inv(H.T @ H) @ H.T
    
#    H_sub = H[[1,2,3,4,6], :]
#    h = np.mean(H_sub, axis=0)

#    h = np.median(H, axis=0)
#    c = H.T @ H * (1/9)
#    c = np.sum([(H[i, :] - h).reshape(-1,1) @ (H[i, :] - h).reshape(-1,1).T for i in range(H.shape[0])], axis=0) / (H.shape[0]-1)
#    c = np.cov(H, rowvar=False)
    c = 1/(H.shape[0]-1) * H.T @ H
    
#    mahalanobis(H[1, :], h, inv(c))
#    h = np.median(H, axis=0)
#    md = np.sqrt(np.array([(H[i, :] - h).reshape(-1,1).T @ np.linalg.inv(c) @ (H[i, :] - h).reshape(-1,1) for i in range(H.shape[0])])).ravel()
    
    md = np.sqrt(np.array([(H[i, :]).reshape(-1,1).T @\
                           np.linalg.inv(c) @ (H[i, :]).reshape(-1,1) for i in range(H.shape[0])])).ravel()
#    
    i = 0
    ps = np.array([np.max([np.abs(H[i, :].reshape(-1,1).T @ H[k, :].reshape(-1,1)) for k in range(H.shape[0])]) for i in range(H.shape[0])])
    ps1 = np.max([np.abs(np.inner(H[0, :], H[k, :])) for k in range(H.shape[0])])
    ps2 = np.max([np.abs(np.inner(H[1, :], H[k, :])) for k in range(H.shape[0])])
    
#    sm1 = 1.1926 * 
#    i = 0
#    inner = [np.abs(H[i, :].T @ H[k, :] + H[j, :].T @ H[k, :] for k in range(9) for j in range (1,9)]
    inners = [np.quantile([np.abs(H[i, :].T @ H[k, :] + H[j, :].T @ H[k, :]) for k in range(H.shape[0]) for j in range (H.shape[0]) if j != i], 5/9) for i in range(9)] 
    beta = 1.1926 * np.quantile(inners, 0.25)
#    22100 / 8.47
    res = ps/beta
#    res = ps/2609.21
    
    
    
    
    
    

        

        
    
    


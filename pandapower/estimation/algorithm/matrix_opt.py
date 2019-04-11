# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra

__all__ = ["WLSEstimatorOpt", "LAVEstimatorOpt",]



class BaseEstimatorOpt:
    def __init__(self, eppci, **hyperparameters):
        self.base_algebra = BaseAlgebra(eppci)

        self.num_non_slack_bus = eppci.num_non_slack_bus
        self.non_slack_buses = eppci.non_slack_buses

        self.v = eppci.v_init.copy()
        self.delta = eppci.delta_init.copy()
        
        self.z = eppci.z
        self.sigma = eppci.r_cov
        self.jac_available = False
        self.hess_available = False

    def cost_function(self, E):
        # Minimize sum(cost(r))
        # r = cost(z - h(x))
        # Must be implemented according to the estimator for the optimization
        pass

    def create_rx_jacobian(self, E):
        pass


class WLSEstimatorOpt(BaseEstimatorOpt):
    def __init__(self, eppci, **hyperparameters):
        super(WLSEstimatorOpt, self).__init__(eppci, **hyperparameters)
        self.jac_available = True

    def cost_function(self, E):
        rx = self.base_algebra.create_rx(E)
        cost = np.sum((1/self.sigma**2) * (rx**2))
        return cost

    def create_rx_jacobian(self, E):
        # dr/dE = drho / dr * d(z-hx) / dE
        # dr/dE = (drho/dr) * - (d(hx)/dE)
        # 2 * rx * -(dhx/dE)
        rx = self.base_algebra.create_rx(E)
        hx_jac = self.base_algebra.create_hx_jacobian(E)
        jac = - np.sum(2 * rx.reshape((-1, 1)) * hx_jac, axis=0)
        return jac


class LAVEstimatorOpt(BaseEstimatorOpt):
    def __init__(self, eppci,  **hyperparameters):
        super(LAVEstimatorOpt, self).__init__(eppci, **hyperparameters)
        self.jac_available = True

    def cost_function(self, E):
        rx = self.base_algebra.create_rx(E)
        cost = np.sum(np.abs(rx))
        return cost

    def create_rx_jacobian(self, E):
        # dr/dE = drho / dr * d(z-hx) / dE
        # dr/dE = (drho/dr) * - (d(hx)/dE)
        # sign(rx) * -(dhx/dE)
        rx = self.base_algebra.create_rx(E)
        hx_jac = self.base_algebra.create_hx_jacobian(E)
        jac = - np.sum(np.sign(rx.reshape((-1, 1))) * hx_jac, axis=0)
        return jac


#class QCEstimatorOpt(BaseEstimatorOpt):
#    def __init__(self, eppci, **hyperparameters):
#        super(QCEstimatorOpt, self).__init__(eppci, **hyperparameters)
#        assert 'a' in hyperparameters
#        self.a = hyperparameters['a']
#        self.jac_available = True
#
#    def cost_function(self, E):
#        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
#        self.v = E[self.num_non_slack_bus:]
#        rx = self.base_algebra.create_rx(self.v, self.delta)
#        cost = (1/self.sigma**2) * (rx**2)
#        if np.any(np.abs(rx/self.sigma) > self.a):
#            cost[np.abs(rx/self.sigma) > self.a] = (self.a**2 / self.sigma**2)[np.abs(rx/self.sigma) > self.a]
##        print(np.sum(cost))
#        return np.sum(cost)
#
#    def create_rx_jacobian(self, E):
#        # dr/dE = drho / dr * d(z-hx) / dE
#        # dr/dE = (drho/dr) * - (d(hx)/dE)
#        # 2 * rx * -(dhx/dE) if np.abs(rx/sigma) < a
#        # 0 else
#        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
#        self.v = E[self.num_non_slack_bus:]
#        rx = self.base_algebra.create_rx(self.v, self.delta)
#        hx_jac = self.base_algebra.create_hx_jacobian(self.v, self.delta)
#        drho = 2 * rx.reshape((-1, 1))
#        if np.any(np.abs(rx/self.sigma) > self.a):
#            drho[np.abs(rx/self.sigma) > self.a] = 0
#        jac = - np.sum(drho * hx_jac, axis=0)
#        return jac
#
#
#class QLEstimatorOpt(BaseEstimatorOpt):
#    def __init__(self, eppci, **hyperparameters):
#        super(QLEstimatorOpt, self).__init__(eppci, **hyperparameters)     
#        assert 'a' in hyperparameters
#        self.a = hyperparameters['a']
#        self.jac_available = True
#        
#    def cost_function(self, E):
#        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
#        self.v = E[self.num_non_slack_bus:]
#        rx = self.base_algebra.create_rx(self.v, self.delta)
#        cost = (1/self.sigma**2) * (rx**2)
#        if np.any(np.abs(rx/self.sigma) > self.a):
#            cost[np.abs(rx/self.sigma) > self.a] = (2*self.a*self.sigma*np.abs(rx) -\
#                self.a**2 * self.sigma**2)[np.abs(rx/self.sigma) > self.a]
##        print(np.sum(cost))
#        return np.sum(cost)
#
#    def create_rx_jacobian(self, E):
#        # dr/dE = drho / dr * d(z-hx) / dE
#        # dr/dE = (drho/dr) * - (d(hx)/dE)
#        # 2 * rx * -(dhx/dE) if np.abs(rx/sigma) < a
#        # 0 else
#        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
#        self.v = E[self.num_non_slack_bus:]
#        rx = self.base_algebra.create_rx(self.v, self.delta)
#        hx_jac = self.base_algebra.create_hx_jacobian(self.v, self.delta)
#        drho = 2 * rx.reshape((-1, 1))
#        if np.any(np.abs(rx/self.sigma) > self.a):
#            drho[np.abs(rx/self.sigma) > self.a] =\
#                - np.sum(2*self.a*self.sigma*np.sign(rx) * hx_jac, axis=0)[np.abs(rx/self.sigma) > self.a]
#        jac = - np.sum(drho * hx_jac, axis=0)  
#        return jac


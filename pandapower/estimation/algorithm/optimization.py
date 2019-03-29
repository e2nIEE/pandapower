# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.optimize import minimize

from pandapower.estimation.algorithm.matrix_opt import \
    (WLSEstimatorOpt, LAVEstimatorOpt, QCEstimatorOpt, QLEstimatorOpt)
from pandapower.estimation.algorithm.wls import WLSAlgorithm

DEFAULT_OPT_METHOD = "Newton-CG"
ESTIMATOR_MAPPING = {'wls': WLSEstimatorOpt,
                     'lav': LAVEstimatorOpt,
                     'qc': QCEstimatorOpt,
                     'ql': QLEstimatorOpt}


class OptAlgorithm(WLSAlgorithm):
    def estimate(self, ppci, opt_vars=None):
        assert 'estimator' in opt_vars and opt_vars['estimator'] in ESTIMATOR_MAPPING
        opt_method = DEFAULT_OPT_METHOD if 'opt_method' not in opt_vars else opt_vars['opt_method']

        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask =\
            self.wls_preprocessing(ppci)

        # matrix calculation object
        estm = ESTIMATOR_MAPPING[opt_vars['estimator']](ppci, non_nan_meas_mask, 
                                z=z, sigma=r_cov, **opt_vars)

        res = minimize(estm.cost_function, x0=E, 
                       method=opt_method, jac=estm.create_rx_jacobian, 
                       tol=self.tolerance, options={'maxiter':100, 'disp': True})
#        res = minimize(estm.cost_function, x0=E, 
#                       method=opt_method, jac=estm.create_rx_jacobian, 
#                       tol=self.tolerance, options={'disp': True})

        self.successful = res.success
        if self.successful:
            E = res.x
            delta[non_slack_buses] = E[:len(non_slack_buses)]
            v_m = E[len(non_slack_buses):]
            V = v_m * np.exp(1j * delta)
            return V
        else:
            raise Exception("Optimiaztion failed! State Estimation not successful!")
            

#from nevergrad.optimization import optimizerlib    
#
#class OptAlgorithm(WLSAlgorithm):
#    def estimate(self, ppci, opt_vars=None):
#        assert 'estimator' in opt_vars and opt_vars['estimator'] in ESTIMATOR_MAPPING
#        opt_method = DEFAULT_OPT_METHOD if 'opt_method' not in opt_vars else opt_vars['opt_method']
#
#        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask =\
#            self.wls_preprocessing(ppci)
#
#        # matrix calculation object
#        estm = ESTIMATOR_MAPPING[opt_vars['estimator']](ppci, non_nan_meas_mask, 
#                                z=z, sigma=r_cov, **opt_vars)
#   
#        optimizer = optimizerlib.CMA(dimension=E.shape[0], budget=10000)
#        res = optimizer.optimize(estm.cost_function, verbosity=True)
#
#        self.successful = True
#
#        E = res
#        delta[non_slack_buses] = E[:len(non_slack_buses)]
#        v_m = E[len(non_slack_buses):]
#        V = v_m * np.exp(1j * delta)
#        return V
##        else:
##            raise Exception("Optimiaztion failed! State Estimation not successful!")
        
if __name__ == '__main__':
    import pandapower as pp
    from pandapower.estimation.state_estimation import estimate
    
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0,
                                   max_i_ka=1)

    pp.create_measurement(net, "p", "line", -0.0011, 0.01, 0, 0)  # p12
    pp.create_measurement(net, "q", "line", 0.024, 0.01, 0, 0)    # q12

    pp.create_measurement(net, "p", "bus", 0.018, 0.01, 2)  # p3
    pp.create_measurement(net, "q", "bus", -0.1, 0.01, 2)   # q3

    pp.create_measurement(net, "v", "bus", 1.08, 0.05, 0)   # u1
    pp.create_measurement(net, "v", "bus", 1.015, 0.05, 2)  # u3

    # 2. Do state estimation
    estimate(net, init='flat')

    # 2. Do state estimation
    from copy import deepcopy
    net_opt = deepcopy(net)
    success = estimate(net_opt, algorithm='opt', estimator="wls", a=10, tolerance=1e-4)


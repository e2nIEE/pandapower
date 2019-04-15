# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.optimize import minimize

from pandapower.estimation.algorithm.matrix_opt import \
    (WLSEstimatorOpt, LAVEstimatorOpt, QCEstimatorOpt, QLEstimatorOpt)
from pandapower.estimation.algorithm.wls import WLSAlgorithm

DEFAULT_OPT_METHOD = "Newton-CG"
#DEFAULT_OPT_METHOD = "BFGS"
#DEFAULT_OPT_METHOD = 'Nelder-Mead'
ESTIMATOR_MAPPING = {'wls': WLSEstimatorOpt,
                     'lav': LAVEstimatorOpt,
                     'qc': QCEstimatorOpt,
                     'ql': QLEstimatorOpt}


class OptAlgorithm(WLSAlgorithm):
    def estimate(self, eppci, **opt_vars):
        assert 'estimator' in opt_vars and opt_vars['estimator'] in ESTIMATOR_MAPPING
        opt_method = DEFAULT_OPT_METHOD if 'opt_method' not in opt_vars else opt_vars['opt_method']

        # matrix calculation object
        estm = ESTIMATOR_MAPPING[opt_vars['estimator']](eppci, **opt_vars)

        jac = estm.create_rx_jacobian
        res = minimize(estm.cost_function, x0=eppci.E, 
                       method=opt_method, jac=jac, tol=self.tolerance,
                       options={'maxiter':1000, 'disp': True})

        self.successful = res.success
        if self.successful:
            E = res.x
            eppci.update_E(E)
            return eppci
        else:
            raise Exception("Optimiaztion failed! State Estimation not successful!")

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.idx_bus import BUS_TYPE, VA, VM
from pandapower.estimation.ppc_conversions import _build_measurement_vectors

from pandapower.estimation.estimator.wls import WLSEstimator
from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraZeroInjectionConstraints
#from pandapower.estimation.estimator.wls_matrix_ops_ori import WLSAlgebra
from scipy.optimize import minimize, linprog


class LAVEstimator(WLSEstimator):
    def estimate(self, ppci):
        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask = self.wls_preprocessing(ppci)

        # matrix calculation object
        sem = WLSAlgebra(ppci, non_nan_meas_mask)

        current_error, cur_it = 100., 0
        H, h_x = None, None

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # create h(x) for the current iteration
                h_x = sem.create_hx(v_m, delta)

                # residual r
                delta_z = z - h_x
                
                # jacobian matrix H
                H = sem.create_hx_jacobian(v_m, delta)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = new_X(H, E, delta_z)
                E += d_E

                # update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = E[len(non_slack_buses):]

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
#                print(cur_it, current_error)
                self.logger.debug("Current error: {:.7f}".format(current_error))

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        V = v_m * np.exp(1j * delta)
        return V
    
def new_X(H, x, delta_z):
    n, m = H.shape[1], H.shape[0]
    zero_n = np.zeros((n, 1))
    one_m = np.ones((m, 1))
    Im = np.eye(m)
    
    c_T = np.r_[zero_n, zero_n, one_m, one_m]
    A = np.c_[H, -H, Im, -Im]
    
#    res = linprog(c_T.ravel(), A_eq=A, b_eq=delta_z ,bounds=[(0, None) for _ in range(A.shape[1])],
#                  method="interior-point")
    res = linprog(c_T.ravel(), A_eq=A, b_eq=delta_z ,bounds=[(0, None) for _ in range(A.shape[1])],
                  method="simplex")
    if res.success:
        d_x = res.x[:n].ravel() - res.x[n:2*n].ravel()
        return d_x
    else:
        raise np.linalg.linalg.LinAlgError

if __name__ == "__main__":
    from pandapower.estimation import estimate
    import pandapower as pp
    
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
    pp.create_measurement(net, "q", "line", 0.024, 0.01, 0, 0)  # q12

    pp.create_measurement(net, "p", "bus", 0.018, 0.01, 2)  # p3
    pp.create_measurement(net, "q", "bus", -0.1, 0.01, 2)  # q3

    pp.create_measurement(net, "v", "bus", 1.08, 0.05, 0)  # u1
    pp.create_measurement(net, "v", "bus", 1.015, 0.05, 2)  # u3

    # 1. Create false voltage measurement for testing bad data detection (-> should be removed)
    pp.create_measurement(net, "v", "bus", 1.3, 0.01, 1)   # V at bus 2

    # 2. Do state estimation
#    success = estimate(net, init='flat')

    # 2. Do state estimation
    success = estimate(net, init='flat', algorithm='lav')
    
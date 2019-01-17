# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.estimation.ppc_conversions import _build_measurement_vectors
from pandapower.estimation.estimator.wls_matrix_ops import WLSAlgebra, WLSAlgebraZeroInjectionConstraints
#from pandapower.estimation.estimator.wls import WLSEstimator
from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_TYPE

try:
    from pandapower.pf.makeYbus import makeYbus
except ImportError:
    from pandapower.pf.makeYbus_pypower import makeYbus


class WLSAlgebraNew:
    def __init__(self, ppci, slack_buses, non_slack_buses):
        np.seterr(divide='ignore', invalid='ignore')
        self.ppci = ppci
        self.slack_buses = slack_buses
        self.non_slack_buses = non_slack_buses
        self.Y_bus = None
        self.Yf = None
        self.Yt = None
        self.G = None
        self.B = None
        self.G_series = None
        self.B_series = None
        self.G_shunt = None
        self.B_shunt = None
        self.keep_ix = None
        self.i_ij = None
        self.fb = None
        self.tb = None
        self.create_y()

    # Function which builds a node admittance matrix out of the topology data
    # In addition, it provides the series admittances of lines as G_series and B_series
    def create_y(self):
        self.fb = self.ppci["branch"][:, 0].real.astype(int)
        self.tb = self.ppci["branch"][:, 1].real.astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_bus, y_f, y_t = makeYbus(self.ppci["baseMVA"], self.ppci["bus"], self.ppci["branch"])
            self.ppci['internal']['Yf'], self.ppci['internal']['Yt'],\
            self.ppci['internal']['Ybus'] = y_f, y_t, y_bus

        # create relevant matrices
        self.Ybus = y_bus.toarray()
        self.Yf = y_f.toarray()
        self.Yt = y_t.toarray()
        self.G = self.Ybus.real
        self.B = self.Ybus.imag
        n = len(self.ppci["bus"])
        self.G_series = - self.G
        np.fill_diagonal(self.G_series, 0.)
        self.B_series = - self.B
        np.fill_diagonal(self.B_series, 0.)
        # In case that's becoming relevant later, G_shunt will not be removed
        self.G_shunt = np.zeros_like(self.G)
        self.B_shunt = np.zeros((n, n))
        from_to = np.concatenate((self.fb, self.tb))
        to_from = np.concatenate((self.tb, self.fb))
        self.B_shunt[from_to, to_from] = np.tile(0.5 * self.ppci["branch"][:, 4].real, 2)

    # Get Y as tuple (real, imaginary)
    def get_y(self):
        return self.G, self.B
    
    def create_hx(self, v, delta, meas_mask):
        f_bus, t_bus = self.fb, self.tb
        V = v * np.exp(1j * delta)
        Sfe = V[f_bus] * np.conj(np.dot(self.Yf, V))
        Ste = V[t_bus] * np.conj(np.dot(self.Yt, V))
        Sbuse = V * np.conj(np.dot(self.Ybus, V))
        Ife = np.abs(Sfe)**2**(0.5)/v[f_bus]
        Ite = np.abs(Ste)**2**(0.5)/v[t_bus]
        hx = np.r_[np.real(Sbuse),
                   np.real(Sfe), 
                   np.real(Ste),
                   np.imag(Sbuse),
                   np.imag(Sfe),
                   np.imag(Ste),
                   v,
                   Ife,
                   Ite]
        return hx[meas_mask]
    
    def _dSbus_dv(self, V, diagV, diagVnorm):
        diagIbus = np.diagflat(np.dot(self.Ybus, V))
        dSbus_dth = 1j * diagV @ np.conj(diagIbus - np.dot(self.Ybus, diagV))
        dSbus_dv = diagV @ np.conj(np.dot(self.Ybus, diagVnorm)) + np.dot(np.conj(diagIbus), diagVnorm)
        return dSbus_dth, dSbus_dv
        
    def _dSbr_dv(self, V, diagV, diagVnorm):
        n_bus, n_branch = self.ppci['bus'].shape[0], self.ppci['branch'].shape[0]
        f_bus, t_bus = self.fb, self.tb
        If, It = np.dot(self.Yf, V), np.dot(self.Yt, V)
        diagIf, diagIt = np.diagflat(If), np.diagflat(It)
        diagVf, diagVt = np.diagflat(V[f_bus]), np.diagflat(V[t_bus])
        
        dSf_dth = 1j * (np.conj(diagIf) * csr_matrix((V[f_bus], (range(n_branch), f_bus)),shape=(n_branch, n_bus)) -\
                        diagVf @ np.conj(np.dot(self.Yf, diagV)))
        dSf_dv = diagVf @ np.conj(np.dot(self.Yf, diagVnorm)) + np.conj(diagIf) *\
                    csr_matrix((V[f_bus], (range(n_branch), f_bus)),shape=(n_branch, n_bus))
        dSt_dth = 1j * (np.conj(diagIt) * csr_matrix((V[t_bus], (range(n_branch), t_bus)),shape=(n_branch, n_bus)) -\
                        diagVt @ np.conj(np.dot(self.Yt, diagV)))
        dSt_dv = diagVt @ np.conj(np.dot(self.Yt, diagVnorm)) + np.conj(diagIt) *\
                    csr_matrix((V[t_bus], (range(n_branch), t_bus)),shape=(n_branch, n_bus))
        return dSf_dth, dSt_dth, dSf_dv, dSt_dv
    
    def _dVbus_dv(self, V):
        dv_dth, dv_dv = np.diag(np.zeros(V.shape)), np.diag(np.ones(V.shape))
        return dv_dth, dv_dv
    
    def _dibr_dv(self, diagV, diagVnorm):
        dif_dth = np.abs(np.dot(self.Yf, 1j * diagV))
        dif_dv = np.abs(np.dot(self.Yf, diagVnorm))
        dit_dth = np.abs(np.dot(self.Yt, 1j * diagV))
        dit_dv = np.abs(np.dot(self.Yt, diagVnorm))
        return dif_dth, dit_dth, dif_dv, dit_dv

    def create_h_jacobian(self, v, delta, meas_mask):
        V = (v * np.exp(1j * delta))
        diagV, diagVnorm = np.diag(V), np.diag(V/np.abs(V))

        dSbus_dth, dSbus_dv = self._dSbus_dv(V, diagV, diagVnorm)
        dSf_dth, dSt_dth, dSf_dv, dSt_dv = self._dSbr_dv(V, diagV, diagVnorm)
        dif_dth, dit_dth, dif_dv, dit_dv = self._dibr_dv(diagV, diagVnorm)
        dv_dth, dv_dv = self._dVbus_dv(V)
        
        h_jac_th = np.r_[np.real(dSbus_dth),
                         np.real(dSf_dth),
                         np.real(dSt_dth),
                         np.imag(dSbus_dth),
                         np.imag(dSf_dth),
                         np.imag(dSt_dth),
                         dv_dth,
                         dif_dth,
                         dit_dth]
        h_jac_v = np.r_[np.real(dSbus_dv),
                        np.real(dSf_dv),
                        np.real(dSt_dv),
                        np.imag(dSbus_dv),
                        np.imag(dSf_dv),
                        np.imag(dSt_dv),
                        dv_dv,
                        dif_dv,
                        dit_dv]
        h_jac = np.c_[h_jac_th, h_jac_v]
        # TODO CHECK!
        bus_pq = np.r_[(self.ppci['bus'][:, BUS_TYPE] != 3) , np.ones(self.ppci['bus'].shape[0], dtype=bool)].ravel()
        return h_jac[meas_mask, :][:, bus_pq]


#class MAlgebra(WLSAlgebra):
#    def ypsilon(self):
#        pass
#    
#    def delta_ypsilon(self):
#        pass
#
#class MEstimator(WLSEstimator):
#    def estimate(self, ppci):
#        slack_buses, non_slack_buses, n_active, r_inv, v_m, delta_masked, delta, z = self.wls_preprocessing(ppci)
#
#        # state vector
#        E = np.concatenate((delta_masked.compressed(), v_m))
#        # matrix calculation object
#        sem = WLSAlgebra(ppci, slack_buses, non_slack_buses)
#
#        current_error = 100.
#        cur_it = 0
#        G_m, r, H, h_x = None, None, None, None
#
#        while current_error > self.tolerance and cur_it < self.max_iterations:
#            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
#            try:
#                # create h(x) for the current iteration
#                h_x = sem.create_hx(v_m, delta)
#
#                # residual r
#                r = csr_matrix(z - h_x).T
#
#                # jacobian matrix H
#                H = csr_matrix(sem.create_jacobian(v_m, delta))
#
#                # gain matrix G_m
#                # G_m = H^t * R^-1 * H
#                G_m = H.T * (r_inv * H)
#
#                # state vector difference d_E
#                # d_E = G_m^-1 * (H' * R^-1 * r)
#                d_E = spsolve(G_m, H.T * (r_inv * r))
#                E += d_E
#
#                # update V/delta
#                delta[non_slack_buses] = E[:len(non_slack_buses)]
#                v_m = np.squeeze(E[len(non_slack_buses):])
#
#                # prepare next iteration
#                cur_it += 1
#                current_error = np.max(np.abs(d_E))
#                self.logger.debug("Current error: {:.7f}".format(current_error))
#
#            except np.linalg.linalg.LinAlgError:
#                self.logger.error("A problem appeared while using the linear algebra methods."
#                                  "Check and change the measurement set.")
#                return False
#
#        # check if the estimation is successfull
#        self.check_result(current_error, cur_it)
#        return delta, v_m
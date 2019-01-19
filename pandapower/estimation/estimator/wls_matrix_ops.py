# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, diags
from scipy.sparse.linalg import spsolve

from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_TYPE

try:
    from pandapower.pf.makeYbus import makeYbus
except ImportError:
    from pandapower.pf.makeYbus_pypower import makeYbus


class WLSAlgebra:
    def __init__(self, ppci, slack_buses, non_slack_buses):
        np.seterr(divide='ignore', invalid='ignore')
        self.ppci = ppci
        self.fb = self.ppci["branch"][:, 0].real.astype(int)
        self.tb = self.ppci["branch"][:, 1].real.astype(int)
#        self.slack_buses = slack_buses
#        self.non_slack_buses = non_slack_buses
        self.Y_bus = None
        self.Yf = None
        self.Yt = None
        self.G = None
        self.B = None
        self.createY()

    # Function which builds a node admittance matrix out of the topology data
    # In addition, it provides the series admittances of lines as G_series and B_series
    def createY(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus(self.ppci["baseMVA"], self.ppci["bus"], self.ppci["branch"])
            self.ppci['internal']['Yf'], self.ppci['internal']['Yt'],\
            self.ppci['internal']['Ybus'] = Yf, Yt, Ybus

        # create relevant matrices
        self.Ybus, self.Yf, self.Yt = Ybus, Yf, Yt
        self.G = self.Ybus.real
        self.B = self.Ybus.imag

    # Get Y as tuple (real, imaginary)
    def get_y(self):
        return self.G, self.B
    
    def create_hx(self, v, delta, meas_mask):
        f_bus, t_bus = self.fb, self.tb
        V = v * np.exp(1j * delta)
        Sfe = V[f_bus] * np.conj(self.Yf * V)
        Ste = V[t_bus] * np.conj(self.Yt * V)
        Sbuse = V * np.conj(self.Ybus * V)
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
        diagIbus = csr_matrix(np.diagflat(self.Ybus * V))
        dSbus_dth = 1j * diagV * np.conj(diagIbus - self.Ybus * diagV)
        dSbus_dv = diagV * np.conj(self.Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
        return dSbus_dth, dSbus_dv

    def _dSbr_dv(self, V, diagV, diagVnorm):
        n_bus, n_branch = self.ppci['bus'].shape[0], self.ppci['branch'].shape[0]
        f_bus, t_bus = self.fb, self.tb
        If, It = self.Yf * V, self.Yt * V
        diagIf, diagIt = np.diagflat(If), np.diagflat(It)
        diagVf, diagVt = np.diagflat(V[f_bus]), np.diagflat(V[t_bus])
        
        dSf_dth = 1j * (np.conj(diagIf) * csr_matrix((V[f_bus], (range(n_branch), f_bus)),shape=(n_branch, n_bus)) -\
                        diagVf * np.conj(self.Yf * diagV))
        dSf_dv = diagVf * np.conj(self.Yf * diagVnorm) + np.conj(diagIf) *\
                    csr_matrix((V[f_bus], (range(n_branch), f_bus)),shape=(n_branch, n_bus))
        dSt_dth = 1j * (np.conj(diagIt) * csr_matrix((V[t_bus], (range(n_branch), t_bus)),shape=(n_branch, n_bus)) -\
                        diagVt * np.conj(self.Yt * diagV))
        dSt_dv = diagVt * np.conj(self.Yt * diagVnorm) + np.conj(diagIt) *\
                    csr_matrix((V[t_bus], (range(n_branch), t_bus)),shape=(n_branch, n_bus))
        return dSf_dth, dSt_dth, dSf_dv, dSt_dv

    def _dVbus_dv(self, V):
        dv_dth, dv_dv = np.diagflat(np.zeros(V.shape)), np.diagflat(np.ones(V.shape))
        return dv_dth, dv_dv

    def _dibr_dv(self, diagV, diagVnorm):
        dif_dth = np.abs(self.Yf * 1j * diagV)
        dif_dv = np.abs(self.Yf * diagVnorm)
        dit_dth = np.abs(self.Yt * 1j * diagV)
        dit_dv = np.abs(self.Yt * diagVnorm)
        return dif_dth, dit_dth, dif_dv, dit_dv

    def create_h_jacobian(self, v, delta, meas_mask):
        # Using sparse matrix in creation sub-jacobian matrix
        V = v * np.exp(1j * delta)
        diagV, diagVnorm = diags(V).tocsr(), diags(V/np.abs(V)).tocsr()

        dSbus_dth, dSbus_dv = self._dSbus_dv(V, diagV, diagVnorm)
        dSf_dth, dSt_dth, dSf_dv, dSt_dv = self._dSbr_dv(V, diagV, diagVnorm)
        dif_dth, dit_dth, dif_dv, dit_dv = self._dibr_dv(diagV, diagVnorm)
        dv_dth, dv_dv = self._dVbus_dv(V)

        h_jac_th = vstack([np.real(dSbus_dth),
                           np.real(dSf_dth),
                           np.real(dSt_dth),
                           np.imag(dSbus_dth),
                           np.imag(dSf_dth),
                           np.imag(dSt_dth),
                           dv_dth,
                           dif_dth,
                           dit_dth])
        h_jac_v = vstack([np.real(dSbus_dv),
                          np.real(dSf_dv),
                          np.real(dSt_dv),
                          np.imag(dSbus_dv),
                          np.imag(dSf_dv),
                          np.imag(dSt_dv),
                          dv_dv,
                          dif_dv,
                          dit_dv])
        h_jac = hstack([h_jac_th, h_jac_v])
        # TODO CHECK!
        bus_pq = np.r_[(self.ppci['bus'][:, BUS_TYPE] != 3),
                       np.ones(self.ppci['bus'].shape[0], dtype=bool)].ravel()
        return h_jac.todense()[meas_mask, :][:, bus_pq]

class WLSAlgebraZeroInjectionConstraints(WLSAlgebra):
    pass
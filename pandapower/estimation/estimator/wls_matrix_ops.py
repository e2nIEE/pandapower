# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, diags

from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE

#try:
#from pandapower.pf.makeYbus import makeYbus
#except ImportError:
from pandapower.pypower.makeYbus import makeYbus


class WLSAlgebra:
    def __init__(self, ppci, non_nan_meas_mask):
        np.seterr(divide='ignore', invalid='ignore')
        self.ppci = ppci
        self.fb = self.ppci["branch"][:, F_BUS].real.astype(int)
        self.tb = self.ppci["branch"][:, T_BUS].real.astype(int)
        self.n_bus = self.ppci['bus'].shape[0]
        self.n_branch = self.ppci['branch'].shape[0]
        self.Y_bus = None
        self.Yf = None
        self.Yt = None
        self.G = None
        self.B = None
        # TODO CHECK!
        # Double the size of the bus to get mask for v_a and v_m
        self.non_nan_meas_mask = non_nan_meas_mask
        self.non_slack_bus_mask = (self.ppci['bus'][:, BUS_TYPE] != 3).ravel()
        self.delta_v_bus_mask = np.r_[self.non_slack_bus_mask,
                                      np.ones(self.non_slack_bus_mask.shape[0], dtype=bool)].ravel()
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
        self.Gshunt = None
        self.Bshunt = None

    # Get Y as tuple (real, imaginary)
    def get_y(self):
        return self.G, self.B

    def create_hx(self, v, delta):
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
        return hx[self.non_nan_meas_mask]

    def create_hx_jacobian(self, v, delta):
        # Using sparse matrix in creation sub-jacobian matrix
        V = v * np.exp(1j * delta)
        Vnorm = V / np.abs(V)
        diagV, diagVnorm = diags(V).tocsr(), diags(Vnorm).tocsr()

        dSbus_dth, dSbus_dv = self._dSbus_dv(V, diagV, diagVnorm)
        dSf_dth, dSt_dth, dSf_dv, dSt_dv = self._dSbr_dv(V, Vnorm, diagV, diagVnorm)
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
        return h_jac.toarray()[self.non_nan_meas_mask, :][:, self.delta_v_bus_mask]

    def _dSbus_dv(self, V, diagV, diagVnorm):
        diagIbus = csr_matrix((self.Ybus * V, (range(self.n_bus), range(self.n_bus))))
        dSbus_dth = 1j * diagV * np.conj(diagIbus - self.Ybus * diagV)
        dSbus_dv = diagV * np.conj(self.Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
        return dSbus_dth, dSbus_dv

    def _dSbr_dv(self, V, Vnorm, diagV, diagVnorm):
        n_bus, n_branch = self.n_bus, self.n_branch
        shape_diag_ft = (range(n_branch), range(n_branch))
        f_bus, t_bus = self.fb, self.tb
        If, It = self.Yf * V, self.Yt * V
        diagIf, diagIt = (csr_matrix((If, shape_diag_ft)),
                          csr_matrix((It, shape_diag_ft)))
        diagVf, diagVt = (csr_matrix((V[f_bus], shape_diag_ft)),
                          csr_matrix((V[t_bus], shape_diag_ft)))

        dSf_dth = 1j * (np.conj(diagIf) * csr_matrix((V[f_bus], (range(n_branch), f_bus)), shape=(n_branch, n_bus)) -
                        diagVf * np.conj(self.Yf * diagV))
        dSt_dth = 1j * (np.conj(diagIt) * csr_matrix((V[t_bus], (range(n_branch), t_bus)), shape=(n_branch, n_bus)) -
                        diagVt * np.conj(self.Yt * diagV))
        dSf_dv = (diagVf * np.conj(self.Yf * diagVnorm) + np.conj(diagIf) *
                  csr_matrix((Vnorm[f_bus], (range(n_branch), f_bus)), shape=(n_branch, n_bus)))
        dSt_dv = (diagVt * np.conj(self.Yt * diagVnorm) + np.conj(diagIt) *
                  csr_matrix((Vnorm[t_bus], (range(n_branch), t_bus)), shape=(n_branch, n_bus)))
        return dSf_dth, dSt_dth, dSf_dv, dSt_dv

    def _dVbus_dv(self, V):
        dv_dth, dv_dv = np.zeros((V.shape[0], V.shape[0])), np.eye(V.shape[0], V.shape[0])
        return dv_dth, dv_dv

    def _dibr_dv(self, diagV, diagVnorm):
        dif_dth = np.abs(self.Yf * 1j * diagV)
        dif_dv = np.abs(self.Yf * diagVnorm)
        dit_dth = np.abs(self.Yt * 1j * diagV)
        dit_dv = np.abs(self.Yt * diagVnorm)
        return dif_dth, dit_dth, dif_dv, dit_dv


class WLSAlgebraZeroInjectionConstraints(WLSAlgebra):
    def create_cx(self, v, delta, p_zero_inj, q_zero_inj):
        V = v * np.exp(1j * delta)
        Sbus = V * np.conj(self.Ybus * V)
        c = np.r_[np.real(Sbus[p_zero_inj]),
                  np.imag(Sbus[q_zero_inj])]
        return c

    def create_cx_jacobian(self, v, delta, p_zero_inj, q_zero_inj):
        V = v * np.exp(1j * delta)
        diagV, diagVnorm = diags(V).tocsr(), diags(V/np.abs(V)).tocsr()
        dSbus_dth, dSbus_dv = self._dSbus_dv(V, diagV, diagVnorm)
        c_jac_th = np.r_[np.real(dSbus_dth.todense())[p_zero_inj],
                         np.imag(dSbus_dth.todense())[q_zero_inj]]
        c_jac_v = np.r_[np.real(dSbus_dv.todense())[p_zero_inj],
                        np.imag(dSbus_dv.todense())[q_zero_inj]]
        c_jac = np.c_[c_jac_th, c_jac_v]
        return c_jac[:, self.delta_v_bus_mask]


class WLSAlgebraOptimization(WLSAlgebra):
    def __init__(self, ppci, non_nan_meas_mask, z, sigma):
        super(WLSAlgebraOptimization, self).__init__(ppci, non_nan_meas_mask)
        self.num_non_slack_bus = np.sum(self.non_slack_bus_mask)
        self.non_slack_buses = np.argwhere(self.non_slack_bus_mask).ravel()
        self.delta = np.zeros(self.n_bus)
        self.v = np.zeros(self.n_bus)
        self.z = z
        self.sigma = sigma
        
    def object_func(self, E):
#        return np.max(self.cost_function(E))
        hx = self.create_hx(self.v, self.delta)
        return hx.reshape(-1,1).T * np.diag(1/self.sigma) @ hx

    def cost_function(self, E):
        rx = self.create_rx(E) 
        cost = (1/self.sigma) * (rx**2)
        return cost
    
    def create_jac(self, E):
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
        self.v = E[self.num_non_slack_bus:]
        rx = self.create_rx(E)
        hx_jac = self.create_hx_jacobian(self.v, self.delta)
        jac = - hx_jac.T @ np.diag(1/self.sigma) @ rx
        return jac

    def create_rx(self, E):
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
        self.v = E[self.num_non_slack_bus:]
        hx = self.create_hx(self.v, self.delta)
        return (self.z - hx).ravel()


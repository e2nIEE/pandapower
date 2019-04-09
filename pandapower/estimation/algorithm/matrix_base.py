# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, diags

from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE, BASE_KV
from pandapower.pypower.makeYbus import makeYbus

from pandapower.estimation.ppc_conversion import ExtendedPPCI

__all__ = ['BaseAlgebra', 'BaseAlgebraZeroInjConstraints']


class BaseAlgebra:
    def __init__(self, e_ppci: ExtendedPPCI):
        np.seterr(divide='ignore', invalid='ignore')
        self.e_ppci = e_ppci

        self.fb = e_ppci.branch[:, F_BUS].real.astype(int)
        self.tb = e_ppci.branch[:, T_BUS].real.astype(int)
        self.n_bus = e_ppci.bus.shape[0]
        self.n_branch = e_ppci.branch.shape[0]
        self.baseMVA = e_ppci.baseMVA
        self.bus_baseKV = e_ppci.bus_baseKV
        self.num_non_slack_bus = e_ppci.num_non_slack_bus
        self.non_slack_buses = e_ppci.non_slack_buses
        self.delta_v_bus_mask = e_ppci.delta_v_bus_mask

        self.z = e_ppci.z
        self.r_cov = e_ppci.r_cov
        self.non_nan_meas_mask = e_ppci.non_nan_meas_mask

        self.Ybus = None
        self.Yf = None
        self.Yt = None
        self.initialize_Y()

        self.v = self.e_ppci.v_init.copy()
        self.delta = self.e_ppci.delta_init.copy()

    # Function which builds a node admittance matrix out of the topology data
    # In addition, it provides the series admittances of lines as G_series and B_series
    def initialize_Y(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus(self.e_ppci["baseMVA"], self.e_ppci["bus"], self.e_ppci["branch"])
            self.e_ppci['internal']['Yf'], self.e_ppci['internal']['Yt'],\
                self.e_ppci['internal']['Ybus'] = Yf, Yt, Ybus

        # create relevant matrices
        self.Ybus, self.Yf, self.Yt = Ybus, Yf, Yt

    def _e2v(self, E):
        self.v = E[self.num_non_slack_bus:]
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
        return self.v, self.delta

    def create_rx(self, E):
        hx = self.create_hx(E)
        return (self.z - hx).ravel()

    def create_hx(self, E):
        v, delta = self._e2v(E)
        f_bus, t_bus = self.fb, self.tb
        V = v * np.exp(1j * delta)
        Sfe = V[f_bus] * np.conj(self.Yf * V) 
        Ste = V[t_bus] * np.conj(self.Yt * V)
        Sbuse = V * np.conj(self.Ybus * V) 
        Ife = np.abs(Sfe)/v[f_bus]
        Ite = np.abs(Ste)/v[t_bus]
        hx = np.r_[np.real(Sbuse) * self.baseMVA,
                   np.real(Sfe) * self.baseMVA,
                   np.real(Ste) * self.baseMVA,
                   np.imag(Sbuse) * self.baseMVA,
                   np.imag(Sfe) * self.baseMVA,
                   np.imag(Ste) * self.baseMVA,
                   v,
                   Ife * self.baseMVA / self.bus_baseKV[f_bus],
                   Ite * self.baseMVA / self.bus_baseKV[t_bus]] 
        return hx[self.non_nan_meas_mask]

    def create_hx_jacobian(self, E):
        v, delta = self._e2v(E)
        # Using sparse matrix in creation sub-jacobian matrix
        f_bus, t_bus = self.fb, self.tb
        V = v * np.exp(1j * delta)
        Vnorm = V / np.abs(V)
        diagV, diagVnorm = diags(V).tocsr(), diags(Vnorm).tocsr()

        dSbus_dth, dSbus_dv = self._dSbus_dv(V, diagV, diagVnorm)
        dSf_dth, dSt_dth, dSf_dv, dSt_dv = self._dSbr_dv(V, Vnorm, diagV, diagVnorm)
        dif_dth, dit_dth, dif_dv, dit_dv = self._dibr_dv(diagV, diagVnorm)
        dv_dth, dv_dv = self._dVbus_dv(V)

        s_jac_th = vstack([np.real(dSbus_dth),
                           np.real(dSf_dth),
                           np.real(dSt_dth),
                           np.imag(dSbus_dth),
                           np.imag(dSf_dth),
                           np.imag(dSt_dth)])
        s_jac_v = vstack([np.real(dSbus_dv),
                          np.real(dSf_dv),
                          np.real(dSt_dv),
                          np.imag(dSbus_dv),
                          np.imag(dSf_dv),
                          np.imag(dSt_dv)])

        s_jac = hstack([s_jac_th, s_jac_v]).toarray() * self.baseMVA
        v_jac = np.c_[dv_dth, dv_dv]
        i_jac = vstack([hstack([dif_dth, dif_dv]),
                        hstack([dit_dth, dit_dv])]).toarray() *\
                        (self.baseMVA / np.r_[self.bus_baseKV[f_bus], 
                                              self.bus_baseKV[t_bus]]).reshape(-1, 1)
        return np.r_[s_jac,
                     v_jac,
                     i_jac][self.non_nan_meas_mask, :][:, self.delta_v_bus_mask]

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


class BaseAlgebraZeroInjConstraints(BaseAlgebra):
    def create_cx(self, E, p_zero_inj, q_zero_inj):
        v, delta = self._e2v(E)
        V = v * np.exp(1j * delta)
        Sbus = V * np.conj(self.Ybus * V)
        c = np.r_[np.real(Sbus[p_zero_inj]),
                  np.imag(Sbus[q_zero_inj])]
        return c

    def create_cx_jacobian(self, E, p_zero_inj, q_zero_inj):
        v, delta = self._e2v(E)
        V = v * np.exp(1j * delta)
        diagV, diagVnorm = diags(V).tocsr(), diags(V/np.abs(V)).tocsr()
        dSbus_dth, dSbus_dv = self._dSbus_dv(V, diagV, diagVnorm)
        c_jac_th = np.r_[np.real(dSbus_dth.todense())[p_zero_inj],
                         np.imag(dSbus_dth.todense())[q_zero_inj]]
        c_jac_v = np.r_[np.real(dSbus_dv.todense())[p_zero_inj],
                        np.imag(dSbus_dv.todense())[q_zero_inj]]
        c_jac = np.c_[c_jac_th, c_jac_v]
        return c_jac[:, self.delta_v_bus_mask]  

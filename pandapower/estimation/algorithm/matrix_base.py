# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
import numpy as np
from scipy.sparse import vstack, hstack, diags
from scipy.sparse import csr_matrix as sparse

from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.dSbr_dV import dSbr_dV
from pandapower.pypower.dIbr_dV import dIbr_dV

from pandapower.estimation.ppc_conversion import ExtendedPPCI

__all__ = ['BaseAlgebra', 'BaseAlgebraZeroInjConstraints']


class BaseAlgebra:
    def __init__(self, eppci: ExtendedPPCI):
        np.seterr(divide='ignore', invalid='ignore')
        self.eppci = eppci

        self.fb = eppci.branch[:, F_BUS].real.astype(int)
        self.tb = eppci.branch[:, T_BUS].real.astype(int)
        self.n_bus = eppci.bus.shape[0]
        self.n_branch = eppci.branch.shape[0]
        self.baseMVA = eppci.baseMVA
        self.bus_baseKV = eppci.bus_baseKV
        self.num_non_slack_bus = eppci.num_non_slack_bus
        self.non_slack_buses = eppci.non_slack_buses
        self.delta_v_bus_mask = eppci.delta_v_bus_mask
        self.non_nan_meas_mask = eppci.non_nan_meas_mask
        self.z = eppci.z
        self.sigma = eppci.r_cov

        self.Ybus = None
        self.Yf = None
        self.Yt = None
        self.initialize_Y()

        self.v = eppci.v_init.copy()
        self.delta = eppci.delta_init.copy()

    # Function which builds a node admittance matrix out of the topology data
    # In addition, it provides the series admittances of lines as G_series and B_series
    def initialize_Y(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus(self.eppci["baseMVA"], self.eppci["bus"], self.eppci["branch"])
            self.eppci['internal']['Yf'], self.eppci['internal']['Yt'],\
                self.eppci['internal']['Ybus'] = Yf, Yt, Ybus

        # create relevant matrices in sparse form
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
        # Using sparse matrix in creation sub-jacobian matrix
        v, delta = self._e2v(E)

        f_bus, t_bus = self.fb, self.tb
        V = v * np.exp(1j * delta)

        dSbus_dth, dSbus_dv = self._dSbus_dv(V)
        dSf_dth, dSf_dv, dSt_dth, dSt_dv = self._dSbr_dv(V)
        dif_dth, dif_dv, dit_dth, dit_dv = self._dibr_dv(V)
        dv_dth, dv_dv = self._dVbus_dv(V)

        s_jac_th = vstack((dSbus_dth.real,
                           dSf_dth.real,
                           dSt_dth.real,
                           dSbus_dth.imag,
                           dSf_dth.imag,
                           dSt_dth.imag))
        s_jac_v = vstack((dSbus_dv.real,
                          dSf_dv.real,
                          dSt_dv.real,
                          dSbus_dv.imag,
                          dSf_dv.imag,
                          dSt_dv.imag))

        s_jac = hstack((s_jac_th, s_jac_v)).toarray() * self.baseMVA
        v_jac = np.c_[dv_dth, dv_dv]
        i_jac = vstack((hstack((dif_dth, dif_dv)),
                        hstack((dit_dth, dit_dv)))).toarray() *\
                        (self.baseMVA / np.r_[self.bus_baseKV[f_bus], 
                                              self.bus_baseKV[t_bus]]).reshape(-1, 1)
        return np.r_[s_jac,
                     v_jac,
                     i_jac][self.non_nan_meas_mask, :][:, self.delta_v_bus_mask]

    def _dSbus_dv(self, V):
        dSbus_dv, dSbus_dth = dSbus_dV(self.Ybus, V)
        return dSbus_dth, dSbus_dv

    def _dSbr_dv(self, V):
        dSf_dth, dSf_dv, dSt_dth, dSt_dv, _, _ = dSbr_dV(self.eppci.branch, self.Yf, self.Yt, V)
        return dSf_dth, dSf_dv, dSt_dth, dSt_dv

    def _dVbus_dv(self, V):
        dv_dth, dv_dv = np.zeros((V.shape[0], V.shape[0])), np.eye(V.shape[0], V.shape[0])
        return dv_dth, dv_dv

    def _dibr_dv(self, V):
        # for current we only interest in the magnitude at the moment
        dif_dth, dif_dv, dit_dth, dit_dv, _, _ = map(np.abs, dIbr_dV(self.eppci.branch, self.Yf, self.Yt, V))
        return dif_dth, dif_dv, dit_dth, dit_dv


class BaseAlgebraZeroInjConstraints(BaseAlgebra):
    def create_cx(self, E, p_zero_inj, q_zero_inj):
        v, delta = self._e2v(E)
        V = v * np.exp(1j * delta)
        Sbus = V * np.conj(self.Ybus * V)
        c = np.r_[Sbus[p_zero_inj].real,
                  Sbus[q_zero_inj].imag] * self.baseMVA
        return c

    def create_cx_jacobian(self, E, p_zero_inj, q_zero_inj):
        v, delta = self._e2v(E)
        V = v * np.exp(1j * delta)
        dSbus_dth, dSbus_dv = self._dSbus_dv(V)
        c_jac_th = np.r_[dSbus_dth.toarray().real[p_zero_inj],
                         dSbus_dth.toarray().imag[q_zero_inj]]
        c_jac_v = np.r_[dSbus_dv.toarray().real[p_zero_inj],
                        dSbus_dv.toarray().imag[q_zero_inj]]
        c_jac = np.c_[c_jac_th, c_jac_v]
        return c_jac[:, self.delta_v_bus_mask]  

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import vstack, hstack

from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.dSbr_dV import dSbr_dV
from pandapower.pypower.dIbr_dV import dIbr_dV_new

from pandapower.estimation.ppc_conversion import ExtendedPPCI

__all__ = ['BaseAlgebra', 'BaseAlgebraZeroInjConstraints']


class BaseAlgebra:
    def __init__(self, eppci: ExtendedPPCI):
        """Object to calculate matrices required in state-estimation iterations."""
        self.eppci = eppci

        self.fb = eppci['branch'][:, F_BUS].real.astype(np.int64)
        self.tb = eppci['branch'][:, T_BUS].real.astype(np.int64)
        self.n_bus = eppci['bus'].shape[0]
        self.n_branch = eppci['branch'].shape[0]

        self.num_non_slack_bus = eppci.num_non_slack_bus
        self.non_slack_buses = eppci.non_slack_buses
        self.delta_v_bus_mask = eppci.delta_v_bus_mask
        self.non_nan_meas_mask = eppci.non_nan_meas_mask

        self.any_i_meas = eppci.any_i_meas
        self.any_degree_meas = eppci.any_degree_meas
        self.delta_v_bus_selector = eppci.delta_v_bus_selector
        self.non_nan_meas_selector = eppci.non_nan_meas_selector
        self.z = eppci.z
        self.sigma = eppci.r_cov

        self.Ybus = None
        self.Yf = None
        self.Yt = None
        self.initialize_Y()

    def initialize_Y(self):
        self.Ybus, self.Yf, self.Yt = self.eppci.get_Y()

    def create_rx(self, E):
        hx = self.create_hx(E)
        return (self.z - hx).ravel()

    def create_hx(self, E):
        f_bus, t_bus = self.fb, self.tb
        if self.eppci.algorithm == "af-wls":
            num_clusters = len(self.eppci["clusters"])
            E1 = E[:-num_clusters]
            E2 = E[-num_clusters:]
        else:
            E1 = E
        V = self.eppci.E2V(E1)
        Sfe = V[f_bus] * np.conj(self.Yf * V)
        Ste = V[t_bus] * np.conj(self.Yt * V)
        Sbuse = V * np.conj(self.Ybus * V)
        hx = np.r_[np.real(Sbuse),
                   np.real(Sfe),
                   np.real(Ste),
                   np.imag(Sbuse),
                   np.imag(Sfe),
                   np.imag(Ste),
                   np.abs(V)]

        # if self.any_i_meas or self.any_degree_meas:
        va = np.angle(V)
        Ife = self.Yf * V
        ifem = np.abs(Ife)
        ifea = np.angle(Ife)
        Ite = self.Yt * V
        item = np.abs(Ite)
        itea = np.angle(Ite)
        hx = np.r_[hx,
                    va,
                    ifem,
                    item,
                    ifea,
                    itea]

        if self.eppci.algorithm == "af-wls":
            Pbuse2 = np.sum(np.multiply(E2,self.eppci["rated_power_clusters"][:,:num_clusters]),axis=1)
            Qbuse2 = np.sum(np.multiply(E2,self.eppci["rated_power_clusters"][:,num_clusters:2*num_clusters]),axis=1)
            hx = np.r_[hx,
                       np.real(Sbuse)-Pbuse2,
                       np.imag(Sbuse)-Qbuse2,
                       E2]
        
        return hx[self.non_nan_meas_selector]

    def create_hx_jacobian(self, E):
        # Using sparse matrix in creation sub-jacobian matrix
        if self.eppci.algorithm == "af-wls":
            num_clusters = len(self.eppci["clusters"])
            E1 = E[:-num_clusters]
        else:
            E1 = E

        V = self.eppci.E2V(E1)

        dSbus_dth, dSbus_dv = self._dSbus_dv(V)
        dSf_dth, dSf_dv, dSt_dth, dSt_dv = self._dSbr_dv(V)
        dvm_dth, dvm_dv = self._dvmbus_dV(V)

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

        s_jac = hstack((s_jac_th, s_jac_v)).toarray()
        vm_jac = np.c_[dvm_dth, dvm_dv]
        jac = np.r_[s_jac,
                    vm_jac]

        # if self.any_i_meas or self.any_degree_meas:
        dva_dth, dva_dv = self._dvabus_dV(V)
        va_jac = np.c_[dva_dth, dva_dv]
        difm_dth, difm_dv, ditm_dth, ditm_dv,\
            difa_dth, difa_dv, dita_dth, dita_dv = self._dimiabr_dV(V)
        im_jac_th = np.r_[difm_dth,
                            ditm_dth]
        im_jac_v = np.r_[difm_dv,
                            ditm_dv]
        ia_jac_th = np.r_[difa_dth,
                            dita_dth]
        ia_jac_v = np.r_[difa_dv,
                            dita_dv]

        im_jac = np.c_[im_jac_th, im_jac_v]
        ia_jac = np.c_[ia_jac_th, ia_jac_v]

        jac = np.r_[jac,
                    va_jac,
                    im_jac,
                    ia_jac]

        if self.eppci.algorithm == "af-wls":
            p_eq_bal_jac_E1 = hstack((dSbus_dth.real, dSbus_dv.real)).toarray()
            q_eq_bal_jac_E1 = hstack((dSbus_dth.imag, dSbus_dv.imag)).toarray()
            af_vmeas_E1 = np.zeros((num_clusters,jac.shape[1]))

            jac_E2 = np.zeros((jac.shape[0],num_clusters))
            p_eq_bal_jac_E2 = - self.eppci["rated_power_clusters"][:,:num_clusters]
            q_eq_bal_jac_E2 = - self.eppci["rated_power_clusters"][:,num_clusters:2*num_clusters]
            af_vmeas_E2 = np.identity(num_clusters)

            jac = np.r_[jac,
                        p_eq_bal_jac_E1,
                        q_eq_bal_jac_E1,
                        af_vmeas_E1]

            jac_E2 = np.r_[jac_E2,
                        p_eq_bal_jac_E2,
                        q_eq_bal_jac_E2,
                        af_vmeas_E2]

            jac = jac[self.non_nan_meas_selector, :][:, self.delta_v_bus_selector]
            jac_E2 = jac_E2[self.non_nan_meas_selector, :][:]
            jac = np.c_[jac, jac_E2]

        else:
            jac = jac[self.non_nan_meas_selector, :][:, self.delta_v_bus_selector]

        return jac

    def _dSbus_dv(self, V):
        dSbus_dv, dSbus_dth = dSbus_dV(self.Ybus, V)
        return dSbus_dth, dSbus_dv

    def _dSbr_dv(self, V):
        dSf_dth, dSf_dv, dSt_dth, dSt_dv, _, _ = dSbr_dV(self.eppci['branch'], self.Yf, self.Yt, V)
        return dSf_dth, dSf_dv, dSt_dth, dSt_dv

    @staticmethod
    def _dvmbus_dV(V):
        dvm_dth, dvm_dv = np.zeros((V.shape[0], V.shape[0])), np.eye(V.shape[0], V.shape[0])
        return dvm_dth, dvm_dv

    @staticmethod
    def _dvabus_dV(V):
        dva_dth, dva_dv = np.eye(V.shape[0], V.shape[0]), np.zeros((V.shape[0], V.shape[0]))
        return dva_dth, dva_dv

    def _dimiabr_dV(self, V):
        # for current we only interest in the magnitude at the moment
        difm_dth, difm_dv, ditm_dth, ditm_dv = dIbr_dV_new(self.eppci['branch'], self.Yf, self.Yt, V)
        difm_dth, difm_dv, ditm_dth, ditm_dv = map(lambda m: m.toarray(), (difm_dth, difm_dv, ditm_dth, ditm_dv))
        difa_dth, difa_dv, dita_dth, dita_dv = 0*difm_dth, 0*difm_dv, 0*ditm_dth, 0*ditm_dv
        # dif_dth, dif_dv, dit_dth, dit_dv, If, It = dIbr_dV(self.eppci['branch'], self.Yf, self.Yt, V)
        # dif_dth, dif_dv, dit_dth, dit_dv = map(lambda m: m.toarray(), (dif_dth, dif_dv, dit_dth, dit_dv))
        # difm_dth = (np.abs(1e-5 * dif_dth + If.reshape((-1, 1))) - np.abs(If.reshape((-1, 1))))/1e-5
        # difm_dv = (np.abs(1e-5 * dif_dv + If.reshape((-1, 1))) - np.abs(If.reshape((-1, 1))))/1e-5
        # ditm_dth = (np.abs(1e-5 * dit_dth + It.reshape((-1, 1))) - np.abs(It.reshape((-1, 1))))/1e-5
        # ditm_dv = (np.abs(1e-5 * dit_dv + It.reshape((-1, 1))) - np.abs(It.reshape((-1, 1))))/1e-5
        # difa_dth = (np.angle(1e-5 * dif_dth + If.reshape((-1, 1))) - np.angle(If.reshape((-1, 1))))/1e-5
        # difa_dv = (np.angle(1e-5 * dif_dv + If.reshape((-1, 1))) - np.angle(If.reshape((-1, 1))))/1e-5
        # dita_dth = (np.angle(1e-5 * dit_dth + It.reshape((-1, 1))) - np.angle(It.reshape((-1, 1))))/1e-5
        # dita_dv = (np.angle(1e-5 * dit_dv + It.reshape((-1, 1))) - np.angle(It.reshape((-1, 1))))/1e-5
        return difm_dth, difm_dv, ditm_dth, ditm_dv, difa_dth, difa_dv, dita_dth, dita_dv


class BaseAlgebraZeroInjConstraints(BaseAlgebra):
    def create_cx(self, E, p_zero_inj, q_zero_inj):
        V = self.eppci.E2V(E)
        Sbus = V * np.conj(self.Ybus * V)
        c = np.r_[Sbus[p_zero_inj].real,
                  Sbus[q_zero_inj].imag] * self.eppci['baseMVA']
        return c

    def create_cx_jacobian(self, E, p_zero_inj, q_zero_inj):
        V = self.eppci.E2V(E)
        dSbus_dth, dSbus_dv = self._dSbus_dv(V)
        c_jac_th = np.r_[dSbus_dth.toarray().real[p_zero_inj],
                         dSbus_dth.toarray().imag[q_zero_inj]]
        c_jac_v = np.r_[dSbus_dv.toarray().real[p_zero_inj],
                        dSbus_dv.toarray().imag[q_zero_inj]]
        c_jac = np.c_[c_jac_th, c_jac_v]
        return c_jac[:, self.delta_v_bus_mask]

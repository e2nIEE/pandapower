# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from numpy import conj, arange, int64
from scipy.sparse import vstack, hstack
from scipy.sparse import eye, csr_matrix as sparse
from pandapower.pypower.idx_brch import F_BUS, T_BUS

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
        
        meas_mask = self.eppci.non_nan_meas_mask
        V = self.eppci.E2V(E1)
        Sfe = V[f_bus] * np.conj(self.Yf * V)
        Ste = V[t_bus] * np.conj(self.Yt * V)
        Sbuse = V * np.conj(self.Ybus * V)
        Ife = self.Yf * V
        Ite = self.Yt * V

        Pbuse = np.real(Sbuse)[meas_mask["pbus"]]
        Pfe = np.real(Sfe)[meas_mask["pfrom"]]
        Pte = np.real(Ste)[meas_mask["pto"]]
        Qbuse = np.imag(Sbuse)[meas_mask["qbus"]]
        Qfe = np.imag(Sfe)[meas_mask["qfrom"]]
        Qte = np.imag(Ste)[meas_mask["qto"]]
        Vm = np.abs(V)[meas_mask["vm"]]
        Va = np.angle(V)[meas_mask["va"]]
        Imfe = np.abs(Ife)[meas_mask["ifrom"]]
        Imte = np.abs(Ite)[meas_mask["ito"]]
        # Iafe = np.angle(Ife)[meas_mask["iafrom"]]
        # Iate = np.angle(Ite)[meas_mask["iato"]]
        
        hx = np.r_[Pbuse, Qbuse, Pfe, Qfe, Pte, Qte, Vm, Va, Imfe, Imte]

        if self.eppci.algorithm == "af-wls":
            Pb2 = np.real(Sbuse) - np.sum(np.multiply(E2,self.eppci["rated_power_clusters"][:,:num_clusters]),axis=1)
            Qb2 = np.imag(Sbuse) - np.sum(np.multiply(E2,self.eppci["rated_power_clusters"][:,num_clusters:2*num_clusters]),axis=1)
            Pbuse2 = Pb2[meas_mask["pbalance"]]
            Qbuse2 = Qb2[meas_mask["qbalance"]]
            E2e = E2[meas_mask["afactor"]]
            hx = np.r_[hx, Pbuse2, Qbuse2, E2e]
        
        return hx

    def create_hx_jacobian(self, E):
        # Using sparse matrix in creation sub-jacobian matrix
        if self.eppci.algorithm == "af-wls":
            num_clusters = len(self.eppci["clusters"])
            E1 = E[:-num_clusters]
        else:
            E1 = E

        meas_mask = self.eppci.non_nan_meas_mask
        V = self.eppci.E2V(E1)
        nvar = 2*len(V)
        jac = sparse((0, nvar))

        if len(meas_mask["pbus"])+len(meas_mask["qbus"])>0:
            dPbus, dQbus = self._dSbus_dv(V, meas_mask["pbus"], meas_mask["qbus"])
            jac = vstack((jac, dPbus, dQbus))

        if len(meas_mask["pfrom"])+len(meas_mask["qfrom"])>0:
            dPf, dQf = self._dSbr_dv(V, "from", meas_mask["pfrom"], meas_mask["qfrom"])
            jac = vstack((jac, dPf, dQf))

        if len(meas_mask["pto"])+len(meas_mask["qto"])>0:
            dPt, dQt = self._dSbr_dv(V, "to", meas_mask["pto"], meas_mask["qto"])
            jac = vstack((jac, dPt, dQt))

        dVm = self._dVmbus_dV(V, meas_mask["vm"])
        jac = vstack((jac, dVm))

        if len(meas_mask["va"])>0:
            dVa = self._dVabus_dV(V, meas_mask["va"])
            jac = vstack((jac, dVa))

        if len(meas_mask["ifrom"])>0:
            dIfm = self._dImbr_dV(V, "from", meas_mask["ifrom"])
            jac = vstack((jac, dIfm))

        if len(meas_mask["ito"])>0:
            dItm = self._dImbr_dV(V, "to", meas_mask["ito"])
            jac = vstack((jac, dItm))

        # dIfa = self._dIabr_dV(V, "from")
        # dIta = self._dIabr_dV(V, "to")

        if self.eppci.algorithm == "af-wls":
            p_bal_jac_E1, q_bal_jac_E1 = self._dSbus_dv(V, meas_mask["pbalance"], meas_mask["qbalance"])
            af_vmeas_E1 = sparse((num_clusters,jac.shape[1])) 

            jac_E2 = sparse((jac.shape[0],num_clusters))
            p_bal_jac_E2 = sparse(- self.eppci["rated_power_clusters"][:,:num_clusters])
            q_bal_jac_E2 = sparse(- self.eppci["rated_power_clusters"][:,num_clusters:2*num_clusters])
            af_vmeas_E2 = eye(num_clusters, num_clusters, format='csr')
            p_bal_jac_E2 = p_bal_jac_E2[meas_mask["pbalance"]]
            q_bal_jac_E2 = q_bal_jac_E2[meas_mask["qbalance"]]
            af_vmeas_E2 = af_vmeas_E2[meas_mask["afactor"]]

            jac = vstack((jac, p_bal_jac_E1, q_bal_jac_E1, af_vmeas_E1))
            jac = jac[:][:, self.delta_v_bus_selector]
            jac_E2 = vstack((jac_E2, p_bal_jac_E2, q_bal_jac_E2, af_vmeas_E2))

            jac = hstack((jac, jac_E2))

        else:
            jac = jac[:][:, self.delta_v_bus_selector]

        return jac

    def _dSbus_dv(self, V, maskP=None, maskQ=None):
        Ybus = self.Ybus
        Ibus = Ybus * V
        b = len(V)
        ib = arange(b)

        maskS, maskP, maskQ = self._define_mask(ib, maskP, maskQ)

        diagV = sparse((V, (ib, ib)))
        diagV2 = diagV[maskS,:]
        diagIbus = sparse((Ibus, (ib, ib)))
        diagIbus2 = diagIbus[maskS,:]
        diagVnorm = sparse((V / abs(V), (ib, ib)))

        dS_dVm = diagV2 @ conj(Ybus @ diagVnorm) + conj(diagIbus2) @ diagVnorm
        dS_dVa = 1j * diagV2 @ conj(diagIbus - Ybus @ diagV)

        dP_dth = dS_dVa.real[maskP,:]
        dP_dv = dS_dVm.real[maskP,:]
        dQ_dth = dS_dVa.imag[maskQ,:]
        dQ_dv = dS_dVm.imag[maskQ,:]

        dP = hstack((dP_dth, dP_dv))
        dQ = hstack((dQ_dth, dQ_dv))

        return dP, dQ
     

    def _dSbr_dv(self, V, side, maskP=None, maskQ=None):
        branch = self.eppci['branch']
        f = branch[:, F_BUS].real.astype(int64)       ## list of "from" buses
        t = branch[:, T_BUS].real.astype(int64)       ## list of "to" buses
        nl = len(f)
        nb = len(V)
        il = arange(nl)
        ib = arange(nb)
        maskS, maskP, maskQ = self._define_mask(il, maskP, maskQ)

        if side == "from":
            Y = self.Yf
            s = f
        elif side == "to":
            Y = self.Yt
            s = t

        I = Y * V
        Vnorm = V / abs(V)

        diagVs = sparse((V[s], (il, il)))
        diagVs2 = diagVs[maskS,:]
        diagI = sparse((I, (il, il)))
        diagI2 = diagI[maskS,:]
        diagV  = sparse((V, (ib, ib)))
        diagVnorm = sparse((Vnorm, (ib, ib)))

        shape = (nl, nb)
        # Partial derivative of S w.r.t voltage phase angle.
        dS_dVa = 1j * (conj(diagI2) @
            sparse((V[s], (il, s)), shape) - diagVs2 @ conj(Y @ diagV))
        # Partial derivative of S w.r.t. voltage amplitude.
        dS_dVm = diagVs2 @ conj(Y @ diagVnorm) + conj(diagI2) @ \
            sparse((Vnorm[s], (il, s)), shape)
        
        dP_dth = dS_dVa.real[maskP,:]
        dP_dv = dS_dVm.real[maskP,:]
        dQ_dth = dS_dVa.imag[maskQ,:]
        dQ_dv = dS_dVm.imag[maskQ,:]

        dP = hstack((dP_dth, dP_dv))
        dQ = hstack((dQ_dth, dQ_dv))

        return dP, dQ

    def _dVmbus_dV(self, V, maskVm=None):
        V_shape = V.shape[0]
        if maskVm is None:
            maskVm = arange(V_shape)

        dvm_dth = sparse((len(maskVm), V_shape))  # Sparse zero matrix
        dvm_dv = eye(V_shape, V_shape, format='csr')
        dvm_dv = dvm_dv[maskVm,:]
        dvm = hstack((dvm_dth, dvm_dv))
        return dvm

    def _dVabus_dV(self, V, maskVa=None):
        V_shape = V.shape[0]
        if maskVa is None:
            maskVa = arange(V_shape)

        dva_dth = eye(V_shape, V_shape, format='csr')  # Sparse identity matrix
        dva_dth = dva_dth[maskVa,:]
        dva_dv = sparse((len(maskVa), V_shape))  # Sparse zero matrix
        dva = hstack((dva_dth, dva_dv))
        return dva
    
    def _dImbr_dV(self, V, side, maskI=None):
        nl = len(self.eppci['branch'])
        nb = len(V)
        il = arange(nl)
        vb = arange(nb)

        if maskI is None:
            maskI = il

        if side == "from":
            Y = self.Yf
        elif side == "to":
            Y = self.Yt

        # Compute branch currents.
        I = Y * V

        diagV = sparse((V, (vb, vb)))
        diagVnorm = sparse((V / abs(V), (vb, vb)))
        idx = abs(I) != 0
        I = I[idx]
        il = il[idx]
        diagInorm = sparse((conj(I) / abs(I), (il, il)), shape=(nl,nl))
        diagInorm2 = diagInorm[maskI,:]
        a = diagInorm2 @ Y @ diagV
        b = diagInorm2 @ Y @ diagVnorm
        dIm_dth = - a.imag
        dIm_dv = b.real
        dIm = hstack((dIm_dth, dIm_dv))

        return dIm
    
    def _define_mask(self, idx, mask1, mask2):
        if mask1 is None:
            mask1 = idx
        if mask2 is None:
            mask2 = idx
        masktot, mask1, mask2 = self._merge_mask(mask1, mask2)

        return masktot, mask1, mask2
    
    @staticmethod
    def _merge_mask(mask1, mask2):
        new_mask_tot = np.unique(np.concatenate((mask1,mask2),0))
        new_mask1 = np.in1d(new_mask_tot, mask1)
        new_mask2 = np.in1d(new_mask_tot, mask2)
        return new_mask_tot, new_mask1, new_mask2


class BaseAlgebraZeroInjConstraints(BaseAlgebra):
    def create_cx(self, E, p_zero_inj, q_zero_inj):
        V = self.eppci.E2V(E)
        Sbus = V * np.conj(self.Ybus * V)
        c = np.r_[Sbus[p_zero_inj].real,
                  Sbus[q_zero_inj].imag] * self.eppci['baseMVA']
        return c

    def create_cx_jacobian(self, E, p_zero_inj, q_zero_inj):
        V = self.eppci.E2V(E)
        dPbus, dQbus = self._dSbus_dv(V, p_zero_inj, q_zero_inj)
        c_jac = vstack((dPbus,dQbus))
        return c_jac[:][:, self.delta_v_bus_mask]
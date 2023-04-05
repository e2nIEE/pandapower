# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix
from pandapower.pypower.idx_brch import BR_R, BR_X, F_BUS, T_BUS
from pandapower.pypower.dSbus_dV import dSbus_dV


def calc_y_svc(x_control, svc_x_l_pu, svc_x_cvar_pu, v_base_kv, baseMVA):
    x_control = np.deg2rad(x_control)
    z_base_ohm = np.square(v_base_kv) / baseMVA
    svc_x_l_pu = svc_x_l_pu / z_base_ohm
    svc_x_cvar_pu = svc_x_cvar_pu / z_base_ohm
    y_svc = calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu)
    y_svc /= z_base_ohm
    return y_svc


def calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu):
    y_svc = (2 * (np.pi - x_control) + np.sin(2 * x_control) + np.pi * svc_x_l_pu / svc_x_cvar_pu) / (np.pi * svc_x_l_pu)
    return y_svc


def create_J_modification_svc(J, svc_buses, pvpq, pq, pq_lookup, V, x_control, svc_x_l_pu, svc_x_cvar_pu,
                              nsvc, nsvc_controllable, svc_controllable):
    J_m = np.zeros_like(J.toarray())
    controllable = np.arange(nsvc)[svc_controllable]

    in_pq = np.isin(svc_buses, pq)

    y_svc = calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu)
    q_svc = np.abs(V[svc_buses]) ** 2 * y_svc

    # J_C_Q_u
    J_C_Q_u = np.zeros(shape=(len(pq), len(pq)), dtype=np.float64)
    J_C_Q_u[pq_lookup[svc_buses[in_pq]], pq_lookup[svc_buses[in_pq]]] = 2 * q_svc[in_pq]
    # count pvpq rows and pvpq columns from top left
    J_m[len(pvpq):len(pvpq) + len(pq), len(pvpq):len(pvpq) + len(pq)] = J_C_Q_u

    # J_C_Q_c
    if np.any(svc_controllable):
        J_C_Q_c = np.zeros(shape=(len(pq), nsvc), dtype=np.float64)
        values = 2 * np.abs(V[svc_buses]) ** 2 * (np.cos(2 * x_control) - 1) / (np.pi * svc_x_l_pu)
        J_C_Q_c[pq_lookup[svc_buses[in_pq & svc_controllable]], controllable] = values[in_pq & svc_controllable]
        # count pvpq rows and pvpq columns from top left
        J_m[len(pvpq):len(pvpq) + len(pq),
            len(pvpq) + len(pq):len(pvpq) + len(pq) + nsvc_controllable] = J_C_Q_c[:, controllable]

    # J_C_C_u
    # d(Ui - Ui,set)/d(Ui) = dUi/dUi = 1
    if np.any(svc_controllable):
        J_C_C_u = np.zeros(shape=(nsvc, len(pq)), dtype=np.float64)
        J_C_C_u[controllable, pq_lookup[svc_buses[in_pq & controllable]]] = 1
        # count pvpq rows and pvpq columns from top left
        J_m[len(pvpq) + len(pq):len(pvpq) + len(pq) + nsvc_controllable,
            len(pvpq):len(pvpq) + len(pq)] = J_C_C_u[controllable, :]

    J_m = csr_matrix(J_m)

    return J_m


def create_J_modification_tcsc(V, Ybus_tcsc, x_control, svc_controllable, tcsc_controllable,
                               tcsc_x_l_pu, tcsc_x_cvar_pu, f, t, pvpq, pq, pvpq_lookup, pq_lookup, nsvc, ntcsc):
    Y_TCSC = calc_y_svc_pu(x_control, tcsc_x_l_pu, tcsc_x_cvar_pu)
    # S_tcsc_pu = V * (Ybus_tcsc.conj() @ V.conj())
    dY_TCSC_dx = 2 * (np.cos(2 * x_control) - 1) / (np.pi * tcsc_x_l_pu)

    Vf = V[f]
    Vt = V[t]

    Vmf = np.abs(Vf)
    Vmt = np.abs(Vt)

    S_Fii = Vf * np.conj(Ybus_tcsc.toarray()[f, f] * Vf)
    S_Fkk = Vt * np.conj(Ybus_tcsc.toarray()[t, t] * Vt)

    S_Fik = Vf * np.conj(Ybus_tcsc.toarray()[f, t] * Vt)
    S_Fki = Vt * np.conj(Ybus_tcsc.toarray()[t, f] * Vf)

    # seems like it is not used:
    # S_ii = np.abs(V[f]) ** 2 * np.abs(Ybus[f, f]) * np.exp(1j * np.angle(Ybus[f, f].conj()))  ####
    # S_kk = np.abs(V[t]) ** 2 * np.abs(Ybus[t, t]) * np.exp(1j * np.angle(Ybus[t, t].conj()))  ####
    #
    # S_ij = Sbus[f] - S_ii
    # S_kj = Sbus[t] - S_kk

    S_Fi_dx = dY_TCSC_dx / Y_TCSC * (S_Fii + S_Fik)
    S_Fk_dx = dY_TCSC_dx / Y_TCSC * (S_Fkk + S_Fki)

    f_in_pq = np.isin(f, pq)
    t_in_pq = np.isin(t, pq)
    f_in_pvpq = np.isin(f, pvpq)
    t_in_pvpq = np.isin(t, pvpq)

    # todo: use _sum_by_group what multiple elements start (or end) at the same bus?
    J_C_P_d = np.zeros(shape=(len(pvpq), len(pvpq)), dtype=np.float64)
    if np.any(f_in_pvpq):
        J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq]]] = -S_Fik.imag
    if np.any(f_in_pvpq & t_in_pvpq):
        J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]] = S_Fik.imag
        J_C_P_d[pvpq_lookup[t[t_in_pvpq]], pvpq_lookup[f[f_in_pvpq]]] = S_Fki.imag
    if np.any(t_in_pvpq):
        J_C_P_d[pvpq_lookup[t[t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]] = -S_Fki.imag
    # J_C_P_d = np.array([[-S_Fik.imag, S_Fik.imag],
    #                     [S_Fki.imag, -S_Fki.imag]]).reshape(2, 2)  # todo: generalize shapes to work with many TCSC

    # J_C_P_u = np.array([[S_Fik.real / Vm[f], S_Fik.real / Vm[t]],
    #                     [S_Fki.real / Vm[f], S_Fki.real / Vm[t]]]).reshape(2, 2)
    J_C_P_u = np.zeros(shape=(len(pvpq), len(pq)), dtype=np.float64)
    if np.any(f_in_pvpq & f_in_pq):
        J_C_P_u[pvpq_lookup[f[f_in_pvpq]], pq_lookup[f[f_in_pq]]] = S_Fik.real / Vmf
    if np.any(f_in_pvpq & t_in_pq):
        J_C_P_u[pvpq_lookup[f[f_in_pvpq]], pq_lookup[t[t_in_pq]]] = S_Fik.real / Vmt
    if np.any(t_in_pvpq & f_in_pq):
        J_C_P_u[pvpq_lookup[t[t_in_pvpq]], pq_lookup[f[f_in_pq]]] = S_Fki.real / Vmf
    if np.any(t_in_pvpq & t_in_pq):
        J_C_P_u[pvpq_lookup[t[t_in_pvpq]], pq_lookup[t[t_in_pq]]] = S_Fki.real / Vmt

    J_C_Q_d = np.zeros(shape=(len(pq), len(pvpq)), dtype=np.float64)
    if np.any(f_in_pvpq & f_in_pq):
        J_C_Q_d[pq_lookup[f[f_in_pq]], pvpq_lookup[f[f_in_pvpq]]] = S_Fik.real
    if np.any(f_in_pq & t_in_pvpq):
        J_C_Q_d[pq_lookup[f[f_in_pq]], pvpq_lookup[t[t_in_pvpq]]] = -S_Fik.real
    if np.any(t_in_pq & f_in_pvpq):
        J_C_Q_d[pq_lookup[t[t_in_pq]], pvpq_lookup[f[f_in_pvpq]]] = -S_Fki.real
    if np.any(t_in_pvpq & t_in_pq):
        J_C_Q_d[pq_lookup[t[t_in_pq]], pvpq_lookup[t[t_in_pvpq]]] = S_Fki.real
    # J_C_Q_d = np.array([[S_Fik.real, -S_Fik.real],
    #                     [-S_Fki.real, S_Fki.real]]).reshape(2, 2)

    J_C_Q_u = np.zeros(shape=(len(pq), len(pq)), dtype=np.float64)
    if np.any(f_in_pq):
        J_C_Q_u[pq_lookup[f[f_in_pq]], pq_lookup[f[f_in_pq]]] = (2 * S_Fii.imag + S_Fik.imag) / Vmf
    if np.any(f_in_pq & t_in_pq):
        J_C_Q_u[pq_lookup[f[f_in_pq]], pq_lookup[t[t_in_pq]]] = S_Fik.imag / Vmt
        J_C_Q_u[pq_lookup[t[t_in_pq]], pq_lookup[f[f_in_pq]]] = S_Fki.imag / Vmf
    if np.any(t_in_pq):
        J_C_Q_u[pq_lookup[t[t_in_pq]], pq_lookup[t[t_in_pq]]] = (2 * S_Fkk.imag + S_Fki.imag) / Vmt
    # J_C_Q_u = np.array([[(2 * S_Fii.imag + S_Fik.imag) / Vm[f], S_Fik.imag / Vm[f]],
    #                     [S_Fki.imag / Vm[t], (2 * S_Fkk.imag + S_Fki.imag) / Vm[f]]]).reshape(2, 2)

    J_C_P_c = np.zeros(shape=(len(pvpq), nsvc + ntcsc), dtype=np.float64)
    if np.any(f_in_pvpq):
        J_C_P_c[pvpq_lookup[f[f_in_pvpq]], (nsvc+np.arange(ntcsc))[f_in_pvpq]] = S_Fi_dx.real
    if np.any(t_in_pvpq):
        J_C_P_c[pvpq_lookup[t[t_in_pvpq]], (nsvc+np.arange(ntcsc))[t_in_pvpq]] = S_Fk_dx.real
    # J_C_P_c = np.array([[S_Fi_dx.real], [S_Fk_dx.real]]).reshape(2, 1)

    J_C_Q_c = np.zeros(shape=(len(pq), nsvc + ntcsc), dtype=np.float64)
    if np.any(f_in_pq):
        J_C_Q_c[pq_lookup[f[f_in_pq]], (nsvc+np.arange(ntcsc))[f_in_pq]] = S_Fi_dx.imag
    if np.any(t_in_pq):
        J_C_Q_c[pq_lookup[t[t_in_pq]], (nsvc+np.arange(ntcsc))[t_in_pq]] = S_Fk_dx.imag
    # J_C_Q_c = np.array([[S_Fi_dx.imag], [S_Fk_dx.imag]]).reshape(2, 1)

    # the signs are opposite here for J_C_C_d, J_C_C_u, J_C_C_c and I don't know why
    # main mode of operation - set point for p_to_mw:
    # J_C_C_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    J_C_C_d = np.zeros(shape=(nsvc+ntcsc, len(pvpq)), dtype=np.float64)
    if np.any(f_in_pvpq):
        J_C_C_d[(nsvc+np.arange(ntcsc))[f_in_pvpq], pvpq_lookup[f[f_in_pvpq]]] = S_Fik.imag
    if np.any(t_in_pvpq):
        J_C_C_d[(nsvc+np.arange(ntcsc))[t_in_pvpq], pvpq_lookup[t[t_in_pvpq]]] = -S_Fik.imag

    J_C_C_u = np.zeros(shape=(nsvc+ntcsc, len(pq)), dtype=np.float64)
    if np.any(f_in_pq):
        J_C_C_u[(nsvc+np.arange(ntcsc))[f_in_pq], pq_lookup[f[f_in_pq]]] = S_Fik.real / Vmf
    if np.any(t_in_pq):
        J_C_C_u[(nsvc+np.arange(ntcsc))[t_in_pq], pq_lookup[t[t_in_pq]]] = S_Fik.real / Vmt

    J_C_C_c = np.zeros(shape=(nsvc+ntcsc, nsvc+ntcsc), dtype=np.float64)
    J_C_C_c[np.r_[nsvc:nsvc+ntcsc], np.r_[nsvc:nsvc+ntcsc]] = -S_Fi_dx.real  # .flatten()?

    # alternative mode of operation: for Vm at to bus (mismatch and setpoint also must be adjusted):
    # J_C_C_d = np.zeros(shape=(len(x_control), len(pvpq)), dtype=np.float64)
    # J_C_C_u = np.zeros(shape=(len(x_control), len(pq)), dtype=np.float64)
    # J_C_C_u[np.arange(len(x_control)), pq_lookup[t]] = 1
    # J_C_C_c = np.zeros((len(x_control), len(x_control)), dtype=np.float64)

    if np.all(tcsc_controllable):
        J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u, J_C_P_c]),
                         np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c]),
                         np.hstack([J_C_C_d, J_C_C_u, J_C_C_c])])
    elif np.any(tcsc_controllable) or np.any(svc_controllable):
        relevant = np.r_[np.arange(nsvc), nsvc+np.arange(ntcsc)[tcsc_controllable]]
        J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u, J_C_P_c[:, relevant]]),
                         np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c[:, relevant]]),
                         np.hstack([J_C_C_d[relevant, :], J_C_C_u[relevant, :],
                                    J_C_C_c[:, relevant][relevant, :]])])
    else:
        J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u]),
                         np.hstack([J_C_Q_d, J_C_Q_u])])

    J_m = csr_matrix(J_m)

    return J_m


# todo delete
def create_J_modification_tcsc_old(J, branch, pvpq_lookup, pq_lookup, Ybus_tcsc, V, tcsc_fb, tcsc_tb, pvpq, pq, tcsc_branches,
                               x_control, x_control_lookup, tcsc_x_l_pu, tcsc_x_cvar_pu,
                               tcsc_in_pq_f, tcsc_in_pq_t, tcsc_in_pvpq_f, tcsc_in_pvpq_t):
    # Sbus = V * np.conj(Ybus_tcsc * V)
    # p_tcsc_ij = Sbus[tcsc_tb].real
    # q_tcsc_ij = Sbus[tcsc_tb].imag
    # phi_tcsc_ij = np.angle(Ybus[tcsc_fb, tcsc_tb])

    dS_dVm, dS_dVa = dSbus_dV(Ybus=Ybus_tcsc, V=V)

    rows_pvpq = np.array([pvpq]).T
    cols_pvpq = pvpq
    J11 = dS_dVa[rows_pvpq, cols_pvpq].real
    J12 = dS_dVm[rows_pvpq, pq].real
    J21 = dS_dVa[np.array([pq]).T, cols_pvpq].imag
    J22 = dS_dVm[np.array([pq]).T, pq].imag

    y_tcsc = calc_y_svc_pu(x_control[tcsc_tb], tcsc_x_l_pu[tcsc_tb], tcsc_x_cvar_pu[tcsc_tb])

    mf_pvpq = np.r_[branch[tcsc_branches[tcsc_in_pvpq_f], F_BUS].real.astype(int)]
    mt_pvpq = np.r_[branch[tcsc_branches[tcsc_in_pvpq_t], T_BUS].real.astype(int)]
    nf_pvpq = np.r_[branch[tcsc_branches[tcsc_in_pvpq_f], T_BUS].real.astype(int)]
    nt_pvpq = np.r_[branch[tcsc_branches[tcsc_in_pvpq_t], F_BUS].real.astype(int)]

    mf_pq = np.r_[branch[tcsc_branches[tcsc_in_pq_f], F_BUS].real.astype(int)]
    mt_pq = np.r_[branch[tcsc_branches[tcsc_in_pq_t], T_BUS].real.astype(int)]
    nf_pq = np.r_[branch[tcsc_branches[tcsc_in_pq_f], T_BUS].real.astype(int)]
    nt_pq = np.r_[branch[tcsc_branches[tcsc_in_pq_t], F_BUS].real.astype(int)]

    # phi_tcsc_ij = np.zeros(shape=(Ybus_tcsc.shape[0], 1), dtype=np.float64)
    # for i, (f, t) in enumerate(zip(tcsc_fb, tcsc_tb)):
    #     phi_tcsc_ij[i] = np.angle(Ybus_tcsc[f, t])

    Sbus_tcsc = V * np.conj(Ybus_tcsc * V)
    p_tcsc_ij = Sbus_tcsc.real
    q_tcsc_ii = np.zeros(shape=(Ybus_tcsc.shape[0]), dtype=np.float64)
    # todo: array calc
    for i in range(Ybus_tcsc.shape[0]):
        q_tcsc_ii[i] = np.square(np.abs(V[i])) * np.abs(Ybus_tcsc[i, i])
    q_tcsc_ij = Sbus_tcsc.imag - q_tcsc_ii

    # J_C_P_d
    # J_C_P_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    # for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
    #     i = pvpq_lookup[m]
    #     j = pvpq_lookup[n]
    #     J_C_P_d[i, i] = -q_tcsc_ij
    #     J_C_P_d[j, i] = q_tcsc_ij    #### q_tcsc_ij = q_tcsc_ji
    #     # J_C_P_d[i, i] = -Sbus[tcsc_tb].imag
    #     # J_C_P_d[j, i] = Sbus[tcsc_tb].imag
    # # J_C_P_d[0:len(tcsc_branches), tcsc_i] = -q_tcsc_ij
    # # J_C_P_d[0:len(tcsc_branches), tcsc_j] = q_tcsc_ij

    J_C_P_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    for in_pvpq, m, n in zip([tcsc_in_pvpq_f], [mf_pvpq], [nf_pvpq]):
        i = pvpq_lookup[m]
        j = pvpq_lookup[n]
        J_C_P_d[i, i] = -q_tcsc_ij[m]
        J_C_P_d[i, j] = q_tcsc_ij[m]
    for in_pvpq, m, n in zip([tcsc_in_pvpq_t], [mt_pvpq], [nt_pvpq]):
        i = pvpq_lookup[m]
        j = pvpq_lookup[n]
        J_C_P_d[i, i] = -q_tcsc_ij[m]
        J_C_P_d[i, j] = q_tcsc_ij[m]
    # J_C_P_d = J11.toarray()
    # J_C_P_u
    # J_C_P_u = np.zeros(shape=(len(pvpq), len(pq)))
    # for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pq), (tcsc_in_pvpq_t, mt_pvpq, nt_pq)):
    #     i = pvpq_lookup[m]
    #     j = pq_lookup[n]
    #     J_C_P_u[i, i] = p_tcsc_ij
    #     J_C_P_u[i, j] = p_tcsc_ij
    #     # J_C_P_u[i, i] = Sbus[tcsc_tb].real
    #     # J_C_P_u[i, j] = Sbus[tcsc_tb].real
    # checking is wrong here. add an if-clause?
    #J_C_P_u[0:len(tcsc_branches), tcsc_i[np.isin(tcsc_i, pq)]] = p_tcsc_ij
    #J_C_P_u[0:len(tcsc_branches), tcsc_j[np.isin(tcsc_j, pq)]] = p_tcsc_ij

    J_C_P_u = np.zeros(shape=(len(pvpq), len(pq)))
    for in_pvpq, m, n in zip([tcsc_in_pvpq_f], [mf_pvpq], [nf_pq]):
        i = pvpq_lookup[m]
        j = pq_lookup[n]
        J_C_P_u[i, i] = p_tcsc_ij[m]
        J_C_P_u[i, j] = p_tcsc_ij[m]
    for in_pvpq, m, n in zip([tcsc_in_pvpq_t], [mt_pvpq], [nt_pq]):
        i = pvpq_lookup[m]
        j = pq_lookup[n]
        J_C_P_u[i, i] = p_tcsc_ij[m]
        J_C_P_u[i, j] = p_tcsc_ij[m]
    # J_C_P_u = J12.toarray()

    #
    # # J_C_Q_d
    # J_C_Q_d = np.zeros(shape=(len(pq), len(pvpq)))
    # for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pq, nf_pvpq), (tcsc_in_pvpq_t, mt_pq, nt_pvpq)):
    #     i = pq_lookup[m]
    #     j = pvpq_lookup[n]
    #     J_C_Q_d[i, i] = p_tcsc_ij
    #     J_C_Q_d[i, j] = -p_tcsc_ij
    #     # J_C_Q_d[i, i] = Sbus[tcsc_tb].real
    #     # J_C_Q_d[i, j] = -Sbus[tcsc_tb].real
    # #J_C_Q_d[0:len(tcsc_branches), tcsc_i] = p_tcsc_ij
    # #J_C_Q_d[0:len(tcsc_branches), tcsc_j] = -p_tcsc_ij
    #

    J_C_Q_d = np.zeros(shape=(len(pq), len(pvpq)))
    for in_pvpq, m, n in zip([tcsc_in_pvpq_f], [mf_pq], [nf_pvpq]):
        i = pq_lookup[m]
        j = pvpq_lookup[n]
        J_C_Q_d[i, i] = p_tcsc_ij[m]
        J_C_Q_d[i, j] = -p_tcsc_ij[m]
    for in_pvpq, m, n in zip([tcsc_in_pvpq_t], [mt_pq], [nt_pvpq]):
        i = pq_lookup[m]
        j = pvpq_lookup[n]
        J_C_Q_d[i, i] = p_tcsc_ij[m]
        J_C_Q_d[i, j] = -p_tcsc_ij[m]
    # J_C_Q_d = J21.toarray()
    # J_C_Q_u
    # J_C_Q_u = np.zeros(shape=(len(pq), len(pq)))
    # for in_pq, m, n in ((tcsc_in_pq_f, mf_pq, nf_pq ), (tcsc_in_pq_t, mt_pq, nt_pq )):
    #     i = pq_lookup[m]
    #     j = pq_lookup[n]
    #     J_C_Q_u[i, i] = 2*q_tcsc_ii + q_tcsc_ij
    #     J_C_Q_u[i, j] = q_tcsc_ij
    #
    #     if len(m) == 0 or len(n) == 0: continue
    #     # J_C_Q_u[i, i] = 2*np.square(np.abs(V[m])) * np.abs(Ybus[m, m]) + Sbus[tcsc_tb].imag
    #     # J_C_Q_u[i, j] = Sbus[tcsc_tb].imag
    # #J_C_Q_u[0:len(tcsc_branches), tcsc_i] = 2*q_tcsc_ii + q_tcsc_ij
    # #J_C_Q_u[0:len(tcsc_branches), tcsc_j] = q_tcsc_ij


    J_C_Q_u = np.zeros(shape=(len(pq), len(pq)))
    for in_pq, m, n in zip([tcsc_in_pq_f], [mf_pq], [nf_pq]):
        i = pq_lookup[m]
        j = pq_lookup[n]
        J_C_Q_u[i, i] = 2*q_tcsc_ii[m] + q_tcsc_ij[m]
        J_C_Q_u[i, j] = q_tcsc_ij[m]

        if len(m) == 0 or len(n) == 0: continue
    for in_pq, m, n in zip([tcsc_in_pq_t], [mt_pq], [nt_pq]):
        i = pq_lookup[m]
        j = pq_lookup[n]
        J_C_Q_u[i, i] = 2*q_tcsc_ii[m] + q_tcsc_ij[m]
        J_C_Q_u[i, j] = q_tcsc_ij[m]

        #if len(m) == 0 or len(n) == 0: continue
    # J_C_Q_u = J22.toarray()
    # J_C_P_c
    # J_C_P_c = np.zeros(shape=(len(pvpq), len(x_control)))
    # aux_term = 2*(np.cos(2*x_control[x_control_lookup==1] - 1)) / (np.pi * tcsc_x_l_pu * y_tcsc)
    #
    # for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
    #     i = pvpq_lookup[m]
    #     J_C_P_c[i, x_control_lookup==1] = aux_term * p_tcsc_ij
    #     # J_C_P_c[i, x_control_lookup==1] = aux_term * Sbus[tcsc_tb].real
    #     #J_C_P_c[i, j] = aux_term * p_tcsc_ij
    # #J_C_P_c[0:len(tcsc_branches), x_control_lookup==1] = aux_term * p_tcsc_ij

    J_C_P_c = np.zeros(shape=(len(pvpq), len(pvpq)))
    aux_term = 2*(np.cos(2*x_control[x_control_lookup==1][pvpq])- 1) / (np.pi * tcsc_x_l_pu[pvpq] * y_tcsc)
    for in_pvpq, m in zip([tcsc_in_pvpq_f], [mf_pvpq]):
        i = pvpq_lookup[m]
        J_C_P_c[i, i] = aux_term[i] * p_tcsc_ij[m]
    for in_pvpq, m in zip([tcsc_in_pvpq_t], [mt_pvpq]):
        i = pvpq_lookup[m]
        J_C_P_c[i, i] = aux_term[i] * p_tcsc_ij[m]

#
#     # J_C_Q_c
#     J_C_Q_c = np.zeros(shape=(len(pq), len(x_control)))
#     for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
#         i = pvpq_lookup[m]
#         if len(m) == 0: continue
#         #J_C_Q_c[i, x_control_lookup==1] = aux_term * (q_tcsc_ii + q_tcsc_ij)
#         J_C_Q_c[i, x_control_lookup==1] = aux_term * (np.square(np.abs(V[m])) * np.abs(Ybus[m, m]) + Sbus[tcsc_tb].imag)
# #    J_C_Q_c[0:len(tcsc_branches), x_control_lookup==1] = aux_term * (q_tcsc_ii + q_tcsc_ij)
#

    J_C_Q_c = np.zeros(shape=(len(pq), len(pq)))
    for in_pvpq, m in zip([tcsc_in_pvpq_f], [mf_pq]):
        i = pq_lookup[m]
        #if len(m) == 0: continue
        J_C_Q_c[i, i] = aux_term[i] * (q_tcsc_ii[m] + q_tcsc_ij[m])
    for in_pvpq, m in zip([tcsc_in_pvpq_t], [mt_pq]):
        i = pq_lookup[m]
        #if len(m) == 0: continue
        J_C_Q_c[i, i] = aux_term[i] * (q_tcsc_ii[m] + q_tcsc_ij[m])


    # J_C_C_d
#     J_C_C_d = np.zeros(shape=(len(x_control), len(pvpq)))
#     for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
#         i = pvpq_lookup[m]
#         j = pvpq_lookup[n]
#         J_C_C_d[x_control_lookup==1, i] = A * np.sin(np.angle(V[n]) + phi_tcsc_ij - np.angle(V[m]))
#         # J_C_C_d[x_control_lookup==1, j] = A * np.sin(np.angle(V[m]) + phi_tcsc_ij - np.angle(V[n]))
#
# #    J_C_C_d[:, tcsc_i] = A * np.sin(np.angle(V[tcsc_j]) + phi_tcsc_ij - np.angle(V[tcsc_i]))
# #    J_C_C_d[:, tcsc_j] = A * np.sin(np.angle(V[tcsc_i]) - phi_tcsc_ij - np.angle(V[tcsc_j]))

    J_C_C_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
        i = pvpq_lookup[m]
        j = pvpq_lookup[n]
        # todo: it becomes negative for i
        J_C_C_d[i, i] = -np.abs(V[m]) * np.abs(Ybus_tcsc[m, n]) * np.abs(V[n]) * np.sin(np.angle(V[m]) - np.angle(V[n]) + np.angle(np.array(Ybus_tcsc[m,n])))  #### A_ij = A_ji    phi_tcsc_ij = phi_tcsc_ji

    # # J_C_C_u
    # J_C_C_u = np.zeros(shape=(len(x_control), len(pq)))
    # for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
    #     i = pvpq_lookup[m]
    #     j = pvpq_lookup[n]
    #     if len(m) == 0: continue
    #     J_C_C_u[x_control_lookup==1, i] = A / abs(V[m]) * np.cos(np.angle(V[n]) - np.angle(V[m]) + phi_tcsc_ij)
    #
    # #J_C_C_u[:, tcsc_i] = A / abs(V[tcsc_i]) * np.cos(np.angle(V[tcsc_j]) - np.angle(V[tcsc_i]) + phi_tcsc_ij)
    # #J_C_C_u[:, tcsc_j] = A / abs(V[tcsc_j]) * np.cos(np.angle(V[tcsc_j]) - np.angle(V[tcsc_i]) + phi_tcsc_ij)
    #

    J_C_C_u = np.zeros(shape=(len(pvpq), len(pq)))
    for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pq, nf_pq), (tcsc_in_pvpq_t, mt_pq, nt_pq)):
        i = pq_lookup[m]
        j = pq_lookup[n]
        if len(m) == 0: continue
        J_C_C_u[i, i] = np.abs(V[n]) * np.abs(Ybus_tcsc[m, n]) * np.cos(np.angle(V[m]) - np.angle(V[n]) + np.angle(np.array(Ybus_tcsc[m,n])))

    # J_C_C_c
    J_C_C_c = np.zeros(shape=(len(pvpq), len(pvpq)))
    for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
        #i = pvpq_lookup[m]
        #j = pvpq_lookup[n]
        #m = mf_pvpq
        #n = nf_pvpq
        if len(m) ==0 or len(n) ==0:
            continue

        # todo: angles
        # B = abs(V[m]) * abs(V[n]) * np.cos(np.angle(V[m]) - np.angle(V[n]) + np.angle(Ybus_tcsc[m,n]))
        # B = abs(V[m]) * abs(V[n]) * np.cos(np.angle(V[m]) - np.angle(V[n]) + np.angle(Ybus_tcsc[m,n]))
        # J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] =  2 * B * (np.cos(2 * x_control[x_control_lookup==1])-1) / (np.pi * tcsc_x_l_pu)

        x = x_control[x_control_lookup==1]
        J_C_C_c[i, i] = 2 * np.abs(V[m]) * np.abs(V[n]) * (np.cos(2 * x[m]) - 1) * np.cos(np.angle([V[m]]) - np.angle(V[n]) + np.angle(np.array(Ybus_tcsc[m,n]))) / (np.pi * tcsc_x_l_pu[m])

        # J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] = (2*abs(V[n]) * abs(V[m]) * (np.cos(2 * x_control[x_control_lookup==1])-1)*np.cos(np.angle(V[n]) - np.angle(V[m]) + np.angle(np.array(Ybus_tcsc[n,m]))))/ (np.pi * tcsc_x_l_pu)
    # J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] = (2*abs(V[tcsc_fb]) * abs(V[tcsc_tb]) * (np.cos(2 * x_control[x_control_lookup==1])-1)*np.cos(np.angle(V[tcsc_tb]) - np.angle(V[tcsc_fb]) + np.angle(Ybus_tcsc[tcsc_fb,tcsc_tb])))/ (np.pi * tcsc_x_l_pu)
    #print("JCCc", J_C_C_c)
    # J_C_C_c = np.array([[2.755]])
    #J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] = np.array(x_control[x_control_lookup==1])
    #J_C_C_c[:, tcsc_i] = - 2 * B * np.cos(2 * x_control[x_control_lookup==1]-1) / (np.pi * tcsc_x_l_pu)
    #J_C_C_c[:, tcsc_j] = - 2 * B * np.cos(2 * x_control[x_control_lookup==1]-1) / (np.pi * tcsc_x_l_pu)

    J_m = np.vstack(
        [np.hstack([J_C_P_d, J_C_P_u, J_C_P_c]),
         np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c]),
         np.hstack([J_C_C_d, J_C_C_u, J_C_C_c])])

    #print("J", J_m)

    J_m = csr_matrix(J_m)

    return J_m
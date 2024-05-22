# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags
from pandapower.pf.makeYbus_facts import calc_y_svc_pu

SMALL_NUMBER = 1e-9

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
        J_C_P_c[pvpq_lookup[f[f_in_pvpq]], (nsvc + np.arange(ntcsc))[f_in_pvpq]] = S_Fi_dx.real
    if np.any(t_in_pvpq):
        J_C_P_c[pvpq_lookup[t[t_in_pvpq]], (nsvc + np.arange(ntcsc))[t_in_pvpq]] = S_Fk_dx.real
    # J_C_P_c = np.array([[S_Fi_dx.real], [S_Fk_dx.real]]).reshape(2, 1)

    J_C_Q_c = np.zeros(shape=(len(pq), nsvc + ntcsc), dtype=np.float64)
    if np.any(f_in_pq):
        J_C_Q_c[pq_lookup[f[f_in_pq]], (nsvc + np.arange(ntcsc))[f_in_pq]] = S_Fi_dx.imag
    if np.any(t_in_pq):
        J_C_Q_c[pq_lookup[t[t_in_pq]], (nsvc + np.arange(ntcsc))[t_in_pq]] = S_Fk_dx.imag
    # J_C_Q_c = np.array([[S_Fi_dx.imag], [S_Fk_dx.imag]]).reshape(2, 1)

    # the signs are opposite here for J_C_C_d, J_C_C_u, J_C_C_c and I don't know why
    # main mode of operation - set point for p_to_mw:
    # J_C_C_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    J_C_C_d = np.zeros(shape=(nsvc + ntcsc, len(pvpq)), dtype=np.float64)
    if np.any(f_in_pvpq):
        J_C_C_d[(nsvc + np.arange(ntcsc))[f_in_pvpq], pvpq_lookup[f[f_in_pvpq]]] = S_Fik.imag
    if np.any(t_in_pvpq):
        J_C_C_d[(nsvc + np.arange(ntcsc))[t_in_pvpq], pvpq_lookup[t[t_in_pvpq]]] = -S_Fik.imag

    J_C_C_u = np.zeros(shape=(nsvc + ntcsc, len(pq)), dtype=np.float64)
    if np.any(f_in_pq):
        J_C_C_u[(nsvc + np.arange(ntcsc))[f_in_pq], pq_lookup[f[f_in_pq]]] = S_Fik.real / Vmf
    if np.any(t_in_pq):
        J_C_C_u[(nsvc + np.arange(ntcsc))[t_in_pq], pq_lookup[t[t_in_pq]]] = S_Fik.real / Vmt

    J_C_C_c = np.zeros(shape=(nsvc + ntcsc, nsvc + ntcsc), dtype=np.float64)
    J_C_C_c[np.r_[nsvc:nsvc + ntcsc], np.r_[nsvc:nsvc + ntcsc]] = -S_Fi_dx.real  # .flatten()?

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
        relevant = np.r_[np.arange(nsvc), nsvc + np.arange(ntcsc)[tcsc_controllable]]
        J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u, J_C_P_c[:, relevant]]),
                         np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c[:, relevant]]),
                         np.hstack([J_C_C_d[relevant, :], J_C_C_u[relevant, :],
                                    J_C_C_c[:, relevant][relevant, :]])])
    else:
        J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u]),
                         np.hstack([J_C_Q_d, J_C_Q_u])])

    J_m = csr_matrix(J_m)

    return J_m


def create_J_modification_ssc(J, V, Ybus_ssc, f, t, pvpq, pq, pvpq_lookup, pq_lookup, control_v, control_delta):
    """
    creates the modification Jacobian matrix for SSC (STATCOM)

    Parameters
    ----------
    control_delta
    V
        array of np.complex128
    Ybus_ssc
        scipy.sparse.csr_matrix
    f
        array of np.int64
    t
        array of np.int64
    pvpq
        array of np.int64
    pq
        array of np.int64
    pvpq_lookup
        array of np.int64
    pq_lookup
        array of np.int64

    Returns
    -------
    J_m
        scipy.sparse.csr_matrix

    """
    #

    Vf = V[f]

    Vt = V[t]

    Vmf = np.abs(Vf)
    Vmt = np.abs(Vt)

    S_Fii = Vf * np.conj(Ybus_ssc.toarray()[f, f] * Vf)
    S_Fkk = Vt * np.conj(Ybus_ssc.toarray()[t, t] * Vt)
    P_Fii, Q_Fii = S_Fii.real, S_Fii.imag
    P_Fkk, Q_Fkk = S_Fkk.real, S_Fkk.imag

    S_Fik = Vf * np.conj(Ybus_ssc.toarray()[f, t] * Vt)
    S_Fki = Vt * np.conj(Ybus_ssc.toarray()[t, f] * Vf)
    P_Fik, Q_Fik = S_Fik.real, S_Fik.imag
    P_Fki, Q_Fki = S_Fki.real, S_Fki.imag

    # seems like it is not used:
    # S_ii = np.abs(V[f]) ** 2 * np.abs(Ybus[f, f]) * np.exp(1j * np.angle(Ybus[f, f].conj()))  ####
    # S_kk = np.abs(V[t]) ** 2 * np.abs(Ybus[t, t]) * np.exp(1j * np.angle(Ybus[t, t].conj()))  ####
    #
    # S_ij = Sbus[f] - S_ii
    # S_kj = Sbus[t] - S_kk


    f_in_pq = np.isin(f, pq)
    f_in_pvpq = np.isin(f, pvpq)
    t_in_pq = np.isin(t, pq)
    t_in_pvpq = np.isin(t, pvpq)

    zero = SMALL_NUMBER * np.ones(shape=S_Fii.shape, dtype=np.float64)
    one = np.ones(shape=S_Fii.shape, dtype=np.float64)

    rows = np.array([], dtype=np.float64)
    cols = np.array([], dtype=np.float64)
    data = np.array([], dtype=np.float64)

    # # todo: use _sum_by_group what multiple elements start (or end) at the same bus?
    # # J_C_P_d = np.zeros(shape=(len(pvpq) + len(x_control), len(pvpq) + len(x_control)), dtype=np.float64)
    # # J_C_P_d = np.zeros(shape=(len(pvpq), len(pvpq)), dtype=np.float64)
    # #if np.any(f_in_pvpq):
    #     # J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq]]] = -S_Fik.imag
    #     # # J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[f_in_pvpq]]+ len(x_control)] = S_Fik.imag
    #     # J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[f_in_pvpq]]] = S_Fik.imag
    #     #
    #     # # J_C_P_d[pvpq_lookup[t[f_in_pvpq]] + len(x_control), pvpq_lookup[f[f_in_pvpq]]] = S_Fki.imag
    #     # # J_C_P_d[pvpq_lookup[t[f_in_pvpq]] + len(x_control), pvpq_lookup[t[f_in_pvpq]]+ len(x_control)] = -S_Fki.imag
    #     #
    #     # J_C_P_d[pvpq_lookup[t[f_in_pvpq]] , pvpq_lookup[f[f_in_pvpq]]] = S_Fki.imag
    #     # J_C_P_d[pvpq_lookup[t[f_in_pvpq]] , pvpq_lookup[t[f_in_pvpq]]] = -S_Fki.imag
    #
    # if np.any(f_in_pvpq):
    #     J_m[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq]]] = -S_Fik.imag
    # if np.any(f_in_pvpq & t_in_pvpq):
    #     J_m[pvpq_lookup[f[f_in_pvpq & t_in_pvpq]], pvpq_lookup[t[f_in_pvpq & t_in_pvpq]]] = S_Fik.imag
    #     J_m[pvpq_lookup[t[f_in_pvpq & t_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pvpq]]] = S_Fki.imag
    # if np.any(t_in_pvpq):
    #     J_m[pvpq_lookup[t[t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]] = -S_Fki.imag

    rows = np.r_[rows, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pvpq]], pvpq_lookup[t[f_in_pvpq & t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]]
    cols = np.r_[cols, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[f_in_pvpq & t_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]]
    # todo
    # data = np.r_[data, -S_Fik.imag[f_in_pvpq], S_Fik.imag[f_in_pvpq & t_in_pvpq], S_Fki.imag[f_in_pvpq & t_in_pvpq], -S_Fki.imag[t_in_pvpq]]
    # data = np.r_[data, -S_Fik.imag[f_in_pvpq], S_Fik.imag[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], zero[t_in_pvpq]]
    data = np.r_[data, -Q_Fik[f_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq], np.where(control_delta[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], Q_Fki[f_in_pvpq & t_in_pvpq]), np.where(control_delta[t_in_pvpq], zero[t_in_pvpq], -Q_Fki[t_in_pvpq])]


    # # J_C_P_u = np.zeros(shape=(len(pvpq), len(pq)), dtype=np.float64)
    # # J_C_P_u = np.zeros(shape=(len(pvpq)+ len(x_control), len(pq)+ len(x_control)), dtype=np.float64)
    #
    # #if np.any(f_in_pvpq & f_in_pq):  ## TODO check if this conditon includes all cases, and check trough tests
    #     # J_C_P_u[pvpq_lookup[f[f_in_pvpq]], pq_lookup[f[f_in_pq]]] = (2 * S_Fii.real + S_Fik.real) / Vmf
    #     # J_C_P_u[pvpq_lookup[f[f_in_pvpq]], pq_lookup[t[f_in_pq]] ] = S_Fik.real/Vmt
    #     #
    #     # J_C_P_u[pvpq_lookup[t[f_in_pvpq]], pq_lookup[f[f_in_pq]]] = S_Fki.real/Vmf
    #     # J_C_P_u[pvpq_lookup[t[f_in_pvpq]], pq_lookup[t[f_in_pq]]] = (2 * S_Fkk.real + S_Fki.real) / Vmt
    #
    # if np.any(f_in_pvpq & f_in_pq):
    #     J_m[pvpq_lookup[f[f_in_pvpq]], len(pvpq)+pq_lookup[f[f_in_pq]]] = (2 * S_Fii.real + S_Fik.real) / Vmf
    # if np.any(f_in_pvpq & t_in_pq):
    #     J_m[pvpq_lookup[f[f_in_pvpq & t_in_pq]], len(pvpq)+pq_lookup[t[f_in_pvpq & t_in_pq]] ] = S_Fik.real/Vmt
    #     J_m[pvpq_lookup[t[f_in_pvpq & t_in_pq]], len(pvpq)+pq_lookup[f[f_in_pvpq & t_in_pq]]] = S_Fki.real/Vmf
    # if np.any(t_in_pvpq & t_in_pq):
    #     J_m[pvpq_lookup[t[t_in_pvpq & t_in_pq]], len(pvpq)+pq_lookup[t[t_in_pvpq & t_in_pq]]] = (2 * S_Fkk.real + S_Fki.real) / Vmt

    rows = np.r_[rows, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pq]], pvpq_lookup[t[f_in_pvpq & t_in_pq]], pvpq_lookup[t[t_in_pvpq & t_in_pq]]]
    cols = np.r_[cols, len(pvpq)+pq_lookup[f[f_in_pq]], len(pvpq)+pq_lookup[t[f_in_pvpq & t_in_pq]], len(pvpq)+pq_lookup[f[f_in_pvpq & t_in_pq]], len(pvpq)+pq_lookup[t[t_in_pvpq & t_in_pq]]]
    # data = np.r_[data, ((2 * S_Fii.real + S_Fik.real) / Vmf)[f_in_pvpq], (S_Fik.real/Vmt)[f_in_pvpq & t_in_pq], (S_Fki.real/Vmf)[f_in_pvpq & t_in_pq], ((2 * S_Fkk.real + S_Fki.real) / Vmt)[t_in_pvpq & t_in_pq]]
    data = np.r_[data, ((2 * P_Fii + P_Fik) / Vmf)[f_in_pvpq], (P_Fik/Vmt)[f_in_pvpq & t_in_pq], np.where(control_delta[f_in_pvpq & t_in_pq], zero[f_in_pvpq & t_in_pq], (P_Fki/Vmf)[f_in_pvpq & t_in_pq]), np.where(control_delta[t_in_pvpq & t_in_pq], zero[t_in_pvpq & t_in_pq], ((2 * P_Fkk + P_Fki) / Vmt)[t_in_pvpq & t_in_pq])]


    # # J_C_Q_d = np.zeros(shape=(len(pq), len(pvpq)), dtype=np.float64)
    # # J_C_Q_d = np.zeros(shape=(len(pq)+ len(x_control), len(pvpq)+ len(x_control)), dtype=np.float64)
    # #if np.any(f_in_pvpq & f_in_pq):
    #     # J_C_Q_d[pq_lookup[f[f_in_pq]], pvpq_lookup[f[f_in_pvpq]]] = S_Fik.real
    #     # J_C_Q_d[pq_lookup[f[f_in_pq]], pvpq_lookup[t[f_in_pvpq]]] = -S_Fik.real
    #     #
    #     # # J_C_Q_d[pq_lookup[t[f_in_pq]]+ len(x_control), pvpq_lookup[f[f_in_pvpq]]] = 0
    #     # # J_C_Q_d[pq_lookup[t[f_in_pq]]+ len(x_control), pvpq_lookup[t[f_in_pvpq]]+ len(x_control)] = 0
    # if np.any(f_in_pvpq & f_in_pq):
    #     J_m[len(pvpq) + pq_lookup[f[f_in_pq]], pvpq_lookup[f[f_in_pvpq]]] = S_Fik.real  # todo: should it be & or |?
    # if np.any(f_in_pq & t_in_pvpq):
    #     J_m[len(pvpq) + pq_lookup[f[f_in_pq & t_in_pvpq]], pvpq_lookup[t[f_in_pq & t_in_pvpq]]] = -S_Fik.real
    # if np.any(f_in_pvpq & t_in_pq):
    #     J_m[len(pvpq) + pq_lookup[t[f_in_pvpq & t_in_pq]], pvpq_lookup[f[f_in_pvpq & t_in_pq]]] = np.where(control_mode_v, SMALL_NUMBER, S_Fik.real)  # control mode V or Q
    # if np.any(t_in_pq & t_in_pvpq):
    #     J_m[len(pvpq) + pq_lookup[t[t_in_pq]], pvpq_lookup[t[t_in_pvpq]]] = np.where(control_mode_v, SMALL_NUMBER, -S_Fik.real) # control mode V or Q

    rows = np.r_[rows, len(pvpq) + pq_lookup[f[f_in_pq]], len(pvpq) + pq_lookup[f[f_in_pq & t_in_pvpq]], len(pvpq) + pq_lookup[t[f_in_pvpq & t_in_pq]], len(pvpq) + pq_lookup[t[t_in_pq]]]
    cols = np.r_[cols, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[f_in_pq & t_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pq]], pvpq_lookup[t[t_in_pvpq]]]
    data = np.r_[data, P_Fik[f_in_pq], -P_Fik[f_in_pq & t_in_pvpq], np.where(control_v[f_in_pvpq & t_in_pq], zero[f_in_pvpq & t_in_pq], P_Fik[f_in_pvpq & t_in_pq]), np.where(control_v[t_in_pq], zero[t_in_pq], -P_Fik[t_in_pq])]



    # # J_C_Q_u = np.zeros(shape=(len(pq), len(pq)), dtype=np.float64)
    # # J_C_Q_u = np.zeros(shape=(len(pq)+ len(x_control), len(pq)+ len(x_control)), dtype=np.float64)
    # # if np.any(f_in_pq):
    #     # J_C_Q_u[pq_lookup[f[f_in_pq]], pq_lookup[f[f_in_pq]]] = (2 * S_Fii.imag + S_Fik.imag) / Vmf
    #     # J_C_Q_u[pq_lookup[f[f_in_pq]], pq_lookup[t[f_in_pq]]] = S_Fik.imag/Vmt
    #     #
    #     # J_C_Q_u[pq_lookup[t[f_in_pq]], pq_lookup[f[f_in_pq]]] = 1
    #     # J_C_Q_u[pq_lookup[t[f_in_pq]], pq_lookup[t[f_in_pq]]] = 0
    # if np.any(f_in_pq):
    #     J_m[len(pvpq)+pq_lookup[f[f_in_pq]], len(pvpq)+pq_lookup[f[f_in_pq]]] = (2 * S_Fii.imag + S_Fik.imag) / Vmf
    # if np.any(f_in_pq & t_in_pq):
    #     J_m[len(pvpq)+pq_lookup[f[f_in_pq & t_in_pq]], len(pvpq)+pq_lookup[t[f_in_pq & t_in_pq]]] = S_Fik.imag / Vmt
    #     J_m[len(pvpq)+pq_lookup[t[f_in_pq & t_in_pq]], len(pvpq)+pq_lookup[f[f_in_pq & t_in_pq]]] = np.where(control_mode_v, 1, (2 * S_Fii.imag + S_Fik.imag) / Vmf)  # control mode V or Q
    # if np.any(t_in_pq):
    #     J_m[len(pvpq)+pq_lookup[t[t_in_pq]], len(pvpq)+pq_lookup[t[t_in_pq]]] = np.where(control_mode_v, SMALL_NUMBER, S_Fik.imag / Vmt)  # control mode V or Q
    #     #
    #     # J_C_Q_u[pq_lookup[t[f_in_pq]]+ len(x_control), pq_lookup[f[f_in_pq]]] = 1
    #     # J_C_Q_u[pq_lookup[t[f_in_pq]]+ len(x_control), pq_lookup[t[f_in_pq]]+ len(x_control)] = 0

    rows = np.r_[rows, len(pvpq)+pq_lookup[f[f_in_pq]], len(pvpq)+pq_lookup[f[f_in_pq & t_in_pq]], len(pvpq)+pq_lookup[t[f_in_pq & t_in_pq]], len(pvpq)+pq_lookup[t[t_in_pq]]]
    cols = np.r_[cols, len(pvpq)+pq_lookup[f[f_in_pq]], len(pvpq)+pq_lookup[t[f_in_pq & t_in_pq]], len(pvpq)+pq_lookup[f[f_in_pq & t_in_pq]], len(pvpq)+pq_lookup[t[t_in_pq]]]
    data = np.r_[data, ((2 * Q_Fii + Q_Fik) / Vmf)[f_in_pq], (Q_Fik / Vmt)[f_in_pq & t_in_pq], np.where(control_v[f_in_pq & t_in_pq], one[f_in_pq & t_in_pq], ((2 * Q_Fii + Q_Fik) / Vmf)[f_in_pq & t_in_pq]), np.where(control_v[t_in_pq], zero[t_in_pq], (Q_Fik / Vmt)[t_in_pq])]

    # for vsc ac slack buses:
    # rows = np.r_[rows, ac_slack]
    # cols = np.r_[cols, ac_slack]
    # data = np.r_[data, np.ones_like(ac_slack, dtype=np.float64)]

    # J_C_P_c = np.zeros(shape=(len(pvpq), nsvc + ntcsc), dtype=np.float64)
    # J_C_Q_c = np.zeros(shape=(len(pq), nsvc + ntcsc + 2 * nssc), dtype=np.float64)
    # J_C_C_d = np.zeros(shape=(nsvc + ntcsc + 2 * nssc, len(pvpq)), dtype=np.float64)
    # J_C_C_u = np.zeros(shape=(nsvc + ntcsc + 2 * nssc, len(pq)), dtype=np.float64)
    # J_C_C_c = np.zeros(shape=(nsvc + ntcsc + 2 *nssc, nsvc + ntcsc + 2 *nssc), dtype=np.float64)



    # if np.any(tcsc_controllable) or np.any(svc_controllable):  # todo
    #     relevant = np.r_[np.arange(nsvc), nsvc + np.arange(ntcsc)[tcsc_controllable]]
    #     J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u, J_C_P_c[:, relevant]]),
    #                      np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c[:, relevant]]),
    #                      np.hstack([J_C_C_d[relevant, :], J_C_C_u[relevant, :],
    #                                 J_C_C_c[:, relevant][relevant, :]])])
    # else:
    #     J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u]),
    #                      np.hstack([J_C_Q_d, J_C_Q_u])])

    #J_m = csr_matrix(J_m)
    J_m = csr_matrix((data, (rows, cols)), shape=J.shape, dtype=np.float64)

    return J_m


def create_J_modification_hvdc(J, V_dc, Ybus_hvdc, Ybus_vsc_dc, dc_p, dc_p_lookup, vsc_dc_fb, vsc_dc_tb, vsc_dc_slack):
    # Calculate the first matrix for all elements
    J_all = (Ybus_hvdc - Ybus_vsc_dc).multiply(V_dc)
    # Calculate the second matrix for diagonal elements only
    J_diag = diags(V_dc).multiply((Ybus_hvdc - Ybus_vsc_dc).dot(V_dc))

    # Calculate DC modification VSC Jacobian
    P_Fii = V_dc[vsc_dc_fb] * Ybus_vsc_dc.toarray()[vsc_dc_fb, vsc_dc_fb]#.dot(V_dc[vsc_dc_fb])
    P_Fkk = V_dc[vsc_dc_tb] * Ybus_vsc_dc.toarray()[vsc_dc_tb, vsc_dc_tb]#.dot(V_dc[vsc_dc_tb])

    P_Fik = V_dc[vsc_dc_fb] * Ybus_vsc_dc.toarray()[vsc_dc_fb, vsc_dc_tb]#.dot(V_dc[vsc_dc_tb])
    P_Fki = V_dc[vsc_dc_tb] * Ybus_vsc_dc.toarray()[vsc_dc_tb, vsc_dc_fb]#.dot(V_dc[vsc_dc_fb])

    zero = SMALL_NUMBER * np.ones(shape=P_Fii.shape, dtype=np.float64)
    one = np.ones(shape=P_Fii.shape, dtype=np.float64)

    rows = np.r_[dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_tb], dc_p_lookup[vsc_dc_tb]]
    cols = np.r_[dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_tb], dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_tb]]
    # data = np.r_[2 * P_Fii + P_Fik, P_Fik, np.where(vsc_dc_slack, one, P_Fki), np.where(vsc_dc_slack, zero, 2 * P_Fkk + P_Fki)]
    data = np.r_[2 * P_Fii + P_Fik, P_Fik, np.where(vsc_dc_slack, one, P_Fik), np.where(vsc_dc_slack, zero, 2 * P_Fii + P_Fik)]
    # data = np.r_[P_Fik / V_dc[vsc_dc_fb], P_Fik / V_dc[vsc_dc_tb], np.where(vsc_dc_slack, one, P_Fki / V_dc[vsc_dc_fb]), np.where(vsc_dc_slack, zero, P_Fki / V_dc[vsc_dc_tb])]
    # data = np.r_[-Q_Fik[f_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq], np.where(control_delta[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], Q_Fki[f_in_pvpq & t_in_pvpq]), np.where(control_delta[t_in_pvpq], zero[t_in_pvpq], -Q_Fki[t_in_pvpq])]

    J_m_vsc = csr_matrix((data, (rows, cols)), shape=J_all.shape, dtype=np.float64)
    # Combine them to form the Jacobian
    J_combined = J_all + J_diag + J_m_vsc

    num_p = len(dc_p)
    offset = J.shape[0] - num_p
    # Create an initial zero sparse matrix for J_m
    J_m = lil_matrix(J.shape)
    J_m[offset:offset+num_p, offset:offset+num_p] = J_combined[dc_p, :][:, dc_p]

    return J_m.tocsr()


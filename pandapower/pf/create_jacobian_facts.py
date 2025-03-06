# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags
from pandapower.pf.makeYbus_facts import calc_y_svc_pu

SMALL_NUMBER = 1e-9


def create_J_modification_svc(J, svc_buses, pvpq, pq, pq_lookup, Vm, x_control, svc_x_l_pu, svc_x_cvar_pu,
                              nsvc_controllable, svc_controllable):

    in_pq = np.isin(svc_buses, pq)
    lpvpq = len(pvpq)
    lpq = len(pq)

    one = np.ones(shape=len(svc_buses), dtype=np.float64)

    y_svc = calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu)
    q_svc = np.square(Vm[svc_buses]) * y_svc

    rows = np.array([], dtype=np.float64)
    cols = np.array([], dtype=np.float64)
    data = np.array([], dtype=np.float64)

    # J_m = np.zeros_like(J.toarray())
    # controllable = np.arange(nsvc)[svc_controllable]

    # # J_C_Q_u
    # J_C_Q_u = np.zeros(shape=(len(pq), len(pq)), dtype=np.float64)
    # J_C_Q_u[pq_lookup[svc_buses[in_pq]], pq_lookup[svc_buses[in_pq]]] = 2 * q_svc[in_pq]
    # # count pvpq rows and pvpq columns from top left
    # J_m[len(pvpq):len(pvpq) + len(pq), len(pvpq):len(pvpq) + len(pq)] = J_C_Q_u

    rows = np.r_[rows, lpvpq + pq_lookup[svc_buses[in_pq]]]
    cols = np.r_[cols, lpvpq + pq_lookup[svc_buses[in_pq]]]
    data = np.r_[data, 2 * q_svc[in_pq]]

    # # J_C_Q_c
    # if np.any(svc_controllable):
    #     J_C_Q_c = np.zeros(shape=(len(pq), nsvc), dtype=np.float64)
    #     values = 2 * np.abs(Vm[svc_buses]) ** 2 * (np.cos(2 * x_control) - 1) / (np.pi * svc_x_l_pu)
    #     J_C_Q_c[pq_lookup[svc_buses[in_pq & svc_controllable]], controllable] = values[in_pq & svc_controllable]
    #     # count pvpq rows and pvpq columns from top left
    #     J_m[len(pvpq):len(pvpq) + len(pq),
    #     len(pvpq) + len(pq):len(pvpq) + len(pq) + nsvc_controllable] = J_C_Q_c[:, controllable]

    if np.any(svc_controllable):
        jcqc_values = 2* np.square(np.abs(Vm[svc_buses])) * (np.cos(2 * x_control) - 1) / (np.pi * svc_x_l_pu)
        rows = np.r_[rows, lpvpq + pq_lookup[svc_buses[in_pq & svc_controllable]]]
        cols = np.r_[cols, lpvpq + lpq:lpvpq + lpq + nsvc_controllable]
        data = np.r_[data, jcqc_values[in_pq & svc_controllable]]

    # # J_C_C_u
    # # d(Ui - Ui,set)/d(Ui) = dUi/dUi = 1
    # if np.any(svc_controllable):
    #     J_C_C_u = np.zeros(shape=(nsvc, len(pq)), dtype=np.float64)
    #     J_C_C_u[controllable, pq_lookup[svc_buses[in_pq & controllable]]] = 1
    #     # count pvpq rows and pvpq columns from top left
    #     J_m[len(pvpq) + len(pq):len(pvpq) + len(pq) + nsvc_controllable,
    #     len(pvpq):len(pvpq) + len(pq)] = J_C_C_u[controllable, :]

    if np.any(svc_controllable):
        rows = np.r_[rows, lpvpq + lpq:lpvpq + lpq + nsvc_controllable]
        cols = np.r_[cols, lpvpq + pq_lookup[svc_buses[in_pq & svc_controllable]]]
        data = np.r_[data, one[in_pq & svc_controllable]]

    J_m = csr_matrix((data, (rows, cols)), shape=J.shape, dtype=np.float64)

    return J_m


def create_J_modification_tcsc(J, V, y_tcsc_pu, x_control, tcsc_controllable,
                               tcsc_x_l_pu, f, t, pvpq, pq, pvpq_lookup, pq_lookup, nsvc):
    # y_tcsc_pu = calc_y_svc_pu(x_control, tcsc_x_l_pu, tcsc_x_cvar_pu)
    # S_tcsc_pu = V * (Ybus_tcsc.conj() @ V.conj())
    dY_TCSC_dx = 2 * (np.cos(2 * x_control) - 1) / (np.pi * tcsc_x_l_pu)

    Vf = V[f]
    Vt = V[t]

    Vmf = np.abs(Vf)
    Vmt = np.abs(Vt)

    # S_Fii = Vf * np.conj(Ybus_tcsc.toarray()[f, f] * Vf)
    # S_Fkk = Vt * np.conj(Ybus_tcsc.toarray()[t, t] * Vt)
    S_Fii = Vf * np.conj(y_tcsc_pu * Vf)
    S_Fkk = Vt * np.conj(y_tcsc_pu * Vt)
    P_Fii, Q_Fii = S_Fii.real, S_Fii.imag
    P_Fkk, Q_Fkk = S_Fkk.real, S_Fkk.imag

    # S_Fik = Vf * np.conj(Ybus_tcsc.toarray()[f, t] * Vt)
    # S_Fki = Vt * np.conj(Ybus_tcsc.toarray()[t, f] * Vf)
    S_Fik = Vf * np.conj(-y_tcsc_pu * Vt)
    S_Fki = Vt * np.conj(-y_tcsc_pu * Vf)
    P_Fik, Q_Fik = S_Fik.real, S_Fik.imag
    P_Fki, Q_Fki = S_Fki.real, S_Fki.imag


    # seems like it is not used:
    # S_ii = np.abs(V[f]) ** 2 * np.abs(Ybus[f, f]) * np.exp(1j * np.angle(Ybus[f, f].conj()))  ####
    # S_kk = np.abs(V[t]) ** 2 * np.abs(Ybus[t, t]) * np.exp(1j * np.angle(Ybus[t, t].conj()))  ####
    #
    # S_ij = Sbus[f] - S_ii
    # S_kj = Sbus[t] - S_kk

    S_Fi_dx = -1j * dY_TCSC_dx / y_tcsc_pu * (S_Fii + S_Fik)  # ybus_tcsc_pu already has -1j but we want it without -1j
    S_Fk_dx = -1j * dY_TCSC_dx / y_tcsc_pu * (S_Fkk + S_Fki)
    P_Fi_dx, Q_Fi_dx = S_Fi_dx.real, S_Fi_dx.imag
    P_Fk_dx, Q_Fk_dx = S_Fk_dx.real, S_Fk_dx.imag

    f_in_pq = np.isin(f, pq)
    t_in_pq = np.isin(t, pq)
    f_in_pvpq = np.isin(f, pvpq)
    t_in_pvpq = np.isin(t, pvpq)

    lpvpq = len(pvpq)
    lpq = len(pq)

    rows = np.array([], dtype=np.float64)
    cols = np.array([], dtype=np.float64)
    data = np.array([], dtype=np.float64)

    # J_C_P_d = np.zeros(shape=(len(pvpq), len(pvpq)), dtype=np.float64)
    # if np.any(f_in_pvpq):
    #     J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq]]] = -Q_Fik
    # if np.any(f_in_pvpq & t_in_pvpq):
    #     J_C_P_d[pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]] = Q_Fik
    #     J_C_P_d[pvpq_lookup[t[t_in_pvpq]], pvpq_lookup[f[f_in_pvpq]]] = Q_Fki
    # if np.any(t_in_pvpq):
    #     J_C_P_d[pvpq_lookup[t[t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]] = -Q_Fki
    # # J_C_P_d = np.array([[-S_Fik.imag, S_Fik.imag],
    # #                     [S_Fki.imag, -S_Fki.imag]]).reshape(2, 2)  # todo: generalize shapes to work with many TCSC
    #
    # # J_C_P_u = np.array([[S_Fik.real / Vm[f], S_Fik.real / Vm[t]],
    # #                     [S_Fki.real / Vm[f], S_Fki.real / Vm[t]]]).reshape(2, 2)

    rows = np.r_[
        rows, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pvpq]],
        pvpq_lookup[t[t_in_pvpq & f_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]]
    cols = np.r_[
        cols, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[f_in_pvpq & t_in_pvpq]],
        pvpq_lookup[f[t_in_pvpq & f_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]]
    data = np.r_[
        data, -Q_Fik[f_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq],
        Q_Fki[t_in_pvpq & f_in_pvpq], -Q_Fki[t_in_pvpq]]

    # J_C_P_u = np.zeros(shape=(len(pvpq), len(pq)), dtype=np.float64)
    # if np.any(f_in_pvpq & f_in_pq):
    #     J_C_P_u[pvpq_lookup[f[f_in_pvpq]], pq_lookup[f[f_in_pq]]] = P_Fik / Vmf
    # if np.any(f_in_pvpq & t_in_pq):
    #     J_C_P_u[pvpq_lookup[f[f_in_pvpq]], pq_lookup[t[t_in_pq]]] = P_Fik / Vmt
    # if np.any(t_in_pvpq & f_in_pq):
    #     J_C_P_u[pvpq_lookup[t[t_in_pvpq]], pq_lookup[f[f_in_pq]]] = P_Fki / Vmf
    # if np.any(t_in_pvpq & t_in_pq):
    #     J_C_P_u[pvpq_lookup[t[t_in_pvpq]], pq_lookup[t[t_in_pq]]] = P_Fki / Vmt

    rows = np.r_[
        rows, pvpq_lookup[f[f_in_pvpq & f_in_pq]], pvpq_lookup[f[f_in_pvpq & t_in_pq]],
        pvpq_lookup[t[t_in_pvpq & f_in_pq]], pvpq_lookup[t[t_in_pvpq & t_in_pq]]]
    cols = np.r_[
        cols, lpvpq + pq_lookup[f[f_in_pvpq & f_in_pq]], lpvpq + pq_lookup[t[f_in_pvpq & t_in_pq]],
        lpvpq + pq_lookup[f[t_in_pvpq & f_in_pq]], lpvpq + pq_lookup[t[t_in_pvpq & t_in_pq]]]
    data = np.r_[
        data, (P_Fik / Vmf)[f_in_pvpq & f_in_pq], (P_Fik / Vmt)[f_in_pvpq & t_in_pq],
        (P_Fki / Vmf)[t_in_pvpq & f_in_pq], (P_Fki / Vmt)[t_in_pvpq & t_in_pq]]

    # J_C_Q_d = np.zeros(shape=(len(pq), len(pvpq)), dtype=np.float64)
    # if np.any(f_in_pvpq & f_in_pq):
    #     J_C_Q_d[pq_lookup[f[f_in_pq]], pvpq_lookup[f[f_in_pvpq]]] = P_Fik
    # if np.any(f_in_pq & t_in_pvpq):
    #     J_C_Q_d[pq_lookup[f[f_in_pq]], pvpq_lookup[t[t_in_pvpq]]] = -P_Fik
    # if np.any(t_in_pq & f_in_pvpq):
    #     J_C_Q_d[pq_lookup[t[t_in_pq]], pvpq_lookup[f[f_in_pvpq]]] = -P_Fki
    # if np.any(t_in_pvpq & t_in_pq):
    #     J_C_Q_d[pq_lookup[t[t_in_pq]], pvpq_lookup[t[t_in_pvpq]]] = P_Fki
    # # J_C_Q_d = np.array([[S_Fik.real, -S_Fik.real],
    # #                     [-S_Fki.real, S_Fki.real]]).reshape(2, 2)

    rows = np.r_[
        rows, lpvpq + pq_lookup[f[f_in_pq & f_in_pvpq]], lpvpq + pq_lookup[f[f_in_pq & t_in_pvpq]],
        lpvpq + pq_lookup[t[t_in_pq & f_in_pvpq]], lpvpq + pq_lookup[t[t_in_pq & t_in_pvpq]]]
    cols = np.r_[
        cols, pvpq_lookup[f[f_in_pq & f_in_pvpq]], pvpq_lookup[t[f_in_pq & t_in_pvpq]],
        pvpq_lookup[f[t_in_pq & f_in_pvpq]], pvpq_lookup[t[t_in_pq & t_in_pvpq]]]
    data = np.r_[
        data, P_Fik[f_in_pq & f_in_pvpq], -P_Fik[f_in_pq & t_in_pvpq],
        -P_Fki[t_in_pq & f_in_pvpq], P_Fki[t_in_pq & t_in_pvpq]]

    # J_C_Q_u = np.zeros(shape=(len(pq), len(pq)), dtype=np.float64)
    # if np.any(f_in_pq):
    #     J_C_Q_u[pq_lookup[f[f_in_pq]], pq_lookup[f[f_in_pq]]] = (2 * Q_Fii + Q_Fik) / Vmf
    # if np.any(f_in_pq & t_in_pq):
    #     J_C_Q_u[pq_lookup[f[f_in_pq]], pq_lookup[t[t_in_pq]]] = Q_Fik / Vmt
    #     J_C_Q_u[pq_lookup[t[t_in_pq]], pq_lookup[f[f_in_pq]]] = Q_Fki / Vmf
    # if np.any(t_in_pq):
    #     J_C_Q_u[pq_lookup[t[t_in_pq]], pq_lookup[t[t_in_pq]]] = (2 * Q_Fkk + Q_Fki) / Vmt
    # # J_C_Q_u = np.array([[(2 * S_Fii.imag + S_Fik.imag) / Vm[f], S_Fik.imag / Vm[f]],
    # #                     [S_Fki.imag / Vm[t], (2 * S_Fkk.imag + S_Fki.imag) / Vm[f]]]).reshape(2, 2)

    rows = np.r_[
        rows, lpvpq + pq_lookup[f[f_in_pq]], lpvpq + pq_lookup[f[f_in_pq & t_in_pq]],
        lpvpq + pq_lookup[t[t_in_pq & f_in_pq]], lpvpq + pq_lookup[t[t_in_pq]]]
    cols = np.r_[
        cols, lpvpq + pq_lookup[f[f_in_pq]], lpvpq + pq_lookup[t[f_in_pq & t_in_pq]],
        lpvpq + pq_lookup[f[t_in_pq & f_in_pq]], lpvpq + pq_lookup[t[t_in_pq]]]
    data = np.r_[
        data, ((2 * Q_Fii + Q_Fik) / Vmf)[f_in_pq], (Q_Fik / Vmt)[f_in_pq & t_in_pq],
        (Q_Fki / Vmf)[t_in_pq & f_in_pq], ((2 * Q_Fkk + Q_Fki) / Vmt)[t_in_pq]]

    # J_C_P_c = np.zeros(shape=(len(pvpq), nsvc + ntcsc), dtype=np.float64)
    # if np.any(f_in_pvpq):
    #     J_C_P_c[pvpq_lookup[f[f_in_pvpq]], (nsvc + np.arange(ntcsc))[f_in_pvpq]] = S_Fi_dx.real
    # if np.any(t_in_pvpq):
    #     J_C_P_c[pvpq_lookup[t[t_in_pvpq]], (nsvc + np.arange(ntcsc))[t_in_pvpq]] = S_Fk_dx.real
    # # J_C_P_c = np.array([[S_Fi_dx.real], [S_Fk_dx.real]]).reshape(2, 1)
    rows = np.r_[
        rows, pvpq_lookup[f[f_in_pvpq & tcsc_controllable]], pvpq_lookup[t[t_in_pvpq & tcsc_controllable]]]
    cols = np.r_[
        cols, lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(f_in_pvpq & tcsc_controllable),
        lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(t_in_pvpq & tcsc_controllable)]
    data = np.r_[
        data, P_Fi_dx[f_in_pvpq & tcsc_controllable], P_Fk_dx[t_in_pvpq & tcsc_controllable]]

    # J_C_Q_c = np.zeros(shape=(len(pq), nsvc + ntcsc), dtype=np.float64)
    # if np.any(f_in_pq):
    #     J_C_Q_c[pq_lookup[f[f_in_pq]], (nsvc + np.arange(ntcsc))[f_in_pq]] = S_Fi_dx.imag
    # if np.any(t_in_pq):
    #     J_C_Q_c[pq_lookup[t[t_in_pq]], (nsvc + np.arange(ntcsc))[t_in_pq]] = S_Fk_dx.imag
    # # J_C_Q_c = np.array([[S_Fi_dx.imag], [S_Fk_dx.imag]]).reshape(2, 1)
    rows = np.r_[
        rows, lpvpq + pq_lookup[f[f_in_pq & tcsc_controllable]], lpvpq + pq_lookup[t[t_in_pq & tcsc_controllable]]]
    cols = np.r_[
        cols, lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(f_in_pq & tcsc_controllable),
        lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(t_in_pq & tcsc_controllable)]
    data = np.r_[
        data, Q_Fi_dx[f_in_pq & tcsc_controllable], Q_Fk_dx[t_in_pq & tcsc_controllable]]

    # the signs are opposite here for J_C_C_d, J_C_C_u, J_C_C_c and I don't know why
    # main mode of operation - set point for p_to_mw:
    # J_C_C_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    # J_C_C_d = np.zeros(shape=(nsvc + ntcsc, len(pvpq)), dtype=np.float64)
    # if np.any(f_in_pvpq):
    #     J_C_C_d[(nsvc + np.arange(ntcsc))[f_in_pvpq], pvpq_lookup[f[f_in_pvpq]]] = Q_Fik
    # if np.any(t_in_pvpq):
    #     J_C_C_d[(nsvc + np.arange(ntcsc))[t_in_pvpq], pvpq_lookup[t[t_in_pvpq]]] = -Q_Fik
    rows = np.r_[
        rows, lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(f_in_pvpq & tcsc_controllable),
        lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(t_in_pvpq & tcsc_controllable)]
    cols = np.r_[
        cols, pvpq_lookup[f[f_in_pvpq & tcsc_controllable]], pvpq_lookup[t[t_in_pvpq & tcsc_controllable]]]
    data = np.r_[
        data, Q_Fik[f_in_pvpq & tcsc_controllable], -Q_Fik[t_in_pvpq & tcsc_controllable]]


    # J_C_C_u = np.zeros(shape=(nsvc + ntcsc, len(pq)), dtype=np.float64)
    # if np.any(f_in_pq):
    #     J_C_C_u[(nsvc + np.arange(ntcsc))[f_in_pq], pq_lookup[f[f_in_pq]]] = P_Fik / Vmf
    # if np.any(t_in_pq):
    #     J_C_C_u[(nsvc + np.arange(ntcsc))[t_in_pq], pq_lookup[t[t_in_pq]]] = P_Fik / Vmt
    # rows = np.r_[
    #     rows, lpvpq + lpq + nsvc + np.arange(ntcsc)[f_in_pq & tcsc_controllable],
    #     lpvpq + lpq + nsvc + np.arange(ntcsc)[t_in_pq & tcsc_controllable]]
    rows = np.r_[
        rows, lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(f_in_pq & tcsc_controllable),
        lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(t_in_pq & tcsc_controllable)]
    cols = np.r_[
        cols, lpvpq + pq_lookup[f[f_in_pq & tcsc_controllable]], lpvpq + pq_lookup[t[t_in_pq & tcsc_controllable]]]
    data = np.r_[data, (P_Fik / Vmf)[f_in_pq & tcsc_controllable], (P_Fik / Vmt)[t_in_pq & tcsc_controllable]]

    # J_C_C_c = np.zeros(shape=(nsvc + ntcsc, nsvc + ntcsc), dtype=np.float64)
    # J_C_C_c[np.r_[nsvc:nsvc + ntcsc], np.r_[nsvc:nsvc + ntcsc]] = -S_Fi_dx.real  # .flatten()?
    rows = np.r_[rows, lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(tcsc_controllable)]
    cols = np.r_[cols, lpvpq + lpq + nsvc:lpvpq + lpq + nsvc + np.count_nonzero(tcsc_controllable)]
    data = np.r_[data, -P_Fi_dx[tcsc_controllable]]

    # alternative mode of operation: for Vm at to bus (mismatch and setpoint also must be adjusted):
    # J_C_C_d = np.zeros(shape=(len(x_control), len(pvpq)), dtype=np.float64)
    # J_C_C_u = np.zeros(shape=(len(x_control), len(pq)), dtype=np.float64)
    # J_C_C_u[np.arange(len(x_control)), pq_lookup[t]] = 1
    # J_C_C_c = np.zeros((len(x_control), len(x_control)), dtype=np.float64)

    # # todo: implement the "relevant" array to be used as mask
    # # todo: adjust indices with position counters
    # if np.all(tcsc_controllable):
    #     J_m = np.vstack([np.hstack([J_C_P_d, J_C_P_u, J_C_P_c]),
    #                      np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c]),
    #                      np.hstack([J_C_C_d, J_C_C_u, J_C_C_c])])
    # elif np.any(tcsc_controllable) or np.any(svc_controllable):
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


def create_J_modification_ssc_vsc(J, V, Vm, y_pu, f, t, pvpq, pq, pvpq_lookup, pq_lookup, control_v, control_delta):
    """
    creates the modification Jacobian matrix for SSC (STATCOM)

    Parameters
    ----------
    Vm
    control_delta
    V
        array of np.complex128
    y_pu
        admittances of the elements
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

    Vmf = Vm[f]
    Vmt = Vm[t]

    S_Fii = Vf * np.conj(y_pu * Vf)
    S_Fkk = Vt * np.conj(y_pu * Vt)
    P_Fii, Q_Fii = S_Fii.real, S_Fii.imag
    P_Fkk, Q_Fkk = S_Fkk.real, S_Fkk.imag

    S_Fik = Vf * np.conj(-y_pu * Vt)
    S_Fki = Vt * np.conj(-y_pu * Vf)
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

    rows = np.r_[rows, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[f[f_in_pvpq & t_in_pvpq]],
                       pvpq_lookup[t[f_in_pvpq & t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]]
    cols = np.r_[cols, pvpq_lookup[f[f_in_pvpq]], pvpq_lookup[t[f_in_pvpq & t_in_pvpq]],
                       pvpq_lookup[f[f_in_pvpq & t_in_pvpq]], pvpq_lookup[t[t_in_pvpq]]]
    # todo
    # data = np.r_[data, -S_Fik.imag[f_in_pvpq], S_Fik.imag[f_in_pvpq & t_in_pvpq], S_Fki.imag[f_in_pvpq & t_in_pvpq], -S_Fki.imag[t_in_pvpq]]
    # data = np.r_[data, -S_Fik.imag[f_in_pvpq], S_Fik.imag[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], zero[t_in_pvpq]]

    data = np.r_[data, -Q_Fik[f_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq],
                 np.where(control_delta[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], Q_Fki[f_in_pvpq & t_in_pvpq]),
                 np.where(control_delta[t_in_pvpq], zero[t_in_pvpq], -Q_Fki[t_in_pvpq])]
    # data = np.r_[data, -Q_Fik[f_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq],
    #              np.where(control_delta[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq]),
    #              np.where(control_delta[t_in_pvpq], zero[t_in_pvpq], -Q_Fik[t_in_pvpq])]


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

    data = np.r_[data, ((2 * P_Fii + P_Fik) / Vmf)[f_in_pvpq], (P_Fik/Vmt)[f_in_pvpq & t_in_pq],
                 np.where(control_delta[f_in_pvpq & t_in_pq], zero[f_in_pvpq & t_in_pq], (P_Fki / Vmf)[f_in_pvpq & t_in_pq]),
                 np.where(control_delta[t_in_pvpq & t_in_pq], zero[t_in_pvpq & t_in_pq], ((2 * P_Fkk + P_Fki) / Vmt)[t_in_pvpq & t_in_pq])]
    # data = np.r_[data, ((2 * P_Fii + P_Fik) / Vmf)[f_in_pvpq], (P_Fik / Vmt)[f_in_pvpq & t_in_pq],
    #              np.where(control_delta[f_in_pvpq & t_in_pq], zero[f_in_pvpq & t_in_pq], ((2 * P_Fii + P_Fik) / Vmf)[f_in_pvpq & t_in_pq]),
    #              np.where(control_delta[t_in_pvpq & t_in_pq], zero[t_in_pvpq & t_in_pq], (P_Fik / Vmt)[t_in_pvpq & t_in_pq])]


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
    data = np.r_[data, P_Fik[f_in_pq], -P_Fik[f_in_pq & t_in_pvpq],
                 np.where(control_v[f_in_pvpq & t_in_pq], zero[f_in_pvpq & t_in_pq], P_Fik[f_in_pvpq & t_in_pq]),
                 np.where(control_v[t_in_pq], zero[t_in_pq], -P_Fik[t_in_pq])]
    # data = np.r_[data, P_Fik[f_in_pq], -P_Fik[f_in_pq & t_in_pvpq],
    #              np.where(control_v[f_in_pvpq & t_in_pq], zero[f_in_pvpq & t_in_pq], P_Fki[f_in_pvpq & t_in_pq]),
    #              np.where(control_v[t_in_pq], zero[t_in_pq], -P_Fki[t_in_pq])]



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
    data = np.r_[data, ((2 * Q_Fii + Q_Fik) / Vmf)[f_in_pq], (Q_Fik / Vmt)[f_in_pq & t_in_pq],
                 np.where(control_v[f_in_pq & t_in_pq], one[f_in_pq & t_in_pq], ((2 * Q_Fii + Q_Fik) / Vmf)[f_in_pq & t_in_pq]),
                 np.where(control_v[t_in_pq], zero[t_in_pq], (Q_Fik / Vmt)[t_in_pq])]
    # data = np.r_[data, ((2 * Q_Fii + Q_Fik) / Vmf)[f_in_pq], (Q_Fik / Vmt)[f_in_pq & t_in_pq],
    #              np.where(control_v[f_in_pq & t_in_pq], one[f_in_pq & t_in_pq], ((2 * Q_Fkk + Q_Fki) / Vmf)[f_in_pq & t_in_pq]),
    #              np.where(control_v[t_in_pq], zero[t_in_pq], (Q_Fkk / Vmt)[t_in_pq])]

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


@np.errstate(all="raise")
def create_J_modification_hvdc(J, V_dc, Ybus_hvdc, Ybus_vsc_dc, vsc_g_pu, vsc_gl_pu, dc_p, dc_p_lookup,
                               vsc_dc_fb, vsc_dc_tb, vsc_dc_slack, vsc_dc_mode_p):
    # Calculate the first matrix for all elements
    J_all = (Ybus_hvdc - Ybus_vsc_dc).multiply(V_dc)
    # Calculate the second matrix for diagonal elements only
    J_diag = diags(V_dc).multiply((Ybus_hvdc - Ybus_vsc_dc).dot(V_dc))

    # Calculate DC modification VSC Jacobian
    # here we do not use Ybus because it has the y values already summed up for the diagonal entries,
    # and creating the sparse matrix for J also sums values that share the same indices.
    # So effectively we achieve the same effect here as if we had the Ybus matrix,
    # except that we operate at the level of individual VSC still.
    # Only relevant when more than one VSC connected to the same bus.
    Vmf = V_dc[vsc_dc_fb]
    Vmt = V_dc[vsc_dc_tb]
    P_Fii = Vmf * (vsc_g_pu + vsc_gl_pu) * Vmf
    P_Fik = -Vmf * vsc_g_pu * Vmt
    P_Fkk = Vmt * vsc_g_pu * Vmt
    P_Fki = -Vmt * vsc_g_pu * Vmf

    zero = SMALL_NUMBER * np.ones(shape=P_Fii.shape, dtype=np.float64)
    one = np.ones(shape=P_Fii.shape, dtype=np.float64)

    rows = np.r_[dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_tb], dc_p_lookup[vsc_dc_tb]]
    cols = np.r_[dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_tb], dc_p_lookup[vsc_dc_fb], dc_p_lookup[vsc_dc_tb]]
    # data = np.r_[2 * P_Fii + P_Fik, P_Fik, np.where(vsc_dc_slack, one, P_Fki), np.where(vsc_dc_slack, zero, 2 * P_Fkk + P_Fki)]
    # data = np.r_[2 * P_Fii + P_Fik, P_Fik, np.where(vsc_dc_slack, one, 2 * P_Fii + P_Fik), np.where(vsc_dc_slack, zero, P_Fik)]
    # it depends on which power must be controlled, at internal node (AC slack mode) or at external node (DC P mode):
    data = np.r_[(2 * P_Fii + P_Fik) / Vmf, P_Fik / Vmt,
                 np.where(vsc_dc_slack, one, np.where(vsc_dc_mode_p, (2 * P_Fii + P_Fik) / Vmf, P_Fki / Vmf)),
                 np.where(vsc_dc_slack, zero, np.where(vsc_dc_mode_p, P_Fik / Vmt, (2 * P_Fkk + P_Fki) / Vmt))]
    # data = np.r_[P_Fik / V_dc[vsc_dc_fb], P_Fik / V_dc[vsc_dc_tb], np.where(vsc_dc_slack, one, P_Fki / V_dc[vsc_dc_fb]), np.where(vsc_dc_slack, zero, P_Fki / V_dc[vsc_dc_tb])]
    # data = np.r_[-Q_Fik[f_in_pvpq], Q_Fik[f_in_pvpq & t_in_pvpq], np.where(control_delta[f_in_pvpq & t_in_pvpq], one[f_in_pvpq & t_in_pvpq], Q_Fki[f_in_pvpq & t_in_pvpq]), np.where(control_delta[t_in_pvpq], zero[t_in_pvpq], -Q_Fki[t_in_pvpq])]

    J_m_vsc = csr_matrix((data, (rows, cols)), shape=J_all.shape, dtype=np.float64)

    # Combine them to form the Jacobian for DC grid including the VSC elements
    J_combined = J_all + J_diag + J_m_vsc

    # Map the J for the DC system to the overall J
    num_p = len(dc_p)
    offset = J.shape[0] - num_p
    # Create an initial zero sparse matrix for J_m
    J_m = lil_matrix(J.shape)
    # Create the modification J for the DC system to be added to the overall J
    J_m[offset:offset+num_p, offset:offset+num_p] = J_combined[dc_p, :][:, dc_p]

    return J_m.tocsr()


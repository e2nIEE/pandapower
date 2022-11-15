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


def create_J_modification_svc(J, svc_buses, pvpq, pq, pq_lookup, V, x_control, x_control_lookup, svc_x_l_pu, svc_x_cvar_pu):
    # dQ_SVC_i/du_i = 2 * q_svc_i
    # J_C_Q_c = dQ_SVC_i / d_alpha_SVC = 2 * V_i ** 2 * (np.cos(2*alpha_SVC - 1) / np.pi * X_L
    # J_C_C_d, J_C_C_u, J_C_C_c - ?  # "depend on the controlled parameter and the corresponding mismatch equation"
    #y_svc = (2 * (np.pi - x_control) + np.sin(2 * x_control) + np.pi * svc_x_l_pu / svc_x_cvar_pu) / (np.pi * svc_x_l_pu) # * np.exp(-1j * np.pi / 2)
    y_svc = calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu)
    q_svc = abs(V[svc_buses]) ** 2 * y_svc

    J_m = np.zeros_like(J.toarray())

    # J_C_Q_u
    J_C_Q_u = np.zeros(shape=(len(pq), len(pq)))
    J_C_Q_u[pq_lookup[svc_buses], pq_lookup[svc_buses]] = 2 * q_svc
    # count pvpq rows and pvpq columns from top left
    J_m[len(pvpq):len(pvpq)+len(pq), len(pvpq):len(pvpq)+len(pq)] = J_C_Q_u

    # J_C_Q_c
    J_C_Q_c = np.zeros(shape=(len(pq), len(x_control)))
    J_C_Q_c[pq_lookup[svc_buses], :] = 2 * abs(V[svc_buses]) ** 2 * (np.cos(2 * x_control) - 1) / (np.pi * svc_x_l_pu)
    # count pvpq rows and pvpq columns from top left
    J_m[len(pvpq):len(pvpq)+len(pq), len(pvpq)+len(pq):len(pvpq)+len(pq) + len(x_control)] = J_C_Q_c

    # J_C_C_d - will be all zero (ignoring)

    # J_C_C_u
    # d(Ui - Ui,set)/d(Ui) = dUi/dUi = 1
    J_C_C_u = np.zeros(shape=(len(x_control), len(pq)))
    J_C_C_u[:, pq_lookup[svc_buses]] = 1
    # count pvpq rows and pvpq columns from top left
    J_m[len(pvpq)+len(pq):len(pvpq)+len(pq)+len(x_control), len(pvpq):len(pvpq) + len(pq)] = J_C_C_u

    # J_C_C_c
    # d(Ui - Ui,set) / d x_control = 0

    J_m = csr_matrix(J_m)

    return J_m


def create_J_modification_tcsc(J, branch, pvpq_lookup, pq_lookup, Ybus_tcsc, V, tcsc_i, tcsc_j, tcsc_fb, tcsc_tb, pvpq, pq, tcsc_branches,
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

    y_tcsc = calc_y_svc_pu(x_control, tcsc_x_l_pu, tcsc_x_cvar_pu)

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

    J_C_P_c = np.zeros(shape=(len(pvpq), len(x_control)))
    aux_term = 2*(np.cos(2*x_control[x_control_lookup==1] - 1)) / (np.pi * tcsc_x_l_pu * y_tcsc)
    for in_pvpq, m in zip([tcsc_in_pvpq_f], [mf_pvpq]):
        i = pvpq_lookup[m]
        J_C_P_c[i, x_control_lookup==1] = aux_term * p_tcsc_ij[m]
    for in_pvpq, m in zip([tcsc_in_pvpq_t], [mt_pvpq]):
        i = pvpq_lookup[m]
        J_C_P_c[i, x_control_lookup==1] = aux_term * p_tcsc_ij[m]

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

    J_C_Q_c = np.zeros(shape=(len(pq), len(x_control)))
    for in_pvpq, m in zip([tcsc_in_pvpq_f], [mf_pq]):
        i = pq_lookup[m]
        if len(m) == 0: continue
        J_C_Q_c[i, x_control_lookup==1] = aux_term * (q_tcsc_ii[m] + q_tcsc_ij[m])
    for in_pvpq, m in zip([tcsc_in_pvpq_t], [mt_pq]):
        i = pq_lookup[m]
        if len(m) == 0: continue
        J_C_Q_c[i, x_control_lookup==1] = aux_term * (q_tcsc_ii[m] + q_tcsc_ij[m])


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

    J_C_C_d = np.zeros(shape=(len(x_control), len(pvpq)))
    for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pvpq, nf_pvpq), (tcsc_in_pvpq_t, mt_pvpq, nt_pvpq)):
        i = pvpq_lookup[m]
        j = pvpq_lookup[n]
        J_C_C_d[x_control_lookup==1, i] = np.abs(V[m]) * np.abs(Ybus_tcsc[m, n]) * np.abs(V[n]) * np.sin(np.angle(V[m]) - np.angle(V[n]) + np.angle(np.array(Ybus_tcsc[m,n])))  #### A_ij = A_ji    phi_tcsc_ij = phi_tcsc_ji

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

    J_C_C_u = np.zeros(shape=(len(x_control), len(pq)))
    for in_pvpq, m, n in ((tcsc_in_pvpq_f, mf_pq, nf_pq), (tcsc_in_pvpq_t, mt_pq, nt_pq)):
        i = pq_lookup[m]
        j = pq_lookup[n]
        if len(m) == 0: continue
        J_C_C_u[x_control_lookup==1, i] = np.abs(V[m]) * np.abs(Ybus_tcsc[m, n]) * np.abs(V[n]) / abs(V[m]) * np.cos(np.angle(V[m]) - np.angle(V[n]) + np.angle(np.array(Ybus_tcsc[m,n])))

    # J_C_C_c
    J_C_C_c = np.zeros(shape=(len(x_control), len(x_control)))
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

        J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] = (2*abs(V[m]) * abs(V[n]) * (np.cos(2 * x_control[x_control_lookup==1])-1)*np.cos(np.angle(V[m]) - np.angle(V[n]) + np.angle(np.array(Ybus_tcsc[m,n]))))/ (np.pi * tcsc_x_l_pu)
    # J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] = (2*abs(V[tcsc_fb]) * abs(V[tcsc_tb]) * (np.cos(2 * x_control[x_control_lookup==1])-1)*np.cos(np.angle(V[tcsc_tb]) - np.angle(V[tcsc_fb]) + np.angle(Ybus_tcsc[tcsc_fb,tcsc_tb])))/ (np.pi * tcsc_x_l_pu)
    #print("JCCc", J_C_C_c)
    # J_C_C_c = np.array([[2.755]])
    J_C_C_c[x_control_lookup == 1, x_control_lookup == 1] = np.array(x_control[x_control_lookup==1])
    #J_C_C_c[:, tcsc_i] = - 2 * B * np.cos(2 * x_control[x_control_lookup==1]-1) / (np.pi * tcsc_x_l_pu)
    #J_C_C_c[:, tcsc_j] = - 2 * B * np.cos(2 * x_control[x_control_lookup==1]-1) / (np.pi * tcsc_x_l_pu)

    J_m = np.vstack(
        [np.hstack([J_C_P_d, J_C_P_u, J_C_P_c]),
         np.hstack([J_C_Q_d, J_C_Q_u, J_C_Q_c]),
         np.hstack([J_C_C_d, J_C_C_u, J_C_C_c])])

    #print("J", J_m)

    J_m = csr_matrix(J_m)

    return J_m


def calc_tcsc_p_pu(Ybus_tcsc, V, tcsc_fb, tcsc_tb):
    Vm_f = np.abs(V[tcsc_fb])
    Va_f = np.angle(V[tcsc_fb])
    Vm_t = np.abs(V[tcsc_tb])
    Va_t = np.angle(V[tcsc_tb])

    delta_ji = Va_f - Va_t
    delta_ij = Va_t - Va_f
    # y_tcsc = calc_y_svc_pu(x_control[x_control_lookup==1], tcsc_x_l_pu, tcsc_x_cvar_pu)
    # y_tcsc_ij = -y_tcsc
    #phi_tcsc_ij = np.angle(np.array(Ybus[tcsc_fb, tcsc_tb])[0])
    y_ij_pu = Ybus_tcsc[tcsc_fb, tcsc_tb]
    y_ji_pu = Ybus_tcsc[tcsc_tb, tcsc_fb]

    phi_tcsc_ij = np.array(np.angle(y_ij_pu))[0]
    phi_tcsc_ji = np.array(np.angle(y_ji_pu))[0]

    #A = Vm_f * np.abs(np.array(Ybus[tcsc_fb, tcsc_tb])[0]) * Vm_t
    A_ij = np.array(Vm_f * np.abs(y_ij_pu) * Vm_t)[0]
    A_ji = np.array(Vm_f * np.abs(y_ji_pu) * Vm_t)[0]


    p_tcsc_ij = np.array(A_ij * np.cos(delta_ij + phi_tcsc_ij))
    p_tcsc_ji = np.array(A_ji * np.cos(delta_ji + phi_tcsc_ji))

    return p_tcsc_ij, p_tcsc_ji,A_ij,A_ji,phi_tcsc_ij,phi_tcsc_ji


def calc_tcsc_q_pu(Ybus_tcsc, V, tcsc_fb, tcsc_tb):
    Vm_f = np.abs(V[tcsc_fb])
    Va_f = np.angle(V[tcsc_fb])
    Vm_t = np.abs(V[tcsc_tb])
    Va_t = np.angle(V[tcsc_tb])

    delta_ji = Va_f - Va_t
    delta_ij = Va_t - Va_f
    # y_tcsc = calc_y_svc_pu(x_control[x_control_lookup==1], tcsc_x_l_pu, tcsc_x_cvar_pu)
    # y_tcsc_ij = -y_tcsc

    y_ij_pu = Ybus_tcsc[tcsc_fb, tcsc_tb]
    y_ji_pu = Ybus_tcsc[tcsc_tb, tcsc_fb]
    phi_tcsc_ij = np.angle(y_ij_pu)
    phi_tcsc_ji = np.angle(y_ji_pu)

    # phi_tcsc_ij = np.angle(np.array(Ybus[tcsc_fb, tcsc_tb])[0])



    # q_tcsc_ii = np.square(Vm_f) * np.abs(np.array(Ybus[tcsc_fb, tcsc_fb])[0])

    y_ii_pu = Ybus_tcsc[tcsc_fb, tcsc_fb]
    y_jj_pu = Ybus_tcsc[tcsc_tb, tcsc_tb]
    y_ij_pu = Ybus_tcsc[tcsc_fb, tcsc_tb]
    y_ji_pu = Ybus_tcsc[tcsc_tb, tcsc_fb]


    q_tcsc_ii = np.array(np.square(Vm_f) * np.abs(np.array(y_ii_pu)))[0]

    q_tcsc_jj = np.array(np.square(Vm_t) * np.abs(np.array(y_jj_pu)))[0]

    q_tcsc_ij = np.array(Vm_f * np.abs(np.array(y_ij_pu)) * Vm_t * np.sin(delta_ij + phi_tcsc_ij))[0]

    q_tcsc_ji = np.array(Vm_t * np.abs(np.array(y_ji_pu)) * Vm_f * np.sin(delta_ji + phi_tcsc_ji))[0]


    # q_tcsc_ij = Vm_f * np.abs(np.array(Ybus[tcsc_fb, tcsc_tb])[0]) * Vm_t * np.sin(delta_ij + phi_tcsc_ij)


    return q_tcsc_ii, q_tcsc_ij,q_tcsc_jj,q_tcsc_ji

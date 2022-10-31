# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix
from pandapower.pypower.idx_brch import BR_R, BR_X, F_BUS, T_BUS


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


def create_J_modification_tcsc(J, Ybus, V, tcsc_i, tcsc_j, pvpq, pq, tcsc_branches, x_control, x_control_lookup, tcsc_x_l_pu, tcsc_x_cvar_pu):
    p_tcsc_ij, A, phi_tcsc_ij = calc_tcsc_p_pu(Ybus, V, tcsc_i, tcsc_j)
    q_tcsc_ii, q_tcsc_ij = calc_tcsc_q_pu(Ybus, V, tcsc_i, tcsc_j)

    y_tcsc = calc_y_svc_pu(x_control, tcsc_x_l_pu, tcsc_x_cvar_pu)

    J_m = np.zeros_like(J.toarray())

    # J_C_P_d
    J_C_P_d = np.zeros(shape=(len(pvpq), len(pvpq)))
    J_C_P_d[0:len(tcsc_branches), tcsc_i] = -q_tcsc_ij
    J_C_P_d[0:len(tcsc_branches), tcsc_j] = q_tcsc_ij

    # J_C_P_u
    J_C_P_u = np.zeros(shape=(len(pvpq), len(pq)))
    # checking is wrong here. add an if-clause?
    J_C_P_u[0:len(tcsc_branches), tcsc_i[np.isin(tcsc_i, pq)]] = p_tcsc_ij
    J_C_P_u[0:len(tcsc_branches), tcsc_j[np.isin(tcsc_j, pq)]] = p_tcsc_ij

    # J_C_Q_d
    J_C_Q_d = np.zeros(shape=(len(pq), len(pvpq)))
    J_C_Q_d[0:len(tcsc_branches), tcsc_i] = p_tcsc_ij
    J_C_Q_d[0:len(tcsc_branches), tcsc_j] = -p_tcsc_ij

    # J_C_Q_u
    J_C_Q_u = np.zeros(shape=(len(pq), len(pq)))
    J_C_Q_u[0:len(tcsc_branches), tcsc_i] = 2*q_tcsc_ii + q_tcsc_ij
    J_C_Q_u[0:len(tcsc_branches), tcsc_j] = q_tcsc_ij

    # J_C_P_c
    J_C_P_c = np.zeros(shape=(len(pvpq), len(x_control)))
    aux_term = 2*(np.cos(2*x_control[x_control_lookup==1] - 1)) / (np.pi * tcsc_x_l_pu * y_tcsc)
    J_C_P_c[0:len(tcsc_branches), x_control_lookup==1] = aux_term * p_tcsc_ij

    # J_C_Q_c
    J_C_Q_c = np.zeros(shape=(len(pq), len(x_control)))
    J_C_Q_c[0:len(tcsc_branches), x_control_lookup==1] = aux_term * (q_tcsc_ii + q_tcsc_ij)

    # J_C_C_d
    J_C_C_d = np.zeros(shape=(len(x_control), len(pvpq)))
    J_C_C_d[:, tcsc_i] = A * np.sin(np.angle(V[tcsc_j]) + phi_tcsc_ij - np.angle(V[tcsc_i]))
    J_C_C_d[:, tcsc_j] = A * np.sin(np.angle(V[tcsc_i]) - phi_tcsc_ij - np.angle(V[tcsc_j]))

    # J_C_C_u
    J_C_C_u = np.zeros(shape=(len(x_control), len(pq)))
    J_C_C_u[:, tcsc_i] = A / abs(V[tcsc_i]) * np.cos(np.angle(V[tcsc_j]) - np.angle(V[tcsc_i]) + phi_tcsc_ij)
    J_C_C_u[:, tcsc_j] = A / abs(V[tcsc_j]) * np.cos(np.angle(V[tcsc_j]) - np.angle(V[tcsc_i]) + phi_tcsc_ij)

    # J_C_C_c
    J_C_C_c = np.zeros(shape=(len(x_control), len(x_control)))
    B = abs(V[tcsc_i]) * abs(V[tcsc_j]) * np.cos(np.angle(V[tcsc_j]) - np.angle(V[tcsc_i]) + phi_tcsc_ij)
    J_C_C_c[:, tcsc_i] = - 2 * B * np.cos(2 * x_control[x_control_lookup==1]-1) / (np.pi * tcsc_x_l_pu)
    J_C_C_c[:, tcsc_j] = - 2 * B * np.cos(2 * x_control[x_control_lookup==1]-1) / (np.pi * tcsc_x_l_pu)

    J_m = csr_matrix(J_m)

    return J_m


def calc_tcsc_p_pu(Ybus, V, tcsc_i, tcsc_j):
    Vm_f = np.abs(V[tcsc_i])
    Va_f = np.angle(V[tcsc_i])
    Vm_t = np.abs(V[tcsc_j])
    Va_t = np.angle(V[tcsc_j])

    delta_ij = Va_t - Va_f
    # y_tcsc = calc_y_svc_pu(x_control[x_control_lookup==1], tcsc_x_l_pu, tcsc_x_cvar_pu)
    # y_tcsc_ij = -y_tcsc
    phi_tcsc_ij = np.angle(np.array(Ybus[tcsc_i, tcsc_j])[0])

    A = Vm_f * np.abs(np.array(Ybus[tcsc_i, tcsc_j])[0]) * Vm_t
    p_tcsc = A * np.cos(delta_ij + phi_tcsc_ij)
    return p_tcsc, A, phi_tcsc_ij


def calc_tcsc_q_pu(Ybus, V, tcsc_i, tcsc_j):
    Vm_f = np.abs(V[tcsc_i])
    Va_f = np.angle(V[tcsc_i])
    Vm_t = np.abs(V[tcsc_j])
    Va_t = np.angle(V[tcsc_j])

    delta_ij = Va_t - Va_f
    # y_tcsc = calc_y_svc_pu(x_control[x_control_lookup==1], tcsc_x_l_pu, tcsc_x_cvar_pu)
    # y_tcsc_ij = -y_tcsc
    phi_tcsc_ij = np.angle(np.array(Ybus[tcsc_i, tcsc_j])[0])

    q_tcsc_ii = np.square(Vm_f) * np.abs(np.array(Ybus[tcsc_i, tcsc_i])[0])
    q_tcsc_ij = Vm_f * np.abs(np.array(Ybus[tcsc_i, tcsc_j])[0]) * Vm_t * np.sin(delta_ij + phi_tcsc_ij)
    return q_tcsc_ii, q_tcsc_ij

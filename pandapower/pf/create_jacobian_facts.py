# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix


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


def create_J_modification_svc(J, svc_buses, pvpq, pq, pq_lookup, V, x_control, svc_x_l_pu, svc_x_cvar_pu):
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
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix


def makeYbus_svc(Ybus, x_control_svc, svc_x_l_pu, svc_x_cvar_pu, svc_buses):
    Y_SVC = -1j * calc_y_svc_pu(x_control_svc, svc_x_l_pu, svc_x_cvar_pu)
    Ybus_svc = csr_matrix((Y_SVC, (svc_buses, svc_buses)), shape=Ybus.shape, dtype=np.complex128)
    return Ybus_svc


def makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb):
    Ybus_tcsc = np.zeros(Ybus.shape, dtype=np.complex128)
    Y_TCSC = -1j * calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)

    # for y_tcsc_pu_i, i, j in zip(Y_TCSC, tcsc_fb, tcsc_tb):
    #     Ybus_tcsc[i, i] += y_tcsc_pu_i
    #     Ybus_tcsc[i, j] += -y_tcsc_pu_i
    #     Ybus_tcsc[j, i] += -y_tcsc_pu_i
    #     Ybus_tcsc[j, j] += y_tcsc_pu_i

    Ybus_tcsc[tcsc_fb, tcsc_tb] = -Y_TCSC
    Ybus_tcsc[tcsc_tb, tcsc_fb] = -Y_TCSC
    Ybus_tcsc[np.diag_indices_from(Ybus_tcsc)] = -Ybus_tcsc.sum(axis=1)

    return csr_matrix(Ybus_tcsc)


def makeYbus_ssc(Ybus, ssc_y_pu, ssc_fb, ssc_tb, any_ssc):
    Ybus_ssc = np.zeros(Ybus.shape, dtype=np.complex128)

    if any_ssc:

        # size_y = Ybus.shape[0]
        # K_Y = vstack([eye(size_y, format="csr"),
        #               csr_matrix((num_ssc, size_y))], format="csr")
        # Ybus = K_Y * Ybus * K_Y.T  # this extends the Ybus matrix with 0-rows and 0-columns for the "q"-bus of SSC



        # for y_tcsc_pu_i, i, j in zip(Y_TCSC, tcsc_fb, tcsc_tb):
        #     Ybus_tcsc[i, i] += y_tcsc_pu_i
        #     Ybus_tcsc[i, j] += -y_tcsc_pu_i
        #     Ybus_tcsc[j, i] += -y_tcsc_pu_i
        #     Ybus_tcsc[j, j] += y_tcsc_pu_i

        Ybus_ssc[ssc_fb, ssc_fb] = ssc_y_pu
        Ybus_ssc[ssc_fb, ssc_tb] = -ssc_y_pu
        Ybus_ssc[ssc_tb, ssc_fb] = -ssc_y_pu
        Ybus_ssc[ssc_tb, ssc_tb] = ssc_y_pu

    return csr_matrix(Ybus_ssc)


def makeYft_tcsc(Ybus_tcsc, tcsc_fb, tcsc_tb):
    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    Y = Ybus_tcsc.toarray()
    nl = len(tcsc_fb)
    nb = Ybus_tcsc.shape[0]
    i = np.hstack([range(nl), range(nl)])  ## double set of row indices

    Yft = Y[tcsc_fb, tcsc_tb]
    Yff = -Yft
    Ytf = Y[tcsc_tb, tcsc_fb]
    Ytt = -Ytf

    Yf = csr_matrix((np.hstack([Yff, Yft]), (i, np.hstack([tcsc_fb, tcsc_tb]))), (nl, nb))
    Yt = csr_matrix((np.hstack([Ytf, Ytt]), (i, np.hstack([tcsc_fb, tcsc_tb]))), (nl, nb))
    return Yf, Yt


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
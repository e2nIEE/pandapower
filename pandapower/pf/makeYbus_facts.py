# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
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


def makeYbus_ssc_vsc(Ybus, internal_y_pu, fb, tb, controllable):
    Ybus_controllable = np.zeros(Ybus.shape, dtype=np.complex128)
    Ybus_not_controllable = np.zeros(Ybus.shape, dtype=np.complex128)

    for flag, Ybus_flag in zip([~controllable, controllable], [Ybus_not_controllable, Ybus_controllable]):
        if not np.any(flag):
            continue

        # size_y = Ybus.shape[0]
        # K_Y = vstack([eye(size_y, format="csr"),
        #               csr_matrix((num_ssc, size_y))], format="csr")
        # Ybus = K_Y * Ybus * K_Y.T  # this extends the Ybus matrix with 0-rows and 0-columns for the "q"-bus of SSC



        # for y_tcsc_pu_i, i, j in zip(Y_TCSC, tcsc_fb, tcsc_tb):
        #     Ybus_tcsc[i, i] += y_tcsc_pu_i
        #     Ybus_tcsc[i, j] += -y_tcsc_pu_i
        #     Ybus_tcsc[j, i] += -y_tcsc_pu_i
        #     Ybus_tcsc[j, j] += y_tcsc_pu_i

        # Ybus[ssc_fb[flag], ssc_fb[flag]] = ssc_y_pu[flag]
        Ybus_flag[fb[flag], tb[flag]] = -internal_y_pu[flag]
        Ybus_flag[tb[flag], fb[flag]] = -internal_y_pu[flag]
        # Ybus[ssc_tb[flag], ssc_tb[flag]] = ssc_y_pu[flag]
        Ybus_flag[np.diag_indices_from(Ybus_flag)] = -Ybus_flag.sum(axis=1)

    return csr_matrix(Ybus_not_controllable), csr_matrix(Ybus_controllable), \
        csr_matrix(Ybus_not_controllable + Ybus_controllable)


def makeYbus_hvdc(hvdc_y_pu, hvdc_fb, hvdc_tb):
    if len(hvdc_fb) == 0:
        return csr_matrix([], dtype=np.float64)
    num_hvdc = np.max(np.r_[hvdc_fb, hvdc_tb]) + 1
    Ybus_hvdc = np.zeros(shape=(num_hvdc, num_hvdc), dtype=np.float64)
    Ybus_hvdc[hvdc_fb, hvdc_tb] = -hvdc_y_pu
    Ybus_hvdc[hvdc_tb, hvdc_fb] = -hvdc_y_pu
    Ybus_hvdc[np.diag_indices_from(Ybus_hvdc)] = -Ybus_hvdc.sum(axis=1)

    return csr_matrix(Ybus_hvdc)


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


def make_Ybus_facts(from_bus, to_bus, y_pu, n):
    """
    Construct the bus admittance matrix with an added FACTS device for a power grid.

    This function creates an additional Ybus matrix, which models the connectivity of
    additional FACTS (Flexible AC Transmission System) devices in the power grid.

    Parameters:
    - from_bus (array-like): Array of source bus indices for each element.
    - to_bus (array-like): Array of destination bus indices for each element.
    - y_pu (array-like): Array of per-unit admittances for each element.
    - n (int): Total number of buses in the grid.

    Returns:
    - csr_matrix: A compressed sparse row representation of the bus admittance matrix
                  incorporating FACTS.
    """
    # Combine diagonal elements with off-diagonal elements
    # Concatenate row and column indices, as well as data
    # Explanation: first pair is for diagonal data, second pair is for off-diagonal data
    # At creation of the csr_matrix, data elements at the coinciding indices are added
    # This feature is useful when you have repeated indices, and you want their values to be aggregated
    row_indices = np.concatenate([from_bus, to_bus, from_bus, to_bus])
    col_indices = np.concatenate([from_bus, to_bus, to_bus, from_bus])
    data = np.concatenate([y_pu, y_pu, -y_pu, -y_pu])

    # Create and return the Ybus matrix using the compressed sparse row format
    Ybus_facts = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
    return Ybus_facts


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


def make_Yft_facts(from_bus, to_bus, y_pu, n):
    """
    Construct the Yf and Yt admittance matrices for branches with FACTS devices.

    This function creates the Yf and Yt matrices, which allow to compute
    the currents injected at the "from" and "to" ends of each branch,
    respectively. The matrices represent the FACTS (Flexible AC Transmission
    System) devices in the grid.

    Parameters:
    - from_bus (array-like): Array of source bus indices for each branch.
    - to_bus (array-like): Array of destination bus indices for each branch.
    - y_pu (array-like): Array of per-unit admittances for each branch.
    - n (int): Total number of buses in the grid.

    Returns:
    - Yf (csr_matrix): Admittance matrix for the "from" end of each branch.
    - Yt (csr_matrix): Admittance matrix for the "to" end of each branch.
    """

    # Number of lines (or branches)
    nl = len(from_bus)
    rows = np.arange(nl)

    # Common row indices for Yf and Yt matrices
    row_indices = np.concatenate([rows, rows])
    # Common column indices for Yf and Yt matrices
    col_indices = np.concatenate([from_bus, to_bus])

    # Construct Yf and Yt matrices using the CSR format
    Yf = csr_matrix((np.concatenate([y_pu, -y_pu]), (row_indices, col_indices)), shape=(nl, n))
    Yt = csr_matrix((np.concatenate([-y_pu, y_pu]), (row_indices, col_indices)), shape=(nl, n))

    return Yf, Yt

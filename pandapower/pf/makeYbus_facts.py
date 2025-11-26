# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix


def calc_y_svc(x_control_degree, svc_x_l_ohm, svc_x_cvar_ohm, v_base_kv, baseMVA):
    """
    Computes the admittance of SVC (Static VAR Compensator) components based on their control angles
    and reactance values in ohms.

    Parameters:
    -----------
    x_control_degree : array-like
        Thyristor firing angles for the SVCs in degrees.
    svc_x_l_ohm : array-like
        Reactance of the SVC's inductor in ohms.
    svc_x_cvar_ohm : array-like
        Reactance of the SVC's capacitor in ohms.
    v_base_kv : array-like
        Base voltage levels in kV for corresponding SVC elements.
    baseMVA : float
        System's base power in MVA.

    Returns:
    --------
    array-like
        Computed admittance values of the SVCs in ohms.
    """
    x_control = np.deg2rad(x_control_degree)
    z_base_ohm = np.square(v_base_kv) / baseMVA
    svc_x_l_pu = svc_x_l_ohm / z_base_ohm
    svc_x_cvar_pu = svc_x_cvar_ohm / z_base_ohm
    y_svc = calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu)
    y_svc *= z_base_ohm
    return y_svc


def calc_y_svc_pu(x_control, svc_x_l_pu, svc_x_cvar_pu):
    """
    Calculates the admittance of SVC elements (Static VAR Compensator) in per unit.

    Parameters:
    -----------
    x_control : array-like
        Thyristor firing angles in radians for the SVCs.
    svc_x_l_pu : array-like
        Reactance of the SVC's inductor in per-unit.
    svc_x_cvar_pu : array-like
        Reactance of the SVC's capacitor in per-unit.

    Returns:
    --------
    array-like
        Admittance values of the SVCs in per-unit.
    """
    y_svc = (2 * (np.pi - x_control) + np.sin(2 * x_control) + np.pi * svc_x_l_pu / svc_x_cvar_pu) / (np.pi * svc_x_l_pu)
    return y_svc


@np.errstate(all="raise")
def makeYbus_svc(Ybus, x_control_svc, svc_x_l_pu, svc_x_cvar_pu, svc_buses):
    """
    Constructs the SVC (Static VAR Compensator) admittance matrix.

    Parameters:
    -----------
    Ybus : csr_matrix
        The base admittance matrix of the system.
    x_control_svc : array-like
        Thyristor firing angles for the SVCs in degrees.
    svc_x_l_pu : array-like
        Reactance of the SVC's inductor in per-unit.
    svc_x_cvar_pu : array-like
        Reactance of the SVC's capacitor in per-unit.
    svc_buses : array-like
        Bus indices where SVCs are connected.

    Returns:
    --------
    csr_matrix
        The admittance matrix for the SVCs.
    """
    Y_SVC = -1j * calc_y_svc_pu(x_control_svc, svc_x_l_pu, svc_x_cvar_pu)
    Ybus_svc = csr_matrix((Y_SVC, (svc_buses, svc_buses)), shape=Ybus.shape, dtype=np.complex128)
    return Ybus_svc


def makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb):
    """
    Constructs the TCSC (Thyristor Controlled Series Capacitor) admittance matrix.

    Parameters:
    -----------
    Ybus : csr_matrix
        The base admittance matrix of the system.
    x_control_tcsc : array-like
        Thyristor firing angles for the TCSCs in radians.
    tcsc_x_l_pu : array-like
        Reactance of the TCSC's inductor in per-unit.
    tcsc_x_cvar_pu : array-like
        Reactance of the TCSC's capacitor in per-unit.
    tcsc_fb : array-like
        From-bus indices where TCSCs are connected.
    tcsc_tb : array-like
        To-bus indices where TCSCs are connected.

    Returns:
    --------
    csr_matrix
        The updated admittance matrix considering the TCSCs.
    """
    Y_TCSC = -1j * calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)
    Ybus_tcsc = make_Ybus_facts(tcsc_fb, tcsc_tb, Y_TCSC, Ybus.shape[0])
    return Ybus_tcsc


def makeYft_tcsc(Ybus_tcsc, tcsc_fb, tcsc_tb, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu):
    """
    Constructs the 'from' and 'to' admittance matrices for the TCSC (Thyristor Controlled Series Capacitor).

    Parameters:
    -----------
    Ybus_tcsc : csr_matrix
        The admittance matrix considering the TCSCs.
    x_control_tcsc : array-like
        Thyristor firing angles for the TCSCs in radians.
    tcsc_x_l_pu : array-like
        Reactance of the TCSC's inductor in per-unit.
    tcsc_x_cvar_pu : array-like
        Reactance of the TCSC's capacitor in per-unit.
    tcsc_fb : array-like
        From-bus indices where TCSCs are connected.
    tcsc_tb : array-like
        To-bus indices where TCSCs are connected.

    Returns:
    --------
    tuple of csr_matrix
        - Yf : csr_matrix
            The 'from' admittance matrix for the TCSCs.
        - Yt : csr_matrix
            The 'to' admittance matrix for the TCSCs.
    """
    Y_TCSC = -1j * calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)
    Yf, Yt = make_Yft_facts(tcsc_fb, tcsc_tb, Y_TCSC, Ybus_tcsc.shape[0])
    return Yf, Yt


def makeYbus_ssc_vsc(Ybus, internal_y_pu, fb, tb, controllable):
    """
    Constructs the admittance matrices for SSCs (Static Synchronous Compensators)
    and VSCs (Voltage Source Converters).

    Parameters:
    -----------
    Ybus : csr_matrix
        The base admittance matrix.
    internal_y_pu : array-like
        Internal transformer admittance values in per-unit.
    fb : array-like
        From bus indices.
    tb : array-like
        To bus indices.
    controllable : array-like
        Boolean flags indicating if elements are controllable.

    Returns:
    --------
    tuple of csr_matrix
        The not controllable, controllable, and combined admittance matrices.
    """
    Ybus_matrices = []

    for flag in [~controllable, controllable]:
        if not np.any(flag):
            Ybus_matrices.append(csr_matrix(Ybus.shape, dtype=np.complex128))
            continue
        Ybus_flag = make_Ybus_facts(fb[flag], tb[flag], internal_y_pu[flag], Ybus.shape[0])
        Ybus_matrices.append(Ybus_flag)

    Ybus_not_controllable, Ybus_controllable = Ybus_matrices[0], Ybus_matrices[1]

    return Ybus_not_controllable, Ybus_controllable, Ybus_not_controllable + Ybus_controllable


def make_Ybus_facts(from_bus, to_bus, y_pu, n, ysf_pu=0, yst_pu=0, dtype=np.complex128):
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
    data = np.concatenate([y_pu + ysf_pu, y_pu + yst_pu, -y_pu, -y_pu])

    # Create and return the Ybus matrix using the compressed sparse row format
    Ybus_facts = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=dtype)
    return Ybus_facts


@np.errstate(all="raise")
def make_Yft_facts(from_bus, to_bus, y_pu, n, ysf_pu=0, yst_pu=0):
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
    Yf = csr_matrix((np.concatenate([y_pu + ysf_pu, -y_pu]), (row_indices, col_indices)), shape=(nl, n))
    Yt = csr_matrix((np.concatenate([-y_pu, y_pu + yst_pu]), (row_indices, col_indices)), shape=(nl, n))

    return Yf, Yt

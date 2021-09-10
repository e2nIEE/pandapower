import numpy as np
from numpy import complex128, float64, int32, r_
from numpy.core.multiarray import zeros, empty, array
from scipy.sparse import csr_matrix as sparse, vstack, hstack

from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.idx_bus import BUS_I
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, QF, PT, QT, BR_STATUS

try:
    # numba functions
    from pandapower.pf.create_jacobian_numba import create_J, create_J2, create_J_ds
    from pandapower.pf.dSbus_dV_numba import dSbus_dV_numba_sparse
except ImportError:
    pass

# todo: makeYbus
# todo: adjust newton

SIGMA = 5.670374419e-8
# ALPHA = 4.03e-3
ALPHA = 4e-3


def calc_r_temp(r_ref_ohm_per_m, t_end, t_ref=20, alpha=ALPHA):
    return r_ref_ohm_per_m * (1 + alpha * (t_end - t_ref))


def calc_t_ss(i_a, a0, a1, a2):
    return a0 + a1 * i_a ** 2 + a2 * i_a ** 4


def calc_t_transient(t_0_degree, t_s, i_a, a0, a1, a2, tau):
    t_ss_degree = calc_t_ss(i_a, a0, a1, a2)
    return t_ss_degree - (t_ss_degree - t_0_degree) * np.exp(-t_s/tau)


def calc_a0_a1_a2_tau(t_amb, t_max, r_ref_ohm_per_m, conductor_outer_diameter_m, mc_joule_per_m_k, v_m_per_s, wind_angle_degree, s_w_per_square_meter=300, alpha=ALPHA, gamma=0.5, epsilon=0.5):
    r_amb_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_amb)
    r_max_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_max)

    h_r = 4 * np.pi * conductor_outer_diameter_m * SIGMA * epsilon * (t_amb + 273) ** 3
    kappa = 6 * np.pi * conductor_outer_diameter_m * SIGMA * epsilon * (t_amb + 273) ** 2

    h_c = calc_h_c(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, t_amb)

    a0 = t_amb + (gamma * conductor_outer_diameter_m * s_w_per_square_meter) / (h_r + h_c)
    a1 = r_amb_ohm_per_m / (h_r + h_c)
    a2 = a1 / (h_r + h_c) * (alpha * r_ref_ohm_per_m - kappa * a1)

    # rho = 2710  # kg/m³ # density of aluminum
    # c = 1.309e6  # J/kg°C
    tau = mc_joule_per_m_k / (h_r + h_c)

    return a0, a1, a2, tau


def calc_h_c(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, t_amb):
    rho_air = 101325 / (287.058 * (t_amb+273))  # pressure 1 atm. / (R_specific * T)
    rho_air_relative = 1.  # relative air density
    # r_f = 0.05 # roughness of conductors
    r_f = 0.1 # roughness of conductors
    w = rho_air * conductor_outer_diameter_m * v_m_per_s

    if v_m_per_s < 0.5:
        K = 0.55
    elif wind_angle_degree < 24:
        K = 0.42 + 0.68 * np.sin(np.deg2rad(wind_angle_degree)) ** 1.08
    else:
        K = 0.42 + 0.58 * np.sin(np.deg2rad(wind_angle_degree)) ** 0.9

    h_cfl = 8.74 * K * w ** 0.471

    h_cfh = 13.44 * K * w ** 0.633 if r_f <= 0.05 else 20.89 * K * w ** 0.8

    h_cn = 8.1 * conductor_outer_diameter_m ** 0.75

    h_c = np.maximum(np.maximum(h_cfl, h_cfh), h_cn)

    return h_c

#
# def calc_a0_a1_a2_old(t_amb, t_max, r_ref_ohm_per_m, conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, s_w_per_square_meter=300, alpha=ALPHA, gamma=0.5, epsilon=0.5):
#     r_amb_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_amb)
#     r_max_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_max)
#     h_r = 4 * SIGMA * epsilon * (t_amb + 273) ** 3
#     # h_r = 4 * np.pi * conductor_outer_diameter_m * SIGMA * epsilon * (t_amb + 273) ** 3
#     h_c = calc_h_c(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, t_amb)
#     # R0 = 1 / (np.pi * conductor_outer_diameter_m * (h_r + h_c))
#     R0 = 1 / (h_r + h_c)
#     kappa = 6 * np.pi * conductor_outer_diameter_m * SIGMA * epsilon * (t_amb + 273) ** 2
#     R1 = 1 / (np.pi * conductor_outer_diameter_m * (h_r + h_c + kappa * (t_max - t_amb)))
#     Ps = gamma * conductor_outer_diameter_m * s_w_per_square_meter
#     #a0 = t_amb + R0 * Ps
#     a0 = t_amb + (gamma * conductor_outer_diameter_m * s_w_per_square_meter) / (h_r + h_c)
#     a1 = R0 * r_amb_ohm_per_m
#     # a2 = R0 * R1 * r_max_ohm_per_m * (alpha * r_ref_ohm_per_m - kappa)
#     a2 = a1 / (h_r + h_c) * (alpha * r_ref_ohm_per_m - kappa * a1)
#     return a0, a1, a2
#
#
# def calc_h_c_old(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree):
#     B1 = 0.178
#     n1 = 0.633
#     # rho_air = 1.2041  # kg_per_m3
#     rho_air = 1.  # relative air density
#
#     if v_m_per_s < 0.5:
#         K = 0.55
#     elif wind_angle_degree < 24:
#         K = 0.42 + 0.68 * np.sin(np.deg2rad(wind_angle_degree)) ** 1.08
#     else:
#         K = 0.42 + 0.58 * np.sin(np.deg2rad(wind_angle_degree)) ** 0.9
#
#     # K = 1.194 - np.cos(np.deg2rad(wind_angle_degree)) + (0.194 * np.cos(2 * np.deg2rad(wind_angle_degree))) + (0.368 * np.sin(2 * np.deg2rad(wind_angle_degree)))
#
#     h_cl = 3.07 / conductor_outer_diameter_m * K * (rho_air * conductor_outer_diameter_m * v_m_per_s) ** 0.471
#
#     h_ch = B1 / conductor_outer_diameter_m * K * (rho_air * conductor_outer_diameter_m * v_m_per_s) ** n1
#
#     h_c = np.maximum(h_cl, h_ch)
#
#     return h_c
#
#
# def calc_tau_old(R0, q_mm2, rho, c, h_r, h_c):
#     # rho = 2710  # kg/m³ # density of aluminum
#     # c = 1.309e6  # J/kg°C
#     q_m2 = q_mm2 * 1e-6
#     m_kg_per_m = rho_kg_per_m3 * area_m2
#     tau = m_kg_per_m * c / (h_r + h_c)
#     return tau


def create_J_tdpf(branch, alpha, r_ref, Yf, Yt, baseMVA, pvpq, pq, pvpq_lookup, pq_lookup, tau, t, a1, a2, V, J):
    Sf, St = get_S_flows(branch, Yf, Yt, baseMVA, V)
    I = Sf / V[pvpq_lookup[branch[:, F_BUS].real.astype(int)]] / np.sqrt(3)

    Vm = np.abs(V)
    Va = np.angle(V)

    A, B = calc_AB(branch, pvpq, pvpq_lookup, Va, Vm)
    C = (1 - np.exp(-t / tau)) * (a1 + a2 * np.square(abs(I)))

    J13 = create_J13(branch, alpha, r_ref, pvpq, pvpq_lookup, Sf, St, A)
    J23 = create_J23(branch, alpha, r_ref, pq, pq_lookup, Sf, St, B)
    J31 = create_J31(branch, alpha, pvpq, pvpq_lookup, B, C)
    J32 = create_J32(branch, alpha, r_ref, pq, pq_lookup, A, C, Vm)
    J33 = create_J33(branch, alpha, r_ref, pvpq, pvpq_lookup, I)

    Jright = np.vstack([J13, J23])
    Jbtm = np.hstack([J31, J32, J33])
    JJ = np.vstack([np.hstack([J.toarray(), Jright]), Jbtm])
    return JJ


def get_S_flows(branch, Yf, Yt, baseMVA, V):
    br = branch[:, BR_STATUS].real.astype(bool)
    Sf = V[np.real(branch[br, F_BUS]).astype(int)] * np.conj(Yf[br, :] * V) * baseMVA
    # complex power injected at "to" bus
    St = V[np.real(branch[br, T_BUS]).astype(int)] * np.conj(Yt[br, :] * V) * baseMVA
    return Sf, St


def calc_AB(branch, pvpq, pvpq_lookup, Va, Vm):
    A = np.zeros(shape=(len(pvpq), len(pvpq)))
    B = np.zeros(shape=(len(pvpq), len(pvpq)))

    for br in range(len(branch)):
        f, t = branch[br, [F_BUS, T_BUS]].real.astype(np.int64)
        m = pvpq_lookup[f]
        i = pvpq_lookup[t]
        A[m, i] = np.square(Vm[m]) - Vm[m] * Vm[i] * np.cos(Va[m] - Va[i])
        B[m, i] = Vm[m] * Vm[i] * np.sin(Va[m] - Va[i])

    # for bus in pvpq:
    #     m = int(pvpq_lookup[bus])
    #     if bus in branch[:, F_BUS].real.astype(int):
    #         other = branch[branch[:, F_BUS] == bus, T_BUS].real.astype(int)
    #         i = pvpq_lookup[other]
    #     elif bus in branch[:, T_BUS].real.astype(int):
    #         other = branch[branch[:, T_BUS] == bus, F_BUS].real.astype(int)
    #         i = pvpq_lookup[other]
    #     else:
    #         continue
    #
    #     A[m, i] = np.square(Vm[m]) - Vm[m] * Vm[i] * np.cos(Va[m]-Va[i])
    #     B[m, i] = Vm[m] * Vm[i] * np.sin(Va[m]-Va[i])

    return A, B


def create_J13(branch, alpha, r_ref, pvpq, pvpq_lookup, Sf, St, A):
    """
         / J11 = dP/dd     J12 = dP/dV     J13 = dP/dT  \
         | (N-1)x(N-1)     (N-1)x(M)       (N-1)x(R)    |
         |                                              |
         | J21 = dQ/dd     J22 = dQ/dV     J23 = dQ/dT  |
         | (M)x(N-1)       (M)x(M)         (M)x(R)      |
         |                                              |
         | J31 = ddT/dd    J32 = ddT/dV    J33 = ddT/dT |
         \ (R)x(N-1)       (R)x(M)         (R)x(R)      /

    N = Number of buses
    M = Number of PQ buses
    R = Number temperature-dependent branches

    shape = (len(branch), len(bus))
    :param pvpq_lookup:

    """
    nrow = len(pvpq)
    ncol = len(branch)
    J13 = np.zeros(shape=(nrow, ncol))

    r = branch[:, BR_R].real
    x = branch[:, BR_X].real
    g = r / (np.square(r) + np.square(x))

    for bus in pvpq:
        m = pvpq_lookup[bus]
        for brch in range(ncol):
            f = branch[brch, F_BUS].real.astype(np.int64)
            t = branch[brch, T_BUS].real.astype(np.int64)
            i = pvpq_lookup[f]
            j = pvpq_lookup[t]
            a = alpha[brch]
            if m == i:
                J13[m, brch] = a * r_ref[brch] * g[brch] * (A[m, j] / r[brch] - 2 * Sf[brch].real)
            elif m == j:
                J13[m, brch] = a * r_ref[brch] * g[brch] * (A[m, i] / r[brch] - 2 * Sf[brch].real)

    return J13


def create_J23(branch, alpha, r_ref, pq, pq_lookup, Sf, St, B):
    """
         / J11 = dP/dd     J12 = dP/dV     J13 = dP/dT  \
         | (N-1)x(N-1)     (N-1)x(M)       (N-1)x(R)    |
         |                                              |
         | J21 = dQ/dd     J22 = dQ/dV     J23 = dQ/dT  |
         | (M)x(N-1)       (M)x(M)         (M)x(R)      |
         |                                              |
         | J31 = ddT/dd    J32 = ddT/dV    J33 = ddT/dT |
         \ (R)x(N-1)       (R)x(M)         (R)x(R)      /

        N = Number of buses
        M = Number of PQ buses
        R = Number temperature-dependent branches

    shape = (len(bus), len(branch))
    :param pq_lookup:

    """

    ncol = len(branch)
    nrow = len(pq)
    J23 = np.zeros(shape=(nrow, ncol))

    r = branch[:, BR_R].real
    x = branch[:, BR_X].real
    g = r / (np.square(r) + np.square(x))

    for bus in pq:
        m = pq_lookup[bus]
        for brch in range(ncol):
            f = branch[brch, F_BUS].real.astype(int)
            t = branch[brch, T_BUS].real.astype(int)
            i = pq_lookup[f]
            j = pq_lookup[t]
            a = alpha[brch]
            if bus == i:
                J23[m, brch] = a * r_ref[brch] * g[brch] * (B[m, j] / r[brch] - 2 * Sf[brch].imag)
            elif bus == j:
                J23[m, brch] = a * r_ref[brch] * g[brch] * (B[m, i] / r[brch] - 2 * Sf[brch].imag)

    return J23


def create_J31(branch, alpha, pvpq, pvpq_lookup, B, C):
    """
         / J11 = dP/dd     J12 = dP/dV     J13 = dP/dT  \
         | (N-1)x(N-1)     (N-1)x(M)       (N-1)x(R)    |
         |                                              |
         | J21 = dQ/dd     J22 = dQ/dV     J23 = dQ/dT  |
         | (M)x(N-1)       (M)x(M)         (M)x(R)      |
         |                                              |
         | J31 = ddT/dd    J32 = ddT/dV    J33 = ddT/dT |
         \ (R)x(N-1)       (R)x(M)         (R)x(R)      /

    N = Number of buses
    M = Number of PQ buses
    R = Number temperature-dependent branches

    shape = (len(branch), len(bus))
    branch elements by row, bus elements by column
    :param pvpq_lookup:

    """

    nrow = len(branch)
    ncol = len(pvpq)

    J31 = np.zeros(shape=(nrow, ncol))

    r = branch[:, BR_R].real
    x = branch[:, BR_X].real
    g = r / (np.square(r) + np.square(x))
    b = -x / (np.square(r) + np.square(x))

    for row in range(nrow):
        f = branch[row, F_BUS].real.astype(int)
        t = branch[row, T_BUS].real.astype(int)
        i = pvpq_lookup[f]
        j = pvpq_lookup[t]
        a = alpha[row]
        for bus in pvpq:
            m = pvpq_lookup[bus]
            if m == i:
                J31[row, m] = (np.square(g[row]) + np.square(b[row])) * C[row] * B[m, j]
            elif m == j:
                J31[row, m] = -(np.square(g[row]) + np.square(b[row])) * C[row] * B[m, i]

    return J31


def create_J32(branch, alpha, r_ref, pq, pq_lookup, A, C, Vm):
    """
         / J11 = dP/dd     J12 = dP/dV     J13 = dP/dT  \
         | (N-1)x(N-1)     (N-1)x(M)       (N-1)x(R)    |
         |                                              |
         | J21 = dQ/dd     J22 = dQ/dV     J23 = dQ/dT  |
         | (M)x(N-1)       (M)x(M)         (M)x(R)      |
         |                                              |
         | J31 = ddT/dd    J32 = ddT/dV    J33 = ddT/dT |
         \ (R)x(N-1)       (R)x(M)         (R)x(R)      /

        N = Number of buses
        M = Number of PQ buses
        R = Number temperature-dependent branches

    shape = (len(branch), len(bus))
    branch elements by row, bus elements by column
    :param pq_lookup:

    """

    nrow = len(branch)
    ncol = len(pq)

    J32 = np.zeros(shape=(nrow, ncol))

    r = branch[:, BR_R].real
    x = branch[:, BR_X].real
    g = r / (np.square(r) + np.square(x))
    b = -x / (np.square(r) + np.square(x))

    for row in range(nrow):
        f = branch[row, F_BUS].real.astype(int)
        t = branch[row, T_BUS].real.astype(int)
        i = pq_lookup[f]
        j = pq_lookup[t]
        a = alpha[row]
        for bus in pq:
            m = pq_lookup[bus]
            if m == i:
                J32[row, m] = 2 * (np.square(g[row]) + np.square(b[row])) * C[row] * A[m, j] / Vm[i]
            elif m == j:
                J32[row, m] = 2 * (np.square(g[row]) + np.square(b[row])) * C[row] * A[m, i] / Vm[j]
            else:
                J32[row, m] = 0

    return J32


def create_J33(branch, alpha, r_ref, pvpq, pvpq_lookup, I):
    """
     / J11 = dP/dd     J12 = dP/dV     J13 = dP/dT  \
     | (N-1)x(N-1)     (N-1)x(M)       (N-1)x(R)    |
     |                                              |
     | J21 = dQ/dd     J22 = dQ/dV     J23 = dQ/dT  |
     | (M)x(N-1)       (M)x(M)         (M)x(R)      |
     |                                              |
     | J31 = ddT/dd    J32 = ddT/dV    J33 = ddT/dT |
     \ (R)x(N-1)       (R)x(M)         (R)x(R)      /

    N = Number of buses
    M = Number of PQ buses
    R = Number temperature-dependent branches

    shape = (len(branch), len(bus))
    branch elements by row, bus elements by column
    :param pvpq_lookup:

    """

    nrow = len(branch)

    J33 = np.zeros(shape=(nrow, nrow))

    r = branch[:, BR_R].real
    x = branch[:, BR_X].real
    g = r / (np.square(r) + np.square(x))
    b = -x / (np.square(r) + np.square(x))

    for mn in range(nrow):
        for ij in range(nrow):
            if mn == ij:
                J33[mn, ij] = -(1 + 2 * alpha[ij] * r_ref[ij] * g[ij] * np.square(abs(I[ij])))

    return J33


if __name__ == "__main__":
    # from pandapower.tdpf.create_jacobian_tdpf import *
    import pandapower as pp
    import pandapower.networks
    from pandapower.pypower.newtonpf import *
    from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
    from pandapower.pypower.makeYbus import makeYbus as makeYbus_pypower
    from pandapower.tdpf.test_system import test_grid

    # net = pp.networks.case9()
    # net.line.loc[net.line.r_ohm_per_km == 0, 'r_ohm_per_km'] = 10.
    net = test_grid()
    pp.runpp(net)
    ppc = net._ppc
    ppci = net._ppc
    options = net._options

    # options
    tol = options['tolerance_mva']
    max_it = options["max_iteration"]
    numba = options["numba"]
    iwamoto = options["algorithm"] == "iwamoto_nr"
    voltage_depend_loads = options["voltage_depend_loads"]
    dist_slack = options["distributed_slack"]
    v_debug = options["v_debug"]
    use_umfpack = options["use_umfpack"]
    permc_spec = options["permc_spec"]

    baseMVA = ppci['baseMVA']
    bus = ppci['bus']
    gen = ppci['gen']
    slack_weights = bus[:, SL_FAC].astype(float64)  ## contribution factors for distributed slack

    baseMVA, bus, gen, branch, ref, pv, pq, _, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    # initialize
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)
    dVa, dVm = None, None

    # set up indexing for updating V
    if dist_slack and len(ref) > 1:
        pv = r_[ref[1:], pv]
        ref = ref[[0]]

    Ybus, Yf, Yt = makeYbus_pypower(baseMVA, bus, branch)

    pvpq = r_[pv, pq]
    refpvpq = r_[ref, pvpq]
    pvpq_lookup = zeros(max(Ybus.indices) + 1, dtype=int)

    if dist_slack:
        # slack bus is relevant for the function createJ_ds
        pvpq_lookup[refpvpq] = arange(len(refpvpq))
    else:
        pvpq_lookup[pvpq] = arange(len(pvpq))

    pq_lookup = zeros(len(pvpq) + 1, dtype=int)
    pq_lookup[pq] = np.arange(len(pq))

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)

    nref = len(ref)
    npv = len(pv)
    npq = len(pq)
    j1 = 0
    j2 = npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses
    j7 = j6
    j8 = j6 + nref  # j7:j8 - slacks

    # make initial guess for the slack
    slack = gen[:, PG].sum() - bus[:, PD].sum()

    t_amb = 40
    r_ref = net.line.r_ohm_per_km.values * 1e-3
    a0, a1, a2, tau = calc_a0_a1_a2_tau(t_amb, 90, r_ref, )

    tau = np.ones(len(branch))
    a1 = np.ones(len(branch))
    a2 = np.ones(len(branch))
    t = 0
    alpha = np.ones(len(branch)) * 0.004
    r_ref = branch[:, BR_R].real.copy()

    J_base = create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

    J = create_J_tdpf(branch, alpha, r_ref, Yf, Yt, baseMVA, pvpq, pq, pvpq_lookup, pq_lookup, tau, t, a1, a2, V, J_base)

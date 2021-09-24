import numpy as np
from numpy import complex128, float64, int32, r_
from numpy.core.multiarray import zeros, empty, array
from scipy.sparse import csr_matrix as sparse, vstack, hstack

from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.idx_bus import BUS_I, BASE_KV
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


def calc_t_transient(t_0_degree, tdpf_delay_s, i_a, a0, a1, a2, tau):
    t_ss_degree = calc_t_ss(i_a, a0, a1, a2)
    return t_ss_degree - (t_ss_degree - t_0_degree) * np.exp(-tdpf_delay_s / tau)


def calc_a0_a1_a2_tau(t_amb, t_max, r_ref_ohm_per_m, conductor_outer_diameter_m, mc_joule_per_m_k, v_m_per_s, wind_angle_degree,
                      s_w_per_square_meter=300, alpha=ALPHA, gamma=0.5, epsilon=0.5):
    r_amb_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_amb)
    r_max_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_max)

    h_r = 4 * np.pi * conductor_outer_diameter_m * SIGMA * epsilon * (t_amb + 273) ** 3
    kappa = 6 * np.pi * conductor_outer_diameter_m * SIGMA * epsilon * (t_amb + 273) ** 2

    h_c = calc_h_c(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, t_amb)

    k2 = r_max_ohm_per_m / (h_r + h_c + kappa)

    a0 = t_amb + (gamma * conductor_outer_diameter_m * s_w_per_square_meter) / (h_r + h_c)
    a1 = r_amb_ohm_per_m / (h_r + h_c)
    a2 = k2 / (h_r + h_c) * (alpha * r_ref_ohm_per_m - kappa * k2)
    # a2 = a1 / (h_r + h_c) * (alpha * r_ref_ohm_per_m - kappa * a1)

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


def create_J_tdpf(branch, alpha, r_ref, pvpq, pq, pvpq_lookup, pq_lookup, tau, tdpf_delay_s, a1, a2, Vm, Va, i_square_pu, r_theta, J, r, x, g, b):

    C = (a1 + 2 * a2 * i_square_pu)
    if tdpf_delay_s is not None and tdpf_delay_s != np.inf:
        C *= (1 - np.exp(-tdpf_delay_s / tau))

    dg_dT = (np.square(x) - np.square(r)) * alpha * r_ref / np.square(np.square(r) + np.square(x))
    db_dT = 2 * x * r * alpha * r_ref / np.square(np.square(r) + np.square(x))

    # todo: optimize and speed-up the code for the matrices (vectorized and numba versions)
    # todo: figure out the indexing for the pv buses
    J13 = create_J13(branch, alpha, r_ref, pvpq, pvpq_lookup, Vm, Va, g, b, dg_dT, db_dT)
    J23 = create_J23(branch, alpha, r_ref, pq, pq_lookup, Vm, Va, g, b, dg_dT, db_dT)
    J31 = create_J31(branch, alpha, pvpq, pvpq_lookup, Vm, Va, C, r_theta, g, b)
    J32 = create_J32(branch, alpha, r_ref, pq, pq_lookup, Vm, Va, C, r_theta, g, b)
    J33 = create_J33(branch, alpha, r_ref, pvpq, pvpq_lookup, i_square_pu, r_theta, Vm, Va, g, b, dg_dT)

    Jright = vstack([J13, J23], format="csr")
    Jbtm = hstack([J31, J32, J33], format="csr")
    JJ = vstack([hstack([J, Jright]), Jbtm], format="csr")
    return JJ


def calc_i_square_pu(branch, Vm, Va):
    r = branch[:, BR_R].real
    x = branch[:, BR_X].real
    g = r / (np.square(r) + np.square(x))
    b = -x / (np.square(r) + np.square(x))

    i = branch[:, F_BUS].real.astype(np.int64)
    j = branch[:, T_BUS].real.astype(np.int64)

    A = np.square(Vm[i]) + np.square(Vm[j]) - 2 * Vm[i] * Vm[j] * np.cos(Va[i] - Va[j])

    i_square_pu = (np.square(g) + np.square(b)) * A

    p_loss_pu = g * A

    return i_square_pu, p_loss_pu


def calc_I(Sf, bus, f_bus, V):
    If = 1e3 * abs(Sf) / (abs(V[f_bus]) * bus[f_bus, BASE_KV].astype(float64)) / np.sqrt(3)
    return If


def get_S_flows(branch, Yf, Yt, baseMVA, V):
    br = branch[:, BR_STATUS].real.astype(bool)
    f_bus = np.real(branch[br, F_BUS]).astype(int)
    Sf = V[f_bus] * np.conj(Yf[br, :] * V) * baseMVA
    # complex power injected at "to" bus
    t_bus = np.real(branch[br, T_BUS]).astype(int)
    St = V[t_bus] * np.conj(Yt[br, :] * V) * baseMVA
    return Sf, St, f_bus, t_bus


def calc_AB(branch, pvpq, pvpq_lookup, Va, Vm):
    # A = np.zeros(shape=(len(pvpq), len(pvpq)))
    # B = np.zeros(shape=(len(pvpq), len(pvpq)))
    A = np.zeros(shape=(len(Vm), len(Vm)))
    B = np.zeros(shape=(len(Vm), len(Vm)))

    # todo: figure out the indexing for the pv buses
    for br in range(len(branch)):
        f, t = branch[br, [F_BUS, T_BUS]].real.astype(np.int64)
        #m = pvpq_lookup[f]
        #i = pvpq_lookup[t]
        m = f
        i = t
        A[m, i] = np.square(Vm[m]) - Vm[m] * Vm[i] * np.cos(Va[m] - Va[i])
        A[i, m] = np.square(Vm[i]) - Vm[m] * Vm[i] * np.cos(Va[i] - Va[m])
        B[m, i] = Vm[m] * Vm[i] * np.sin(Va[m] - Va[i])
        B[i, m] = Vm[m] * Vm[i] * np.sin(Va[i] - Va[m])

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


def create_J13(branch, alpha, r_ref, pvpq, pvpq_lookup, Vm, Va, g, b, dg_dT, db_dT):
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
    :param b:
    :param g:
    :param dg_dT:
    :param db_dT:
    :param Vm:
    :param Va:
    :param pvpq_lookup:

    """
    nrow = len(pvpq)
    ncol = len(branch)
    J13 = np.zeros(shape=(nrow, ncol))

    for m in pvpq:
        mm = pvpq_lookup[m]
        # m = bus
        for ij in range(ncol):
            i = branch[ij, F_BUS].real.astype(np.int64)
            j = branch[ij, T_BUS].real.astype(np.int64)

            if m == i:
                n = j
            elif m == j:
                n = i
            else:
                continue

            A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
            B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
            p_mn = g[ij] * A_mn - b[ij] * B_mn
            # J13[mm, ij] = alpha[ij] * r_ref[ij] * g[ij] * (A_mn / r[ij] - 2 * p_mn)
            J13[mm, ij] = A_mn * dg_dT[ij] - B_mn * db_dT[ij]

    return sparse(J13)


def create_J23(branch, alpha, r_ref, pq, pq_lookup, Vm, Va, g, b, dg_dT, db_dT):
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
    :param g:
    :param b:
    :param dg_dT:
    :param db_dT:
    :param Vm:
    :param Va:
    :param pq_lookup:

    """

    ncol = len(branch)
    nrow = len(pq)
    J23 = np.zeros(shape=(nrow, ncol))

    for m in pq:
        mm = pq_lookup[m]
        # m = bus
        for ij in range(ncol):
            i = branch[ij, F_BUS].real.astype(int)
            j = branch[ij, T_BUS].real.astype(int)

            if m == i:
                n = j
            elif m == j:
                n = i
            else:
                continue

            A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
            B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
            q_mn = -b[ij] * A_mn - g[ij] * B_mn
            # J13[mm, ij] = alpha[ij] * r_ref[ij] * g[ij] * (B_mn / r[ij] - 2 * q_mn)
            J23[mm, ij] = - A_mn * db_dT[ij] - B_mn * dg_dT[ij]

    return sparse(J23)


def create_J31(branch, alpha, pvpq, pvpq_lookup, Vm, Va, C, r_theta, g, b):
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

    shape = (len(branch), len(m))
    branch elements by row, m elements by column
    :param g:
    :param b:
    :param r_theta:
    :param Vm:
    :param Va:
    :param pvpq_lookup:

    """

    nrow = len(branch)
    ncol = len(pvpq)

    J31 = np.zeros(shape=(nrow, ncol))

    for ij in range(nrow):
        i = branch[ij, F_BUS].real.astype(int)
        j = branch[ij, T_BUS].real.astype(int)
        for m in pvpq:
            mm = pvpq_lookup[m]
            if m == i:
                n = j
                sign = 1
            elif m == j:
                n = i
                sign = -1
            else:
                continue

            B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
            # J31[ij, mm] = sign * (g[ij] ** 2 + b[ij] ** 2) * C[ij] * B_mn
            J31[ij, mm] = - 2 * r_theta[ij] * g[ij] * B_mn

    return sparse(J31)


def create_J32(branch, alpha, r_ref, pq, pq_lookup, Vm, Va, C, r_theta, g, b):
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

    shape = (len(branch), len(m))
    branch elements by ij, m elements by column
    :param g:
    :param b:
    :param r_theta:
    :param pq_lookup:

    """

    nrow = len(branch)
    ncol = len(pq)

    J32 = np.zeros(shape=(nrow, ncol))

    for ij in range(nrow):
        i = branch[ij, F_BUS].real.astype(int)
        j = branch[ij, T_BUS].real.astype(int)
        for m in pq:
            mm = pq_lookup[m]
            if m == i:
                n = j
            elif m == j:
                n = i
            else:
                continue

            A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
            # J32[ij, mm] = 2 * (g[ij]**2 + b[ij]**2) * C[ij] * A_mn / Vm[m]
            J32[ij, mm] = - 2 * r_theta[ij] * g[ij] * (Vm[m] - Vm[n] * np.cos(Va[m] - Va[n]))

    return sparse(J32)


def create_J33(branch, alpha, r_ref, pvpq, pvpq_lookup, i_square_pu, r_theta, Vm, Va, g, b, dg_dT):
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
    :param g:
    :param b:
    :param dg_dT:
    :param Vm:
    :param r_theta:
    :param pvpq_lookup:

    """

    nrow = len(branch)

    J33 = np.zeros(shape=(nrow, nrow))

    k = (np.square(g) + np.square(b)) / g
    p_loss_pu = i_square_pu / k

    for mn in range(nrow):
        for ij in range(nrow):
            i, j = branch[ij, [F_BUS, T_BUS]].real.astype(int)
            if mn == ij:
                # J33[mn, ij] = -(1 + 2 * alpha[ij] * r_ref[ij] * g[ij] * i_square_pu[ij])
                J33[mn, ij] = 1 - r_theta[ij] * (Vm[i]**2 + Vm[j]**2 - 2*Vm[i]*Vm[j]*np.cos(Va[i]-Va[j])) * dg_dT[ij]
    return sparse(J33)


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
    pp.set_user_pf_options(net, tdpf=True)
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
    a0, a1, a2, tau = calc_a0_a1_a2_tau(t_amb, 90, r_ref, 18.2e-3, 525, 0.5, 45, 1000)

    tdpf_delay_s = np.inf
    alpha = np.ones(len(branch)) * 0.004
    r_ref = branch[:, BR_R].real.copy()
    Sf, St, f_bus, _ = get_S_flows(branch, Yf, Yt, baseMVA, V)
    I = calc_I(Sf, bus, f_bus, V)

    J_base = create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

    J = create_J_tdpf(branch, alpha, r_ref, pvpq, pq, pvpq_lookup, pq_lookup, tau, t, a1, a2, V, Va, i_square_pu, p_rated_loss_pu, J_base,
                      r, x, g, b)

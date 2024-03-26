# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix as sparse, eye, vstack, hstack
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_STATUS

SIGMA = 5.670374419e-8
# ALPHA = 4.03e-3
ALPHA = 4e-3


def calc_r_theta_from_t_rise(net, t_rise_degree_celsius):
    """
    Calculate thermal resistance of the conductors from an assumed or calculated temperature rise.
    The calculation is implemented according to Frank et al.

    Parameters
    ----------
    net : pandapowerNet
    t_rise_degree_celsius : array
        temperature rise of the conductor

    Returns
    -------
    r_theta_kelvin_per_mw : array
        Thermal resistance of the conductors R_{\Theta}

    References
    ----------
    S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-Dependent Power Flow," in IEEE Transactions on Power Systems,
    vol. 28, no. 4, pp. 4007-4018, Nov. 2013, doi: 10.1109/TPWRS.2013.2266409.
    """
    r_for_t_rated_rise = net.line.r_ohm_per_km * (1 + net.line.alpha * t_rise_degree_celsius) * \
                         net.line.length_km / net.line.parallel
    p_rated_loss_mw = np.square(net.line.max_i_ka * np.sqrt(3)) * r_for_t_rated_rise
    r_theta_kelvin_per_mw = t_rise_degree_celsius / p_rated_loss_mw
    return r_theta_kelvin_per_mw


def calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va):
    """
    Calculate squared current and the active power losses.

    Parameters
    ----------
    branch: np.array(complex)
        ppc["branch"]
    tdpf_lines : np.array(bool)
        array that defines which lines are relevant for TDPF
    g : array
    b : array
    Vm : array
        Bus voltage magnitude (p.u.)
    Va : array
        Bus voltage angle (rad)

    Returns
    -------
    i_square_pu : array
        squared current
    p_loss_pu : array
        active power losses
    """
    i = branch[tdpf_lines, F_BUS].real.astype(np.int64)
    j = branch[tdpf_lines, T_BUS].real.astype(np.int64)

    A = np.square(Vm[i]) + np.square(Vm[j]) - 2 * Vm[i] * Vm[j] * np.cos(Va[i] - Va[j])

    # i_square_pu matches net.res_line.i_ka only if c_nf_per_km == 0
    i_square_pu = (np.square(g) + np.square(b)) * A

    # p_loss_pu here is correct, matches net.res_line.pl_mw
    p_loss_pu = g * A

    return i_square_pu, p_loss_pu


def calc_r_theta(t_air_pu, a0, a1, a2, i_square_pu, p_loss_pu):
    """
    Calculate thermal resistance using the thermal model from Ngoko et al.

    Parameters
    ----------
    t_air_pu : array
        Air temperature in p.u.
    a0 : array
        constant term of the thermal model
    a1 : array
        linear term of the thermal model
    a2 : array
        quadratic term of the thermal model
    i_square_pu : array
        squared current in p.u.
    p_loss_pu : array
        active power losses in p.u.

    Returns
    -------
    r_theta_pu : array
        Thermal resistance of the conductors R_{\Theta}

    References
    ----------
    S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-Dependent Power Flow," in IEEE Transactions on Power Systems,
    vol. 28, no. 4, pp. 4007-4018, Nov. 2013, doi: 10.1109/TPWRS.2013.2266409.

    B. Ngoko, H. Sugihara and T. Funaki, "A Temperature Dependent Power Flow Model Considering Overhead Transmission
    Line Conductor Thermal Inertia Characteristics," 2019 IEEE International Conference on Environment and
    Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC / I&CPS Europe),
    2019, pp. 1-6, doi: 10.1109/EEEIC.2019.8783234.
    """
    t_rise_pu = a0 + a1 * i_square_pu + a2 * np.square(i_square_pu) - t_air_pu
    r_theta_pu = t_rise_pu / np.where(p_loss_pu == 0, 1e-6, p_loss_pu)
    return r_theta_pu


def calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, tdpf_delay_s, T0, tau):
    """
    Calculate overhead line temperature according to the method from Frank et al.
    The calculation of the overhead line temperature is based on their thermal resistance.

    Parameters
    ----------
    p_loss_pu : array
        active power losses in p.u.
    t_air_pu : array
        Air temperature in p.u.
    r_theta_pu : array
        Thermal resistance of the conductors R_{\Theta}
    tdpf_delay_s : float, None
        Delay for the consideration of thermal inertia in seconds. Describes the time passed after a change
        of current in overhead lines that causes a change of temperature.
        Example: tdpf_delay_s = 0 means there is no change in temperature;
        tdpf_delay_s = np.inf leads to obtaining steady-state temperature (also default behavior if tdpf_delay_s = None)
    T0 : array, None
        initial temperature of overhead lines
    tau : array, None
        time constant of the overhead lines; describes the time after a current change after which the temperature
        reaches approx. 63.2 % of the steady-state value

    Returns
    -------
    t_transient : array
        Temperature of the overhead lines, either steady-state or corresponding to the time delay tdpf_delay_s

    References
    ----------
    S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-Dependent Power Flow," in IEEE Transactions on Power Systems,
    vol. 28, no. 4, pp. 4007-4018, Nov. 2013, doi: 10.1109/TPWRS.2013.2266409.
    """
    t_ss = t_air_pu + r_theta_pu * p_loss_pu

    if tdpf_delay_s is None:
        return t_ss

    t_transient = t_ss - (t_ss - T0) * np.exp(-tdpf_delay_s / tau)
    return t_transient


def calc_T_ngoko(i_square_pu, a0, a1, a2, tdpf_delay_s, T0, tau):
    """
    Calculate the overhead line temperature with the approach from Ngoko et al.
    The calculation of the overhead line temperature is based on the simplified model that
    includes a constant term, a linear coefficient and a quadratic coefficient.

    Parameters
    ----------
    i_square_pu : array
        squared current in p.u.
    a0 : array
        constant term of the thermal model
    a1 : array
        linear term of the thermal model
    a2 : array
        quadratic term of the thermal model
    tdpf_delay_s : float, None
        Delay for the consideration of thermal inertia in seconds. Describes the time passed after a change
        of current in overhead lines that causes a change of temperature.
        Example: tdpf_delay_s = 0 means there is no change in temperature;
        tdpf_delay_s = np.inf leads to obtaining steady-state temperature (also default behavior if tdpf_delay_s = None)
    T0 : array, None
        initial temperature of overhead lines
    tau : array, None
        time constant of the overhead lines; describes the time after a current change after which the temperature
        reaches approx. 63.2 % of the steady-state value

    Returns
    -------
    t_transient : array
        Temperature of the overhead lines, either steady-state or corresponding to the time delay tdpf_delay_s

    References
    ----------
    B. Ngoko, H. Sugihara and T. Funaki, "A Temperature Dependent Power Flow Model Considering Overhead Transmission
    Line Conductor Thermal Inertia Characteristics," 2019 IEEE International Conference on Environment and
    Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC / I&CPS Europe),
    2019, pp. 1-6, doi: 10.1109/EEEIC.2019.8783234.
    """
    t_ss = a0 + a1 * i_square_pu + a2 * np.square(i_square_pu)

    if tdpf_delay_s is None:
        return t_ss

    t_transient = t_ss - (t_ss - T0) * np.exp(-tdpf_delay_s / tau)
    return t_transient


def calc_a0_a1_a2_tau(t_air_pu, t_max_pu, t_ref_pu, r_ref_ohm_per_m, conductor_outer_diameter_m,
                      mc_joule_per_m_k, wind_speed_m_per_s, wind_angle_degree, s_w_per_square_meter,
                      alpha_pu=ALPHA, solar_absorptivity=0.5, emissivity=0.5, T_base=1, i_base_a=1):
    """
    Calculate the coefficients for the simplified thermal model according to Ngoko et al.

    Parameters
    ----------
    t_air_pu : array
        Air temperature in p.u.
    t_max_pu : array
        max. rated temperature of the overhead lines
    t_ref_pu : array
        rated temperature at which the reference (datasheet) resistance is provided
    r_ref_ohm_per_m : array
        reference (datasheet) specific resistance of the overhead lines
    conductor_outer_diameter_m : array
        outer diameter of the overhead line conductors (diameter of 1 individual conductor of the overhead line)
    mc_joule_per_m_k : array
        specific thermal capacitance of the overhead line:
        mass per unit length m [kg/m] multiplied by the specific thermal capacity c [J/kg • K]
    wind_speed_m_per_s : array
        wind speed in m/s
    wind_angle_degree : array
        wind angle of attack
    s_w_per_square_meter : array
        solar radiation in W/m²
    alpha_pu : array
        temperature coefficient of resistance in p.u. - alpha multiplied by T_base
    solar_absorptivity : array
    emissivity : array
    T_base : array
        base value for T for calculating T in p.u.
    i_base_a : array
        base value for current for calculating I in p.u.

    Returns
    -------
    a0 : array
        constant term of the thermal model
    a1 : array
        linear term of the thermal model
    a2 : array
        quadratic term of the thermal model
    tau : array
        time constant of the overhead lines; describes the time after a current change after which the temperature
        reaches approx. 63.2 % of the steady-state value

    References
    ----------
    B. Ngoko, H. Sugihara and T. Funaki, "A Temperature Dependent Power Flow Model Considering Overhead Transmission
    Line Conductor Thermal Inertia Characteristics," 2019 IEEE International Conference on Environment and
    Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC / I&CPS Europe),
    2019, pp. 1-6, doi: 10.1109/EEEIC.2019.8783234.
    """
    # alpha here is expected to be for T in pu (alpha multiplied by T_base)
    r_amb_ohm_per_m = r_ref_ohm_per_m * (1 + alpha_pu * (t_air_pu - t_ref_pu))
    r_max_ohm_per_m = r_ref_ohm_per_m * (1 + alpha_pu * (t_max_pu - t_ref_pu))

    h_r = 4 * np.pi * conductor_outer_diameter_m * SIGMA * emissivity * (t_air_pu * T_base + 273) ** 3
    kappa = 6 * np.pi * conductor_outer_diameter_m * SIGMA * emissivity * (t_air_pu * T_base + 273) ** 2

    h_c = calc_h_c(conductor_outer_diameter_m, wind_speed_m_per_s, wind_angle_degree, t_air_pu * T_base)

    k2 = r_max_ohm_per_m / (h_r + h_c + kappa)

    a0 = t_air_pu + (solar_absorptivity * conductor_outer_diameter_m * s_w_per_square_meter) / (h_r + h_c) / T_base
    a1 = r_amb_ohm_per_m / (h_r + h_c) / T_base * np.square(i_base_a)
    a2 = k2 / (h_r + h_c) * (alpha_pu / T_base * r_ref_ohm_per_m - kappa * k2) / T_base * np.power(i_base_a, 4)
    # a2 = a1 / (h_r + h_c) * (alpha/T_base * r_ref_ohm_per_m - kappa * a1) / T_base * np.power(i_base_a, 4)

    # rho = 2710  # kg/m³ # density of aluminum
    # c = 1.309e6  # J/kg°C
    tau = mc_joule_per_m_k / (h_r + h_c)

    return a0, a1, a2, tau


def calc_h_c(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, t_air_degree_celsius):
    rho_air = 101325 / (287.058 * (t_air_degree_celsius + 273))  # pressure 1 atm. / (R_specific * T)
    rho_air_relative = 1.  # relative air density
    # r_f = 0.05 # roughness of conductors
    r_f = 0.1 # roughness of conductors
    w = rho_air * conductor_outer_diameter_m * v_m_per_s

    K = np.where(v_m_per_s < 0.5, 0.55,
                 np.where(v_m_per_s < 24,
                          0.42 + 0.68 * np.sin(np.deg2rad(wind_angle_degree)) ** 1.08,
                          0.42 + 0.58 * np.sin(np.deg2rad(wind_angle_degree)) ** 0.9))

    h_cfl = 8.74 * K * w ** 0.471

    h_cfh = 13.44 * K * w ** 0.633 if r_f <= 0.05 else 20.89 * K * w ** 0.8

    h_cn = 8.1 * conductor_outer_diameter_m ** 0.75

    h_c = np.maximum(np.maximum(h_cfl, h_cfh), h_cn)

    return h_c

#
# def calc_a0_a1_a2_old(t_air_degree_celsius, t_max, r_ref_ohm_per_m, conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, s_w_per_square_meter=300, alpha=ALPHA, solar_absorptivity=0.5, emissivity=0.5):
#     r_amb_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_air_degree_celsius)
#     r_max_ohm_per_m = calc_r_temp(r_ref_ohm_per_m, t_max)
#     h_r = 4 * SIGMA * emissivity * (t_air_degree_celsius + 273) ** 3
#     # h_r = 4 * np.pi * conductor_outer_diameter_m * SIGMA * emissivity * (t_air_degree_celsius + 273) ** 3
#     h_c = calc_h_c(conductor_outer_diameter_m, v_m_per_s, wind_angle_degree, t_air_degree_celsius)
#     # R0 = 1 / (np.pi * conductor_outer_diameter_m * (h_r + h_c))
#     R0 = 1 / (h_r + h_c)
#     kappa = 6 * np.pi * conductor_outer_diameter_m * SIGMA * emissivity * (t_air_degree_celsius + 273) ** 2
#     R1 = 1 / (np.pi * conductor_outer_diameter_m * (h_r + h_c + kappa * (t_max - t_air_degree_celsius)))
#     Ps = solar_absorptivity * conductor_outer_diameter_m * s_w_per_square_meter
#     #a0 = t_air_degree_celsius + R0 * Ps
#     a0 = t_air_degree_celsius + (solar_absorptivity * conductor_outer_diameter_m * s_w_per_square_meter) / (h_r + h_c)
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


def create_J_tdpf(branch, tdpf_lines, alpha_pu, r_ref_pu, pvpq, pq, pvpq_lookup, pq_lookup, tau, tdpf_delay_s, Vm, Va,
                  r_theta_pu, J, r, x, g):
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

    References
    ----------
    S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-Dependent Power Flow," in IEEE Transactions on Power Systems,
    vol. 28, no. 4, pp. 4007-4018, Nov. 2013, doi: 10.1109/TPWRS.2013.2266409.

    B. Ngoko, H. Sugihara and T. Funaki, "A Temperature Dependent Power Flow Model Considering Overhead Transmission
    Line Conductor Thermal Inertia Characteristics," 2019 IEEE International Conference on Environment and
    Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC / I&CPS Europe),
    2019, pp. 1-6, doi: 10.1109/EEEIC.2019.8783234.
    """
    C = np.ones_like(tdpf_lines, dtype=np.float64)
    if tdpf_delay_s is not None and tdpf_delay_s != np.inf:
        C *= (1 - np.exp(-tdpf_delay_s / tau))

    dg_dT = (np.square(x) - np.square(r)) * alpha_pu * r_ref_pu / np.square(np.square(r) + np.square(x))
    db_dT = 2 * x * g * alpha_pu * r_ref_pu / (np.square(r) + np.square(x))

    in_pq_f = np.isin(branch[tdpf_lines, F_BUS].real.astype(np.int64), pq)
    in_pq_t = np.isin(branch[tdpf_lines, T_BUS].real.astype(np.int64), pq)
    in_pvpq_f = np.isin(branch[tdpf_lines, F_BUS].real.astype(np.int64), pvpq)
    in_pvpq_t = np.isin(branch[tdpf_lines, T_BUS].real.astype(np.int64), pvpq)

    # todo: optimize and speed-up the code for the matrices (write numba versions)
    J13 = create_J13(branch, tdpf_lines, in_pvpq_f, in_pvpq_t, pvpq, pvpq_lookup, Vm, Va, dg_dT, db_dT)
    J23 = create_J23(branch, tdpf_lines, in_pq_f, in_pq_t, pq, pq_lookup, Vm, Va, dg_dT, db_dT)
    J31 = create_J31(branch, tdpf_lines, in_pvpq_f, in_pvpq_t, pvpq, pvpq_lookup, Vm, Va, C, r_theta_pu, g)
    J32 = create_J32(branch, tdpf_lines, in_pq_f, in_pq_t, pq, pq_lookup, Vm, Va, C, r_theta_pu, g)
    J33 = create_J33(branch, tdpf_lines, r_theta_pu, Vm, Va, dg_dT)

    Jright = vstack([sparse(J13), sparse(J23)], format="csr")
    Jbtm = hstack([sparse(J31), sparse(J32), sparse(J33)], format="csr")
    JJ = vstack([hstack([J, Jright]), Jbtm], format="csr")
    return JJ


def calc_I(Sf, bus, f_bus, V):
    If = 1e3 * abs(Sf) / (abs(V[f_bus]) * bus[f_bus, BASE_KV].astype(np.float64)) / np.sqrt(3)
    return If


def calc_g_b(r, x):
    g = r / (np.square(r) + np.square(x))
    b = -x / (np.square(r) + np.square(x))
    return g, b


def get_S_flows(branch, Yf, Yt, baseMVA, V):
    br = branch[:, BR_STATUS].real.astype(bool)
    f_bus = np.real(branch[br, F_BUS]).astype(np.int64)
    Sf = V[f_bus] * np.conj(Yf[br, :] * V) * baseMVA
    # complex power injected at "to" bus
    t_bus = np.real(branch[br, T_BUS]).astype(np.int64)
    St = V[t_bus] * np.conj(Yt[br, :] * V) * baseMVA
    return Sf, St, f_bus, t_bus

#
# def calc_AB(branch, tdpf_lines, pvpq, pvpq_lookup, Va, Vm):
#     # A = np.zeros(shape=(len(pvpq), len(pvpq)))
#     # B = np.zeros(shape=(len(pvpq), len(pvpq)))
#     A = np.zeros(shape=(len(Vm), len(Vm)))
#     B = np.zeros(shape=(len(Vm), len(Vm)))
#
#     # figure out the indexing for the pv buses:
#     for br in tdpf_lines:
#         f, t = branch[br, [F_BUS, T_BUS]].real.astype(np.int64)
#         #m = pvpq_lookup[f]
#         #i = pvpq_lookup[t]
#         m = f
#         i = t
#         A[m, i] = np.square(Vm[m]) - Vm[m] * Vm[i] * np.cos(Va[m] - Va[i])
#         A[i, m] = np.square(Vm[i]) - Vm[m] * Vm[i] * np.cos(Va[i] - Va[m])
#         B[m, i] = Vm[m] * Vm[i] * np.sin(Va[m] - Va[i])
#         B[i, m] = Vm[m] * Vm[i] * np.sin(Va[i] - Va[m])
#
#     # for bus in pvpq:
#     #     m = int(pvpq_lookup[bus])
#     #     if bus in branch[:, F_BUS].real.astype(np.int64):
#     #         other = branch[branch[:, F_BUS] == bus, T_BUS].real.astype(np.int64)
#     #         i = pvpq_lookup[other]
#     #     elif bus in branch[:, T_BUS].real.astype(np.int64):
#     #         other = branch[branch[:, T_BUS] == bus, F_BUS].real.astype(np.int64)
#     #         i = pvpq_lookup[other]
#     #     else:
#     #         continue
#     #
#     #     A[m, i] = np.square(Vm[m]) - Vm[m] * Vm[i] * np.cos(Va[m]-Va[i])
#     #     B[m, i] = Vm[m] * Vm[i] * np.sin(Va[m]-Va[i])
#
#     return A, B


def create_J13(branch, tdpf_lines, in_pvpq_f, in_pvpq_t, pvpq, pvpq_lookup, Vm, Va, dg_dT, db_dT):
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
    J13 = np.zeros(shape=(nrow, ncol), dtype=np.float64)

    # for m in pvpq:
    #     mm = pvpq_lookup[m]
    #     # m = bus
    #     #for ij in range(ncol):
    #     for ij_lookup, ij in enumerate(tdpf_lines):
    #         i = branch[ij, F_BUS].real.astype(np.int64)
    #         j = branch[ij, T_BUS].real.astype(np.int64)
    #
    #         if m == i:
    #             n = j
    #         elif m == j:
    #             n = i
    #         else:
    #             continue
    #
    #         A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
    #         B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
    #         # p_mn = g[ij_lookup] * A_mn - b[ij_lookup] * B_mn
    #         # J13[mm, ij] = alpha[ij] * r_ref[ij] * g[ij] * (A_mn / r[ij] - 2 * p_mn)
    #         J13[mm, ij] = A_mn * dg_dT[ij_lookup] - B_mn * db_dT[ij_lookup]

    mf = np.r_[branch[tdpf_lines[in_pvpq_f], F_BUS].real.astype(np.int64)]
    mt = np.r_[branch[tdpf_lines[in_pvpq_t], T_BUS].real.astype(np.int64)]
    nf = np.r_[branch[tdpf_lines[in_pvpq_f], T_BUS].real.astype(np.int64)]
    nt = np.r_[branch[tdpf_lines[in_pvpq_t], F_BUS].real.astype(np.int64)]

    for in_pq, m, n in ((in_pvpq_f, mf, nf), (in_pvpq_t, mt, nt)):
        pq_j = pvpq_lookup[m]
        A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
        B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
        J13[pq_j, tdpf_lines[in_pq]] = A_mn * dg_dT[in_pq] - B_mn * db_dT[in_pq]

    return J13


def create_J23(branch, tdpf_lines, in_pq_f, in_pq_t, pq, pq_lookup, Vm, Va, dg_dT, db_dT):
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
    J23 = np.zeros(shape=(nrow, ncol), dtype=np.float64)

    if nrow == 0:
        return J23

    # for m in pq:
    #     mm = pq_lookup[m]
    #     # m = bus
    #     #for ij in range(ncol):
    #     for ij_lookup, ij in enumerate(tdpf_lines):
    #         i = branch[ij, F_BUS].real.astype(np.int64)
    #         j = branch[ij, T_BUS].real.astype(np.int64)
    #
    #         if m == i:
    #             n = j
    #         elif m == j:
    #             n = i
    #         else:
    #             continue
    #
    #         A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
    #         B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
    #         q_mn = -b[ij_lookup] * A_mn - g[ij_lookup] * B_mn
    #         # J13[mm, ij] = alpha[ij] * r_ref[ij] * g[ij] * (B_mn / r[ij] - 2 * q_mn)
    #         J23[mm, ij] = - A_mn * db_dT[ij_lookup] - B_mn * dg_dT[ij_lookup]

    mf = np.r_[branch[tdpf_lines[in_pq_f], F_BUS].real.astype(np.int64)]
    mt = np.r_[branch[tdpf_lines[in_pq_t], T_BUS].real.astype(np.int64)]
    nf = np.r_[branch[tdpf_lines[in_pq_f], T_BUS].real.astype(np.int64)]
    nt = np.r_[branch[tdpf_lines[in_pq_t], F_BUS].real.astype(np.int64)]

    for in_pq, m, n in ((in_pq_f, mf, nf), (in_pq_t, mt, nt)):
        pq_j = pq_lookup[m]
        A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
        B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
        J23[pq_j, tdpf_lines[in_pq]] = - A_mn * db_dT[in_pq] - B_mn * dg_dT[in_pq]

    return J23


def create_J31(branch, tdpf_lines, in_pvpq_f, in_pvpq_t, pvpq, pvpq_lookup, Vm, Va, C, r_theta_pu, g):
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
    :param r_theta_pu:
    :param Vm:
    :param Va:
    :param pvpq_lookup:

    """

    nrow = len(branch)
    ncol = len(pvpq)

    J31 = np.zeros(shape=(nrow, ncol), dtype=np.float64)

    # #for ij in range(nrow):
    # for ij_lookup, ij in enumerate(tdpf_lines):
    #     i = branch[ij, F_BUS].real.astype(np.int64)
    #     j = branch[ij, T_BUS].real.astype(np.int64)
    #     for m in pvpq:
    #         mm = pvpq_lookup[m]
    #         if m == i:
    #             n = j
    #             sign = 1
    #         elif m == j:
    #             n = i
    #             sign = -1
    #         else:
    #             continue
    #
    #         B_mn = Vm[m] * Vm[n] * np.sin(Va[m] - Va[n])
    #         # J31[ij, mm] = sign * (g[ij] ** 2 + b[ij] ** 2) * C[ij] * B_mn
    #         J31_old[ij, mm] = - 2 * r_theta_pu[ij_lookup] * g[ij_lookup] * B_mn * C[ij_lookup]

    mf = np.r_[branch[tdpf_lines[in_pvpq_f], F_BUS].real.astype(np.int64)]
    mt = np.r_[branch[tdpf_lines[in_pvpq_t], T_BUS].real.astype(np.int64)]
    nf = np.r_[branch[tdpf_lines[in_pvpq_f], T_BUS].real.astype(np.int64)]
    nt = np.r_[branch[tdpf_lines[in_pvpq_t], F_BUS].real.astype(np.int64)]

    for in_pvpq, m, n in ((in_pvpq_f, mf, nf), (in_pvpq_t, mt, nt)):
        pq_j = pvpq_lookup[m]
        J31[tdpf_lines[in_pvpq], pq_j] = - 2 * r_theta_pu[in_pvpq] * g[in_pvpq] * Vm[m] * Vm[n] * np.sin(Va[m] - Va[n]) * C[in_pvpq]

    return J31


def create_J32(branch, tdpf_lines, in_pq_f, in_pq_t, pq, pq_lookup, Vm, Va, C, r_theta_pu, g):
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
    :param r_theta_pu:
    :param pq_lookup:

    """

    nrow = len(branch)
    ncol = len(pq)

    J32 = np.zeros(shape=(nrow, ncol), dtype=np.float64)

    if ncol == 0:
        return J32

    # # #for ij in range(nrow):
    # for ij_lookup, ij in enumerate(tdpf_lines):
    #     i = branch[ij, F_BUS].real.astype(np.int64)
    #     j = branch[ij, T_BUS].real.astype(np.int64)
    #     for m in pq:
    #         mm = pq_lookup[m]
    #         if m == i:
    #             n = j
    #         elif m == j:
    #             n = i
    #         else:
    #             continue
    #
    #         # A_mn = Vm[m] ** 2 - Vm[m] * Vm[n] * np.cos(Va[m] - Va[n])
    #         # J32[ij, mm] = 2 * (g[ij]**2 + b[ij]**2) * C[ij] * A_mn / Vm[m]
    #         J32_old[ij, mm] = - 2 * r_theta_pu[ij_lookup] * g[ij_lookup] * (Vm[m] - Vm[n] * np.cos(Va[m] - Va[n])) * C[ij_lookup]

    mf = np.r_[branch[tdpf_lines[in_pq_f], F_BUS].real.astype(np.int64)]
    mt = np.r_[branch[tdpf_lines[in_pq_t], T_BUS].real.astype(np.int64)]
    nf = np.r_[branch[tdpf_lines[in_pq_f], T_BUS].real.astype(np.int64)]
    nt = np.r_[branch[tdpf_lines[in_pq_t], F_BUS].real.astype(np.int64)]

    for in_pq, m, n in ((in_pq_f, mf, nf), (in_pq_t, mt, nt)):
        pq_j = pq_lookup[m]
        J32[tdpf_lines[in_pq], pq_j] = - 2 * r_theta_pu[in_pq] * g[in_pq] * (Vm[m] - Vm[n] * np.cos(Va[m] - Va[n])) * C[in_pq]

    return J32


def create_J33(branch, tdpf_lines, r_theta_pu, Vm, Va, dg_dT):
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
    :param r_theta_pu:
    :param pvpq_lookup:

    """

    nrow = len(branch)
    J33 = eye(nrow, format="csr", dtype=np.float64)

    # J33 = np.zeros(shape=(nrow, nrow))
    # J33[np.arange(nrow), np.arange(nrow)] = 1

    # k = (np.square(g) + np.square(b)) / g
    # p_loss_pu = i_square_pu / k

    #for mn in range(nrow):
    # for mn in tdpf_lines:
    #     #for ij in range(nrow):
    #     for ij_lookup, ij in enumerate(tdpf_lines):
    #         if mn == ij:
    #             i, j = branch[ij, [F_BUS, T_BUS]].real.astype(np.int64)
    #             # J33[mn, ij] = -(1 + 2 * alpha[ij_lookup] * r_ref[ij_lookup] * g[ij_lookup] * i_square_pu[ij_lookup])
    #             J33[mn, ij] = 1 - r_theta_pu[ij_lookup] * (Vm[i]**2 + Vm[j]**2 - 2*Vm[i]*Vm[j]*np.cos(Va[i]-Va[j])) * dg_dT[ij_lookup]

    #use this instead:
    # for ij_lookup, ij in enumerate(tdpf_lines):
    #    i, j = branch[ij, [F_BUS, T_BUS]].real.astype(np.int64)
    #    # J33[mn, ij] = -(1 + 2 * alpha[ij_lookup] * r_ref[ij_lookup] * g[ij_lookup] * i_square_pu[ij_lookup])
    #    J33[ij, ij] = 1 - r_theta_pu[ij_lookup] * (Vm[i] ** 2 + Vm[j] ** 2 - 2 * Vm[i] * Vm[j] * np.cos(Va[i] - Va[j])) * dg_dT[ij_lookup]

    # vectorized with numpy:
    i = branch[tdpf_lines, F_BUS].real.astype(np.int64)
    j = branch[tdpf_lines, T_BUS].real.astype(np.int64)
    J33[tdpf_lines, tdpf_lines] = 1 - r_theta_pu * (Vm[i] ** 2 + Vm[j] ** 2 - 2 * Vm[i] * Vm[j] * np.cos(Va[i] - Va[j])) * dg_dT

    return J33

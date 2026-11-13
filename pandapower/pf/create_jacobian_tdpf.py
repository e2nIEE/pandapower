# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, T_BUS
from pandapower.pypower.idx_bus import BASE_KV

SIGMA = 5.670374419e-8
# ALPHA = 4.03e-3
ALPHA_TDPF = 4e-3


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
        Thermal resistance of the conductors R_{\\Theta}

    References
    ----------
    S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-Dependent Power Flow," in IEEE Transactions on Power Systems,
    vol. 28, no. 4, pp. 4007-4018, Nov. 2013, doi: 10.1109/TPWRS.2013.2266409.
    """
    r_for_t_rated_rise = (
        net.line.r_ohm_per_km * (1 + net.line.alpha * t_rise_degree_celsius) * net.line.length_km / net.line.parallel
    )
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
        Thermal resistance of the conductors R_{\\Theta}

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
        Thermal resistance of the conductors R_{\\Theta}
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


def calc_a0_a1_a2_tau(
    t_air_pu,
    t_max_pu,
    t_ref_pu,
    r_ref_ohm_per_m,
    conductor_outer_diameter_m,
    mc_joule_per_m_k,
    wind_speed_m_per_s,
    wind_angle_degree,
    s_w_per_square_meter,
    alpha_pu=ALPHA_TDPF,
    solar_absorptivity=0.5,
    emissivity=0.5,
    T_base=1,
    i_base_a=1,
):
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
    r_f = 0.1  # roughness of conductors
    w = rho_air * conductor_outer_diameter_m * v_m_per_s

    K = np.where(
        v_m_per_s < 0.5,
        0.55,
        np.where(
            v_m_per_s < 24,
            0.42 + 0.68 * np.sin(np.deg2rad(wind_angle_degree)) ** 1.08,
            0.42 + 0.58 * np.sin(np.deg2rad(wind_angle_degree)) ** 0.9,
        ),
    )

    h_cfl = 8.74 * K * w**0.471

    h_cfh = 13.44 * K * w**0.633 if r_f <= 0.05 else 20.89 * K * w**0.8

    h_cn = 8.1 * conductor_outer_diameter_m**0.75

    h_c = np.maximum(np.maximum(h_cfl, h_cfh), h_cn)

    return h_c


def create_J_tdpf(
    branch,
    tdpf_lines,
    alpha_pu,
    r_ref_pu,
    pvpq,
    pq,
    pvpq_lookup,
    pq_lookup,
    tau,
    tdpf_delay_s,
    Vm,
    Va,
    r_theta_pu,
    J,
    r,
    x,
    g,
):
    """
             | J11 = dP/dd     J12 = dP/dV     J13 = dP/dT  |
             | (N-1)x(N-1)     (N-1)x(M)       (N-1)x(R)    |
             |                                              |
             | J21 = dQ/dd     J22 = dQ/dV     J23 = dQ/dT  |
             | (M)x(N-1)       (M)x(M)         (M)x(R)      |
             |                                              |
             | J31 = ddT/dd    J32 = ddT/dV    J33 = ddT/dT |
             | (R)x(N-1)       (R)x(M)         (R)x(R)      |

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
    n_branch = len(branch)
    n_pvpq = len(pvpq)
    n_pq = len(pq)

    C = np.ones_like(tdpf_lines, dtype=np.float64)
    if tdpf_delay_s is not None and tdpf_delay_s != np.inf:
        C *= 1 - np.exp(-tdpf_delay_s / tau)

    dg_dT = (np.square(x) - np.square(r)) * alpha_pu * r_ref_pu / np.square(np.square(r) + np.square(x))
    db_dT = 2 * x * g * alpha_pu * r_ref_pu / (np.square(r) + np.square(x))

    # bus masks instead of four np.isin calls
    mask_pvpq = np.zeros(len(Vm), dtype=bool)
    mask_pvpq[pvpq] = True
    mask_pq = np.zeros(len(Vm), dtype=bool)
    mask_pq[pq] = True

    f = branch[tdpf_lines, F_BUS].real.astype(np.int64)
    t = branch[tdpf_lines, T_BUS].real.astype(np.int64)

    # trig terms shared by all four coupling blocks
    Vf = Vm[f]
    Vt = Vm[t]
    delta = Va[f] - Va[t]
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    VfVt = Vf * Vt
    # A_mn = Vm[m]^2 - Vm[m] * Vm[n] * cos(Va[m] - Va[n]), B_mn = Vm[m] * Vm[n] * sin(Va[m] - Va[n])
    A_f = np.square(Vf) - VfVt * cos_d
    A_t = np.square(Vt) - VfVt * cos_d
    B_f = VfVt * sin_d
    B_t = -B_f

    in_pvpq_f = mask_pvpq[f]
    in_pvpq_t = mask_pvpq[t]
    in_pq_f = mask_pq[f]
    in_pq_t = mask_pq[t]

    # one entry per TDPF line and side; the branch index is a coordinate, so no duplicates arise
    br_pvpq = np.r_[tdpf_lines[in_pvpq_f], tdpf_lines[in_pvpq_t]]
    br_pq = np.r_[tdpf_lines[in_pq_f], tdpf_lines[in_pq_t]]
    row_pvpq = np.r_[pvpq_lookup[f[in_pvpq_f]], pvpq_lookup[t[in_pvpq_t]]]
    row_pq = np.r_[pq_lookup[f[in_pq_f]], pq_lookup[t[in_pq_t]]]

    A_pvpq = np.r_[A_f[in_pvpq_f], A_t[in_pvpq_t]]
    B_pvpq = np.r_[B_f[in_pvpq_f], B_t[in_pvpq_t]]
    dg_pvpq = np.r_[dg_dT[in_pvpq_f], dg_dT[in_pvpq_t]]
    db_pvpq = np.r_[db_dT[in_pvpq_f], db_dT[in_pvpq_t]]
    r_theta_pvpq = np.r_[r_theta_pu[in_pvpq_f], r_theta_pu[in_pvpq_t]]
    g_pvpq = np.r_[g[in_pvpq_f], g[in_pvpq_t]]
    C_pvpq = np.r_[C[in_pvpq_f], C[in_pvpq_t]]

    A_pq = np.r_[A_f[in_pq_f], A_t[in_pq_t]]
    B_pq = np.r_[B_f[in_pq_f], B_t[in_pq_t]]
    dg_pq = np.r_[dg_dT[in_pq_f], dg_dT[in_pq_t]]
    db_pq = np.r_[db_dT[in_pq_f], db_dT[in_pq_t]]
    r_theta_pq = np.r_[r_theta_pu[in_pq_f], r_theta_pu[in_pq_t]]
    g_pq = np.r_[g[in_pq_f], g[in_pq_t]]
    C_pq = np.r_[C[in_pq_f], C[in_pq_t]]

    J13 = csr_matrix(
        (A_pvpq * dg_pvpq - B_pvpq * db_pvpq, (row_pvpq, br_pvpq)),
        shape=(n_pvpq, n_branch),
    )
    J23 = csr_matrix(
        (-A_pq * db_pq - B_pq * dg_pq, (row_pq, br_pq)),
        shape=(n_pq, n_branch),
    )
    J31 = csr_matrix(
        (-2 * r_theta_pvpq * g_pvpq * B_pvpq * C_pvpq, (br_pvpq, row_pvpq)),
        shape=(n_branch, n_pvpq),
    )

    # J32 uses Vm[m] - Vm[n] * cos(Va[m] - Va[n]) per side
    dV_pq = np.r_[Vf[in_pq_f] - Vt[in_pq_f] * cos_d[in_pq_f], Vt[in_pq_t] - Vf[in_pq_t] * cos_d[in_pq_t]]
    J32 = csr_matrix(
        (-2 * r_theta_pq * g_pq * dV_pq * C_pq, (br_pq, row_pq)),
        shape=(n_branch, n_pq),
    )

    # J33 is the identity for non-TDPF branches and gets the thermal derivative on the TDPF diagonal
    j33_diag = np.ones(n_branch, dtype=np.float64)
    j33_diag[tdpf_lines] = 1 - r_theta_pu * (np.square(Vf) + np.square(Vt) - 2 * VfVt * cos_d) * dg_dT
    diagonal = np.arange(n_branch)
    J33 = csr_matrix((j33_diag, (diagonal, diagonal)), shape=(n_branch, n_branch))

    Jright = vstack([J13, J23], format="csr")
    Jbtm = hstack([J31, J32, J33], format="csr")
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

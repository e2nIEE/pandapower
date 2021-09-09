import pandas as pd
from pandapower.tdpf.create_jacobian_tdpf import *

if __name__ == "__main__":
    # from the 1st paper
    t_amb = 40
    t_max = 90
    r_ref_ohm_per_m = 0.1824e-3
    conductor_outer_diameter_m = 18.2e-3
    v_m_per_s = 0.5
    wind_angle_degree = 45
    s_w_per_square_meter = 1000
    alpha = ALPHA
    gamma = epsilon = 0.5
    mc_joule_per_m_k = 525

    # heat capacity mc = 525 J/mK
    # static line rating = 453 A

    # from the 2nd paper
    t_amb = 40
    t_max = 90
    r_ref_ohm_per_m = 0.0705e-3
    conductor_outer_diameter_m = 38.4e-3
    v_m_per_s = 0.5
    wind_angle_degree = 45
    s_w_per_square_meter = 1000
    alpha = ALPHA
    gamma = epsilon = 0.5
    mc_joule_per_m_k = 525

    a0, a1, a2, tau = calc_a0_a1_a2_tau(t_amb=t_amb, t_max=t_max, r_ref_ohm_per_m=r_ref_ohm_per_m,
                                        conductor_outer_diameter_m=conductor_outer_diameter_m, mc_joule_per_m_k=mc_joule_per_m_k,
                                        v_m_per_s=v_m_per_s, wind_angle_degree=wind_angle_degree, s_w_per_square_meter=s_w_per_square_meter)

    t_ss = calc_t_ss(600, a0, a1, a2)

    t = pd.DataFrame(columns=['i', 't'])
    for i in range(601):
        t.loc[i, 't2'] = calc_t_ss(i, a0, a1, a2)
        t.loc[i, ['i', 't']] = i, calc_t_ss(i, a0, a1, a2)
        # print(calc_t_ss(i, a0, a1, a2))
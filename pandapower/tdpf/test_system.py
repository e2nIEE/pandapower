import pandapower as pp
from pandapower.pypower.idx_brch import BR_R, F_BUS, BR_R_REF_OHM_PER_KM


def test_grid(load_scaling=1, sgen_scaling=1, with_gen=False, distributed_slack=False):
    s_base = 100
    net = pp.create_empty_network(sn_mva=s_base)
    std_type = "490-AL1/64-ST1A 110.0"
    # r = 0.1188
    # std_type = "490-AL1/64-ST1A 220.0"
    r = 0.059
    v_base = 132
    z_base = v_base ** 2 / s_base

    pp.create_buses(net, 5, v_base, geodata=((0,1), (-1,0.5), (0,0), (1,0.5), (0,0.5)))

    pp.create_line(net, 0, 1, 0.84e-2 * z_base / r, std_type, name="1-2")
    pp.create_line(net, 0, 3, 0.84e-2 * z_base / r, std_type, name="1-4")
    pp.create_line(net, 1, 2, 0.67e-2 * z_base / r, std_type, name="2-3")
    pp.create_line(net, 1, 4, 0.42e-2 * z_base / r, std_type, name="2-5")
    pp.create_line(net, 2, 3, 0.67e-2 * z_base / r, std_type, name="3-4")
    pp.create_line(net, 3, 4, 0.42e-2 * z_base / r, std_type, name="4-5")
    net.line.c_nf_per_km = 0

    net.line["temperature_degree_celsius"] = 20
    net.line["ambient_temperature_degree_celsius"] = 35
    net.line["alpha"] = 0.004
    net.line["outer_diameter_m"] = 30.6e-3
    net.line["mc_joule_per_m_k"] = 1490
    net.line["wind_speed_m_per_s"] = 0.6
    net.line["wind_angle_degree"] = 45
    net.line["solar_radiation_w_per_sq_m"] = 900
    net.line["gamma"] = 0.5
    net.line["epsilon"] = 0.5
    net.line["tdpf"] = True

    pp.create_ext_grid(net, 3, 1.05, name="G1")
    pp.create_sgen(net, 0, 200, scaling=sgen_scaling, name="R1")
    pp.create_sgen(net, 1, 250, scaling=sgen_scaling, name="R2")
    if with_gen:
        idx=pp.create_gen(net, 2, 600, 1., scaling=sgen_scaling, name="G3")
        pp.create_gen(net, 4, 300, 1., scaling=sgen_scaling, name="G5")
    else:
        idx=pp.create_sgen(net, 2, 600, scaling=sgen_scaling, name="G3")
        pp.create_sgen(net, 4, 300, scaling=sgen_scaling, name="G5")

    if distributed_slack:
        net["gen" if with_gen else "sgen"].at[idx, 'slack_weight'] = 1
        pp.set_user_pf_options(net, distributed_slack=True)

    pp.create_load(net, 1, 600, 240, scaling=load_scaling)
    pp.create_load(net, 3, 1000, 400, scaling=load_scaling)
    pp.create_load(net, 4, 400, 160, scaling=load_scaling)

    return net


if __name__ == '__main__':
    # from pandapower.tdpf.test_system import *
    # first text steady-state results
    net = test_grid(load_scaling=0.25, sgen_scaling=0.5)
    pp.runpp(net, tdpf=True, max_iteration=100)

    net2 = test_grid(load_scaling=0.25, sgen_scaling=0.5)
    net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
    pp.runpp(net2, consider_line_temperature=True, tolerance_mva=1e-6)

    from pandapower.test.toolbox import assert_res_equal
    net.res_line.drop(["temperature_degree_celsius"], axis=1, inplace=True)
    assert_res_equal(net, net2)

    # now test transient results
    pp.runpp(net, tdpf=True, tdpf_delay_s=5*60, max_iteration=100, tolerance_mva=1e-6)

    net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
    pp.runpp(net2, consider_line_temperature=True, tolerance_mva=1e-6)

    net.res_line.drop(["temperature_degree_celsius"], axis=1, inplace=True)
    assert_res_equal(net, net2)

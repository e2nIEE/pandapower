# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Networks that are used for tutorials AND tests are located here so that both tutorials and tests have access to them.
"""

import numpy as np

from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_gen,
    create_line,
    create_lines,
    create_load,
    create_sgen,
    create_transformer,
    create_transformer_from_parameters,
)
from pandapower.network import pandapowerNet
from pandapower.pf.create_jacobian_tdpf import ALPHA_TDPF
from pandapower.run import set_user_pf_options


def facts_case_study_grid():
    net = pandapowerNet(name="facts_case_study_grid")

    b1, b2, b3, b4, b5, b6, b7, b8 = create_buses(
        net, 8, name=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"], vn_kv=[18, 16.5, 230, 230, 230, 230, 230, 230]
    )

    create_ext_grid(net, bus=b1, vm_pu=1, va_degree=0)

    create_lines(
        net,
        name=["L1", "L2"],
        from_buses=[b3, b3],
        to_buses=[b4, b4],
        length_km=30,
        line_params={
            "r_ohm_per_km": 0.049,
            "x_ohm_per_km": 0.136,
            "g_us_per_km": 0,
            "c_nf_per_km": 142,
            "max_i_ka": 1.5,
        },
    )

    create_lines(
        net,
        name=["L3", "L4", "L5", "L6", "L7", "L8"],
        from_buses=[b4, b4, b5, b6, b5, b7],
        to_buses=[b5, b6, b7, b8, b6, b8],
        length_km=[100, 100, 220, 140, 180, 180],
        line_params={
            "r_ohm_per_km": 0.081,
            "x_ohm_per_km": 0.312,
            "g_us_per_km": 0,
            "c_nf_per_km": 11,
            "max_i_ka": 1.5,
        },
    )

    # create_line_from_parameters(net,name="L9",from_bus=3,to_bus=4,length_km=100, r_ohm_per_km=0.312,
    # x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11)

    create_transformer_from_parameters(
        net,
        name="trafo1",
        hv_bus=b8,
        lv_bus=b1,
        sn_mva=192,
        vn_hv_kv=230,
        vn_lv_kv=18,
        vkr_percent=0,
        vector_group="Yy0",
        pfe_kw=0,
        vk_percent=12,
        i0_percent=0,
    )
    create_transformer_from_parameters(
        net,
        name="trafo2",
        hv_bus=b3,
        lv_bus=b2,
        sn_mva=500,
        vn_hv_kv=230,
        vn_lv_kv=16.5,
        vkr_percent=0,
        vector_group="Yy0",
        pfe_kw=0,
        vk_percent=16,
        i0_percent=0,
    )

    create_gen(net, bus=b2, p_mw=500, vm_pu=1)
    # create_sgen(net,bus = 2, p_mw=500,name="WT")
    #
    create_load(net, bus=b4, p_mw=130, q_mvar=50)
    create_load(net, bus=b5, p_mw=120, q_mvar=50)
    create_load(net, bus=b6, p_mw=80, q_mvar=25)
    create_load(net, bus=b7, p_mw=50, q_mvar=25)

    return net


def simplest_test_grid(generator_type, step_up_trafo=False):
    net = pandapowerNet(name="simplest_test_grid", sn_mva=6)
    if step_up_trafo:
        b0, b1 = create_buses(net, 2, [20, 0.4])
        create_transformer(net, b0, b1, "0.25 MVA 20/0.4 kV", parallel=10)
    else:
        (b0,) = create_buses(net, 1, 20)
        b1 = b0

    create_ext_grid(net, b0, s_sc_max_mva=1e-12, rx_max=0)
    if generator_type == "async_doubly_fed":
        create_sgen(
            net, b1, 0, 0, 2.5, current_source=False, generator_type=generator_type, max_ik_ka=0.388, kappa=1.7, rx=0.1
        )
    elif generator_type == "current_source":
        create_sgen(net, b1, 0, 0, 2.5, generator_type=generator_type, current_source=True, k=1.3, rx=0.1)
    elif generator_type == "async":
        create_sgen(net, b1, 0, 0, 2.5, generator_type=generator_type, current_source=False, rx=0.1, lrc_pu=5)
    else:
        raise NotImplementedError(
            f"unknown sgen generator type {generator_type}, can be one of "
            f"'full_size_converter', 'async', 'async_doubly_fed'"
        )
    return net


def simple_test_grid(load_scaling=1.0, sgen_scaling=1.0, with_gen=False, distributed_slack=False):
    s_base = 100

    net = pandapowerNet(name="simple_test_grid", sn_mva=s_base)
    std_type = "490-AL1/64-ST1A 110.0"
    r = 0.059
    v_base = 132
    z_base = v_base**2 / s_base

    create_buses(net, 5, v_base, geodata=[(0, 1), (-1, 0.5), (0, 0), (1, 0.5), (0, 0.5)])

    create_line(net, 0, 1, 0.84e-2 * z_base / r, std_type, name="1-2")
    create_line(net, 0, 3, 0.84e-2 * z_base / r, std_type, name="1-4")
    create_line(net, 1, 2, 0.67e-2 * z_base / r, std_type, name="2-3")
    create_line(net, 1, 4, 0.42e-2 * z_base / r, std_type, name="2-5")
    create_line(net, 2, 3, 0.67e-2 * z_base / r, std_type, name="3-4")
    create_line(net, 3, 4, 0.42e-2 * z_base / r, std_type, name="4-5")
    net.line.c_nf_per_km = 0

    net.line["temperature_degree_celsius"] = 20
    net.line["reference_temperature_degree_celsius"] = 20
    net.line["air_temperature_degree_celsius"] = 35
    net.line["alpha"] = ALPHA_TDPF
    net.line["conductor_outer_diameter_m"] = 30.6e-3
    net.line["mc_joule_per_m_k"] = 1490
    net.line["wind_speed_m_per_s"] = 0.6
    net.line["wind_angle_degree"] = 45
    net.line["solar_radiation_w_per_sq_m"] = 900
    net.line["solar_absorptivity"] = 0.5
    net.line["emissivity"] = 0.5
    net.line["tdpf"] = True

    create_ext_grid(net, 3, 1.05, name="G1")
    create_sgen(net, 0, 200, scaling=sgen_scaling, name="R1")
    create_sgen(net, 1, 250, scaling=sgen_scaling, name="R2")
    if with_gen:
        idx = create_gen(net, 2, 600, 1.0, scaling=sgen_scaling, name="G3")
        create_gen(net, 4, 300, 1.0, scaling=sgen_scaling, name="G5")
    else:
        idx = create_sgen(net, 2, 600, scaling=sgen_scaling, name="G3")
        create_sgen(net, 4, 300, scaling=sgen_scaling, name="G5")

    if distributed_slack:
        if with_gen:  # distributed slack is currently not supported for sgen.
            net["gen"]["slack_weight"] = 0.0
            net["gen"].at[idx, "slack_weight"] = 1
        set_user_pf_options(net, distributed_slack=True)
        net.sn_mva = 1000  # otherwise numerical issues

    create_load(net, 1, 600, 240, scaling=load_scaling)
    create_load(net, 3, 1000, 400, scaling=load_scaling)
    create_load(net, 4, 400, 160, scaling=load_scaling)

    return net


def vde_232():
    net = pandapowerNet(name="vde_232", sn_mva=12)
    # hv buses
    create_buses(net, 2, [110, 21], geodata=[(0, 0), (1, 0)])

    create_ext_grid(
        net,
        0,
        s_sc_max_mva=13.61213 * 110 * np.sqrt(3),
        rx_max=0.20328,
        x0x_max=3.47927,
        r0x0_max=3.03361 * 0.20328 / 3.47927,
    )
    create_transformer_from_parameters(
        net,
        0,
        1,
        150,
        115,
        21,
        0.5,
        16,
        pfe_kw=0,
        i0_percent=0,
        tap_step_percent=1,
        tap_max=12,
        tap_min=-12,
        tap_neutral=0,
        tap_side="hv",
        vector_group="YNd",
        vk0_percent=np.sqrt(np.square(0.95 * 15.99219) + np.square(0.5)),
        vkr0_percent=0.5,
        mag0_percent=10000,
        mag0_rx=0,
        si0_hv_partial=0.9,
        pt_percent=12,
        oltc=True,
        power_station_unit=True,
        xn_ohm=22,
        tap_changer_type="Ratio",
    )

    create_gen(
        net, 1, 150, 1, 150, vn_kv=21, xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, power_station_trafo=0, pg_percent=5
    )

    return net

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower import pandapowerNet
from pandapower.run import set_user_pf_options
from pandapower.create import (
    create_bus, create_ext_grid, create_line, create_load, create_switch, create_sgen, create_transformer,
    create_transformer_from_parameters, create_gen, create_buses
)
from pandapower.pf.create_jacobian_tdpf import ALPHA_TDPF


def panda_four_load_branch():
    """
    This function creates a simple six bus system with four radial low voltage nodes connected to \
    a medium voltage slack bus. At every low voltage node the same load is connected.

    OUTPUT:
         **net** - Returns the required four load system

    EXAMPLE:
        >>> from pandapower.networks.simple_pandapower_test_networks import panda_four_load_branch
        >>> net_four_load = panda_four_load_branch()
    """
    net = pandapowerNet(name='four_load_branch')

    busnr1 = create_bus(net, name="bus1", vn_kv=10., geodata=(0, 0))
    busnr2 = create_bus(net, name="bus2", vn_kv=.4, geodata=(0, -1))
    busnr3 = create_bus(net, name="bus3", vn_kv=.4, geodata=(0, -2))
    busnr4 = create_bus(net, name="bus4", vn_kv=.4, geodata=(0, -3))
    busnr5 = create_bus(net, name="bus5", vn_kv=.4, geodata=(0, -4))
    busnr6 = create_bus(net, name="bus6", vn_kv=.4, geodata=(0, -5))

    create_ext_grid(net, busnr1)

    create_transformer(net, busnr1, busnr2, std_type="0.25 MVA 10/0.4 kV")

    create_line(net, busnr2, busnr3, name="line1", length_km=0.05,
                std_type="NAYY 4x120 SE")
    create_line(net, busnr3, busnr4, name="line2", length_km=0.05,
                std_type="NAYY 4x120 SE")
    create_line(net, busnr4, busnr5, name="line3", length_km=0.05,
                std_type="NAYY 4x120 SE")
    create_line(net, busnr5, busnr6, name="line4", length_km=0.05,
                std_type="NAYY 4x120 SE")

    create_load(net, busnr3, 0.030, 0.010)
    create_load(net, busnr4, 0.030, 0.010)
    create_load(net, busnr5, 0.030, 0.010)
    create_load(net, busnr6, 0.030, 0.010)
    return net


def four_loads_with_branches_out():
    """
    This function creates a simple ten bus system with four radial low voltage nodes connected to \
    a medium voltage slack bus. At every of the four radial low voltage nodes another low voltage \
    node with a load is connected via cable.

    OUTPUT:
         **net** - Returns the required four load system with branches

    EXAMPLE:
        >>> from pandapower.networks.simple_pandapower_test_networks import four_loads_with_branches_out
        >>> net_four_load_with_branches = four_loads_with_branches_out()
    """
    net = pandapowerNet(name='four_loads_with_branches_out')

    busnr1 = create_bus(net, name="bus1ref", vn_kv=10., geodata=(0, 0))
    create_ext_grid(net, busnr1)
    busnr2 = create_bus(net, name="bus2", vn_kv=.4, geodata=(0, -1))
    create_transformer(net, busnr1, busnr2, std_type="0.25 MVA 10/0.4 kV")
    busnr3 = create_bus(net, name="bus3", vn_kv=.4, geodata=(0, -2))
    create_line(net, busnr2, busnr3, name="line1", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr4 = create_bus(net, name="bus4", vn_kv=.4, geodata=(0, -3))
    create_line(net, busnr3, busnr4, name="line2", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr5 = create_bus(net, name="bus5", vn_kv=.4, geodata=(0, -4))
    create_line(net, busnr4, busnr5, name="line3", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr6 = create_bus(net, name="bus6", vn_kv=.4, geodata=(0, -5))
    create_line(net, busnr5, busnr6, name="line4", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr7 = create_bus(net, name="bus7", vn_kv=.4, geodata=(1, -3))
    create_line(net, busnr3, busnr7, name="line5", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr8 = create_bus(net, name="bus8", vn_kv=.4, geodata=(1, -4))
    create_line(net, busnr4, busnr8, name="line6", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr9 = create_bus(net, name="bus9", vn_kv=.4, geodata=(1, -5))
    create_line(net, busnr5, busnr9, name="line7", length_km=0.05,
                std_type="NAYY 4x120 SE")
    busnr10 = create_bus(net, name="bus10", vn_kv=.4, geodata=(1, -6))
    create_line(net, busnr6, busnr10, name="line8", length_km=0.05,
                std_type="NAYY 4x120 SE")

    create_load(net, busnr7, p_mw=0.030, q_mvar=0.010)
    create_load(net, busnr8, p_mw=0.030, q_mvar=0.010)
    create_load(net, busnr9, p_mw=0.030, q_mvar=0.010)
    create_load(net, busnr10, p_mw=0.030, q_mvar=0.010)
    return net


def simple_four_bus_system():
    """
    This function creates a simple four bus system with two radial low voltage nodes connected to \
    a medium voltage slack bus. At both low voltage nodes the a load and a static generator is \
    connected.

    OUTPUT:
         **net** - Returns the required four bus system

    EXAMPLE:
        >>> from pandapower.networks.simple_pandapower_test_networks import simple_four_bus_system
        >>> net_simple_four_bus = simple_four_bus_system()
    """
    net = pandapowerNet(name='simple_four_bus_system')
    busnr1 = create_bus(net, name="bus1ref", vn_kv=10, geodata=(0, 0))
    create_ext_grid(net, busnr1)
    busnr2 = create_bus(net, name="bus2", vn_kv=.4, geodata=(0, -1))
    create_transformer(net, busnr1, busnr2, name="transformer", std_type="0.25 MVA 10/0.4 kV")
    busnr3 = create_bus(net, name="bus3", vn_kv=.4, geodata=(0, -2))
    create_line(net, busnr2, busnr3, name="line1", length_km=0.50000, std_type="NAYY 4x50 SE")
    busnr4 = create_bus(net, name="bus4", vn_kv=.4, geodata=(0, -3))
    create_line(net, busnr3, busnr4, name="line2", length_km=0.50000, std_type="NAYY 4x50 SE")
    create_load(net, busnr3, 0.030, 0.010, name="load1")
    create_load(net, busnr4, 0.030, 0.010, name="load2")
    create_sgen(net, busnr3, p_mw=0.020, q_mvar=0.005, name="pv1", sn_mva=0.03)
    create_sgen(net, busnr4, p_mw=0.015, q_mvar=0.002, name="pv2", sn_mva=0.02)
    return net


def simple_mv_open_ring_net():
    """
    This function creates a simple medium voltage open ring network with loads at every medium \
    voltage node.
    As an example this function is used in the topology and diagnostic docu.

    OUTPUT:
         **net** - Returns the required simple medium voltage open ring network

    EXAMPLE:
         >>> from pandapower.networks.simple_pandapower_test_networks import simple_mv_open_ring_net
         >>> net_simple_open_ring = simple_mv_open_ring_net()
    """

    net = pandapowerNet(name='simple_mv_open_ring_net')

    create_bus(net, name="110 kV bar", vn_kv=110, type='b', geodata=(0, 0))
    create_bus(net, name="20 kV bar", vn_kv=20, type='b', geodata=(0, -1))
    create_bus(net, name="bus 2", vn_kv=20, type='b', geodata=(-0.5, -2))
    create_bus(net, name="bus 3", vn_kv=20, type='b', geodata=(-0.5, -3))
    create_bus(net, name="bus 4", vn_kv=20, type='b', geodata=(-0.5, -4))
    create_bus(net, name="bus 5", vn_kv=20, type='b', geodata=(0.5, -4))
    create_bus(net, name="bus 6", vn_kv=20, type='b', geodata=(0.5, -3))

    create_ext_grid(net, 0, vm_pu=1)

    create_line(net, name="line 0", from_bus=1, to_bus=2, length_km=1,
                std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    create_line(net, name="line 1", from_bus=2, to_bus=3, length_km=1,
                std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    create_line(net, name="line 2", from_bus=3, to_bus=4, length_km=1,
                std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    create_line(net, name="line 3", from_bus=4, to_bus=5, length_km=1,
                std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    create_line(net, name="line 4", from_bus=5, to_bus=6, length_km=1,
                std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    create_line(net, name="line 5", from_bus=6, to_bus=1, length_km=1,
                std_type="NA2XS2Y 1x185 RM/25 12/20 kV")

    create_transformer(net, hv_bus=0, lv_bus=1, std_type="25 MVA 110/20 kV")

    create_load(net, 2, p_mw=1, q_mvar=0.200, name="load 0")
    create_load(net, 3, p_mw=1, q_mvar=0.200, name="load 1")
    create_load(net, 4, p_mw=1, q_mvar=0.200, name="load 2")
    create_load(net, 5, p_mw=1, q_mvar=0.200, name="load 3")
    create_load(net, 6, p_mw=1, q_mvar=0.200, name="load 4")

    create_switch(net, bus=1, element=0, et='l')
    create_switch(net, bus=2, element=0, et='l')
    create_switch(net, bus=2, element=1, et='l')
    create_switch(net, bus=3, element=1, et='l')
    create_switch(net, bus=3, element=2, et='l')
    create_switch(net, bus=4, element=2, et='l')
    create_switch(net, bus=4, element=3, et='l', closed=0)
    create_switch(net, bus=5, element=3, et='l')
    create_switch(net, bus=5, element=4, et='l')
    create_switch(net, bus=6, element=4, et='l')
    create_switch(net, bus=6, element=5, et='l')
    create_switch(net, bus=1, element=5, et='l')
    return net


def vde_232():
    net = pandapowerNet(name="vde_232", sn_mva=12)
    # hv buses
    create_bus(net, 110, geodata=(0, 0))
    create_bus(net, 21, geodata=(1, 0))

    create_ext_grid(net, 0, s_sc_max_mva=13.61213 * 110 * np.sqrt(3), rx_max=0.20328, x0x_max=3.47927,
                    r0x0_max=3.03361 * 0.20328 / 3.47927)
    create_transformer_from_parameters(net, 0, 1, 150, 115, 21, 0.5, 16, pfe_kw=0, i0_percent=0, tap_step_percent=1,
                                       tap_max=12, tap_min=-12, tap_neutral=0, tap_side='hv', vector_group="YNd",
                                       vk0_percent=np.sqrt(np.square(0.95 * 15.99219) + np.square(0.5)),
                                       vkr0_percent=0.5, mag0_percent=10000, mag0_rx=0, si0_hv_partial=0.9,
                                       pt_percent=12,
                                       oltc=True, power_station_unit=True, xn_ohm=22, tap_changer_type="Ratio")

    create_gen(net, 1, 150, 1, 150, vn_kv=21, xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, power_station_trafo=0,
               pg_percent=5)

    return net


def simplest_test_grid(generator_type, step_up_trafo=False):
    net = pandapowerNet(name="simplest_test_grid", sn_mva=6)
    if step_up_trafo:
        b0 = create_bus(net, 20)
        b1 = create_bus(net, 0.4)
        create_transformer(net, b0, b1, "0.25 MVA 20/0.4 kV", parallel=10)
    else:
        b0 = b1 = create_bus(net, 20)

    create_ext_grid(net, b0, s_sc_max_mva=1e-12, rx_max=0)
    if generator_type == "async_doubly_fed":
        create_sgen(net, b1, 0, 0, 2.5, current_source=False,
                    generator_type=generator_type, max_ik_ka=0.388, kappa=1.7, rx=0.1)
    elif generator_type == "current_source":
        create_sgen(net, b1, 0, 0, 2.5, generator_type=generator_type, current_source=True, k=1.3, rx=0.1)
    elif generator_type == "async":
        create_sgen(net, b1, 0, 0, 2.5, generator_type=generator_type, current_source=False, rx=0.1, lrc_pu=5)
    else:
        raise NotImplementedError(f"unknown sgen generator type {generator_type}, can be one of "
                                  f"'full_size_converter', 'async', 'async_doubly_fed'")
    return net


def simple_test_grid(load_scaling=1., sgen_scaling=1., with_gen=False, distributed_slack=False):
    s_base = 100

    net = pandapowerNet(name="simple_test_grid", sn_mva=s_base)
    std_type = "490-AL1/64-ST1A 110.0"
    r = 0.059
    v_base = 132
    z_base = v_base ** 2 / s_base

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
        idx = create_gen(net, 2, 600, 1., scaling=sgen_scaling, name="G3")
        create_gen(net, 4, 300, 1., scaling=sgen_scaling, name="G5")
    else:
        idx = create_sgen(net, 2, 600, scaling=sgen_scaling, name="G3")
        create_sgen(net, 4, 300, scaling=sgen_scaling, name="G5")

    if distributed_slack:
        if with_gen:  # distributed slack is currently not supported for sgen.
            net["gen"]["slack_weight"] = 0.0
            net["gen"].at[idx, 'slack_weight'] = 1
        set_user_pf_options(net, distributed_slack=True)
        net.sn_mva = 1000  # otherwise numerical issues

    create_load(net, 1, 600, 240, scaling=load_scaling)
    create_load(net, 3, 1000, 400, scaling=load_scaling)
    create_load(net, 4, 400, 160, scaling=load_scaling)

    return net

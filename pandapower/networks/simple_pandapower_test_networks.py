# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.create import create_bus, create_ext_grid, create_line, create_load, create_switch, create_sgen, \
    create_transformer, create_empty_network


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
    net = create_empty_network()

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
    net = create_empty_network()

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
    net = create_empty_network()
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

    net = create_empty_network()

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

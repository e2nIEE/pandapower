# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_lines,
    create_loads,
    create_sgen,
    create_switches,
    create_transformer,
)
from pandapower.network import pandapowerNet


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

    geo = [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5)]
    names = [f"bus{i}" for i in range(2, 7)]
    (busnr1,) = create_buses(net, 1, name="bus1", vn_kv=10.0, geodata=(0, 0))
    buses = create_buses(net, 5, name=names, vn_kv=0.4, geodata=geo)

    create_ext_grid(net, busnr1)

    create_transformer(net, busnr1, buses[0], std_type="0.25 MVA 10/0.4 kV")

    names = [f"line{i}" for i in range(1, 5)]
    create_lines(net, buses[:4], buses[1:5], name=names, length_km=0.05, line_params="NAYY 4x120 SE")

    create_loads(net, buses[1:5], 0.030, 0.010)
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

    geo = [(0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (1, -3), (1, -4), (1, -5), (1, -6)]
    names = [f"bus{i}" for i in range(2, 11)]
    (busnr1,) = create_buses(net, 1, name="bus1ref", vn_kv=10.0, geodata=(0, 0))
    buses = create_buses(net, 9, name=names, vn_kv=0.4, geodata=geo)

    create_ext_grid(net, busnr1)
    create_transformer(net, busnr1, buses[0], std_type="0.25 MVA 10/0.4 kV")

    from_buses = [buses[i] for i in (0, 1, 2, 3, 1, 2, 3, 4)]
    to_buses = buses[1:]
    names = [f"line{i}" for i in range(1, 9)]
    create_lines(net, from_buses, to_buses, name=names, length_km=0.05, line_params="NAYY 4x120 SE")

    create_loads(net, buses[-4:], p_mw=0.030, q_mvar=0.010)
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
    names = ["bus2", "bus3", "bus4"]
    geo = [(0, -1), (0, -2), (0, -3)]
    (busnr1,) = create_buses(net, 1, name="bus1ref", vn_kv=10, geodata=(0, 0))
    busnr2, busnr3, busnr4 = create_buses(net, 3, name=names, vn_kv=0.4, geodata=geo)

    create_ext_grid(net, busnr1)
    create_transformer(net, busnr1, busnr2, name="transformer", std_type="0.25 MVA 10/0.4 kV")
    create_lines(
        net, [busnr2, busnr3], [busnr3, busnr4], name=["line1", "line2"], length_km=0.50000, line_params="NAYY 4x50 SE"
    )
    create_loads(net, [busnr3, busnr4], 0.030, 0.010, name=["load1", "load2"])
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

    names = ["20 kV bar", "bus 2", "bus 3", "bus 4", "bus 5", "bus 6"]
    geo = [(0, -1), (-0.5, -2), (-0.5, -3), (-0.5, -4), (0.5, -4), (0.5, -3)]
    create_buses(net, 1, name="110 kV bar", vn_kv=110, type="b", geodata=(0, 0))
    create_buses(net, 6, name=names, vn_kv=20, type="b", geodata=geo)

    create_ext_grid(net, 0, vm_pu=1)

    names = [f"line {i}" for i in range(6)]
    from_buses = list(range(1, 7))
    to_buses = list(range(2, 7)) + [1]
    create_lines(net, from_buses, to_buses, 1, "NA2XS2Y 1x185 RM/25 12/20 kV", names)

    create_transformer(net, hv_bus=0, lv_bus=1, std_type="25 MVA 110/20 kV")

    buses = [2, 3, 4, 5, 6]
    names = [f"load {i}" for i in range(5)]
    create_loads(net, buses, p_mw=1, q_mvar=0.200, name=names)

    buses = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1]
    elements = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    *_, cs, _, _, _, _, _ = create_switches(net, buses, elements, et="l")
    net.switch.at[cs, "closed"] = False
    return net

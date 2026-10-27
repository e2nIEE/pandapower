# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = [
    "create_cigre_network_hv",
    "create_cigre_network_lv",
    "create_cigre_network_mv",
]

import logging

from numpy import nan

from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_gen,
    create_lines,
    create_load,
    create_load_from_cosphi,
    create_sgen,
    create_shunt,
    create_storage,
    create_switch,
    create_transformer_from_parameters,
)
from pandapower.network import pandapowerNet
from pandapower.std_types import create_std_type

logger = logging.getLogger(__name__)


def create_cigre_network_hv(length_km_6a_6b=0.1):
    """
    Create the CIGRE HV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OPTIONAL:
        **length_km_6a_6b** (float, 0.1) - Length of the line segment 9 between buses 6a and 6b
            which is intended to be optional with user-definable geometrical configuration, length
            and installation type.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net_cigre_hv = pandapowerNet(name="CIGRE HV Grid")

    # Linedata
    # Line220kV
    line_data = {"c_nf_per_km": 9.08, "r_ohm_per_km": 0.0653, "x_ohm_per_km": 0.398, "max_i_ka": 1.14, "type": "ol"}

    create_std_type(net_cigre_hv, line_data, "Line220kV", element="line")

    # Line380kV
    line_data = {"c_nf_per_km": 11.5, "r_ohm_per_km": 0.0328, "x_ohm_per_km": 0.312, "max_i_ka": 1.32, "type": "ol"}

    create_std_type(net_cigre_hv, line_data, "Line380kV", element="line")

    # Busses
    bus1, bus2, bus3, bus4, bus5, bus6a, bus6b = create_buses(
        net_cigre_hv,
        7,
        name=["Bus 1", "Bus 2", "Bus 3", "Bus 4", "Bus 5", "Bus 6a", "Bus 6b"],
        vn_kv=220,
        type="b",
        zone="CIGRE_HV",
    )

    bus7, bus8 = create_buses(net_cigre_hv, 2, name=["Bus 7", "Bus 8"], vn_kv=380, type="b", zone="CIGRE_HV")

    bus9, bus10, bus11, bus12 = create_buses(
        net_cigre_hv, 4, name=["Bus 9", "Bus 10", "Bus 11", "Bus 12"], vn_kv=22, type="b", zone="CIGRE_HV"
    )

    # Lines
    names = ["Line 1-2", "Line 1-6a", "Line 2-5", "Line 3-4", "Line 3-4_2", "Line 4-5", "Line 4-6a"]
    length_km = [100, 300, 300, 100, 100, 300, 300]
    from_buses = [bus1, bus1, bus2, bus3, bus3, bus4, bus4]
    to_buses = [bus2, bus6a, bus5, bus4, bus4, bus5, bus6a]
    create_lines(net_cigre_hv, from_buses, to_buses, length_km, "Line220kV", names)

    create_lines(net_cigre_hv, bus7, bus8, length_km=600, line_params="Line380kV", name="Line 7-8")
    create_lines(net_cigre_hv, bus6a, bus6b, length_km=length_km_6a_6b, line_params="Line220kV", name="Line 6a-6b")

    # Trafos
    create_transformer_from_parameters(
        net_cigre_hv,
        bus7,
        bus1,
        sn_mva=1000,
        vn_hv_kv=380,
        vn_lv_kv=220,
        vkr_percent=0.0,
        vk_percent=13.0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0.0,
        name="Trafo 1-7",
    )
    create_transformer_from_parameters(
        net_cigre_hv,
        bus8,
        bus3,
        sn_mva=1000,
        vn_hv_kv=380,
        vn_lv_kv=220,
        vkr_percent=0.0,
        vk_percent=13.0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=0.0,
        name="Trafo 3-8",
    )

    create_transformer_from_parameters(
        net_cigre_hv,
        bus1,
        bus9,
        sn_mva=1000,
        vn_hv_kv=220,
        vn_lv_kv=22,
        vkr_percent=0.0,
        vk_percent=13.0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=330.0,
        name="Trafo 9-1",
    )
    create_transformer_from_parameters(
        net_cigre_hv,
        bus2,
        bus10,
        sn_mva=1000,
        vn_hv_kv=220,
        vn_lv_kv=22,
        vkr_percent=0.0,
        vk_percent=13.0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=330.0,
        name="Trafo 10-2",
    )
    create_transformer_from_parameters(
        net_cigre_hv,
        bus3,
        bus11,
        sn_mva=1000,
        vn_hv_kv=220,
        vn_lv_kv=22,
        vkr_percent=0.0,
        vk_percent=13.0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=330.0,
        name="Trafo 11-3",
    )
    create_transformer_from_parameters(
        net_cigre_hv,
        bus6b,
        bus12,
        sn_mva=500,
        vn_hv_kv=220,
        vn_lv_kv=22,
        vkr_percent=0.0,
        vk_percent=13.0,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=330.0,
        name="Trafo 12-6b",
    )

    # Loads
    create_load(net_cigre_hv, bus2, p_mw=285, q_mvar=200, name="Load 2")
    create_load(net_cigre_hv, bus3, p_mw=325, q_mvar=244, name="Load 3")
    create_load(net_cigre_hv, bus4, p_mw=326, q_mvar=244, name="Load 4")
    create_load(net_cigre_hv, bus5, p_mw=103, q_mvar=62, name="Load 5")
    create_load(net_cigre_hv, bus6a, p_mw=435, q_mvar=296, name="Load 6a")

    # External grid
    create_ext_grid(net_cigre_hv, bus9, vm_pu=1.03, va_degree=0, name="Generator 9")

    # Generators
    create_gen(net_cigre_hv, bus10, vm_pu=1.03, p_mw=500, name="Generator 10")
    create_gen(net_cigre_hv, bus11, vm_pu=1.03, p_mw=200, name="Generator 11")
    create_gen(net_cigre_hv, bus12, vm_pu=1.03, p_mw=300, name="Generator 12")

    # Shunts
    create_shunt(net_cigre_hv, bus4, p_mw=0.0, q_mvar=-160, name="Shunt 4")
    create_shunt(net_cigre_hv, bus5, p_mw=0.0, q_mvar=-80, name="Shunt 5")
    create_shunt(net_cigre_hv, bus6a, p_mw=0.0, q_mvar=-180, name="Shunt 6a")

    # Bus geo data
    bus_geodata = [
        f'{{"type":"Point", "coordinates":[{xy[0]}, {xy[1]}]}}'
        for xy in zip(
            [4, 8, 20, 16, 12, 8, 12, 4, 20, 0, 8, 24, 16],
            [8.0, 8.0, 8.0, 8.0, 8.0, 6.0, 4.5, 1.0, 1.0, 8.0, 12.0, 8.0, 4.5],
        )
    ]
    # Match bus.index
    net_cigre_hv.bus["geo"] = bus_geodata

    return net_cigre_hv


def create_cigre_network_mv(with_der=False):
    """
    Create the CIGRE MV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OPTIONAL:
        **with_der** (boolean or str, False) - Range of DER consideration, which should be in
            (False, "pv_wind", "all"). The DER types, dimensions and locations are taken from CIGRE
            CaseStudy: "DER in Medium Voltage Systems"

    OUTPUT:
        **net** - The pandapower format network.
    """
    if with_der is True:
        raise ValueError("'with_der=True' is deprecated. Please use 'with_der=pv_wind'")
    if with_der not in [False, "pv_wind", "all"]:
        raise ValueError("'with_der' is unknown. It should be in [False, 'pv_wind', 'all'].")

    net_cigre_mv = pandapowerNet(name="CIGRE MV Grid")

    # Linedata
    line_data = {"c_nf_per_km": 151.1749, "r_ohm_per_km": 0.501, "x_ohm_per_km": 0.716, "max_i_ka": 0.145, "type": "cs"}
    create_std_type(net_cigre_mv, line_data, name="CABLE_CIGRE_MV", element="line")

    line_data = {"c_nf_per_km": 10.09679, "r_ohm_per_km": 0.510, "x_ohm_per_km": 0.366, "max_i_ka": 0.195, "type": "ol"}
    create_std_type(net_cigre_mv, line_data, name="OHL_CIGRE_MV", element="line")

    # Busses
    bus_geodata = list(
        zip(
            [7.0, 4.0, 4.0, 4.0, 2.5, 1.0, 1.0, 8.0, 8.0, 6.0, 4.0, 4.0, 10.0, 10.0, 10.0],
            [16.0, 15.0, 13.0, 11.0, 9.0, 7.0, 3.0, 3.0, 5.0, 5.0, 5.0, 7.0, 15.0, 11.0, 5.0],
        )
    )
    (bus0,) = create_buses(net_cigre_mv, 1, name="Bus 0", vn_kv=110, type="b", zone="CIGRE_MV", geodata=bus_geodata[0])
    buses = create_buses(
        net_cigre_mv,
        14,
        name=[f"Bus {i}" for i in range(1, 15)],
        vn_kv=20,
        type="b",
        zone="CIGRE_MV",
        geodata=bus_geodata[1:],
    )

    # Lines
    from_buses = [buses[i] for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 2]]
    to_buses = [buses[i] for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 7]]
    length_km = [2.82, 4.42, 0.61, 0.56, 1.54, 1.67, 0.32, 0.77, 0.33, 1.3]
    names = [
        "Line 1-2",
        "Line 2-3",
        "Line 3-4",
        "Line 4-5",
        "Line 5-6",
        "Line 7-8",
        "Line 8-9",
        "Line 9-10",
        "Line 10-11",
        "Line 3-8",
    ]
    create_lines(net_cigre_mv, from_buses, to_buses, length_km, "CABLE_CIGRE_MV", names)

    create_lines(
        net_cigre_mv,
        [buses[11], buses[12]],
        [buses[12], buses[13]],
        length_km=[4.89, 2.99],
        line_params="OHL_CIGRE_MV",
        name=["Line 12-13", "Line 13-14"],
    )

    line6_7, line4_11 = create_lines(
        net_cigre_mv,
        [buses[5], buses[10]],
        [buses[6], buses[3]],
        length_km=[0.24, 0.49],
        line_params="CABLE_CIGRE_MV",
        name=["Line 6-7", "Line 11-4"],
    )

    (line8_14,) = create_lines(
        net_cigre_mv, buses[13], buses[7], length_km=2.0, line_params="OHL_CIGRE_MV", name="Line 14-8"
    )

    # Ext-Grid
    create_ext_grid(
        net_cigre_mv, bus0, vm_pu=1.03, va_degree=0.0, s_sc_max_mva=5000, s_sc_min_mva=5000, rx_max=0.1, rx_min=0.1
    )

    # Trafos
    trafo0 = create_transformer_from_parameters(
        net_cigre_mv,
        bus0,
        buses[0],
        sn_mva=25,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.16,
        vk_percent=12.00107,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=30.0,
        name="Trafo 0-1",
    )
    trafo1 = create_transformer_from_parameters(
        net_cigre_mv,
        bus0,
        buses[11],
        sn_mva=25,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.16,
        vk_percent=12.00107,
        pfe_kw=0,
        i0_percent=0,
        shift_degree=30.0,
        name="Trafo 0-12",
    )

    # Switches
    # S2
    create_switch(net_cigre_mv, buses[5], line6_7, et="l", closed=True, type="LBS")
    create_switch(net_cigre_mv, buses[6], line6_7, et="l", closed=False, type="LBS", name="S2")
    # S3
    create_switch(net_cigre_mv, buses[3], line4_11, et="l", closed=False, type="LBS", name="S3")
    create_switch(net_cigre_mv, buses[10], line4_11, et="l", closed=True, type="LBS")
    # S1
    create_switch(net_cigre_mv, buses[7], line8_14, et="l", closed=False, type="LBS", name="S1")
    create_switch(net_cigre_mv, buses[13], line8_14, et="l", closed=True, type="LBS")
    # trafos
    create_switch(net_cigre_mv, bus0, trafo0, et="t", closed=True, type="CB")
    create_switch(net_cigre_mv, bus0, trafo1, et="t", closed=True, type="CB")

    # Loads
    # Residential
    create_load_from_cosphi(net_cigre_mv, buses[0], 15.3, 0.98, "underexcited", name="Load R1")
    create_load_from_cosphi(net_cigre_mv, buses[2], 0.285, 0.97, "underexcited", name="Load R3")
    create_load_from_cosphi(net_cigre_mv, buses[3], 0.445, 0.97, "underexcited", name="Load R4")
    create_load_from_cosphi(net_cigre_mv, buses[4], 0.750, 0.97, "underexcited", name="Load R5")
    create_load_from_cosphi(net_cigre_mv, buses[5], 0.565, 0.97, "underexcited", name="Load R6")
    create_load_from_cosphi(net_cigre_mv, buses[7], 0.605, 0.97, "underexcited", name="Load R8")
    create_load_from_cosphi(net_cigre_mv, buses[9], 0.490, 0.97, "underexcited", name="Load R10")
    create_load_from_cosphi(net_cigre_mv, buses[10], 0.340, 0.97, "underexcited", name="Load R11")
    create_load_from_cosphi(net_cigre_mv, buses[11], 15.3, 0.98, "underexcited", name="Load R12")
    create_load_from_cosphi(net_cigre_mv, buses[13], 0.215, 0.97, "underexcited", name="Load R14")

    # Commercial / Industrial
    create_load_from_cosphi(net_cigre_mv, buses[0], 5.1, 0.95, "underexcited", name="Load CI1")
    create_load_from_cosphi(net_cigre_mv, buses[2], 0.265, 0.85, "underexcited", name="Load CI3")
    create_load_from_cosphi(net_cigre_mv, buses[6], 0.090, 0.85, "underexcited", name="Load CI7")
    create_load_from_cosphi(net_cigre_mv, buses[8], 0.675, 0.85, "underexcited", name="Load CI9")
    create_load_from_cosphi(net_cigre_mv, buses[9], 0.080, 0.85, "underexcited", name="Load CI10")
    create_load_from_cosphi(net_cigre_mv, buses[11], 5.28, 0.95, "underexcited", name="Load CI12")
    create_load_from_cosphi(net_cigre_mv, buses[12], 0.04, 0.85, "underexcited", name="Load CI13")
    create_load_from_cosphi(net_cigre_mv, buses[13], 0.390, 0.85, "underexcited", name="Load CI14")

    # Optional distributed energy recources
    if with_der in ["pv_wind", "all"]:
        create_sgen(net_cigre_mv, buses[2], 0.02, q_mvar=0, sn_mva=0.02, name="PV 3", type="PV")
        create_sgen(net_cigre_mv, buses[3], 0.020, q_mvar=0, sn_mva=0.02, name="PV 4", type="PV")
        create_sgen(net_cigre_mv, buses[4], 0.030, q_mvar=0, sn_mva=0.03, name="PV 5", type="PV")
        create_sgen(net_cigre_mv, buses[5], 0.030, q_mvar=0, sn_mva=0.03, name="PV 6", type="PV")
        create_sgen(net_cigre_mv, buses[7], 0.030, q_mvar=0, sn_mva=0.03, name="PV 8", type="PV")
        create_sgen(net_cigre_mv, buses[8], 0.030, q_mvar=0, sn_mva=0.03, name="PV 9", type="PV")
        create_sgen(net_cigre_mv, buses[9], 0.040, q_mvar=0, sn_mva=0.04, name="PV 10", type="PV")
        create_sgen(net_cigre_mv, buses[10], 0.010, q_mvar=0, sn_mva=0.01, name="PV 11", type="PV")
        create_sgen(net_cigre_mv, buses[6], 1.5, q_mvar=0, sn_mva=1.5, name="WKA 7", type="WP")
        if with_der == "all":
            create_storage(
                net_cigre_mv,
                bus=buses[4],
                p_mw=0.6,
                max_e_mwh=nan,
                sn_mva=0.2,
                name="Battery 1",
                type="Battery",
                max_p_mw=0.6,
                min_p_mw=-0.6,
            )
            create_sgen(
                net_cigre_mv,
                bus=buses[4],
                p_mw=0.033,
                sn_mva=0.033,
                name="Residential fuel cell 1",
                type="Residential fuel cell",
            )
            create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.310, sn_mva=0.31, name="CHP diesel 1", type="CHP diesel")
            create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.212, sn_mva=0.212, name="Fuel cell 1", type="Fuel cell")
            create_storage(
                net_cigre_mv,
                bus=buses[9],
                p_mw=0.200,
                max_e_mwh=nan,
                sn_mva=0.2,
                name="Battery 2",
                type="Battery",
                max_p_mw=0.2,
                min_p_mw=-0.2,
            )
            create_sgen(
                net_cigre_mv,
                bus=buses[9],
                p_mw=0.014,
                sn_mva=0.014,
                name="Residential fuel cell 2",
                type="Residential fuel cell",
            )

    return net_cigre_mv


def create_cigre_network_lv():
    """
    Create the CIGRE LV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net_cigre_lv = pandapowerNet(name="CIGRE LV Grid")

    # Linedata
    # UG1
    line_data = {"c_nf_per_km": 0.0, "r_ohm_per_km": 0.162, "x_ohm_per_km": 0.0832, "max_i_ka": 1.0, "type": "cs"}
    create_std_type(net_cigre_lv, line_data, name="UG1", element="line")

    # UG2
    line_data = {"c_nf_per_km": 0.0, "r_ohm_per_km": 0.2647, "x_ohm_per_km": 0.0823, "max_i_ka": 1.0, "type": "cs"}
    create_std_type(net_cigre_lv, line_data, name="UG2", element="line")

    # UG3
    line_data = {"c_nf_per_km": 0.0, "r_ohm_per_km": 0.822, "x_ohm_per_km": 0.0847, "max_i_ka": 1.0, "type": "cs"}
    create_std_type(net_cigre_lv, line_data, name="UG3", element="line")

    # OH1
    line_data = {"c_nf_per_km": 0.0, "r_ohm_per_km": 0.4917, "x_ohm_per_km": 0.2847, "max_i_ka": 1.0, "type": "ol"}
    create_std_type(net_cigre_lv, line_data, name="OH1", element="line")

    # OH2
    line_data = {"c_nf_per_km": 0.0, "r_ohm_per_km": 1.3207, "x_ohm_per_km": 0.321, "max_i_ka": 1.0, "type": "ol"}
    create_std_type(net_cigre_lv, line_data, name="OH2", element="line")

    # OH3
    line_data = {"c_nf_per_km": 0.0, "r_ohm_per_km": 2.0167, "x_ohm_per_km": 0.3343, "max_i_ka": 1.0, "type": "ol"}
    create_std_type(net_cigre_lv, line_data, name="OH3", element="line")

    # Busses
    bus0, busR0, busR1 = create_buses(
        net_cigre_lv, 3, name=["Bus 0", "Bus R0", "Bus R1"], vn_kv=[20.0, 20.0, 0.4], type="b", zone="CIGRE_LV"
    )

    name = [
        "Bus R2",
        "Bus R3",
        "Bus R4",
        "Bus R5",
        "Bus R6",
        "Bus R7",
        "Bus R8",
        "Bus R9",
        "Bus R10",
        "Bus R11",
        "Bus R12",
        "Bus R13",
        "Bus R14",
        "Bus R15",
        "Bus R16",
        "Bus R17",
        "Bus R18",
    ]
    (
        busR2,
        busR3,
        busR4,
        busR5,
        busR6,
        busR7,
        busR8,
        busR9,
        busR10,
        busR11,
        busR12,
        busR13,
        busR14,
        busR15,
        busR16,
        busR17,
        busR18,
    ) = create_buses(net_cigre_lv, 17, name=name, vn_kv=0.4, type="m", zone="CIGRE_LV")

    busI0, busI1, busI2 = create_buses(
        net_cigre_lv,
        3,
        name=["Bus I0", "Bus I1", "Bus I2"],
        vn_kv=[20.0, 0.4, 0.4],
        type=["b", "b", "m"],
        zone="CIGRE_LV",
    )

    busC0, busC1 = create_buses(
        net_cigre_lv, 2, name=["Bus C0", "Bus C1"], vn_kv=[20.0, 0.4], type="b", zone="CIGRE_LV"
    )

    names = [
        "Bus C2",
        "Bus C3",
        "Bus C4",
        "Bus C5",
        "Bus C6",
        "Bus C7",
        "Bus C8",
        "Bus C9",
        "Bus C10",
        "Bus C11",
        "Bus C12",
        "Bus C13",
        "Bus C14",
        "Bus C15",
        "Bus C16",
        "Bus C17",
        "Bus C18",
        "Bus C19",
        "Bus C20",
    ]
    (
        busC2,
        busC3,
        busC4,
        busC5,
        busC6,
        busC7,
        busC8,
        busC9,
        busC10,
        busC11,
        busC12,
        busC13,
        busC14,
        busC15,
        busC16,
        busC17,
        busC18,
        busC19,
        busC20,
    ) = create_buses(net_cigre_lv, 19, name=names, vn_kv=0.4, type="m", zone="CIGRE_LV")

    # Lines
    from_buses = [busR1, busR2, busR3, busR4, busR5, busR6, busR7, busR8, busR9]
    to_buses = [busR2, busR3, busR4, busR5, busR6, busR7, busR8, busR9, busR10]
    names = [
        "Line R1-R2",
        "Line R2-R3",
        "Line R3-R4",
        "Line R4-R5",
        "Line R5-R6",
        "Line R6-R7",
        "Line R7-R8",
        "Line R8-R9",
        "Line R9-R10",
    ]
    create_lines(net_cigre_lv, from_buses, to_buses, length_km=0.035, line_params="UG1", name=names)

    from_buses = [busR3, busR4, busR12, busR13, busR14, busR6, busR9, busR10]
    to_buses = [busR11, busR12, busR13, busR14, busR15, busR16, busR17, busR18]
    length_km = [0.030, 0.035, 0.035, 0.035, 0.030, 0.030, 0.030, 0.030]
    names = [
        "Line R3-R11",
        "Line R4-R12",
        "Line R12-R13",
        "Line R13-R14",
        "Line R14-R15",
        "Line R6-R16",
        "Line R9-R17",
        "Line R10-R18",
    ]
    create_lines(net_cigre_lv, from_buses, to_buses, length_km, "UG3", names)

    create_lines(net_cigre_lv, busI1, busI2, length_km=0.2, line_params="UG2", name="Line I1-I2")

    from_buses = [
        busC1,
        busC2,
        busC3,
        busC4,
        busC5,
        busC6,
        busC7,
        busC8,
        busC3,
        busC10,
        busC11,
        busC11,
        busC10,
        busC5,
        busC15,
        busC15,
        busC16,
        busC8,
        busC9,
    ]
    to_buses = [
        busC2,
        busC3,
        busC4,
        busC5,
        busC6,
        busC7,
        busC8,
        busC9,
        busC10,
        busC11,
        busC12,
        busC13,
        busC14,
        busC15,
        busC16,
        busC17,
        busC18,
        busC19,
        busC20,
    ]
    std_types = [
        "OH1",
        "OH1",
        "OH1",
        "OH1",
        "OH1",
        "OH1",
        "OH1",
        "OH1",
        "OH2",
        "OH2",
        "OH3",
        "OH3",
        "OH3",
        "OH2",
        "OH2",
        "OH3",
        "OH3",
        "OH3",
        "OH3",
    ]
    names = [
        "Line C1-C2",
        "Line C2-C3",
        "Line C3-C4",
        "Line C4-C5",
        "Line C5-C6",
        "Line C6-C7",
        "Line C7-C8",
        "Line C8-C9",
        "Line C3-C10",
        "Line C10-C11",
        "Line C11-C12",
        "Line C11-C13",
        "Line C10-C14",
        "Line C5-C15",
        "Line C15-C16",
        "Line C15-C17",
        "Line C16-C18",
        "Line C8-C19",
        "Line C9-C20",
    ]
    create_lines(net_cigre_lv, from_buses, to_buses, length_km=0.030, line_params=std_types, name=names)

    # Trafos
    create_transformer_from_parameters(
        net_cigre_lv,
        busR0,
        busR1,
        sn_mva=0.5,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=1.0,
        vk_percent=4.123106,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=30.0,
        tap_pos=0.0,
        name="Trafo R0-R1",
    )

    create_transformer_from_parameters(
        net_cigre_lv,
        busI0,
        busI1,
        sn_mva=0.15,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=1.003125,
        vk_percent=4.126896,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=30.0,
        tap_pos=0.0,
        name="Trafo I0-I1",
    )

    create_transformer_from_parameters(
        net_cigre_lv,
        busC0,
        busC1,
        sn_mva=0.3,
        vn_hv_kv=20.0,
        vn_lv_kv=0.4,
        vkr_percent=0.993750,
        vk_percent=4.115529,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=30.0,
        tap_pos=0.0,
        name="Trafo C0-C1",
    )

    # External grid
    create_ext_grid(
        net_cigre_lv, bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=100.0, s_sc_min_mva=100.0, rx_max=1.0, rx_min=1.0
    )

    # Loads
    create_load(net_cigre_lv, busR1, p_mw=0.19, q_mvar=0.062449980, name="Load R1")
    create_load(net_cigre_lv, busR11, p_mw=0.01425, q_mvar=0.004683748, name="Load R11")
    create_load(net_cigre_lv, busR15, p_mw=0.0494, q_mvar=0.016236995, name="Load R15")
    create_load(net_cigre_lv, busR16, p_mw=0.05225, q_mvar=0.017173744, name="Load R16")
    create_load(net_cigre_lv, busR17, p_mw=0.03325, q_mvar=0.010928746, name="Load R17")
    create_load(net_cigre_lv, busR18, p_mw=0.04465, q_mvar=0.014675745, name="Load R18")
    create_load(net_cigre_lv, busI2, p_mw=0.0850, q_mvar=0.052678269, name="Load I2")
    create_load(net_cigre_lv, busC1, p_mw=0.1080, q_mvar=0.052306787, name="Load C1")
    create_load(net_cigre_lv, busC12, p_mw=0.018, q_mvar=0.008717798, name="Load C12")
    create_load(net_cigre_lv, busC13, p_mw=0.018, q_mvar=0.008717798, name="Load C13")
    create_load(net_cigre_lv, busC14, p_mw=0.0225, q_mvar=0.010897247, name="Load C14")
    create_load(net_cigre_lv, busC17, p_mw=0.0225, q_mvar=0.010897247, name="Load C17")
    create_load(net_cigre_lv, busC18, p_mw=0.0072, q_mvar=0.003487119, name="Load C18")
    create_load(net_cigre_lv, busC19, p_mw=0.0144, q_mvar=0.006974238, name="Load C19")
    create_load(net_cigre_lv, busC20, p_mw=0.0072, q_mvar=0.003487119, name="Load C20")

    # Switches
    create_switch(net_cigre_lv, bus0, busR0, et="b", closed=True, type="CB", name="S1")
    create_switch(net_cigre_lv, bus0, busI0, et="b", closed=True, type="CB", name="S2")
    create_switch(net_cigre_lv, bus0, busC0, et="b", closed=True, type="CB", name="S3")

    # Bus geo data
    bus_geodata = [
        f'{{"type":"Point", "coordinates":[{x}, {y}]}}'
        for x, y in zip(
            [
                0.2,
                0.2,
                -1.4583333333,
                -1.4583333333,
                -1.4583333333,
                -1.9583333333,
                -2.7083333333,
                -2.7083333333,
                -3.2083333333,
                -3.2083333333,
                -3.2083333333,
                -3.7083333333,
                -0.9583333333,
                -1.2083333333,
                -1.2083333333,
                -1.2083333333,
                -1.2083333333,
                -2.2083333333,
                -2.7083333333,
                -3.7083333333,
                0.2,
                0.2,
                0.2,
                0.2,
                1.9166666667,
                1.9166666667,
                1.9166666667,
                0.5416666667,
                0.5416666667,
                -0.2083333333,
                -0.2083333333,
                -0.2083333333,
                -0.7083333333,
                3.2916666667,
                2.7916666667,
                2.2916666667,
                3.2916666667,
                3.7916666667,
                1.2916666667,
                0.7916666667,
                1.7916666667,
                0.7916666667,
                0.2916666667,
                -0.7083333333,
            ],
            [
                1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                8.0,
                11.0,
                12.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                5.0,
                6.0,
                7.0,
                7.0,
                6.0,
                7.0,
                8.0,
                8.0,
                9.0,
                10.0,
                11.0,
            ],
        )
    ]
    # Match bus.index
    net_cigre_lv.bus["geo"] = bus_geodata
    return net_cigre_lv

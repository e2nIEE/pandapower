import pandapower as pp
import time
from pandapower.grid_equivalents.auxiliary import drop_internal_branch_elements, \
    _runpp_except_voltage_angles
import pandas as pd
import numpy as np

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _calculate_ward_and_impedance_parameters(Ybus_eq, bus_lookups, show_computing_time, power_eq=0):
    """calculates the wards and equivalente impedance to represente the
    external network"""
    t_start = time.perf_counter()
    # --- calculate ward paramter
    b_buses_ppc = bus_lookups["bus_lookup_ppc"]["b_area_buses"]
    b_buses_pd = bus_lookups["bus_lookup_pd"]["b_area_buses"]
    nb_b_buses_ppc = len(b_buses_ppc)

    ward_parameter = pd.DataFrame(columns=["bus_pd", "bus_ppc", "shunt", "power_eq"])
    ward_parameter["bus_ppc"] = b_buses_ppc
    ward_parameter["bus_pd"] = b_buses_pd
    ward_parameter["shunt"] = Ybus_eq.sum(axis=1)[-nb_b_buses_ppc:]
    ward_parameter["power_eq"] = 0 + 1j*0  # power_eq.power_eq.values

    # --- calculate impedance paramter
    params = Ybus_eq[-nb_b_buses_ppc:, -nb_b_buses_ppc:]
    nl = (nb_b_buses_ppc) * (nb_b_buses_ppc - 1) // 2
    impedance_parameter = pd.DataFrame(
        np.arange(nl * 6).reshape((nl, 6)), columns=["from_bus", "to_bus", "rft_pu", "xft_pu",
                                                     "rtf_pu", "xtf_pu"], dtype=float)
    k = 0
    for i in range(nb_b_buses_ppc):
        for j in range(nb_b_buses_ppc):
            if j > i:
                if np.abs(params[i, j]) > 1e-10:
                    impedance_parameter.from_bus[k] = b_buses_pd[i]
                    impedance_parameter.to_bus[k] = b_buses_pd[j]
                    impedance_parameter.rft_pu[k] = (-1 / params[i, j]).real
                    impedance_parameter.xft_pu[k] = (-1 / params[i, j]).imag
                    impedance_parameter.rtf_pu[k] = (-1 / params[j, i]).real
                    impedance_parameter.xtf_pu[k] = (-1 / params[j, i]).imag
                    k += 1
                else:
                    impedance_parameter = impedance_parameter[:-1]
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"calculate_ward_and_impedance_parameters\" finished in %s seconds:" % round((
                    t_end-t_start), 2))
    return ward_parameter, impedance_parameter


def _calculate_xward_and_impedance_parameters(net_external, Ybus_eq, bus_lookups,
                                              show_computing_time, power_eq=0):
    """calculates the xwards and the equivalent impedance"""
    t_start = time.perf_counter()
    xward_parameter, impedance_parameter = \
        _calculate_ward_and_impedance_parameters(Ybus_eq, bus_lookups, False)
    xward_parameter["r_ohm"] = 0
    xward_parameter["x_ohm"] = -1/xward_parameter.shunt.values.imag / \
        net_external.sn_mva*net_external.bus.vn_kv[xward_parameter.bus_pd].values**2 #/2
        # np.square(net_external.bus.vn_kv[xward_parameter.bus_pd.values].values) / \
        # net_external.sn_mva/2
    xward_parameter["vm_pu"] = net_external.res_bus.vm_pu[xward_parameter.bus_pd.values].values
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"calculate_xward_and_impedance_parameters\" finished in %s seconds:" % round((
                    t_end-t_start), 2))
    return xward_parameter, impedance_parameter


def create_passive_external_net_for_ward_admittance(
    net, all_external_buses, boundary_buses, calc_volt_angles=True,
    runpp_fct=_runpp_except_voltage_angles, **kwargs):
    """
    This function replace the wards and xward in external network by internal
    elements, and replace the power injections in external area by shunts
    if necessary.

    INPUT:
        **net** - The pandapower format network

        **all_external_buses** (list) -  list of all external buses

        **boundary_buses** (list) -  list of boundary bus indices, by which the
            original network are divide into an internal area and an external
            area
    """
    # --- replace power injection in external net by shunts to creat a passiv network
    v_m = net.res_bus.vm_pu[all_external_buses].values
    current_injections = (net.res_bus.p_mw[all_external_buses].values -
                          1j * net.res_bus.q_mvar[all_external_buses].values)
    shunt_params = list(current_injections / v_m**2)
    # creats shunts
    for i in range(len(all_external_buses)):
        if abs(np.nan_to_num(shunt_params[i])) != 0:
            pp.create_shunt(net, all_external_buses[i], -shunt_params[i].imag,
                            shunt_params[i].real)
    # drops all power injections
    for elm in ["sgen", "gen", "load", "storage"]:
        target_idx = net[elm].index[net[elm].bus.isin(all_external_buses)]
        net[elm] = net[elm].drop(target_idx)
    runpp_fct(net, calculate_voltage_angles=calc_volt_angles, **kwargs)


def _replace_external_area_by_wards(net_external, bus_lookups, ward_parameter_no_power,
                                    impedance_parameter, ext_buses_with_xward,
                                    show_computing_time, calc_volt_angles=True,
                                    runpp_fct=_runpp_except_voltage_angles, **kwargs):
    t_start = time.perf_counter()
    """replaces the external networks by wards and equivalent impedance"""
    # --- drop all external elements
    e_buses_pd = bus_lookups["bus_lookup_pd"]["e_area_buses"]
    pp.drop_buses(net_external, e_buses_pd)
    drop_internal_branch_elements(net_external, bus_lookups["boundary_buses_inclusive_bswitch"])
#    runpp_fct(net_external, calculate_voltage_angles=True, **kwargs)

    # --- drop shunt elements attached to boundary buses
    traget_shunt_idx = net_external.shunt.index[net_external.shunt.bus.isin(bus_lookups[
        "boundary_buses_inclusive_bswitch"])]
    net_external.shunt = net_external.shunt.drop(traget_shunt_idx)

    # --- creat impedance
    sn = net_external.sn_mva
    for idx in impedance_parameter.index:
        from_bus = impedance_parameter.from_bus[idx]
        to_bus = impedance_parameter.to_bus[idx]
        if abs(impedance_parameter.rft_pu[idx]) > 1e-8 or \
            abs(impedance_parameter.xft_pu[idx]) > 1e-8 or \
           abs(impedance_parameter.rtf_pu[idx]) > 1e-8 or \
           abs(impedance_parameter.xtf_pu[idx]) > 1e-8:
               pp.create_impedance(net_external, from_bus, to_bus,
                                   impedance_parameter.rft_pu[idx],
                                   impedance_parameter.xft_pu[idx],
                                   sn_mva=sn,
                                   rtf_pu=impedance_parameter.rtf_pu[idx],
                                   xtf_pu=impedance_parameter.xtf_pu[idx],
                                   name="eq_impedance")
        else:
            pp.create_switch(net_external, from_bus, to_bus, "b", name="eq_switch")

    # --- creata ward
    for i in ward_parameter_no_power.index:
        target_bus = ward_parameter_no_power.bus_pd[i]
        pp.create_ward(net_external, target_bus,
                       0.0,  # np.nan_to_num(-ward_parameter.power_eq[i].real),
                       0.0,  # np.nan_to_num(-ward_parameter.power_eq[i].imag),
                       ward_parameter_no_power.shunt[i].real * sn / (net_external.res_bus.vm_pu[target_bus] ** 2),
                       -ward_parameter_no_power.shunt[i].imag * sn / (net_external.res_bus.vm_pu[target_bus] ** 2),
                       name="network_equivalent")

    eq_power = net_external.res_ext_grid.copy()
    eq_power["bus"] = net_external.ext_grid.bus.values
    eq_power["elm"] = "ext_grid"
    slack_gen = net_external.gen.index[net_external.gen.slack]
    if len(slack_gen) != 0:
        for i in slack_gen:
            new_eq_power = \
            [net_external.res_gen.p_mw[i], net_external.res_gen.q_mvar[i],\
             net_external.gen.bus[i], "gen"]
            eq_power.loc[len(eq_power)] = new_eq_power
    assert len(eq_power.bus) == len(set(eq_power.bus))  # only one slack at individual bus

    runpp_fct(net_external, calculate_voltage_angles=calc_volt_angles, **kwargs)

    eq_power.p_mw -= \
        pd.concat([net_external.res_ext_grid.p_mw, net_external.res_gen.p_mw[slack_gen]])
    eq_power.q_mvar -= \
        pd.concat([net_external.res_ext_grid.q_mvar, net_external.res_gen.q_mvar[slack_gen]])
    for bus in eq_power.bus:
        net_external.ward.ps_mw[net_external.ward.bus==bus] = \
            eq_power.p_mw[eq_power.bus==bus].values
        net_external.ward.qs_mvar[net_external.ward.bus==bus] = \
            eq_power.q_mvar[eq_power.bus==bus].values

    net_external.poly_cost = net_external.poly_cost[0:0]
    net_external.pwl_cost = net_external.pwl_cost[0:0]
    if len(ext_buses_with_xward):
        pp.drop_buses(net_external,
                      net_external.bus.index.tolist()[-(len(ext_buses_with_xward)):])
    # net_external.ward.qs_mvar[i] = eq_power.q_mvar[
    #     net_external.ext_grid.bus == ward_parameter_no_power.bus_pd[i]]
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"replace_external_area_by_wards\" finished in %s seconds:" % round((
                    t_end-t_start), 2))


def _replace_external_area_by_xwards(net_external, bus_lookups, xward_parameter_no_power,
                                     impedance_parameter, ext_buses_with_xward,
                                     show_computing_time, calc_volt_angles=True,
                                     runpp_fct=_runpp_except_voltage_angles, **kwargs):
    """replaces the external networks by xwards and equivalent impedance"""
    t_start = time.perf_counter()
    # --- drop all external elements
    e_buses_pd = bus_lookups["bus_lookup_pd"]["e_area_buses"]
    pp.drop_buses(net_external, e_buses_pd)
    drop_internal_branch_elements(net_external, bus_lookups["boundary_buses_inclusive_bswitch"])
    # --- drop shunt elements attached to boundary buses
    traget_shunt_idx = net_external.shunt.index[net_external.shunt.bus.isin(bus_lookups[
        "boundary_buses_inclusive_bswitch"])]
    net_external.shunt = net_external.shunt.drop(traget_shunt_idx)

    # --- creat impedance
    sn = net_external.sn_mva
    for idx in impedance_parameter.index:
        from_bus = impedance_parameter.from_bus[idx]
        to_bus = impedance_parameter.to_bus[idx]
        if abs(impedance_parameter.rft_pu[idx]) > 1e-8 or \
            abs(impedance_parameter.xft_pu[idx]) > 1e-8 or \
           abs(impedance_parameter.rtf_pu[idx]) > 1e-8 or \
           abs(impedance_parameter.xtf_pu[idx]) > 1e-8:
            pp.create_impedance(net_external, from_bus, to_bus,
                                impedance_parameter.rft_pu[idx],
                                impedance_parameter.xft_pu[idx],
                                sn_mva=net_external.sn_mva,
                                rtf_pu=impedance_parameter.rtf_pu[idx],
                                xtf_pu=impedance_parameter.xtf_pu[idx],
                                name="eq_impedance")
        else:
            pp.create_switch(net_external, from_bus, to_bus, "b", name="eq_switch")
    # --- creata xward
    for i in xward_parameter_no_power.index:
        target_bus = xward_parameter_no_power.bus_pd[i]
        pp.create_xward(net_external, target_bus,
                        0.0,  # np.nan_to_num(-xward_parameter.power_eq[i].real),
                        0.0,  # np.nan_to_num(-xward_parameter.power_eq[i].imag),
                        xward_parameter_no_power.shunt[i].real * sn / xward_parameter_no_power.vm_pu[i]**2,
                        0.0,
                        xward_parameter_no_power.r_ohm[i],
                        np.nan_to_num(xward_parameter_no_power.x_ohm[i]),  # neginf=1e100 is commented since this led to error
                        xward_parameter_no_power.vm_pu[i],
                        name="network_equivalent")

    eq_power = net_external.res_ext_grid.copy()
    eq_power["bus"] = net_external.ext_grid.bus.values
    eq_power["elm"] = "ext_grid"
    slack_gen = net_external.gen.index[net_external.gen.slack]
    if len(slack_gen) != 0:
        for i in slack_gen:
            new_eq_power = \
            [net_external.res_gen.p_mw[i], net_external.res_gen.q_mvar[i],\
             net_external.gen.bus[i], "gen"]
            eq_power.loc[len(eq_power)] = new_eq_power
    assert len(eq_power.bus) == len(set(eq_power.bus))  # only one slack at individual bus

    runpp_fct(net_external, calculate_voltage_angles=calc_volt_angles,
             tolerance_mva=1e-6, max_iteration=100, **kwargs)

    eq_power.p_mw -= \
        pd.concat([net_external.res_ext_grid.p_mw, net_external.res_gen.p_mw[slack_gen]])
    eq_power.q_mvar -= \
        pd.concat([net_external.res_ext_grid.q_mvar, net_external.res_gen.q_mvar[slack_gen]])
    for bus in eq_power.bus:
        net_external.xward.ps_mw[net_external.xward.bus==bus] = \
            eq_power.p_mw[eq_power.bus==bus].values
        net_external.xward.qs_mvar[net_external.xward.bus==bus] = \
            eq_power.q_mvar[eq_power.bus==bus].values

    net_external.poly_cost=net_external.poly_cost[0:0]
    net_external.pwl_cost=net_external.pwl_cost[0:0]
    if len(ext_buses_with_xward):
        pp.drop_buses(net_external,
                      net_external.bus.index.tolist()[-(len(ext_buses_with_xward)):])
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"replace_external_area_by_xwards\" finished in %s seconds:" % round((
                    t_end-t_start), 2))
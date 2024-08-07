import pandapower as pp
import pandapower.topology as top
import time
from copy import deepcopy
from pandapower.grid_equivalents.auxiliary import drop_assist_elms_by_creating_ext_net, \
    drop_internal_branch_elements, add_ext_grids_to_boundaries, \
    _ensure_unique_boundary_bus_names, match_controller_and_new_elements, \
    match_cost_functions_and_eq_net, _check_network, _runpp_except_voltage_angles
from pandapower.grid_equivalents.rei_generation import _create_net_zpbn, \
    _get_internal_and_external_nets, _calculate_equivalent_Ybus, \
    _create_bus_lookups, _calclate_equivalent_element_params, \
    _replace_ext_area_by_impedances_and_shunts
from pandapower.grid_equivalents.ward_generation import \
    _calculate_ward_and_impedance_parameters, \
    _calculate_xward_and_impedance_parameters, \
    create_passive_external_net_for_ward_admittance, \
    _replace_external_area_by_wards, _replace_external_area_by_xwards

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def get_equivalent(net, eq_type, boundary_buses, internal_buses,
                   return_internal=True, show_computing_time=False,
                   ward_type="ward_injection", adapt_va_degree=False,
                   calculate_voltage_angles=True,
                   allow_net_change_for_convergence=False,
                   runpp_fct=_runpp_except_voltage_angles, **kwargs):
    """
    This function calculates and implements the rei or ward/xward network
    equivalents.

    ATTENTION:

        - Power flow results in the given pandapower net are mandatory.

    Known REI equivalents problems:

        - shift_degree != 0 of trafos and trafo3ws lead to errors or wrong results

        - despite 'adapt_va_degree', va_degree at the slack bus showed deviations within large grids

        - with large, real grids small deviations in the power flow results occured, in small grids \
            the results fit perfectly

    INPUT:
        **net** - The pandapower network including power flow results will not be changed during this function.

        **eq_type** (string) - type of the eqvalent network

            The following methods are available:

                - "rei": The idea of rei equivalent is to aggregate the power \
                        and current injection of the external buses to one or \
                        more fictitious radial, equivalent and independent \
                        (rei) nodes. There are three types of the rei-node in \
                        this routine, i.e. the reserved load, the reserved sgen \
                        and the reserved gen (also ext_grid). According to the \
                        demand, these elements (load, sgen and gen) are \
                        considered in the function "_create_net_zpbn" \
                        integrated or separately.

                - "ward": The ward-type equivalent represents the external \
                        network with some equivalent impedance, shunts and \
                        power injections at boundary buses. The equivalent \
                        power injections represent the power of the elements \
                        (load, sgen and gen), which are removed by the \
                        equivalent calculation.

                - "xward": The xward equivalent is an extended variation of \
                        the ward equivalent. Addition to the \
                        ward-representation, a fictitious PV node (generator) \
                        is added with zero active power injection at every \
                        boundary bus. The voltage of the PV node is set \
                        according to the boundary bus voltage.

                ward/xward has two mothods to develop an equivalent circuits, \
                i.e. the injection method and the admittance method. In the \
                admittance method, all the bus power injections in the external \
                networks are converted into shunt admittance before network \
                equivalent. That is the only difference between these two \
                methods. With the help of the function "adapt_net", these \
                methods are configurable.

        **boundary_buses** (iterable) - list of boundary bus indices, by which \
            the original network are divided into two networks, i.e. an \
            internal network and an external network.

        **internal_buses** (iterable) - list of bus indices, which are within \
            the internal network. The program will add buses which are \
            connected to this list of internal buses without passing boundary \
            buses. If 'internal_buses' is an empty list or None, the whole \
            grid is treated as external network.

    OPTIONAL:
        **return_internal** (bool, True) - Reservation of the internal network

             If True, the internal network is reserved in the final equivalent \
             network; otherwise only the external network is the output.

        **show_computing_time** (bool, False) - show computing time of each step

        **ward_type** (str, "ward_injection") - Type of ward and xward

            ward and xward proposed two mothods called the Ward Injection \
            method and the Ward Admittance method to develop equivalent \
            circuits. The only difference between these methods is that in \
            the Ward Admittance method, all bus power in the external networks \
            are converted into shunt admittances before network equivalent.

        **adapt_va_degree** (bool, None) - if True, in add_ext_grids_to_boundaries(), the va_degree \
            of the additional ext_grids (if needed) at the boundary buses will be increased or \
            decreased to values that minimize the difference to the given res_bus.va_degree values.

        **allow_net_change_for_convergence** (bool, False) - if the net doesn't converge at the \
            first internal power flow, which is in add_ext_grids_to_boundaries(), and this parameter is \
            True, the code tests if changes to unusual impedance values solve the divergence issue.

        **calculate_voltage_angles** (bool, True) - parameter passed to internal runpp() runs.

        ****kwargs** - key word arguments, such as sgen_separate, load_separate, gen_separate, \
        group_name.

    OUTPUT:
         **net_eq** - The equivalent network in pandapower format

    """

    time_start = time.perf_counter()
    eq_type = eq_type.lower()
    net = deepcopy(net)
    if not len(boundary_buses):
        raise ValueError("No boundary buses are given.")
    _check_network(net)
    logger.info(eq_type + " equivalent calculation started.")

    # --- determine interal buses, external buses, buses connected to boundary buses via
    #     bus-bus-switch and update boundary buses by external slack buses

    all_internal_buses, all_external_buses, boundary_buses_inclusive_bswitch, boundary_buses = \
        _determine_bus_groups(net, boundary_buses, internal_buses, show_computing_time)
    if not len(all_external_buses):
        logger.warning("There are no external buses so that no equivalent grid can be calculated.")
        return None
    return_internal &= bool(len(all_internal_buses))

    # --- ensure unique boundary bus names
    _ensure_unique_boundary_bus_names(net, boundary_buses_inclusive_bswitch)

    # --- create reference buses
    orig_slack_gens = add_ext_grids_to_boundaries(
        net, boundary_buses, adapt_va_degree, calc_volt_angles=calculate_voltage_angles,
        allow_net_change_for_convergence=allow_net_change_for_convergence, runpp_fct=runpp_fct,
        **kwargs)

    # --- replace ward and xward elements by internal elements (load, shunt, impedance, gen)
    ext_buses_with_ward = net.ward.bus[net.ward.bus.isin(all_external_buses)]
    ext_buses_with_xward = net.xward.bus[net.xward.bus.isin(all_external_buses)]
    if len(ext_buses_with_ward):
        logger.debug("ward elements of the external network are replaced by internal elements.")
        pp.replace_ward_by_internal_elements(net, wards=ext_buses_with_ward.index)
    if len(ext_buses_with_xward):
        logger.debug("xward elements of the external network are replaced by internal elements.")
        pp.replace_xward_by_internal_elements(net, xwards=ext_buses_with_xward.index)

    # --- switch from ward injection to ward addmittance if requested
    if eq_type in ["ward", "xward"] and ward_type == "ward_admittance":
        create_passive_external_net_for_ward_admittance(
            net, all_external_buses, boundary_buses, runpp_fct=runpp_fct, **kwargs)

    # --- rei calculations
    if eq_type == "rei":
        # --- create zero power balance network
        net_zpbn, net_internal, _ = _create_net_zpbn(
                net, boundary_buses, all_internal_buses,
                all_external_buses, calc_volt_angles=calculate_voltage_angles,
                runpp_fct=runpp_fct, **kwargs)

        # --- determine bus-lookups for the following calculation
        bus_lookups = _create_bus_lookups(
            net_zpbn, boundary_buses, all_internal_buses,
            all_external_buses, boundary_buses_inclusive_bswitch,
            show_computing_time)

        # --- calculate equivalent Ybus according to gaussian elimination
        Ybus_eq = _calculate_equivalent_Ybus(net_zpbn, bus_lookups,
                                             eq_type, show_computing_time,
                                             **kwargs)

        # --- calculate equivalent impedance and shunts
        shunt_params, impedance_params = \
            _calclate_equivalent_element_params(net_zpbn, Ybus_eq, bus_lookups,
                                                show_computing_time, **kwargs)

        # --- replace external network by equivalent elements
        _replace_ext_area_by_impedances_and_shunts(
            net_zpbn, bus_lookups, impedance_params, shunt_params,
            net_internal, return_internal, show_computing_time,
            calc_volt_angles=calculate_voltage_angles, runpp_fct=runpp_fct)
        net_eq = net_zpbn

    # --- ward and xward calculations
    elif eq_type in ["ward", "xward"]:
        net_internal, net_external = _get_internal_and_external_nets(
            net, boundary_buses, all_internal_buses, all_external_buses,
            calc_volt_angles=calculate_voltage_angles, runpp_fct=runpp_fct)

        # --- remove buses without power flow results in net_eq
        pp.drop_buses(net_external, net_external.res_bus.index[net_external.res_bus.vm_pu.isnull()])

        # --- determine bus-lookups for the following calculation
        bus_lookups = _create_bus_lookups(
            net_external, boundary_buses, all_internal_buses,
            all_external_buses, boundary_buses_inclusive_bswitch)

        # --- cacluate equivalent Ybus accourding to gaussian elimination
        Ybus_eq = _calculate_equivalent_Ybus(net_external,  bus_lookups,
                                             eq_type, show_computing_time,
                                             check_validity=False)

        if eq_type == "ward":
            # --- calculate equivalent impedance and wards
            ward_parameter_no_power, impedance_parameter = \
                _calculate_ward_and_impedance_parameters(Ybus_eq, bus_lookups,
                                                         show_computing_time)

            # --- replace external network by equivalent elements
            _replace_external_area_by_wards(net_external, bus_lookups,
                                            ward_parameter_no_power,
                                            impedance_parameter,
                                            ext_buses_with_xward,
                                            show_computing_time,
                                            calc_volt_angles=calculate_voltage_angles,
                                            runpp_fct=runpp_fct)
        else:  # eq_type == "xward"
            # --- calculate equivalent impedance and xwards
            xward_parameter_no_power, impedance_parameter = \
                _calculate_xward_and_impedance_parameters(net_external,
                                                          Ybus_eq,
                                                          bus_lookups,
                                                          show_computing_time)

            # --- replace external network by equivalent elements
            _replace_external_area_by_xwards(net_external, bus_lookups,
                                             xward_parameter_no_power,
                                             impedance_parameter,
                                             ext_buses_with_xward,
                                             show_computing_time,
                                             calc_volt_angles=calculate_voltage_angles,
                                             runpp_fct=runpp_fct)
        net_eq = net_external
    else:
        raise NotImplementedError(f"The {eq_type=} is unknown.")

    net_eq["bus_lookups"] = bus_lookups

    if return_internal:
        logger.debug("Merging of internal and equivalent network begins.")
        if len(kwargs.get("central_controller_types", [])):
            net_internal.controller.drop([idx for idx in net_internal.controller.index if any([
                isinstance(net_internal.controller.object.at[idx], central_controller_type) for
                central_controller_type in kwargs["central_controller_types"]])], inplace=True)
        net_eq = merge_internal_net_and_equivalent_external_net(
            net_eq, net_internal, show_computing_time=show_computing_time,
            calc_volt_angles=calculate_voltage_angles)
        if len(orig_slack_gens):
            net_eq.gen.loc[net_eq.gen.index.intersection(orig_slack_gens), "slack"] = True
        # run final power flow calculation
        net_eq = runpp_fct(net_eq, calculate_voltage_angles=calculate_voltage_angles)
    else:
        drop_assist_elms_by_creating_ext_net(net_eq)
        logger.debug("Only the equivalent net is returned.")

    # match the controller and the new elements
    match_controller_and_new_elements(net_eq, net)
    # delete bus in poly_cost
    match_cost_functions_and_eq_net(net_eq, boundary_buses, eq_type)

    time_end = time.perf_counter()
    logger.info("%s equivalent finished in %.2f seconds." % (eq_type, time_end-time_start))

    if kwargs.get("add_group", True):
        # declare a group for the new equivalent
        ib_buses_after_merge, be_buses_after_merge = \
            _get_buses_after_merge(net_eq, net_internal, bus_lookups, return_internal)
        eq_elms = dict()
        for elm in ["bus", "gen", "impedance", "load", "sgen", "shunt",
                    "switch", "ward", "xward"]:
            if "ward" in elm:
                new_idx = net_eq[elm].index[net_eq[elm].name == "network_equivalent"].difference(
                    net[elm].index[net[elm].name == "network_equivalent"])
            else:
                names = net_eq[elm].name.astype(str)
                if elm in ["bus", "sgen", "gen", "load"]:
                    buses = net_eq.bus.index if elm == "bus" else net_eq[elm].bus
                    new_idx = net_eq[elm].index[names.str.contains("_integrated") |
                                                names.str.contains("_separate") &
                                                ~buses.isin(ib_buses_after_merge)]
                elif elm in ["impedance"]:
                    fr_buses = net_eq[elm].from_bus
                    to_buses = net_eq[elm].to_bus
                    new_idx = net_eq[elm].index[names.str.startswith("eq_%s" % elm) |
                                                (fr_buses.isin(be_buses_after_merge) &
                                                 to_buses.isin(be_buses_after_merge))]
                else:
                    buses = net_eq[elm].bus
                    new_idx = net_eq[elm].index[names.str.startswith("eq_%s" % elm) &
                                                buses.isin(be_buses_after_merge)]

                # don't include eq elements to this Group if these are already included to other
                # groups
                # ATTENTION: If there are eq elements (elements that fit to the above query of
                # new_idx) which already exist in net but are not included to other groups, they
                # will be considered here which is wrong. Furthermore, the indices may have changed
                # from net to net_eq, so that already existing groups with reference_columns == None
                # may fail their functionality
                new_idx = new_idx[~pp.isin_group(net_eq, elm, new_idx)]

            if len(new_idx):
                eq_elms[elm] = list(new_idx)

        gr_idx = pp.create_group_from_dict(net_eq, eq_elms, name=kwargs.get("group_name", eq_type))
        reference_column = kwargs.get("reference_column", None)
        if reference_column is not None:
            pp.set_group_reference_column(net_eq, gr_idx, reference_column)

    return net_eq


def merge_internal_net_and_equivalent_external_net(
        net_eq, net_internal, fuse_bus_column="auto", show_computing_time=False, **kwargs):
    """
    Merges the internal network and the equivalent external network.
    It is expected that the boundaries occur in both, equivalent net and
    internal net. Therefore, the boundaries are first dropped in the
    internal net before merging.

    INPUT:
        **net_eq** - equivalent external area

        **net_internal** - internal area

    OPTIONAL:
        **fuse_bus_column**  (str, "auto) - the function expects boundary buses to be in net_eq and
        in net_internal. These duplicate buses get fused. To identify these buses, the given column is used. Option "auto" provides backward compatibility which is: use "name_equivalent" if
        existing and "name" otherwise

        **show_computing_time** (bool, False)

        ****kwargs** - key word arguments for pp.merge_nets()

    OUTPUT:
        **merged_net** - equivalent network within the internal area

    """
    kwargs = deepcopy(kwargs)
    t_start = time.perf_counter()
    net_internal["bus_lookups"] = net_eq["bus_lookups"]
    boundary_buses_inclusive_bswitch = net_eq.bus_lookups["boundary_buses_inclusive_bswitch"]

    # --- drop all branch elements between boundary buses in the internal net
    drop_internal_branch_elements(net_internal, boundary_buses_inclusive_bswitch)

    # --- drop bus elements attached to boundary buses in the internal net
    if kwargs.pop("drop_boundary_buses", True):
        pp.drop_elements_at_buses(net_internal, boundary_buses_inclusive_bswitch,
                                  branch_elements=False)

    # --- merge equivalent external net and internal net
    merged_net = pp.merge_nets(
        net_internal, net_eq, validate=kwargs.pop("validate", False),
        net2_reindex_log_level=kwargs.pop("net2_reindex_log_level", "debug"), **kwargs)

    # --- fuse or combine the boundary buses in external and internal nets
    if fuse_bus_column == "auto":
        if fuse_bus_column in merged_net.bus.columns:
            raise ValueError(
                f"{fuse_bus_column=} is ambiguous since the column 'auto' exists in net.bus")
        if "name_equivalent" in merged_net.bus.columns:
            fuse_bus_column = "name_equivalent"
        else:
            fuse_bus_column = "name"
    for bus in boundary_buses_inclusive_bswitch:
        try:
            name = merged_net.bus[fuse_bus_column].loc[bus]
        except:
            print(fuse_bus_column)
            print(merged_net.bus.columns)
            print()
        target_buses = merged_net.bus.index[merged_net.bus[fuse_bus_column] == name]
        if len(target_buses) != 2:
            raise ValueError(
                "The code expects all boundary buses to occur double. One because "
                "of net_eq and one because of net_internal. However target_buses is "
                "'%s'." % str(target_buses))
        pp.fuse_buses(merged_net, target_buses[0], target_buses[1])

    # --- drop assist elements
    drop_assist_elms_by_creating_ext_net(merged_net)

    # --- drop repeated characteristic
    drop_repeated_characteristic(merged_net)

    # --- reindex buses named with "total" (done by REI)
    is_total_bus = merged_net.bus.name.astype(str).str.contains("total", na=False)
    if sum(is_total_bus):
        max_non_total_bus_idx = merged_net.bus[~is_total_bus].index.values.max()
        lookup = dict(zip(merged_net.bus.index[is_total_bus], range(
            max_non_total_bus_idx+1, max_non_total_bus_idx + sum(is_total_bus)+1)))
        pp.reindex_buses(merged_net, lookup)

    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("'merge_int_and_eq_net' finished in %s seconds." %
                    round((t_end-t_start), 2))

    return merged_net


def drop_repeated_characteristic(net):
    idxs = []
    repeated_idxs = []
    for m in net.characteristic.index:
        idx = net.characteristic.object[m].__dict__["index"]
        if idx in idxs:
            repeated_idxs.append(m)
        else:
            idxs.append(idx)
    net.characteristic = net.characteristic.drop(repeated_idxs)


def _determine_bus_groups(net, boundary_buses, internal_buses,
                          show_computing_time=False):
    """
    Defines bus groups according to the given boundary buses and internal
    buses.

    INPUT:
        **net** - original network

        **boundary_buses** (iterable) - list of boundary bus indices

        **internal_buses** (iterable) - some of the internal bus indices

    OUTPUT:
        **all_internal_buses** (list) - list of all internal bus indices

        **all_external_buses** (list) - list of all external bus indices

        **boundary_buses_inclusive_bswitch** (list) - list of boundary buses
            and the connected buses via bus-bus-switch

        **boundary_buses** (list) - list of boundary bus indices (only
            supplied buses are considered)

    """
    # --- check internal and boundary buses
    t_start = time.perf_counter()
    if internal_buses is None:
        internal_buses = set()
    else:
        internal_buses = set(internal_buses)

    boundary_buses = set(boundary_buses)

    unsupplied_buses = set(net.res_bus.index[net.res_bus.vm_pu.isnull()])
    unsupplied_boundary_buses = boundary_buses & unsupplied_buses
    if len(unsupplied_boundary_buses):
        raise ValueError(
            "get_equivalent() do not allow unsupplied boundary " +
            "buses, these have no voltage results (possibly because power " +
            "flow results miss, the buses are isolated " +
            "or out of service): " + str(sorted(unsupplied_boundary_buses)) +
            ". Remove these buses from the boundary buses and try again.")

    if internal_buses & boundary_buses:
        logger.info("Some internal buses are also contained in the boundary buses, " +
                       "this could cause small inaccuracy.")

    # --- determine buses connected to boundary buses via bus-bus-switch
    boundary_buses_inclusive_bswitch = set()
    mg_sw = top.create_nxgraph(net, respect_switches=True, include_lines=False, include_impedances=False,
                               include_tcsc=False, include_trafos=False, include_trafo3ws=False)
    for bbus in boundary_buses:
        boundary_buses_inclusive_bswitch |= set(top.connected_component(mg_sw, bbus))
    if len(boundary_buses_inclusive_bswitch) > len(boundary_buses):
        logger.info("There are some buses connected to the boundary buses via "+
                    "bus-bus-switches. They could be the nodes on the same bus bar " +
                    "of the boundary buses. It is suggested to consider all these " +
                    "buses (the connected buses and the given boundary buses) " +
                    "as the boundary. They are: %s." % boundary_buses_inclusive_bswitch)

    # --- determine all internal buses
    all_internal_buses = set()
    if internal_buses is None:
        internal_buses = set()
    else:
        internal_buses = set(internal_buses)
        mg = top.create_nxgraph(net)
        cc = top.connected_components(mg, notravbuses=boundary_buses_inclusive_bswitch)
        while True:
            try:
                buses = next(cc)
            except StopIteration:
                break
            if len(buses & set(internal_buses)):
                all_internal_buses |= buses
        all_internal_buses -= boundary_buses_inclusive_bswitch

    # --- determine all external buses
    all_external_buses = set(net.bus.index) - unsupplied_buses - \
        all_internal_buses - boundary_buses_inclusive_bswitch

    # --- move all slack buses from external net to boundary if no slack is
    #     in boundary or internal
    slack_buses = set(net.ext_grid.bus[net.ext_grid.in_service]) |\
        set(net.gen.bus[net.gen.in_service & net.gen.slack])
    if len(all_internal_buses) and not len((
            all_internal_buses | boundary_buses_inclusive_bswitch) &
            slack_buses):
        if not len(slack_buses):
            raise ValueError("There is no active slack in the net.")
        for bbus in slack_buses & all_external_buses:
            boundary_buses |= {bbus}
            bbus_bswitch = set(top.connected_component(mg_sw, bbus))
            boundary_buses_inclusive_bswitch |= bbus_bswitch
            all_external_buses -= {bbus} | bbus_bswitch

    # --- function endings
    _check_bus_groups(all_internal_buses, all_external_buses, internal_buses,
                      boundary_buses)
    t_end = time.perf_counter()
    if show_computing_time:
        logger.info("\"determine_bus_groups\" finished in %s seconds." %
                    round((t_end-t_start), 2))

    return sorted(all_internal_buses), sorted(all_external_buses), \
        sorted(boundary_buses_inclusive_bswitch), sorted(boundary_buses)


def _check_bus_groups(all_internal_buses, all_external_buses, internal_buses,
                      boundary_buses):
    """
    Checks the plausibility of the bus groups.
    """
    missing_internals = internal_buses - all_internal_buses
    if len(missing_internals) and not (missing_internals & boundary_buses):
        raise ValueError("These internal buses miss in 'all_internal_buses': " + str(sorted(
                missing_internals)))
    in_and_extern_buses = all_internal_buses & all_external_buses
    if len(in_and_extern_buses):
        raise ValueError("These buses are in 'all_internal_buses' and 'all_external_buses': " + str(
                sorted(in_and_extern_buses)))


def _get_buses_after_merge(net_eq, net_internal, bus_lookups, return_internal):
    """
    Finds bus groups according to the new index after merge
    """
    if return_internal:
        bus_list = net_eq.bus.index.tolist()
        bus_list_old = net_internal.bus.index.tolist()
        ib_buses_after_merge = bus_list[:sum(~net_internal.bus.name.str.contains("assist_", na=False))]
        i_pos = [bus_list_old.index(i) for i in bus_lookups["origin_all_internal_buses"]]
        be_buses_after_merge = [x for i, x in enumerate(bus_list) if i not in i_pos]
    else:
        ib_buses_after_merge = bus_lookups["bus_lookup_pd"]["b_area_buses"]
        be_buses_after_merge = bus_lookups["bus_lookup_pd"]["b_area_buses"] + \
            bus_lookups["bus_lookup_pd"]["e_area_buses"]
    return ib_buses_after_merge, be_buses_after_merge


if __name__ == "__main__":
    """ --- quick test --- """
    # pp.logger.setLevel(logging.ERROR)
    # logger.setLevel(logging.DEBUG)
    import pandapower.networks as pn
    net = pn.case9()
    net.ext_grid.vm_pu = 1.04
    net.gen.vm_pu[0] = 1.025
    net.gen.vm_pu[1] = 1.025

    net.poly_cost = net.poly_cost.drop(net.poly_cost.index)
    net.pwl_cost = net.pwl_cost.drop(net.pwl_cost.index)
    # pp.replace_gen_by_sgen(net)
    # net.sn_mva = 109.00
    boundary_buses = [4, 8]
    internal_buses = [0]
    return_internal = True
    show_computing_time = False
    pp.runpp(net, calculate_voltage_angles=True)
    net_org = deepcopy(net)
    eq_type = "rei"
    net_eq = get_equivalent(net, eq_type, boundary_buses,
                            internal_buses,
                            return_internal=return_internal,
                            show_computing_time=False,
                            calculate_voltage_angles=True)
    print(net.res_bus)
    # print(net_eq.res_bus.loc[[0,3]])
    # print(net_eq.ward.loc[0])

    # net_eq.sn_mva = 10
    # pp.runpp(net_eq, calculate_voltage_angles=True)
    # print(net_eq.res_bus.loc[[0,3]])





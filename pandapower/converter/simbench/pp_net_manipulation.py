import numpy as np
import pandas as pd
import pandapower as pp
from copy import deepcopy

try:
    import pplog as logging
except ImportError:
    import logging

from pandapower.converter.simbench.auxiliary import idx_in_2nd_array, column_indices, \
    append_str_by_underline_count, get_unique_duplicated_dict
from pandapower.converter.simbench.voltLvl import get_voltlvl
from pandapower.converter.simbench.format_information import get_columns

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def pp_profile_names():
    return ["load", "powerplants", "renewables", "storage"]


def _extend_pandapower_net_columns(net):
    """
    This functions adds new columns to pandapower element tables as well as new tables. These new
    information space will be filled by simbench csv format data.
    """
    elms_to_extend = ["bus", "ext_grid", "gen", "sgen", "load", "storage", "shunt", "line",
                      "dcline", "trafo", "trafo3w", "switch", "ward", "xward", 'substation',
                      'measurement']
    subnet_voltLvl = ['subnet', 'voltLvl']
    add_columns = dict.fromkeys(elms_to_extend, subnet_voltLvl)
    elms_with_min_max = ["ext_grid", "gen", "sgen", "load", "storage"]
    min_max = ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']
    add_columns = {x: (min_max+y if x in elms_with_min_max else y) for x, y in add_columns.items()}
    add_columns["bus"] = ['min_vm_pu', 'max_vm_pu', 'substation'] + add_columns["bus"]
    add_columns["ext_grid"] = ['dspf', 'phys_type', 'type', 'profile', 'sn_mva'] + \
        add_columns["ext_grid"]
    add_columns["gen"] = ['dspf', 'phys_type', 'profile'] + add_columns["gen"]
    add_columns["sgen"] = ['dspf', 'phys_type', 'profile'] + add_columns["sgen"]
    add_columns["load"] = ['profile'] + add_columns["load"]
    add_columns["storage"] = ['profile', "efficiency_percent", "self-discharge_percent_per_day",
                              "resistance_ohm"] + add_columns["storage"]
    add_columns["line"] = ['max_loading_percent'] + add_columns["dcline"]
    add_columns["dcline"] = ['std_type', 'length_km', 'max_loading_percent'] + add_columns["dcline"]
    add_columns["trafo"] = ['max_loading_percent', 'autoTap', 'autoTapSide', 'autoTapSetp',
                            'substation'] + add_columns["trafo"]
    add_columns["trafo3w"] = ['max_loading_percent', 'autoTap', 'autoTapSide', 'autoTapSetp',
                              'substation'] + add_columns["trafo3w"]
    add_columns["switch"] = ['substation'] + add_columns["switch"]
    add_columns["substation"] = ['name'] + add_columns["substation"]

    for elm, columns in add_columns.items():
        if elm in net.keys():
            add_col = list(set(columns) - set(net[elm].columns))
            net[elm] = pd.concat((net[elm], pd.DataFrame([], columns=add_col)), axis=1)
        else:
            net[elm] = pd.DataFrame([], columns=columns)

    if "profiles" not in net.keys():
        net['profiles'] = {profilename: pd.DataFrame([], columns=["time"]) for profilename in
                           pp_profile_names()}


def convert_parallel_branches(net, multiple_entries=True, elm_to_convert=["line", "trafo"]):
    """ Converts parallel branch elements into multiple branches in pandapower net or other way
    around. """
    for element in elm_to_convert:  # in pp "trafo3w" currently is no parallel parameter
        # --- changes net[element].parallel to 1 and adds entries
        if multiple_entries:
            parallels = net[element].index[net[element].parallel > 1]
            while len(parallels):
                n_elm = net[element].shape[0]
                # add parallel elements
                net[element].parallel.loc[parallels] -= 1
                elm_to_append = net[element].loc[parallels]
                parallels_in_res = parallels[parallels.isin(net["res_"+element].index)]
                res_elm_to_append = net["res_"+element].loc[parallels_in_res]
                elm_to_append["parallel"] = 1
                num_par = list(net[element].parallel.loc[parallels])
                elm_to_append["name"] += [("_" + str(num)) for num in num_par]
                net[element] = net[element].append(
                    pd.DataFrame(elm_to_append.values, columns=net[element].columns),
                    ignore_index=True)
                net["res_"+element] = net["res_"+element].append(
                    pd.DataFrame(res_elm_to_append.values, columns=net["res_"+element].columns),
                    ignore_index=True)
                # add parallel switches
                for i, par in pd.Series(parallels).iteritems():
                    sw_to_append = net.switch.loc[(net.switch.element == par) & (
                        net.switch.et == element[0])]  # do not work for trafo3w
                    sw_to_append["element"] = n_elm + i
                    sw_to_append["name"] += "_" + str(num_par[i])
                    net["switch"] = net["switch"].append(
                            pd.DataFrame(sw_to_append.values, columns=net["switch"].columns),
                            ignore_index=True)
                # update parallels
                parallels = net[element].index[net[element].parallel > 1]
        # --- changes net[element].parallel to >= 1 and drops entries
        else:
            col2comp = list(set(net[element].columns) - {"name"})
            parallels = net[element].loc[net[element][col2comp].duplicated()]
            for idx, elm in parallels.iterrows():
                col2comp2 = list(set(elm.dropna().index) - {"name"})
                orig = net[element].loc[(net[element][col2comp2] == elm[col2comp2]).all(axis=1)]
                orig = orig.loc[orig.index[0]]
                # consider switches
                bus_names = {"line": ["from_bus", "to_bus"], "trafo": ["hv_bus", "lv_bus"]}[element]
                switches_differ = False
                switch_dupl = []
                for bn in bus_names:
                    sw_dupl = net.switch.loc[(net.switch.element == idx) & (
                            net.switch.et == element[0]) & (net.switch.bus == elm[bn])]
                    sw_orig = net.switch.loc[(net.switch.element == orig.name) & (
                            net.switch.et == element[0]) & (net.switch.bus == orig[bn])]
                    if sw_dupl.shape[0]:
                        if sw_dupl.shape[0] == sw_orig.shape[0]:
                            if (sw_dupl.closed.iloc[0] != sw_orig.closed.iloc[0]) | \
                               ((sw_dupl.type.iloc[0] != sw_orig.type.iloc[0]) & (
                                    (sw_dupl.type.iloc[0] is not None) |
                                    (sw_orig.type.iloc[0] is not None))):
                                switches_differ = True
                            else:
                                switch_dupl.append(sw_dupl.index[0])
                        else:
                            switches_differ = True
                if switches_differ:
                    continue
                # drop duplicated elements
                net[element].parallel.loc[orig.name] += elm.parallel
                net[element].drop(idx, inplace=True)
                net["switch"].drop(switch_dupl, inplace=True)


def merge_busbar_coordinates(net):
    """ merges x and y coordinates of busbar node connected via bus-bus switches """
    bb_nodes_set = set(net.bus.index[net.bus.type == "b"])
    bb_nodes = sorted(bb_nodes_set)
    all_connected_buses = set()
    for bb_node in bb_nodes:
        if bb_node in all_connected_buses:
            continue
        connected_nodes = pp.get_connected_buses(net, bb_node, consider=("t", "s"))
        if len(connected_nodes):
            net.bus_geodata.x.loc[connected_nodes] = net.bus_geodata.x.at[bb_node]
            net.bus_geodata.y.loc[connected_nodes] = net.bus_geodata.y.at[bb_node]
            all_connected_buses |= connected_nodes


def provide_subnet_col(net):
    """ This function provides 'subnet' column in all DataFrames of net. While
    csv2pp() writes all subnet information into pandapower_net[element]["subnet"], this function
    allows pp2csv() to consider pandapower_net[element]["zone"] if pandapower_net[element]["subnet"]
    is not available. """
    # --- provide subnet column in net.bus and fill with data of net.bus.zone
    if "subnet" not in net.bus.columns or net.bus["subnet"].isnull().all():
        # copy zone to subnet
        net.bus["subnet"] = net.bus.zone
    else:  # fill subnet nan values with zone
        net.bus.subnet.loc[net.bus.subnet.isnull()] = net.bus.zone.loc[net.bus.subnet.isnull()]

    # --- for all elements: if subnet is not avialable but zone, take it from zone.
    for element in net.keys():
        if isinstance(net[element], pd.DataFrame):
            if "subnet" not in net[element].columns or net[element]["subnet"].isnull().all():
                if "zone" in net[element].columns and not net[element]["zone"].isnull().all():
                    net[element]["subnet"] = net[element]["zone"].values

    # --- If both, subnet and zone, are not available, take subnet from bus
    # add subnet column from node to all elements but "trafo"
    pp.add_column_from_node_to_elements(
        net, "subnet", replace=False, elements=pp.pp_elements(bus=False)-{"trafo"},
        branch_bus=["from_bus", "lv_bus"])
    # add subnet column from node to trafo without verbose
    pp.add_column_from_node_to_elements(
        net, "subnet", replace=False, elements={"trafo"}, branch_bus=["from_bus", "lv_bus"],
        verbose=False)
    pp.add_column_from_element_to_elements(net, "subnet", replace=False, elements=["measurement"])

    # --- at trafo switches: use subnet from trafo instead of the bus subnet data:
    trafo_sw = net.switch.index[net.switch.et == "t"]
    net.switch.subnet.loc[trafo_sw] = net.trafo.subnet.loc[net.switch.element.loc[trafo_sw]].values

    # --- at measurements: use branch subnet instead of bus subnet data:
    for branch_type in ["line", "trafo"]:
        meas = net.measurement.index[net.measurement.element_type == branch_type]
        net.measurement.subnet.loc[meas] = net[branch_type].subnet.loc[net.measurement.element.loc[
            meas]].values


def provide_voltLvl_col(net):
    """ This function provides 'voltLvl' column in pp_elements DataFrames (not in net.substation).
    """
    if "voltLvl" not in net.bus.columns or net.bus["voltLvl"].isnull().all():
        # set voltLvl considering vn_kv and substations
        net.bus["voltLvl"] = get_voltlvl(net.bus.vn_kv)
    else:  # fill voltLvl nan values with vn_kv information
        idx_nan = net.bus.index[net.bus.voltLvl.isnull()]
        net.bus.voltLvl.loc[idx_nan] = get_voltlvl(net.bus.vn_kv.loc[idx_nan])

    # --- provide voltLvl parameters for all elements
    # add voltLvl column from node to all elements but "trafo"
    pp.add_column_from_node_to_elements(
        net, "voltLvl", replace=False, elements=pp.pp_elements(bus=False)-{"trafo"},
        branch_bus=["to_bus", "hv_bus"])
    # add voltLvl column from node to trafo without verbose
    pp.add_column_from_node_to_elements(
        net, "voltLvl", replace=False, elements={"trafo"}, branch_bus=["to_bus", "hv_bus"],
        verbose=False)
    pp.add_column_from_element_to_elements(net, "voltLvl", replace=False, elements=["measurement"])
    # correct voltLvl for trafos
    net.trafo["voltLvl"] = np.array((net.bus.voltLvl.loc[net.trafo.hv_bus].values +
                                     net.bus.voltLvl.loc[net.trafo.lv_bus].values)/2).astype(int)


def provide_substation_cols(net):
    # --- providing 'substation' column in switches and trafos
    if "substation" in net.bus.columns:
        pp.add_column_from_node_to_elements(net, "substation", replace=False, elements=["switch"])
        pp.add_column_from_node_to_elements(net, "substation", replace=False, elements=["trafo"],
                                            branch_bus=["from_bus", "lv_bus"])
    pp.add_column_from_element_to_elements(net, "substation", replace=False,
                                           elements=["measurement"])


def _add_dspf_calc_type_and_phys_type_columns(net):
    """ Adds 'dspf' and 'calc_type' column to generation elements if missing. """
    gen_tables = ["ext_grid", "gen", "sgen", "ward", "xward"]
    phys_types = ["ExternalNet", "PowerPlant", "RES", None, None]
    calc_types = ["vavm", "pvm", "pq", "Ward", "xWard"]
    for gen_table, phys_type, calc_type in zip(gen_tables, phys_types, calc_types):
        if "dspf" not in net[gen_table].columns or net[gen_table]["dspf"].isnull().all():
            if gen_table == "ext_grid" and net[gen_table].shape[0]:
                net[gen_table]["dspf"] = 1/net[gen_table].shape[0]
            else:
                net[gen_table]["dspf"] = 0
        if phys_type is not None:
            if "phys_type" not in net[gen_table].columns:
                net[gen_table]["phys_type"] = phys_type
            else:
                net[gen_table]["phys_type"].loc[net[gen_table]["phys_type"].isnull()] = phys_type
        net[gen_table]["calc_type"] = calc_type


def _replace_buses_connected_to_busbars(net, buses):
    """ Replaces buses, which are no busbars, connected to a transformer and connected at least one
        busbar via bus-bus switch, by busbars to set transformer setpoints to the busbars. """
    no_busbar = ~net.bus.loc[buses].type.str.contains("busbar")
    non_busbars = buses[no_busbar.values]
    bb_sw = net.switch.loc[net.switch.et == "b"]

    new_buses = pd.Series()
    for X, Y in zip(["element", "bus"], ["bus", "element"]):
        X_in_nonb = bb_sw[X].isin(non_busbars)
        Y_is_busbar = net.bus.loc[bb_sw[Y]].type.str.contains("busbar").values
        idx_sw_to_set = bb_sw.index[X_in_nonb & Y_is_busbar]

        if len(idx_sw_to_set):
            idx_sw_in_nonb = idx_in_2nd_array(bb_sw[X].loc[idx_sw_to_set].values,
                                                 non_busbars.values)

            trafos = non_busbars.index[idx_sw_in_nonb]
            new_buses = pd.concat([new_buses, pd.Series(bb_sw[Y].loc[idx_sw_to_set].values,
                                                        index=trafos)])

    new_buses = pd.concat([buses.loc[~pd.Series(buses.index).isin(new_buses.index).values],
                           new_buses])
    return new_buses



def _add_vm_va_setpoints_to_buses(net):
    """ Adds "vmSetp" and "vaSetp" to pp net bus table and removes vm_pu from net.ext_grid,
        net.gen and net.sgen. """
    net.bus.loc[net.ext_grid.bus, "vmSetp"] = net.ext_grid.vm_pu.values
    net.bus.loc[net.ext_grid.bus, "vaSetp"] = net.ext_grid.va_degree.values
    for elm, bus_type, param in zip(
            ["gen", "dcline", "dcline", "trafo", "trafo3w"],
            ["bus", "from_bus", "to_bus", None, None],
            ["vm_pu", "vm_from_pu", "vm_to_pu", "autoTapSetp", "autoTapSetp"]):
        if "trafo" not in elm:
            buses = net[elm][bus_type]
        else:
            autotap_trafos = net[elm].autoTap.fillna(False).astype(bool)
            autoTapSide = net[elm].autoTapSide.fillna(False).astype(bool)
            if not all(autotap_trafos == autoTapSide):
                logger.warning("'autotap_trafos' is not equal to 'autoTapSide'. \n" +
                               "'autotap_trafos.sum()' is %i, " % autotap_trafos.sum() +
                               "'autoTapSide.sum()' is %i" % autoTapSide.sum())
            autoTapSetp = net[elm].autoTapSetp.fillna(False).astype(bool)
            if not all(autotap_trafos == autoTapSetp):
                logger.warning("'autotap_trafos' is not equal to 'autoTapSetp'. \n" +
                               "'autotap_trafos.sum()' is %i, " % autotap_trafos.sum() +
                               "'autoTapSetp.sum()' is %i" % autoTapSetp.sum())
            bus_type = net[elm].autoTapSide.loc[autotap_trafos].str.lower() + "_bus"
            if bus_type.isnull().any():
                raise ValueError("The 'autoTapSide' values of automated tapped Trafos may not " +
                                 "be NaN.")
            bus_type_col_idx = column_indices(net[elm], bus_type)
            buses = pd.Series(net[elm].values[net[elm].index[autotap_trafos], bus_type_col_idx],
                              index=net[elm].index[autotap_trafos])
            buses = _replace_buses_connected_to_busbars(net, buses)
        no_vm = pd.isnull(net.bus.loc[buses, "vmSetp"]).values
        net.bus.loc[buses.loc[no_vm], "vmSetp"] = net[elm][param].values[net[elm][param].index[buses.index[no_vm]]]
        if sum(~no_vm):
            logger.debug("At buses " + str(list(buses.loc[~no_vm])) + " %s " % elm +
                         "have a vm setpoint which is not considered, since another element " +
                         "(ext_grid or gen) already set a setpoint.")
    # remove vm_pu from ext_grid - needed since vm_pu is taken from xward to ExternalNet which
    # otherwise could also be done from ext_grid mistakenly
    for elm in ["ext_grid", "gen", "sgen"]:
        if "vm_pu" in net[elm].columns:
            del net[elm]["vm_pu"]


def _set_vm_setpoint_to_trafos(net, csv_data):
    """ Adds 'autoTapSetp' to trafo and trafo3w tables. """
    for elm in ["trafo", "trafo3w"]:
        if "autoTap" in net[elm] and "autoTapSide" in net[elm]:
            autotap_trafos = net[elm].autoTap.fillna(False).astype(bool)
            assert all(autotap_trafos == net[elm].autoTapSide.fillna(False).astype(bool))
            if sum(autotap_trafos):
                bus_type = net[elm].autoTapSide.loc[autotap_trafos].str.lower() + "_bus"
                bus_type_col_idx = column_indices(net[elm], bus_type)
                buses = net[elm].values[autotap_trafos, bus_type_col_idx]
                bus_names = net.bus.name.loc[buses]
                idx_node = idx_in_2nd_array(bus_names.values, csv_data["Node*bus"].name.values)
                net[elm].loc[autotap_trafos, "autoTapSetp"] = csv_data["Node*bus"].vmSetp[
                    idx_node].values


def _prepare_res_bus_table(net):
    """ Adds columns to be converted to csv_data. """
    if net.res_bus.shape[0]:
        net.res_bus["node"] = net.res_bus.index
        col_from_bus = ['substation', 'subnet', 'voltLvl']
        net.res_bus[col_from_bus] = net.bus.loc[net.res_bus.index, col_from_bus]
    else:
        for col in ["node", 'substation', 'subnet', 'voltLvl']:
            net["res_bus"][col] = np.nan


def replace_branch_switches(net, reserved_aux_node_names=None):
    """ Instead of directly connect branch elements (trafo and line) to nodes which connection is
        done by switches, this function creates an auxiliary node in between of the branch on the
        one hand and the switch and node on the other hand. """

    # --- determine indices
    idx_t_sw = net.switch.index[net.switch.et == "t"]
    idx_l_sw = net.switch.index[net.switch.et == "l"]
    n_branch_switches = len(idx_t_sw)+len(idx_l_sw)
    idx_bus = net.switch.bus[idx_t_sw | idx_l_sw]

    # --- create auxiliary nodes
    names, reserved_aux_node_names = append_str_by_underline_count(
        net.bus.name[idx_bus], reserved_strings=reserved_aux_node_names)
    if "subnet" in net.switch.columns:
        # if replace_branch_switches() is called by pp2csv_data(), "subnet" is available
        subnets = net.switch.subnet.loc[idx_t_sw | idx_l_sw].values
    else:
        # if replace_branch_switches() is called out of pp2csv_data(), this else statement is given
        subnets = net.bus.zone[idx_bus].values
    geodata = net.bus_geodata.loc[idx_bus, ["x", "y"]].values if net["bus_geodata"].shape[0] else \
        np.empty((len(idx_bus), 2))
    aux_buses = pp.create_buses(
        net, n_branch_switches, net.bus.vn_kv[idx_bus].values, name=names.values, type="auxiliary",
        geodata=geodata, zone=subnets)
    for col in ["min_vm_pu", "max_vm_pu", "substation", "voltLvl"]:
        if col in net.bus.columns:
            net.bus[col].loc[aux_buses] = net.bus[col][idx_bus].values
    if "subnet" in net.bus.columns:
        net.bus.subnet.loc[aux_buses] = subnets
    assert len(idx_bus) == len(aux_buses)

    # --- replace branch bus by new auxiliary node
    for idx_b_sw, branch, bus_types in zip([idx_t_sw, idx_l_sw], ["trafo", "line"],
                                           [["hv_bus", "lv_bus"], ["from_bus", "to_bus"]]):
        idx_elm = net.switch.element[idx_b_sw]
        is_first_bus_type = net[branch][bus_types[0]].loc[idx_elm].values == idx_bus.loc[
            idx_b_sw].values
        # is_first_bus_type == hv_bus resp. from_bus
        pos_in_aux_buses = idx_in_2nd_array(np.array(idx_b_sw[is_first_bus_type]),
                                               np.array(idx_t_sw | idx_l_sw))
        net[branch][bus_types[0]].loc[idx_elm[is_first_bus_type]] = aux_buses[pos_in_aux_buses]
        # ~is_first_bus_type == lv_bus resp. to_bus
        pos_in_aux_buses = idx_in_2nd_array(np.array(idx_b_sw[~is_first_bus_type]),
                                               np.array(idx_t_sw | idx_l_sw))
        net[branch][bus_types[1]].loc[idx_elm[~is_first_bus_type]] = aux_buses[pos_in_aux_buses]

    # --- replace switch element by new auxiliary nodes
    net.switch.element.loc[idx_t_sw | idx_l_sw] = aux_buses
    net.switch.et.loc[idx_t_sw | idx_l_sw] = "b"

    return reserved_aux_node_names


def create_branch_switches(net):
    """ Changes bus-bus switches with auxiliary buses into bus-branch switches and drops all
        auxiliary buses. """
    # initialize DataFrame to store the indices of auxiliary buses ("aux_buses"), the switch indices
    # the auxiliary buses are connected to ("idx_switch"), the bus indices which are connected to
    # auxiliary buses via the switches("connected_buses"), the element type the auxiliary buses are
    # connected to ("et") and the element the auxiliary buses are connected to ("element")
    aux_bus_df = pd.DataFrame([], columns=["idx_switch", "aux_buses", "connected_buses", "et",
                                           "element"])

    # determine the bus indices of all auxiliary buses
    all_aux_buses = net.bus.index[net.bus.type == "auxiliary"]

    # determine the switch indices which are connected to auxiliary buses
    aux_bus_df["idx_switch"] = net.switch.index[net.switch.element.isin(all_aux_buses)]

    # determine the auxiliary bus indices of the switches
    aux_bus_df["aux_buses"] = net.switch.element.loc[aux_bus_df["idx_switch"]].values

    # determine the indices of the buses which are connected to auxiliary buses via switches
    aux_bus_df["connected_buses"] = net.switch.bus.loc[aux_bus_df["idx_switch"]].values

    # determine the element types and element indices which are connected to auxiliary buses
    for branch, bus_types in zip(["trafo", "line"], [["hv_bus", "lv_bus"], ["from_bus", "to_bus"]]):
        for bus_type in bus_types:
            current_branch_bus_type_buses = net[branch][bus_type].astype(int)
            aux_buses_are_cbbtb = aux_bus_df["aux_buses"].isin(current_branch_bus_type_buses)
            current_branch_bus_types_aux_buses = aux_bus_df["aux_buses"][aux_buses_are_cbbtb].values
            aux_bus_df["element"].loc[aux_buses_are_cbbtb] = current_branch_bus_type_buses.index[
                idx_in_2nd_array(current_branch_bus_types_aux_buses,
                                    current_branch_bus_type_buses.values)]  # requirement: only one
            # switch per aux bus
            aux_bus_df["et"].loc[aux_buses_are_cbbtb] = branch[0]

            # replace auxiliary buses in line and trafo tables
            net[branch][bus_type].loc[aux_bus_df["element"].loc[aux_buses_are_cbbtb]] = aux_bus_df[
                "connected_buses"].loc[aux_buses_are_cbbtb].values

    if pd.isnull(aux_bus_df).any().any():
        logger.error("Auxiliary bus replacement fails.")

    # replace auxiliary buses in switch table by branch elements
    for col in ["et", "element"]:
        net.switch[col].loc[aux_bus_df["idx_switch"]] = aux_bus_df[col].values

    # drop all auxiliary buses
    net.bus.drop(aux_bus_df["aux_buses"], inplace=True)
    idx_in_res_bus = aux_bus_df["aux_buses"][aux_bus_df["aux_buses"].isin(net.res_bus.index)]
    net.res_bus.drop(idx_in_res_bus, inplace=True)
    idx_in_bus_geodata = aux_bus_df["aux_buses"][aux_bus_df["aux_buses"].isin(
        net.bus_geodata.index)]
    net.bus_geodata.drop(idx_in_bus_geodata, inplace=True)


def _add_coordID(net, highest_existing_coordinate_number):
    """ adds "id", "subnet" and "voltLvl" to net.bus_geodata, "coordID" to net.bus and
        returns an unique coordinate table. """

    net.bus_geodata.dropna(how="all", inplace=True)

    # add "subnet" column to net.bus_geodata
    net.bus_geodata["subnet"] = net.bus["subnet"]
    net.bus_geodata["voltLvl"] = net.bus["voltLvl"]

    # detect duplicated coordinates
    uniq_dupl_dict = get_unique_duplicated_dict(net.bus_geodata)
    uniq = list(uniq_dupl_dict.keys())

    # add "voltLvl" column to net.bus_geodata
    net.bus_geodata["voltLvl"] = net.bus["voltLvl"]

    # add "id" column to net.bus_geodata
    net.bus_geodata["id"] = np.nan
    net.bus_geodata.id.loc[uniq] = ["coord_%i" % i for i in range(
        highest_existing_coordinate_number+1, highest_existing_coordinate_number+1+len(uniq))]

    # correct bus_geodata of duplicated entries
    for uni, dupl in uniq_dupl_dict.items():
        net.bus_geodata.id.loc[dupl] = net.bus_geodata.id.loc[uni]

    # copy new column "id" to net.bus["coordID"] - used in conversion process of pp2csv_data()
    net.bus["coordID"] = net.bus_geodata.id

    # replace bus_geodata df by unique coordinate table
    net["bus_geodata"] = pd.DataFrame(net.bus_geodata.loc[uniq],
                                      columns=get_columns("Coordinates"))


def move_slack_gens_to_ext_grid(net):
    """
    This function moves all gens with slack==True into ext_grid table to be considered in
    simbench csv format as "vavm" generation element. Since in the simbench csv format is no
    information whether a "vavm" generation element was a ext_grid or a slack, the converter back,
    csv2pp(csv_data) will write all gens with slack==True also in the ext_grid table.
    """
    if "slack" in net.gen.columns and net.gen.slack.any():
        idx_slack = net.gen.index[net.gen.slack]
        slacks = deepcopy(net.gen.loc[idx_slack])
        slacks["va_degree"] = 0
        slack_col_to_remove = [col for col in ["slack", "scaling"] if col in slacks.columns]
        slacks.drop(columns=slack_col_to_remove, inplace=True)
        ext_grid_col_to_set = [("controllable", True)]
        for col_to_set in ext_grid_col_to_set:
            if col_to_set[0] not in net.ext_grid.columns and col_to_set[0] in slacks.columns:
                net.ext_grid[col_to_set[0]] = col_to_set[1]
        net["ext_grid"] = pd.concat([net["ext_grid"], slacks], ignore_index=True)
        net.gen.drop(idx_slack, inplace=True)


def ensure_bus_index_columns_as_int(net):
    """ Ensures that all columns with bus indices, e.g. net.line.from_bus, have int as dtype. """
    ebts = pp.element_bus_tuples(bus_elements=True, branch_elements=True, res_elements=False)
    ebts |= {("switch", "element"), ("measurement", "element")}
    for elm, bus in ebts:
        net[elm][bus] = net[elm][bus].astype(int)

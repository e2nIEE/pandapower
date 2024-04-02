# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd
from pandapower.auxiliary import pandapowerNet, _preserve_dtypes, ensure_iterability, \
    log_to_level, plural_s
from pandapower.std_types import change_std_type
from pandapower.create import create_switch, create_line_from_parameters, \
    create_impedance, create_empty_network, create_gen, create_ext_grid, \
    create_load, create_shunt, create_bus, create_sgen, create_storage
from pandapower.run import runpp
from pandapower.toolbox.element_selection import branch_element_bus_dict, element_bus_tuples, pp_elements, \
    get_connected_elements, get_connected_elements_dict, next_bus
from pandapower.toolbox.result_info import clear_result_tables
from pandapower.toolbox.data_modification import reindex_elements
from pandapower.groups import detach_from_groups, attach_to_group, attach_to_groups, isin_group, \
    check_unique_group_rows, element_associated_groups

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _select_cost_df(net, p2, cost_type):
    isin = np.array([False] * net[cost_type].shape[0])
    for et in net[cost_type].et.unique():
        isin_et = net[cost_type].element.isin(p2[et].index)
        is_et = net[cost_type].et == et
        isin |= isin_et & is_et
    p2[cost_type] = net[cost_type].loc[isin]


def select_subnet(net, buses, include_switch_buses=False, include_results=False,
                  keep_everything_else=False):
    """
    Selects a subnet by a list of bus indices and returns a net with all elements
    connected to them.
    """
    buses = set(buses)
    if include_switch_buses:
        # we add both buses of a connected line, the one selected is not switch.bus
        buses_to_add = set()
        # for all line switches
        for s in net["switch"].query("et=='l'").itertuples():
            # get from/to-bus of the connected line
            fb = net["line"]["from_bus"].at[s.element]
            tb = net["line"]["to_bus"].at[s.element]
            # if one bus of the line is selected and its not the switch-bus, add the other bus
            if fb in buses and s.bus != fb:
                buses_to_add.add(tb)
            if tb in buses and s.bus != tb:
                buses_to_add.add(fb)
        buses |= buses_to_add

    if keep_everything_else:  # Info: keep_everything_else might help to keep controllers but
        # does not help if a part of controllers should be kept
        p2 = copy.deepcopy(net)
        if not include_results:
            clear_result_tables(p2)
    else:
        p2 = create_empty_network(add_stdtypes=False)
        p2["std_types"] = copy.deepcopy(net["std_types"])

        net_parameters = ["name", "f_hz"]
        for net_parameter in net_parameters:
            if net_parameter in net.keys():
                p2[net_parameter] = net[net_parameter]

    p2.bus = net.bus.loc[list(buses)]
    for elm in pp_elements(bus=False, bus_elements=True, branch_elements=False,
                           other_elements=False, res_elements=False):
        p2[elm] = net[elm][net[elm].bus.isin(buses)]

    p2.line = net.line[(net.line.from_bus.isin(buses)) & (net.line.to_bus.isin(buses))]
    p2.dcline = net.dcline[(net.dcline.from_bus.isin(buses)) & (net.dcline.to_bus.isin(buses))]
    p2.trafo = net.trafo[(net.trafo.hv_bus.isin(buses)) & (net.trafo.lv_bus.isin(buses))]
    p2.trafo3w = net.trafo3w[(net.trafo3w.hv_bus.isin(buses)) & (net.trafo3w.mv_bus.isin(buses)) &
                             (net.trafo3w.lv_bus.isin(buses))]
    p2.impedance = net.impedance[(net.impedance.from_bus.isin(buses)) &
                                 (net.impedance.to_bus.isin(buses))]
    p2.measurement = net.measurement[((net.measurement.element_type == "bus") &
                                      (net.measurement.element.isin(buses))) |
                                     ((net.measurement.element_type == "line") &
                                      (net.measurement.element.isin(p2.line.index))) |
                                     ((net.measurement.element_type == "trafo") &
                                      (net.measurement.element.isin(p2.trafo.index))) |
                                     ((net.measurement.element_type == "trafo3w") &
                                      (net.measurement.element.isin(p2.trafo3w.index)))]
    relevant_characteristics = set()
    for col in ("vk_percent_characteristic", "vkr_percent_characteristic"):
        if col in net.trafo.columns:
            relevant_characteristics |= set(net.trafo.loc[~net.trafo[col].isnull(), col].values)
    for col in (f"vk_hv_percent_characteristic", f"vkr_hv_percent_characteristic",
                f"vk_mv_percent_characteristic", f"vkr_mv_percent_characteristic",
                f"vk_lv_percent_characteristic", f"vkr_lv_percent_characteristic"):
        if col in net.trafo3w.columns:
            relevant_characteristics |= set(net.trafo3w.loc[~net.trafo3w[col].isnull(), col].values)
    p2.characteristic = net.characteristic.loc[list(relevant_characteristics)]

    _select_cost_df(net, p2, "poly_cost")
    _select_cost_df(net, p2, "pwl_cost")

    if include_results:
        for table in net.keys():
            if net[table] is None or not isinstance(net[table], pd.DataFrame) or not \
               net[table].shape[0] or not table.startswith("res_") or table[4:] not in \
               net.keys() or not isinstance(net[table[4:]], pd.DataFrame) or not \
               net[table[4:]].shape[0]:
                continue
            elif table == "res_bus":
                p2[table] = net[table].loc[pd.Index(buses).intersection(net[table].index)]
            else:
                p2[table] = net[table].loc[p2[table[4:]].index.intersection(net[table].index)]
    if "bus_geodata" in net:
        p2["bus_geodata"] = net.bus_geodata.loc[p2.bus.index.intersection(
            net.bus_geodata.index)]
    if "line_geodata" in net:
        p2["line_geodata"] = net.line_geodata.loc[p2.line.index.intersection(
            net.line_geodata.index)]

    # switches
    p2["switch"] = net.switch.loc[
        net.switch.bus.isin(p2.bus.index) & pd.concat([
            net.switch[net.switch.et == 'b'].element.isin(p2.bus.index),
            net.switch[net.switch.et == 'l'].element.isin(p2.line.index),
            net.switch[net.switch.et == 't'].element.isin(p2.trafo.index),
        ], sort=False)
        ]

    return pandapowerNet(p2)


def merge_nets(net1, net2, validate=True, merge_results=True, tol=1e-9, **kwargs):
    """Function to concatenate two nets into one data structure. The elements keep their indices
    unless both nets have the same indices. In that case, net2 elements get reindexed. The reindex
    lookup of net2 elements can be retrieved by passing return_net2_reindex_lookup=True.

    Parameters
    ----------
    net1 : pp.pandapowerNet
        first net to concatenate
    net2 : pp.pandapowerNet
        second net to concatenate
    validate : bool, optional
        whether power flow results should be compared against the results of the input nets,
        by default True
    merge_results : bool, optional
        whether results tables should be concatenated, by default True
    tol : float, optional
        tolerance which is allowed to pass the results validate check (relevant if validate is
        True), by default 1e-9
    std_prio_on_net1 : bool, optional
        whether net1 standard type should be kept if net2 has types with same names, by default True
    return_net2_reindex_lookup : bool, optional
        if True, the merged net AND a dict of lookups is returned, by default False
    net2_reindex_log_level : str, optional
        logging level of the message which element types of net2 got reindexed elements. Options
        are, for example "debug", "info", "warning", "error", or None, by default "info"

    Returns
    -------
    pp.pandapowerNet
        net with concatenated element tables

    Raises
    ------
    UserWarning
        if validate is True and power flow results of the merged net deviate from input nets results
    """
    old_params = {"retain_original_indices_in_net1", "create_continuous_bus_indices"}
    new_params = {"std_prio_on_net1", "return_net2_reindex_lookup", "net2_reindex_log_level"}
    msg1 = f"Since pandapower version 2.11.0, merge_nets() keeps element indices " + \
        "and prioritize net1 standard types by default."
    msg2 = f"Parameters {old_params} are deprecated."
    msg3 = "To silence this warning, explicitely pass at least one of the new parameters " + \
        f"{new_params}."

    old_params_passed = len(set(kwargs.keys()).intersection(old_params))
    new_params_passed = len(set(kwargs.keys()).intersection(new_params))

    if old_params_passed:
        raise FutureWarning(msg1 + msg2 + msg3)
    elif not new_params_passed:
        warnings.warn(msg1 + msg3, category=FutureWarning)
    return _merge_nets(net1, net2, validate=validate, merge_results=merge_results, tol=tol,
                           **kwargs)


def _merge_nets(net1, net2, validate=True, merge_results=True, tol=1e-9,
                std_prio_on_net1=True, return_net2_reindex_lookup=False,
                net2_reindex_log_level="info", **runpp_kwargs):
    """Function to concatenate two nets into one data structure. The elements keep their indices
    unless both nets have the same indices. In that case, net2 elements get reindex. The reindex
    lookup of net2 elements can be retrieved by passing return_net2_reindex_lookup=True.
    """
    net = copy.deepcopy(net1)
    net2 = copy.deepcopy(net2)

    if validate:
        runpp(net, **runpp_kwargs)
        net1_res_bus = copy.deepcopy(net.res_bus)
        runpp(net2, **runpp_kwargs)

    # collect element types to copy from net2 to net (output)
    elm_types = [elm_type for elm_type, df in net2.items() if not elm_type.startswith("_") and \
        isinstance(df, pd.DataFrame) and df.shape[0] and elm_type != "dtypes" and \
            (not elm_type.startswith("res_") or (merge_results and not validate))]

    # reindex net2 elements if some indices already exist in net
    reindex_lookup = dict()
    for elm_type in elm_types:
        is_dupl = pd.Series(net2[elm_type].index).isin(net[elm_type].index)
        if any(is_dupl):
            start = max(net1[elm_type].index.max(), net2[elm_type].index[~is_dupl].max()) + 1
            old_indices = net2[elm_type].index[is_dupl]
            if elm_type == "group":
                old_indices = pd.Series(old_indices).loc[~pd.Series(old_indices).duplicated()].tolist()
            new_indices = range(start, start + len(old_indices))
            reindex_lookup[elm_type] = dict(zip(old_indices, new_indices))
            reindex_elements(net2, elm_type, lookup=reindex_lookup[elm_type])
    if len(reindex_lookup.keys()):
        log_to_level("net2 elements of these types has been reindexed by merge_nets() because " + \
            f"these exist already in net1: {list(reindex_lookup.keys())}", logger,
            net2_reindex_log_level)

    # copy dataframes from net2 to net (output)
    for elm_type in elm_types:
        dtypes = net[elm_type].dtypes
        net[elm_type] = pd.concat([net[elm_type], net2[elm_type]])
        _preserve_dtypes(net[elm_type], dtypes)

    # copy standard types of net by data of net2
    for type_ in net.std_types.keys():
        if std_prio_on_net1:
            net.std_types[type_] = {**net2.std_types[type_], **net.std_types[type_]}
        else:
            net.std_types[type_].update(net2.std_types[type_])

    # validate vm results
    if validate:
        runpp(net, **runpp_kwargs)
        dev1 = max(abs(net.res_bus.loc[net1.bus.index].vm_pu.values - net1_res_bus.vm_pu.values))
        dev2 = max(abs(net.res_bus.iloc[len(net1.bus.index):].vm_pu.values -
                       net2.res_bus.vm_pu.values))
        if dev1 > tol or dev2 > tol:
            raise UserWarning("Deviation in bus voltages after merging: %.10f" % max(dev1, dev2))

    if return_net2_reindex_lookup:
        return net, reindex_lookup
    else:
        return net


def set_element_status(net, buses, in_service):
    """
    Sets buses and all elements connected to them in or out of service.
    """
    net.bus.loc[buses, "in_service"] = in_service

    for element in net.keys():
        if element not in ['bus'] and isinstance(net[element], pd.DataFrame) \
                and "in_service" in net[element].columns:
            try:
                idx = get_connected_elements(net, element, buses)
                net[element].loc[list(idx), 'in_service'] = in_service
            except:
                pass


def set_isolated_areas_out_of_service(net, respect_switches=True):
    """
    Set all isolated buses and all elements connected to isolated buses out of service.
    """
    from pandapower.topology import unsupplied_buses
    closed_switches = set()
    unsupplied = unsupplied_buses(net, respect_switches=respect_switches)
    logger.info("set %d of %d unsupplied buses out of service" % (
        len(net.bus.loc[list(unsupplied)].query('~in_service')), len(unsupplied)))
    set_element_status(net, list(unsupplied), False)

    for tr3w in net.trafo3w.index.values:
        tr3w_buses = net.trafo3w.loc[tr3w, ['hv_bus', 'mv_bus', 'lv_bus']].values
        if not all(net.bus.loc[tr3w_buses, 'in_service'].values):
            net.trafo3w.at[tr3w, 'in_service'] = False
        open_tr3w_switches = net.switch.loc[(net.switch.et == 't3') & ~net.switch.closed & (
            net.switch.element == tr3w)]
        if len(open_tr3w_switches) == 3:
            net.trafo3w.at[tr3w, 'in_service'] = False

    for element, et in zip(["line", "trafo"], ["l", "t"]):
        oos_elements = net[element].query("not in_service").index
        oos_switches = net.switch[(net.switch.et == et) & net.switch.element.isin(
            oos_elements)].index

        closed_switches.update([i for i in oos_switches.values if not net.switch.at[i, 'closed']])
        net.switch.loc[oos_switches, "closed"] = True

        for idx, bus in net.switch.loc[~net.switch.closed & (net.switch.et == et)][[
                "element", "bus"]].values:
            if not net.bus.in_service.at[next_bus(net, bus, idx, element)]:
                net[element].at[idx, "in_service"] = False
    if len(closed_switches) > 0:
        logger.info('closed %d switches: %s' % (len(closed_switches), closed_switches))


def repl_to_line(net, idx, std_type, name=None, in_service=False, **kwargs):
    """
    creates a power line in parallel to the existing power line based on the values of the new
    std_type. The new parallel line has an impedance value, which is chosen so that the resulting
    impedance of the new line and the already existing line is equal to the impedance of the
    replaced line. Or for electrical engineers:

        Z0 = impedance of the existing line

        Z1 = impedance of the replaced line

        Z2 = impedance of the created line

    sketch:
    ::

            --- Z2 ---
        ---|          |---   =  --- Z1 ---
            --- Z0 ---

    Parameters
    ----------
    net - pandapower net
    idx (int) - idx of the existing line
    std_type (str) - pandapower standard type
    name (str, None) - name of the new power line
    in_service (bool, False) - if the new power line is in service
    **kwargs - additional line parameters you want to set for the new line

    Returns
    -------
    new_idx (int) - index of the created power line

    """

    # impedance before changing the standard type
    r0 = net.line.at[idx, "r_ohm_per_km"]
    p0 = net.line.at[idx, "parallel"]
    x0 = net.line.at[idx, "x_ohm_per_km"]
    c0 = net.line.at[idx, "c_nf_per_km"]
    g0 = net.line.at[idx, "g_us_per_km"]
    i_ka0 = net.line.at[idx, "max_i_ka"]
    bak = net.line.loc[idx, :].values

    change_std_type(net, idx, std_type)

    # impedance after changing the standard type
    r1 = net.line.at[idx, "r_ohm_per_km"]
    x1 = net.line.at[idx, "x_ohm_per_km"]
    c1 = net.line.at[idx, "c_nf_per_km"]
    g1 = net.line.at[idx, "g_us_per_km"]
    i_ka1 = net.line.at[idx, "max_i_ka"]

    # complex resistance of the line parallel to the existing line
    y1 = 1 / complex(r1, x1)
    y0 = p0 / complex(r0, x0)
    z2 = 1 / (y1 - y0)

    # required parameters
    c_nf_per_km = c1 * 1 - c0 * p0
    r_ohm_per_km = z2.real
    x_ohm_per_km = z2.imag
    g_us_per_km = g1 * 1 - g0 * p0
    max_i_ka = i_ka1 - i_ka0
    name = "repl_" + str(idx) if name is None else name

    # if this line is in service to the existing line, the power flow result should be the same as
    # when replacing the existing line with the desired standard type
    new_idx = create_line_from_parameters(
        net, from_bus=net.line.at[idx, "from_bus"], to_bus=net.line.at[idx, "to_bus"],
        length_km=net.line.at[idx, "length_km"], r_ohm_per_km=r_ohm_per_km,
        x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km, max_i_ka=max_i_ka,
        g_us_per_km=g_us_per_km, in_service=in_service, name=name, **kwargs)
    # restore the previous line parameters before changing the standard type
    net.line.loc[idx, :] = bak

    # check switching state and add line switch if necessary:
    for bus in net.line.at[idx, "to_bus"], net.line.at[idx, "from_bus"]:
        if bus in net.switch[~net.switch.closed & (net.switch.element == idx) & (net.switch.et == "l")].bus.values:
            create_switch(net, bus=bus, element=new_idx, closed=False, et="l", type="LBS")

    return new_idx


def merge_parallel_line(net, idx):
    """
    Changes the impedances of the parallel line so that it equals a single line.

        Z0 = impedance of the existing parallel lines

        Z1 = impedance of the respective single line

    sketch:
    ::

            --- Z0 ---
        ---|          |---   =  --- Z1 ---
            --- Z0 ---

    Parameters
    ----------
        net - pandapower net

        idx (int) - idx of the line to merge

    Returns
    -------
    net
    """
    # impedance before changing the standard type
    r0 = net.line.at[idx, "r_ohm_per_km"]
    p0 = net.line.at[idx, "parallel"]
    x0 = net.line.at[idx, "x_ohm_per_km"]
    c0 = net.line.at[idx, "c_nf_per_km"]
    g0 = net.line.at[idx, "g_us_per_km"]
    i_ka0 = net.line.at[idx, "max_i_ka"]

    # complex resistance of the line to the existing line
    y0 = 1 / complex(r0, x0)
    y1 = p0*y0
    z1 = 1 / y1
    r1 = z1.real
    x1 = z1.imag

    g1 = p0*g0
    c1 = p0*c0
    i_ka1 = p0*i_ka0

    net.line.at[idx, "r_ohm_per_km"] = r1
    net.line.at[idx, "parallel"] = 1
    net.line.at[idx, "x_ohm_per_km"] = x1
    net.line.at[idx, "c_nf_per_km"] = c1
    net.line.at[idx, "g_us_per_km"] = g1
    net.line.at[idx, "max_i_ka"] = i_ka1

    return net


def merge_same_bus_generation_plants(net, add_info=True, error=True,
                                     gen_elms=["ext_grid", "gen", "sgen"]):
    """
    Merge generation plants connected to the same buses so that a maximum of one generation plants
    per node remains.

    ATTENTION:
        * gen_elms should always be given in order of slack (1.), PV (2.) and PQ (3.) elements.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **add_info** (bool, True) - If True, the column 'includes_other_plants' is added to the
        elements dataframes. This column informs about which element table rows are the result of a
        merge of generation plants.

        **error** (bool, True) - If True, raises an Error, if vm_pu values differ with same buses.

        **gen_elms** (list, ["ext_grid", "gen", "sgen"]) - list of elements to be merged by same
        buses. Should be in order of slack (1.), PV (2.) and PQ (3.) elements.
    """
    if add_info:
        for elm in gen_elms:
            # adding column 'includes_other_plants' if missing or overwriting if its no bool column
            if "includes_other_plants" not in net[elm].columns or net[elm][
                    "includes_other_plants"].dtype != bool:
                net[elm]["includes_other_plants"] = False

    # --- construct gen_df with all relevant plants data
    limit_cols = ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]
    cols = pd.Index(["bus", "vm_pu", "p_mw", "q_mvar"]+limit_cols)
    cols_dict = {elm: cols.intersection(net[elm].columns) for elm in gen_elms}
    gen_df = pd.concat([net[elm][cols_dict[elm]] for elm in gen_elms])
    gen_df["elm_type"] = np.repeat(gen_elms, [net[elm].shape[0] for elm in gen_elms])
    gen_df.reset_index(inplace=True)

    # --- merge data and drop duplicated rows - directly in the net tables
    something_merged = False
    for bus in gen_df["bus"].loc[gen_df["bus"].duplicated()].unique():
        idxs = gen_df.index[gen_df.bus == bus]
        if "vm_pu" in gen_df.columns and len(gen_df.vm_pu.loc[idxs].dropna().unique()) > 1:
            message = "Generation plants connected to bus %i have different vm_pu." % bus
            if error:
                raise ValueError(message)
            else:
                logger.error(message + " Only the first value is considered.")
        uniq_et = gen_df["elm_type"].at[idxs[0]]
        uniq_idx = gen_df.at[idxs[0], "index"]

        if add_info:  # add includes_other_plants information
            net[uniq_et].at[uniq_idx, "includes_other_plants"] = True

        # sum p_mw
        col = "p_mw" if uniq_et != "ext_grid" else "p_disp_mw"
        net[uniq_et].at[uniq_idx, col] = gen_df.loc[idxs, "p_mw"].sum()

        if "profiles" in net and col == "p_mw":
            elm = "gen" if "gen" in gen_df["elm_type"].loc[idxs[1:]].unique() else "sgen"
            elm_p = "%s.p_mw" % elm
            net.profiles[elm_p].loc[:, uniq_idx] = net.profiles[elm_p].loc[
                :, gen_df["index"].loc[idxs]].sum(axis=1)
            net.profiles[elm_p] = net.profiles[elm_p].drop(columns=gen_df["index"].loc[idxs[1:]])
            if elm == "gen":
                net.profiles["%s.vm_pu" % elm].drop(columns=gen_df["index"].loc[idxs[1:]],
                                                    inplace=True)

        # sum q_mvar (if available)
        if "q_mvar" in net[uniq_et].columns:
            net[uniq_et].at[uniq_idx, "q_mvar"] = gen_df.loc[idxs, "q_mvar"].sum()

        # sum limits
        for col in limit_cols:
            if col in gen_df.columns and not gen_df.loc[idxs, col].isnull().all():
                if col not in net[uniq_et].columns:
                    net[uniq_et][col] = np.nan
                net[uniq_et].at[uniq_idx, col] = gen_df.loc[idxs, col].sum()

        # drop duplicated elements
        for elm in gen_df["elm_type"].loc[idxs[1:]].unique():
            dupl_idx_elm = gen_df.loc[gen_df.index.isin(idxs[1:]) &
                                      (gen_df.elm_type == elm), "index"].values
            net[elm] = net[elm].drop(dupl_idx_elm)

        something_merged |= True
    return something_merged


def close_switch_at_line_with_two_open_switches(net):
    """
    Finds lines that have opened switches at both ends and closes one of them.
    Function is usually used when optimizing section points to
    prevent the algorithm from ignoring isolated lines.
    """
    closed_switches = set()
    nl = net.switch[(net.switch.et == 'l') & (net.switch.closed == 0)]
    for _, switch in nl.groupby("element"):
        if len(switch.index) > 1:  # find all lines that have open switches at both ends
            # and close on of them
            net.switch.at[switch.index[0], "closed"] = True
            closed_switches.add(switch.index[0])
    if len(closed_switches) > 0:
        logger.info('closed %d switches at line with 2 open switches (switches: %s)' % (
            len(closed_switches), closed_switches))


def fuse_buses(net, b1, b2, drop=True, fuse_bus_measurements=True):
    """
    Reroutes any connections to buses in b2 to the given bus b1. Additionally drops the buses b2,
    if drop=True (default).
    """
    b2 = set(b2) - {b1} if isinstance(b2, Iterable) else [b2]

    # --- reroute element connections from b2 to b1
    for element, value in element_bus_tuples():
        if net[element].shape[0]:
            net[element].loc[net[element][value].isin(b2), value] = b1
    net["switch"].loc[(net["switch"]["et"] == 'b') & (
                      net["switch"]["element"].isin(b2)), "element"] = b1

    # --- reroute bus measurements from b2 to b1
    if fuse_bus_measurements and net.measurement.shape[0]:
        bus_meas = net.measurement.loc[net.measurement.element_type == "bus"]
        bus_meas = bus_meas.index[bus_meas.element.isin(b2)]
        net.measurement.loc[bus_meas, "element"] = b1

    # --- drop b2
    if drop:
        # drop_elements=True is not needed because the elements must be connected to new buses now:
        drop_buses(net, b2, drop_elements=False)
        # branch elements which connected b1 with b2 are now connecting b1 with b1. these branch
        # can now be dropped:
        drop_inner_branches(net, buses=[b1])
        # if there were measurements at b1 and b2, these can be duplicated at b1 now -> drop
        if fuse_bus_measurements and net.measurement.shape[0]:
            drop_duplicated_measurements(net, buses=[b1])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dropping Elements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def drop_elements(net, element_type, element_index, **kwargs):
    """
    Drops element, result and group entries, as well as, associated elements from the pandapower
    net.
    """
    if element_type ==  "bus":
        drop_buses(net, element_index, **kwargs)
    elif "trafo" in element_type:
        drop_trafos(net, element_index, table=element_type)
    elif element_type ==  "line":
        drop_lines(net, element_index)
    else:
        drop_elements_simple(net, element_type, element_index)


def drop_elements_simple(net, element_type, element_index):
    """
    Drops element, result and group entries from the pandapower net.

    See Also
    --------
    drop_elements : providing more generic usage (inter-element connections considered)
    """
    if element_type in ["bus", "line", "trafo", "trafo3w"]:
        logger.warning(f"drop_elements_simple() is not appropriate to drop {element_type}s. "
                       "It is recommended to use drop_elements() instead.")

    element_index = ensure_iterability(element_index)
    detach_from_groups(net, element_type, element_index)
    net[element_type] = net[element_type].drop(element_index)

    # res_element
    res_element_type = "res_" + element_type
    if res_element_type in net.keys() and isinstance(net[res_element_type], pd.DataFrame):
        drop_res_idx = net[res_element_type].index.intersection(element_index)
        net[res_element_type] = net[res_element_type].drop(drop_res_idx)

    # logging
    if number := len(element_index) > 0:
        logger.debug("Dropped %i %s%s!" % (number, element_type, plural_s(number)))


def drop_buses(net, buses, drop_elements=True):
    """
    Drops specified buses, their bus_geodata and by default drops all elements connected to
    them as well.
    """
    detach_from_groups(net, "bus", buses)
    net["bus"] = net["bus"].drop(buses)
    net["bus_geodata"] = net["bus_geodata"].drop(set(buses) & set(net["bus_geodata"].index))
    res_buses = net.res_bus.index.intersection(buses)
    net["res_bus"] = net["res_bus"].drop(res_buses)
    if drop_elements:
        drop_elements_at_buses(net, buses)
        drop_measurements_at_elements(net, "bus", idx=buses)


def drop_trafos(net, trafos, table="trafo"):
    """
    Deletes all trafos and in the given list of indices and removes
    any switches connected to it.
    """
    if table not in ('trafo', 'trafo3w'):
        raise UserWarning("parameter 'table' must be 'trafo' or 'trafo3w'")
    # drop any switches
    et = "t" if table == 'trafo' else "t3"
    # remove any affected trafo or trafo3w switches
    i = net["switch"].index[(net["switch"]["element"].isin(trafos)) & (net["switch"]["et"] == et)]
    detach_from_groups(net, "switch", i)
    net["switch"] = net["switch"].drop(i)
    num_switches = len(i)

    # drop measurements
    drop_measurements_at_elements(net, table, idx=trafos)

    # drop the trafos
    detach_from_groups(net, table, trafos)
    net[table] = net[table].drop(trafos)
    res_trafos = net["res_" + table].index.intersection(trafos)
    net["res_" + table] = net["res_" + table].drop(res_trafos)
    logger.debug("Dropped %i %s%s with %i switches" % (
        len(trafos), table, plural_s(len(trafos)), num_switches))


def drop_lines(net, lines):
    """
    Deletes all lines and their geodata in the given list of indices and removes
    any switches connected to it.
    """
    # drop connected switches
    i = net["switch"][(net["switch"]["element"].isin(lines)) & (net["switch"]["et"] == "l")].index
    detach_from_groups(net, "switch", i)
    net["switch"] = net["switch"].drop(i)

    # drop measurements
    drop_measurements_at_elements(net, "line", idx=lines)

    # drop lines and geodata
    detach_from_groups(net, "line", lines)
    net["line"] = net["line"].drop(lines)
    net["line_geodata"] = net["line_geodata"].drop(set(lines) & set(net["line_geodata"].index))
    res_lines = net.res_line.index.intersection(lines)
    net["res_line"] = net["res_line"].drop(res_lines)
    logger.debug("Dropped %i line%s with %i line switches" % (
        len(lines), plural_s(len(lines)), len(i)))


def drop_elements_at_buses(net, buses, bus_elements=True, branch_elements=True,
                           drop_measurements=True):
    """
    drop elements connected to given buses
    """
    for element_type, column in element_bus_tuples(bus_elements, branch_elements, res_elements=False):
        if element_type == "switch":
            drop_switches_at_buses(net, buses)

        elif any(net[element_type][column].isin(buses)):
            eid = net[element_type][net[element_type][column].isin(buses)].index
            if element_type == 'line':
                drop_lines(net, eid)
            elif element_type == 'trafo' or element_type == 'trafo3w':
                drop_trafos(net, eid, table=element_type)
            else:
                n_el = net[element_type].shape[0]
                detach_from_groups(net, element_type, eid)
                net[element_type] = net[element_type].drop(eid)
                # res_element_type
                res_element_type = "res_" + element_type
                if res_element_type in net.keys() and isinstance(net[res_element_type], pd.DataFrame):
                    res_eid = net[res_element_type].index.intersection(eid)
                    net[res_element_type] = net[res_element_type].drop(res_eid)
                if net[element_type].shape[0] < n_el:
                    logger.debug("Dropped %d %s elements" % (
                        n_el - net[element_type].shape[0], element_type))
                # drop costs for the affected elements
                for cost_elm in ["poly_cost", "pwl_cost"]:
                    net[cost_elm] = net[cost_elm].drop(net[cost_elm].index[
                        (net[cost_elm].et == element_type) &
                        (net[cost_elm].element.isin(eid))])
    if drop_measurements:
        drop_measurements_at_elements(net, "bus", idx=buses)


def drop_switches_at_buses(net, buses):
    i = net["switch"][(net["switch"]["bus"].isin(buses)) |
                      ((net["switch"]["element"].isin(buses)) & (net["switch"]["et"] == "b"))].index
    net["switch"] = net["switch"].drop(i)
    logger.debug("Dropped %d switches" % len(i))


def drop_measurements_at_elements(net, element_type, idx=None, side=None):
    """
    Drop measurements of given element_type and (if given) given elements (idx) and side.
    """
    idx = ensure_iterability(idx) if idx is not None else net[element_type].index
    bool1 = net.measurement.element_type == element_type
    bool2 = net.measurement.element.isin(idx)
    bool3 = net.measurement.side == side if side is not None else np.full(net.measurement.shape[0], 1, dtype=bool)
    to_drop = net.measurement.index[bool1 & bool2 & bool3]
    net.measurement = net.measurement.drop(to_drop)


def drop_controllers_at_elements(net, element_type, idx=None):
    """
    Drop all the controllers for the given elements (idx).
    """
    idx = ensure_iterability(idx) if idx is not None else net[element_type].index
    to_drop = []
    for i in net.controller.index:
        et = net.controller.object[i].__dict__.get("element")
        elm_idx = ensure_iterability(net.controller.object[i].__dict__.get("element_index", [0.1]))
        if element_type == et:
            if set(elm_idx) - set(idx) == set():
                to_drop.append(i)
            else:
                net.controller.object[i].__dict__["element_index"] = list(set(elm_idx) - set(idx))
                net.controller.object[i].__dict__["matching_params"]["element_index"] = list(
                    set(elm_idx) - set(idx))
    net.controller = net.controller.drop(to_drop)


def drop_controllers_at_buses(net, buses):
    """
    Drop all the controllers for the elements connected to the given buses.
    """
    elms = get_connected_elements_dict(net, buses)
    for elm in elms.keys():
        drop_controllers_at_elements(net, elm, elms[elm])


def drop_duplicated_measurements(net, buses=None, keep="first"):
    """
    Drops duplicated measurements at given set of buses. If buses is None, all buses are considered.
    """
    buses = buses if buses is not None else net.bus.index
    # only analyze measurements at given buses
    bus_meas = net.measurement.loc[net.measurement.element_type == "bus"]
    analyzed_meas = bus_meas.loc[net.measurement.element.isin(buses).fillna("nan")]
    # drop duplicates
    if not analyzed_meas.duplicated(subset=[
            "measurement_type", "element_type", "side", "element"], keep=keep).empty:
        idx_to_drop = analyzed_meas.index[analyzed_meas.duplicated(subset=[
            "measurement_type", "element_type", "side", "element"], keep=keep)]
        net.measurement = net.measurement.drop(idx_to_drop)


def _inner_branches(net, buses, task, branch_elements=None):
    """
    Drops or finds branches that connects buses within 'buses' at all branch sides (e.g. 'from_bus'
    and 'to_bus').
    """
    branch_dict = branch_element_bus_dict(include_switch=True)
    if branch_elements is not None:
        branch_dict = {key: branch_dict[key] for key in branch_elements}

    inner_branches = dict()
    for elm, bus_types in branch_dict.items():
        inner = pd.Series(True, index=net[elm].index)
        for bus_type in bus_types:
            inner &= net[elm][bus_type].isin(buses)
        if elm == "switch":
            inner &= net[elm]["element"].isin(buses)
            inner &= net[elm]["et"] == "b"  # bus-bus-switches

        if any(inner):
            if task == "drop":
                if elm == "line":
                    drop_lines(net, net[elm].index[inner])
                elif "trafo" in elm:
                    drop_trafos(net, net[elm].index[inner])
                else:
                    net[elm] = net[elm].drop(net[elm].index[inner])
            elif task == "get":
                inner_branches[elm] = net[elm].index[inner]
            else:
                raise NotImplementedError("task '%s' is unknown." % str(task))
    return inner_branches


def get_inner_branches(net, buses, branch_elements=None):
    """
    Returns indices of branches that connects buses within 'buses' at all branch sides (e.g.
    'from_bus' and 'to_bus').
    """
    return _inner_branches(net, buses, "get", branch_elements=branch_elements)


def drop_inner_branches(net, buses, branch_elements=None):
    """
    Drops branches that connects buses within 'buses' at all branch sides (e.g. 'from_bus' and
    'to_bus').
    """
    _inner_branches(net, buses, "drop", branch_elements=branch_elements)


def drop_out_of_service_elements(net):
    """
    Drop all elements (including corresponding dataframes such as switches, measurements,
    result tables, geodata) with "in_service" is False. Buses that are connected to in-service
    branches are not deleted.
    """

    # --- drop inactive branches
    inactive_lines = net.line[~net.line.in_service].index
    drop_lines(net, inactive_lines)

    inactive_trafos = net.trafo[~net.trafo.in_service].index
    drop_trafos(net, inactive_trafos, table='trafo')

    inactive_trafos3w = net.trafo3w[~net.trafo3w.in_service].index
    drop_trafos(net, inactive_trafos3w, table='trafo3w')

    other_branch_elms = pp_elements(bus=False, bus_elements=False, branch_elements=True,
                                    other_elements=False) - {"line", "trafo", "trafo3w", "switch"}
    for elm in other_branch_elms:
        drop_elements_simple(net, elm, net[elm][~net[elm].in_service].index)

    # --- drop inactive buses (safely)
    # do not delete buses connected to branches
    do_not_delete = set()
    for elm, bus_col in element_bus_tuples(bus_elements=False):
        if elm != "switch":
            do_not_delete |= set(net[elm][bus_col].values)

    # remove inactive buses (safely)
    inactive_buses = set(net.bus[~net.bus.in_service].index) - do_not_delete
    drop_buses(net, inactive_buses, drop_elements=True)

    # --- drop inactive elements other than branches and buses
    for elm in pp_elements(bus=False, bus_elements=True, branch_elements=False,
                           other_elements=True):
        if "in_service" not in net[elm].columns:
            if elm not in ["measurement", "switch"]:
                logger.info("Out-of-service elements cannot be dropped since 'in_service' is " +
                            "not in net[%s].columns" % elm)
        else:
            drop_elements_simple(net, elm, net[elm][~net[elm].in_service].index)


def drop_inactive_elements(net, respect_switches=True):
    """
    Drops any elements not in service AND any elements connected to inactive
    buses.
    """
    set_isolated_areas_out_of_service(net, respect_switches=respect_switches)
    drop_out_of_service_elements(net)


def create_replacement_switch_for_branch(net, element_type, element_index):
    """
    Creates a switch parallel to a branch, connecting the same buses as the branch.
    The switch is closed if the branch is in service and open if the branch is out of service.
    The in_service status of the original branch is not affected and should be set separately,
    if needed.

    :param net: pandapower network
    :param element_type: element_type table e. g. 'line', 'impedance'
    :param element_index: index of the branch e. g. 0
    :return: None
    """
    bus_i = net[element_type].from_bus.at[element_index]
    bus_j = net[element_type].to_bus.at[element_index]
    in_service = net[element_type].in_service.at[element_index]
    if element_type in ['line', 'trafo']:
        is_closed = all(
            net.switch.loc[(net.switch.element == element_index) &
                           (net.switch.et == element_type[0]), 'closed'])
        is_closed = is_closed and in_service
    else:
        is_closed = in_service

    switch_name = 'REPLACEMENT_%s_%d' % (element_type, element_index)
    sid = create_switch(net, name=switch_name, bus=bus_i, element=bus_j, et='b', closed=is_closed,
                        type='CB')
    logger.debug('created switch %s (%d) as replacement for %s %s' %
                 (switch_name, sid, element_type, element_index))
    return sid

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Replacing Elements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def replace_zero_branches_with_switches(net, elements=('line', 'impedance'), zero_length=True,
                                        zero_impedance=True, in_service_only=True, min_length_km=0,
                                        min_r_ohm_per_km=0, min_x_ohm_per_km=0, min_c_nf_per_km=0,
                                        min_rft_pu=0, min_xft_pu=0, min_rtf_pu=0, min_xtf_pu=0,
                                        drop_affected=False):
    """
    Creates a replacement switch for branches with zero impedance (line, impedance) and sets them
    out of service.

    :param net: pandapower network
    :param elements: a tuple of names of element tables e. g. ('line', 'impedance') or (line)
    :param zero_length: whether zero length lines will be affected
    :param zero_impedance: whether zero impedance branches will be affected
    :param in_service_only: whether the branches that are not in service will be affected
    :param drop_affected: wheter the affected branch elements are dropped
    :param min_length_km: threshhold for line length for a line to be considered zero line
    :param min_r_ohm_per_km: threshhold for line R' value for a line to be considered zero line
    :param min_x_ohm_per_km: threshhold for line X' value for a line to be considered zero line
    :param min_c_nf_per_km: threshhold for line C' for a line to be considered zero line
    :param min_rft_pu: threshhold for R from-to value for impedance to be considered zero impedance
    :param min_xft_pu: threshhold for X from-to value for impedance to be considered zero impedance
    :param min_rtf_pu: threshhold for R to-from value for impedance to be considered zero impedance
    :param min_xtf_pu: threshhold for X to-from value for impedance to be considered zero impedance
    :return:
    """

    if not isinstance(elements, tuple):
        raise TypeError(
            'input parameter "elements" must be a tuple, e.g. ("line", "impedance") or ("line")')

    replaced = dict()
    for elm in elements:
        branch_zero = set()
        if elm == 'line' and zero_length:
            branch_zero.update(net[elm].loc[net[elm].length_km <= min_length_km].index.tolist())

        if elm == 'line' and zero_impedance:
            branch_zero.update(net[elm].loc[(net[elm].r_ohm_per_km <= min_r_ohm_per_km) &
                                            (net[elm].x_ohm_per_km <= min_x_ohm_per_km) &
                                            (net[elm].c_nf_per_km <= min_c_nf_per_km)
                                            ].index.tolist())

        if elm == 'impedance' and zero_impedance:
            # using np.abs() here because the impedance parameters can have negative values e.g. after grid reduction:
            branch_zero.update(net[elm].loc[(np.abs(net[elm].rft_pu) <= min_rft_pu) &
                                            (np.abs(net[elm].xft_pu) <= min_xft_pu) &
                                            (np.abs(net[elm].rtf_pu) <= min_rtf_pu) &
                                            (np.abs(net[elm].xtf_pu) <= min_xtf_pu)].index.tolist())

        affected_elements = set()
        for b in branch_zero:
            if in_service_only and ~net[elm].in_service.at[b]:
                continue
            create_replacement_switch_for_branch(net, elm, b)
            net[elm].loc[b, 'in_service'] = False
            affected_elements.add(b)

        replaced[elm] = net[elm].loc[list(affected_elements)]

        if drop_affected:
            if elm == 'line':
                drop_lines(net, affected_elements)
            else:
                net[elm] = net[elm].drop(affected_elements)

            logger.info('replaced %d %ss by switches' % (len(affected_elements), elm))
        else:
            logger.info('set %d %ss out of service' % (len(affected_elements), elm))

    return replaced


def replace_impedance_by_line(net, index=None, only_valid_replace=True, max_i_ka=np.nan):
    """
    Creates lines by given impedances data, while the impedances are dropped.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **index** (index, None) - Index of all impedances to be replaced. If None, all impedances
        will be replaced.

        **only_valid_replace** (bool, True) - If True, impedances will only replaced, if a
        replacement leads to equal power flow results. If False, unsymmetric impedances will
        be replaced by symmetric lines.

        **max_i_ka** (value(s), False) - Data/Information how to set max_i_ka. If 'imp.sn_mva' is
        given, the sn_mva values of the impedances are considered.
    """
    index = list(ensure_iterability(index)) if index is not None else list(net.impedance.index)
    max_i_ka = ensure_iterability(max_i_ka, len(index))
    new_index = []
    for (idx, imp), max_i in zip(net.impedance.loc[index].iterrows(), max_i_ka):
        if not np.isclose(imp.rft_pu, imp.rtf_pu) or not np.isclose(imp.xft_pu, imp.xtf_pu):
            if only_valid_replace:
                index.remove(idx)
                continue
            logger.error("impedance differs in from or to bus direction. lines always " +
                         "parameters always pertain in both direction. only from_bus to " +
                         "to_bus parameters are considered.")
        vn = net.bus.vn_kv.at[imp.from_bus]
        Zni = vn ** 2 / imp.sn_mva
        if max_i == 'imp.sn_mva':
            max_i = imp.sn_mva / vn / np.sqrt(3)
        new_index.append(create_line_from_parameters(

            net, imp.from_bus, imp.to_bus,
            length_km=1,
            r_ohm_per_km=imp.rft_pu * Zni,
            x_ohm_per_km=imp.xft_pu * Zni,
            c_nf_per_km=0,
            max_i_ka=max_i,
            r0_ohm_per_km=imp.rft0_pu * Zni if "rft0_pu" in net.impedance.columns else np.nan,
            x0_ohm_per_km=imp.xft0_pu * Zni if "xft0_pu" in net.impedance.columns else np.nan,
            c0_nf_per_km=0,
            parallel=1,
            name=imp.name, in_service=imp.in_service))
    _replace_group_member_element_type(net, index, "impedance", new_index, "line",
                                       detach_from_gr=False)
    drop_elements_simple(net, "impedance", index)
    return new_index


def _replace_group_member_element_type(
        net, old_elements, old_element_type, new_elements, new_element_type, detach_from_gr=True):
    assert not isinstance(old_element_type, set)
    assert not isinstance(new_element_type, set)
    old_elements = pd.Series(old_elements)
    new_elements = pd.Series(new_elements)

    check_unique_group_rows(net)
    gr_et = net.group.loc[net.group.element_type == old_element_type]
    for gr_index in gr_et.index:
        isin = old_elements.isin(gr_et.at[gr_index, "element"])
        if any(isin):
            attach_to_group(net, gr_index, new_element_type, [new_elements.loc[isin].tolist()],
                            reference_columns=gr_et.at[gr_index, "reference_column"])
    if detach_from_gr:
        detach_from_groups(net, old_element_type, old_elements)  # sometimes done afterwarts when
        # dropping the old elements


def replace_line_by_impedance(net, index=None, sn_mva=None, only_valid_replace=True):
    """
    Creates impedances by given lines data, while the lines are dropped.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **index** (index, None) - Index of all lines to be replaced. If None, all lines
        will be replaced.

        **sn_kva** (list or array, None) - Values of sn_kva for creating the impedances. If None,
        the net.sn_kva is assumed

        **only_valid_replace** (bool, True) - If True, lines will only replaced, if a replacement
        leads to equal power flow results. If False, capacitance and dielectric conductance will
        be neglected.
    """
    index = list(ensure_iterability(index)) if index is not None else list(net.line.index)
    sn_mva = sn_mva or net.sn_mva
    sn_mva = sn_mva if sn_mva != "max_i_ka" else net.line.max_i_ka.loc[index]
    sn_mva = sn_mva if hasattr(sn_mva, "__iter__") else [sn_mva] * len(index)
    if len(sn_mva) != len(index):
        raise ValueError("index and sn_mva must have the same length.")

    parallel = net.line["parallel"].values

    i = 0
    new_index = []
    for idx, line_ in net.line.loc[index].iterrows():
        if line_.c_nf_per_km or line_.g_us_per_km:
            if only_valid_replace:
                index.remove(idx)
                continue
            logger.error(f"Capacitance and dielectric conductance of line {idx} cannot be "
                         "converted to impedances, which do not model such parameters.")
        vn = net.bus.vn_kv.at[line_.from_bus]
        Zni = vn ** 2 / sn_mva[i]
        par = parallel[idx]
        new_index.append(create_impedance(
            net, line_.from_bus, line_.to_bus,
            rft_pu=line_.r_ohm_per_km * line_.length_km / par / Zni,
            xft_pu=line_.x_ohm_per_km * line_.length_km / par / Zni,
            sn_mva=sn_mva[i],
            rft0_pu=line_.r0_ohm_per_km * line_.length_km / par / Zni if "r0_ohm_per_km" in net.line.columns else None,
            xft0_pu=line_.x0_ohm_per_km * line_.length_km / par / Zni if "x0_ohm_per_km" in net.line.columns else None,
            name=line_.name,
            in_service=line_.in_service))
        i += 1
    _replace_group_member_element_type(net, index, "line", new_index, "impedance",
                                       detach_from_gr=False)
    drop_lines(net, index)
    return new_index


def replace_ext_grid_by_gen(net, ext_grids=None, gen_indices=None, slack=False, cols_to_keep=None,
                            add_cols_to_keep=None):
    """
    Replaces external grids by generators.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **ext_grids** (iterable) - indices of external grids which should be replaced

        **gen_indices** (iterable) - required indices of new generators

        **slack** (bool, False) - indicates which value is set to net.gen.slack for the new
        generators

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
        ext_grids. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are always set:
        "bus", "vm_pu", "p_mw", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
        'cols_to_keep' to be kept while replacing ext_grids.
    """
    # --- determine ext_grid index
    if ext_grids is None:
        ext_grids = net.ext_grid.index
    else:
        ext_grids = ensure_iterability(ext_grids)
    if gen_indices is None:
        gen_indices = [None] * len(ext_grids)
    elif len(gen_indices) != len(ext_grids):
        raise ValueError("The length of 'gen_indices' must be the same as 'ext_grids' but is " +
                         "%i instead of %i" % (len(gen_indices), len(ext_grids)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "p_mw", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net.ext_grid.loc[ext_grids].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.gen which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net.gen.columns)
    for col in missing_cols_to_keep:
        net.gen[col] = np.nan

    # --- create gens
    new_idx = []
    for ext_grid, index in zip(net.ext_grid.loc[ext_grids].itertuples(), gen_indices):
        p_mw = 0 if ext_grid.Index not in net.res_ext_grid.index else net.res_ext_grid.at[
            ext_grid.Index, "p_mw"]
        idx = create_gen(net, ext_grid.bus, vm_pu=ext_grid.vm_pu, p_mw=p_mw, name=ext_grid.name,
                         in_service=ext_grid.in_service, controllable=True, index=index)
        new_idx.append(idx)
    net.gen.loc[new_idx, "slack"] = slack
    net.gen.loc[new_idx, existing_cols_to_keep] = net.ext_grid.loc[
        ext_grids, existing_cols_to_keep].values

    _replace_group_member_element_type(net, ext_grids, "ext_grid", new_idx, "gen")

    # --- drop replaced ext_grids
    net.ext_grid = net.ext_grid.drop(ext_grids)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "ext_grid") &
                                         (net[table].element.isin(ext_grids))]
            if len(to_change):
                net[table].loc[to_change, "et"] = "gen"
                net[table].loc[to_change, "element"] = new_idx

    # --- result data
    if net.res_ext_grid.shape[0]:
        in_res = pd.Series(ext_grids).isin(net["res_ext_grid"].index).values
        to_add = net.res_ext_grid.loc[pd.Index(ext_grids)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_gen = pd.concat([net.res_gen, to_add], sort=True)
        net.res_ext_grid = net.res_ext_grid.drop(pd.Index(ext_grids)[in_res])
    return new_idx


def replace_gen_by_ext_grid(net, gens=None, ext_grid_indices=None, cols_to_keep=None,
                            add_cols_to_keep=None):
    """
    Replaces generators by external grids.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **gens** (iterable) - indices of generators which should be replaced

        **ext_grid_indices** (iterable) - required indices of new external grids

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
        gens. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are alway set:
        "bus", "vm_pu", "va_degree", "name", "in_service"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
        'cols_to_keep' to be kept while replacing gens.
    """
    # --- determine gen index
    if gens is None:
        gens = net.gen.index
    else:
        gens = ensure_iterability(gens)
    if ext_grid_indices is None:
        ext_grid_indices = [None] * len(gens)
    elif len(ext_grid_indices) != len(gens):
        raise ValueError("The length of 'ext_grid_indices' must be the same as 'gens' but is " +
                         "%i instead of %i" % (len(ext_grid_indices), len(gens)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "va_degree", "name", "in_service"})

    existing_cols_to_keep = net.gen.loc[gens].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.ext_grid
    missing_cols_to_keep = existing_cols_to_keep.difference(net.ext_grid.columns)
    for col in missing_cols_to_keep:
        net.ext_grid[col] = np.nan

    # --- create ext_grids
    new_idx = []
    for gen, index in zip(net.gen.loc[gens].itertuples(), ext_grid_indices):
        va_degree = 0. if gen.bus not in net.res_bus.index else net.res_bus.va_degree.at[gen.bus]
        idx = create_ext_grid(net, gen.bus, vm_pu=gen.vm_pu, va_degree=va_degree, name=gen.name,
                              in_service=gen.in_service, index=index)
        new_idx.append(idx)
    net.ext_grid.loc[new_idx, existing_cols_to_keep] = net.gen.loc[
        gens, existing_cols_to_keep].values

    _replace_group_member_element_type(net, gens, "gen", new_idx, "ext_grid")

    # --- drop replaced gens
    net.gen = net.gen.drop(gens)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "gen") & (net[table].element.isin(gens))]
            if len(to_change):
                net[table].loc[to_change, "et"] = "ext_grid"
                net[table].loc[to_change, "element"] = new_idx

    # --- result data
    if net.res_gen.shape[0]:
        in_res = pd.Series(gens).isin(net["res_gen"].index).values
        to_add = net.res_gen.loc[pd.Index(gens)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_ext_grid = pd.concat([net.res_ext_grid, to_add], sort=True)
        net.res_gen = net.res_gen.drop(pd.Index(gens)[in_res])
    return new_idx


def replace_gen_by_sgen(net, gens=None, sgen_indices=None, cols_to_keep=None,
                        add_cols_to_keep=None):
    """
    Replaces generators by static generators.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **gens** (iterable) - indices of generators which should be replaced

        **sgen_indices** (iterable) - required indices of new static generators

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
        gens. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are always set:
        "bus", "p_mw", "q_mvar", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
        'cols_to_keep' to be kept while replacing gens.
    """
    # --- determine gen index
    if gens is None:
        gens = net.gen.index
    else:
        gens = ensure_iterability(gens)
    if sgen_indices is None:
        sgen_indices = [None] * len(gens)
    elif len(sgen_indices) != len(gens):
        raise ValueError("The length of 'sgen_indices' must be the same as 'gens' but is " +
                         "%i instead of %i" % (len(sgen_indices), len(gens)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "p_mw", "q_mvar", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net.gen.loc[gens].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add missing columns to net.gen which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net.sgen.columns)
    for col in missing_cols_to_keep:
        net.sgen[col] = np.nan

    # --- create sgens
    new_idx = []
    for gen, index in zip(net.gen.loc[gens].itertuples(), sgen_indices):
        q_mvar = 0. if gen.Index not in net.res_gen.index else net.res_gen.at[gen.Index, "q_mvar"]
        controllable = True if "controllable" not in net.gen.columns else gen.controllable
        idx = create_sgen(net, gen.bus, p_mw=gen.p_mw, q_mvar=q_mvar, name=gen.name,
                          in_service=gen.in_service, controllable=controllable, index=index)
        new_idx.append(idx)
    net.sgen.loc[new_idx, existing_cols_to_keep] = net.gen.loc[
        gens, existing_cols_to_keep].values

    _replace_group_member_element_type(net, gens, "gen", new_idx, "sgen")

    # --- drop replaced gens
    net.gen = net.gen.drop(gens)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "gen") & (net[table].element.isin(gens))]
            if len(to_change):
                net[table].loc[to_change, "et"] = "sgen"
                net[table].loc[to_change, "element"] = new_idx

    # --- result data
    if net.res_gen.shape[0]:
        in_res = pd.Series(gens).isin(net["res_gen"].index).values
        to_add = net.res_gen.loc[pd.Index(gens)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_sgen = pd.concat([net.res_sgen, to_add], sort=True)
        net.res_gen = net.res_gen.drop(pd.Index(gens)[in_res])
    return new_idx


def replace_sgen_by_gen(net, sgens=None, gen_indices=None, cols_to_keep=None,
                        add_cols_to_keep=None):
    """
    Replaces static generators by generators.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **sgens** (iterable) - indices of static generators which should be replaced

        **gen_indices** (iterable) - required indices of new generators

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing
        sgens. If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
        "max_q_mvar", "min_q_mvar". However cols_to_keep is given, these columns are always set:
        "bus", "vm_pu", "p_mw", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
        'cols_to_keep' to be kept while replacing sgens.
    """
    # --- determine sgen index
    if sgens is None:
        sgens = net.sgen.index
    else:
        sgens = ensure_iterability(sgens)
    if gen_indices is None:
        gen_indices = [None] * len(sgens)
    elif len(gen_indices) != len(sgens):
        raise ValueError("The length of 'gen_indices' must be the same as 'sgens' but is " +
                         "%i instead of %i" % (len(gen_indices), len(sgens)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "p_mw", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net.sgen.loc[sgens].dropna(axis=1).columns.intersection(
        cols_to_keep)
    # add columns which should be kept from sgen but miss in gen to net.gen
    missing_cols_to_keep = existing_cols_to_keep.difference(net.gen.columns)
    for col in missing_cols_to_keep:
        net.gen[col] = np.nan

    # --- create gens
    new_idx = []
    log_warning = False
    for sgen, index in zip(net.sgen.loc[sgens].itertuples(), gen_indices):
        if sgen.bus in net.res_bus.index:
            vm_pu = net.res_bus.at[sgen.bus, "vm_pu"]
        else:  # no result information to get vm_pu -> use net.gen.vm_pu or net.ext_grid.vm_pu or
            # set 1.0
            if sgen.bus in net.gen.bus.values:
                vm_pu = net.gen.vm_pu.loc[net.gen.bus == sgen.bus].values[0]
            elif sgen.bus in net.ext_grid.bus.values:
                vm_pu = net.ext_grid.vm_pu.loc[net.ext_grid.bus == sgen.bus].values[0]
            else:
                vm_pu = 1.0
                log_warning = True
        controllable = False if "controllable" not in net.sgen.columns else sgen.controllable
        idx = create_gen(net, sgen.bus, vm_pu=vm_pu, p_mw=sgen.p_mw, name=sgen.name,
                         in_service=sgen.in_service, controllable=controllable, index=index)
        new_idx.append(idx)
    net.gen.loc[new_idx, existing_cols_to_keep] = net.sgen.loc[
        sgens, existing_cols_to_keep].values

    if log_warning:
        logger.warning("In replace_sgen_by_gen(), for some generator 'vm_pu' is assumed as 1.0 " +
                       "since no power flow results were available.")

    _replace_group_member_element_type(net, sgens, "sgen", new_idx, "gen")

    # --- drop replaced sgens
    net.sgen = net.sgen.drop(sgens)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == "sgen") &
                                         (net[table].element.isin(sgens))]
            if len(to_change):
                net[table].loc[to_change, "et"] = "gen"
                net[table].loc[to_change, "element"] = new_idx

    # --- result data
    if net.res_sgen.shape[0]:
        in_res = pd.Series(sgens).isin(net["res_sgen"].index).values
        to_add = net.res_sgen.loc[pd.Index(sgens)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net.res_gen = pd.concat([net.res_gen, to_add], sort=True)
        net.res_sgen = net.res_sgen.drop(pd.Index(sgens)[in_res])
    return new_idx


def replace_pq_elmtype(net, old_element_type, new_element_type, old_indices=None, new_indices=None,
                       cols_to_keep=None, add_cols_to_keep=None):
    """
    Replaces e.g. static generators by loads or loads by storages and so forth.

    INPUT:
        **net** - pandapower net

        **old_element_type** (str) - element type of which elements should be replaced. Should be in [
            "sgen", "load", "storage"]

        **new_element_type** (str) - element type of which elements should be created. Should be in [
            "sgen", "load", "storage"]

    OPTIONAL:
        **old_indices** (iterable) - indices of the elements which should be replaced

        **new_indices** (iterable) - required indices of the new elements

        **cols_to_keep** (list, None) - list of column names which should be kept while replacing.
        If None these columns are kept if values exist: "max_p_mw", "min_p_mw",
        "max_q_mvar", "min_q_mvar". Independent whether cols_to_keep is given, these columns are
        always set: "bus", "p_mw", "q_mvar", "name", "in_service", "controllable"

        **add_cols_to_keep** (list, None) - list of column names which should be added to
        'cols_to_keep' to be kept while replacing.

    OUTPUT:
        **new_idx** (list) - list of indices of the new elements
    """
    if old_element_type == new_element_type:
        logger.warning(f"'old_element_type' and 'new_element_type' are both '{old_element_type}'. "
                       "No replacement is done.")
        return old_indices
    if old_indices is None:
        old_indices = net[old_element_type].index
    else:
        old_indices = ensure_iterability(old_indices)
    if not len(old_indices):
        return []
    if new_indices is None:
        new_indices = [None] * len(old_indices)
    elif len(new_indices) != len(old_indices):
        raise ValueError("The length of 'new_indices' must be the same as of 'old_indices' but " +
                         "is %i instead of %i" % (len(new_indices), len(old_indices)))

    # --- determine which columns should be kept while replacing
    cols_to_keep = cols_to_keep if cols_to_keep is not None else [
        "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]
    if isinstance(add_cols_to_keep, list) and len(add_cols_to_keep):
        cols_to_keep += add_cols_to_keep
    elif add_cols_to_keep is not None:
        raise ValueError("'add_cols_to_keep' must be a list or None but is a %s" % str(type(
            add_cols_to_keep)))
    cols_to_keep = list(set(cols_to_keep) - {"bus", "vm_pu", "p_mw", "name", "in_service",
                                             "controllable"})

    existing_cols_to_keep = net[old_element_type].loc[old_indices].dropna(
        axis=1).columns.intersection(cols_to_keep)
    # add missing columns to net[new_element_type] which should be kept
    missing_cols_to_keep = existing_cols_to_keep.difference(net[new_element_type].columns)
    for col in missing_cols_to_keep:
        net[new_element_type][col] = np.nan

    # --- create new_element_type
    already_considered_cols = set()
    new_idx = []
    for oelm, index in zip(net[old_element_type].loc[old_indices].itertuples(), new_indices):
        controllable = False if "controllable" not in net[old_element_type].columns else oelm.controllable
        sign = -1 if old_element_type in ["sgen"] else 1
        args = dict()
        if new_element_type == "load":
            fct = create_load
        elif new_element_type == "sgen":
            fct = create_sgen
            sign *= -1
        elif new_element_type == "storage":
            fct = create_storage
            already_considered_cols |= {"max_e_mwh"}
            args = {"max_e_mwh": 1 if "max_e_mwh" not in net[old_element_type].columns else net[
                old_element_type].max_e_kwh.loc[old_indices]}
        idx = fct(net, oelm.bus, p_mw=sign*oelm.p_mw, q_mvar=sign*oelm.q_mvar, name=oelm.name,
                  in_service=oelm.in_service, controllable=controllable, index=index, **args)
        new_idx.append(idx)

    if sign == -1:
        for col1, col2 in zip(["max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"],
                              ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]):
            if col1 in existing_cols_to_keep:
                net[new_element_type].loc[new_idx, col2] = sign * net[old_element_type].loc[
                    old_indices, col1].values
                already_considered_cols |= {col1}
    net[new_element_type].loc[new_idx, existing_cols_to_keep.difference(
        already_considered_cols)] = net[old_element_type].loc[
            old_indices, existing_cols_to_keep.difference(already_considered_cols)].values

    _replace_group_member_element_type(net, old_indices, old_element_type, new_idx, new_element_type)

    # --- drop replaced old_indices
    net[old_element_type] = net[old_element_type].drop(old_indices)

    # --- adapt cost data
    for table in ["pwl_cost", "poly_cost"]:
        if net[table].shape[0]:
            to_change = net[table].index[(net[table].et == old_element_type) &
                                         (net[table].element.isin(old_indices))]
            if len(to_change):
                net[table].loc[to_change, "et"] = new_element_type
                net[table].loc[to_change, "element"] = new_idx

    # --- result data
    if net["res_" + old_element_type].shape[0]:
        in_res = pd.Series(old_indices).isin(net["res_" + old_element_type].index).values
        to_add = net["res_" + old_element_type].loc[pd.Index(old_indices)[in_res]]
        to_add.index = pd.Index(new_idx)[in_res]
        net["res_" + new_element_type] = pd.concat([net["res_" + new_element_type], to_add], sort=True)
        net["res_" + old_element_type] = net["res_" + old_element_type].drop(pd.Index(old_indices)[in_res])
    return new_idx


def replace_ward_by_internal_elements(net, wards=None, log_level="warning"):
    """
    Replaces wards by loads and shunts.

    INPUT:
        **net** - pandapower net

    OPTIONAL:
        **wards** (iterable) - indices of xwards which should be replaced

    OUTPUT:
        No output - the given wards in pandapower net are replaced by loads and shunts

    """
    # --- determine wards index
    if wards is None:
        wards = net.ward.index
    else:
        wards = ensure_iterability(wards)

    ass = element_associated_groups(net, "ward", wards)

    # --- create loads and shunts
    new_load_idx = []
    new_shunt_idx = []
    for ward in net.ward.loc[wards].itertuples():
        load_idx = create_load(net, ward.bus, ward.ps_mw, ward.qs_mvar,
                               in_service=ward.in_service, name=ward.name)
        shunt_idx = create_shunt(net, ward.bus, q_mvar=ward.qz_mvar, p_mw=ward.pz_mw,
                                 in_service=ward.in_service, name=ward.name)
        new_load_idx.append(load_idx)
        new_shunt_idx.append(shunt_idx)

        attach_to_groups(net, ass[ward.Index], ["load", "shunt"], [[load_idx], [shunt_idx]])

    # --- result data
    if net.res_ward.shape[0]:
        sign_in_service = np.multiply(net.ward.in_service.loc[wards].values, 1)
        sign_not_isolated = np.multiply(net.res_ward.vm_pu.loc[wards].values != 0, 1)
        to_add_load = net.res_ward.loc[wards, ["p_mw", "q_mvar"]]
        to_add_load.index = new_load_idx
        to_add_load.p_mw = net.ward.ps_mw.loc[wards].values * sign_in_service * sign_not_isolated
        to_add_load.q_mvar = net.ward.qs_mvar.loc[wards].values * sign_in_service * \
            sign_not_isolated
        net.res_load = pd.concat([net.res_load, to_add_load])

        to_add_shunt = net.res_ward.loc[wards, ["p_mw", "q_mvar", "vm_pu"]]
        to_add_shunt.index = new_shunt_idx
        to_add_shunt.p_mw = net.res_ward.vm_pu.loc[wards].values ** 2 * net.ward.pz_mw.loc[
            wards].values * sign_in_service * sign_not_isolated
        to_add_shunt.q_mvar = net.res_ward.vm_pu.loc[wards].values ** 2 * net.ward.qz_mvar.loc[
            wards].values * sign_in_service * sign_not_isolated
        to_add_shunt.vm_pu = net.res_ward.vm_pu.loc[wards].values
        net.res_shunt = pd.concat([net.res_shunt, to_add_shunt])

    # --- drop replaced wards
    drop_elements_simple(net, "ward", wards)


def replace_xward_by_internal_elements(net, xwards=None, set_xward_bus_limits=False,
                                       log_level="warning"):
    """
    Replaces xward by loads, shunts, impedance and generators

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    xwards : iterable, optional
        indices of xwards which should be replaced. If None, all xwards are replaced, by default None
    set_xward_bus_limits : bool, optional
        if True, the buses internal in xwards get vm limits from the connected buses
    log_level : str, optional
        logging level of the message which element types of net2 got reindexed elements. Options
        are, for example "debug", "info", "warning", "error", or None, by default "info"

    Returns
    -------
    None
        the given xwards in pandapower are replaced by buses, loads, shunts, impadance and generators
    """
    # --- determine xwards index
    if xwards is None:
        xwards = net.xward.index
    else:
        xwards = ensure_iterability(xwards)

    ass = element_associated_groups(net, "xward", xwards)
    default_vm_lims = [0, 2]

    # --- create buses, loads, shunts, gens and impedances
    for xward in net.xward.loc[xwards].itertuples():
        vn = net.bus.vn_kv.at[xward.bus]
        vm_lims = net.bus.loc[xward.bus, ["min_vm_pu", "max_vm_pu"]].tolist() if \
            set_xward_bus_limits else default_vm_lims
        new_bus = create_bus(net, net.bus.vn_kv[xward.bus], in_service=xward.in_service,
                             name=xward.name, min_vm_pu=vm_lims[0], max_vm_pu=vm_lims[1])
        new_load = create_load(net, xward.bus, xward.ps_mw, xward.qs_mvar,
            in_service=xward.in_service, name=xward.name)
        new_shunt = create_shunt(net, xward.bus, q_mvar=xward.qz_mvar, p_mw=xward.pz_mw,
            in_service=xward.in_service, name=xward.name)
        new_gen = create_gen(net, new_bus, 0, xward.vm_pu, in_service=xward.in_service,
            name=xward.name)
        new_imp = create_impedance(net, xward.bus, new_bus, xward.r_ohm / (vn ** 2),
            xward.x_ohm / (vn ** 2), net.sn_mva, in_service=xward.in_service, name=xward.name)

        attach_to_groups(net, ass[xward.Index], ["bus", "load", "shunt", "gen", "impedance"], [
            [new_bus], [new_load], [new_shunt], [new_gen], [new_imp]])

    # --- result data
    if net.res_xward.shape[0]:
        log_to_level("Implementations to move xward results to new internal elements are missing.",
                     logger, log_level)
        net.res_xward = net.res_xward.drop(xwards)

    # --- drop replaced wards
    drop_elements_simple(net, "xward", xwards)

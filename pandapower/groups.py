# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import numpy as np
import pandas as pd
import pandas.testing as pdt
import uuid

from pandapower.auxiliary import ensure_iterability
from pandapower.create import create_empty_network, _group_parameter_list, _set_multiple_entries, \
    _check_elements_existence
from pandapower.toolbox import pp_elements, group_element_index, group_row, \
    element_bus_tuples, branch_element_bus_dict, get_connected_elements_dict

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# ====================================
# ADAPT GROUP MEMBERS
# ====================================


def append_to_group(net, index, element_types, elements, reference_columns=None):
    """Appends the group by the elements given as dict of indices in 'elements_dict_to_append'.
    If net is given, elements_dict_to_append is checked and updated by _update_elements_dict().
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    element_types : str or list of strings
        defines, together with 'elements', which net elements belong to the group
    elements : list of list of indices
        defines, together with 'element_types', which net elements belong to the group
    reference_columns : string or list of strings, optional
        If given, the elements_dict should
        not refer to DataFrames index but to another column. It is highly relevant that the
        reference_column exists in all DataFrames of the grouped elements and have the same dtype,
        by default None
    """
    if index not in net.group.index:
        raise ValueError(
            f"{index} is not in net.group.index. Correct index or used create_group() instead.")

    element_types, elements, reference_columns = _group_parameter_list(
        element_types, elements, reference_columns)

    complete_new = {col: list() for col in ["name", "element_type", "element", "reference_column"]}
    name = group_name(net, index)

    is_group_index = np.isin(net.group.index.values, index)

    for et, elm, rc in zip(element_types, elements, reference_columns):

        group_et = is_group_index & (net.group.element_type == et).values
        no_row = np.sum(group_et)

        # --- element entry of existing is appended
        if no_row == 1:
            if net.group.reference_column.loc[group_et].at[index] != rc:
                raise UserWarning(
                    f"The reference column of existing group {index} for element "
                    f"type '{et}' and of the elements to append differ. Use "
                    "set_reference_column() to change the reference column of net.group before, or"
                    "pass appropriate data to append_to_group().")
            prev_elm = net.group.element.loc[group_et].at[index]
            prev_elm = [prev_elm] if isinstance(prev_elm, str) or not hasattr(
                prev_elm, "__iter__") else list(prev_elm)
            net.group.element.loc[group_et] = [prev_elm + elm]

        # --- prepare adding new rows to net.group (because no other elements of element type et
        # --- already belongs to the group)
        elif no_row == 0:
            complete_new["name"].append(name)
            complete_new["element_type"].append(et)
            complete_new["element"].append(elm)
            complete_new["reference_column"].append(rc)

        else:
            raise ValueError(f"Multiple {et} rows for group {index}")

    # --- add new rows to net.group
    if len(complete_new["name"]):
        _check_elements_existence(net, element_types, elements, reference_columns)
        _set_multiple_entries(net, "group", [index]*len(complete_new["name"]), **complete_new)
        net.group.sort_index(inplace=True)


# =================================================
# ACCESS GROUP DATA AND EVALUATE MEMBERSHIP
# =================================================


def _get_lists_from_df(df, cols):
    return [df[col].tolist() for col in cols]


def group_element_lists(net, index):
    return tuple(_get_lists_from_df(net.group.loc[[index]],
                                    ["element_type", "element", "reference_column"]))


def group_name(net, index):
    """Returns the name of the group and checks that all group rows include the same name
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the group
    verbose : bool, optional
        Setting to False, accelerate the code but don't check inconsistent names, by default True
    """
    names = net.group.name.loc[[index]]
    if len(set(names)) != 1 and not pd.isnull(names).all():
        raise ValueError(f"group {index} has different values in net.group.name.loc[index]")
    return names.values[0]


def isin_group(net, element_type, element_index, index=None, drop_empty_lines=True):
    """Returns whether elements are in group(s).
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_type : str
        element type of the elements to be found in the groups
    element_index : int or list of integers
        indices of the element table which should be found in the groups
    index : int or list of integers, optional
        Can narrow the number of groups in which the elements are searched, by default None
    drop_empty_lines : bool, optional
        This parameter decides whether empty entries should be removed (the complete row in
        net.group), by default True
    Returns
    -------
    boolean or boolean numpy.array
        Information whether the element are in any group
    """
    if isinstance(element_index, str):
        raise ValueError("element_index must be an integer or a list of integers.")
    if not hasattr(element_index, "__iter__"):
        single_group = True
        element_index = [element_index]
    else:
        element_index = list(element_index)
        single_group = False
    if index is None:
        index = list(set(net.group.index))
    else:
        index = ensure_iterability(index)

    ensure_lists_in_group_element_column(net, drop_empty_lines=drop_empty_lines)

    member_idx = pd.Index([], dtype=int)
    for idx in index:
        member_idx = member_idx.union(group_element_index(net, idx, element_type))

    isin = np.isin(element_index, member_idx)

    if single_group:
        return isin[0]
    else:
        return isin


def count_group_elements(net, index):
    """Returns a Series concluding the number of included elements in self.elements_dict
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    """
    return pd.Series({
        et: len(elm) if hasattr(elm, "__iter__") and not isinstance(elm, str) else 1 for
        et, elm in zip(*_get_lists_from_df(net.group.loc[[index]], ["element_type", "element"]))},
        dtype=int)


# =================================================
# COMPARE GROUPS
# =================================================


def groups_equal(net, index1, index2, **kwargs):
    """Returns a boolean whether both group are equal.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index1 : int
        index of the first group to compare
    index2 : int
        index of the second group to compare
    """
    df1 = net.group.loc[[index1]].set_index("name")
    df2 = net.group.loc[[index2]].set_index("name")
    try:
        pdt.assert_frame_equal(df1, df2, **kwargs)
        return True
    except AssertionError:
        return False


def compare_group_elements(net, index1, index2):
    """allow_cross_reference_column_comparison
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index1 : int
        index of the first group to compare
    index2 : int
        index of the second group to compare
    """
    ensure_lists_in_group_element_column(net)
    et1 = net.group.loc[[index1], "element_type"].tolist()
    et2 = net.group.loc[[index2], "element_type"].tolist()
    if len(et1) != len(set(et1)):
        raise ValueError(f"The Group of index {index1} has duplicated element_types.")
    if len(et2) != len(set(et2)):
        raise ValueError(f"The Group of index {index2} has duplicated element_types.")
    if set(et1) != set(et2):
        return False

    gr1 = net.group.loc[[index1]].set_index("element_type")
    gr2 = net.group.loc[[index2]].set_index("element_type")
    for et in et1:
        if gr1.reference_column.at[et] == gr2.reference_column.at[et]:
            if len(pd.Index(gr1.element.at[et]).symmetric_difference(gr2.element.at[et])):
                return False
        else:
            if len(group_element_index(net, index1, et).symmetric_difference(group_element_index(net, index2, et))):
                return False
    return True

# =================================================
# FIX GROUP DATA
# =================================================


def check_unique_group_names(net, raise_=False):
    """Checks whether all groups have unique names. raise_ decides whether duplicated names lead
    to error or log message.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    raise_ : bool, optional
        decides whether duplicated names lead to error or log message., by default False
    """
    df = net.group[["name", "element_type"]].reset_index()
    if df.duplicated().any():
        raise ValueError("There are multiple groups with same index, name and element_type.")
    single_name_per_index = [len(names) == 1 for names in net.group.reset_index().groupby("index")[
        "name"].agg(set)]
    if not all(single_name_per_index):
        warn = "There are multiple groups with same index and name."
        if raise_:
            raise UserWarning(warn)
        else:
            logger.warning(warn)


def remove_not_existing_group_members(net, verbose=True):
    """Remove group members from net.group that do not exist in the elements tables.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    verbose : bool, optional
        Steers amount of logging messages, by default True
    """
    gr_idx_before = set(net.group.index)
    keep = np.ones(net.group.shape[0], dtype=bool)
    not_existing_et = set()
    empty_et = set()
    for i in range(net.group.shape[0]):
        et = net.group.element_type.iat[i]
        if et not in net.keys() or not isinstance(net[et], pd.DataFrame):
            not_existing_et |= {et}
            keep[i] = False
        elif not net[et].shape[0]:
            empty_et |= {et}
            keep[i] = False
        else:
            not_exist_bool = ~group_entries_exist_in_element_table(
                net, net.group.index[i], et)
            if np.all(not_exist_bool):
                keep[i] = False
                if verbose:
                    logger.info(f"net.group row {i} is dropped because no fitting elements exist in"
                                f" net[{et}].")
            elif np.any(not_exist_bool):
                net.group.element.iat[i] = list(np.array(net.group.element.iat[i])[~not_exist_bool])
                if verbose:
                    logger.info(
                        f"{np.sum(not_exist_bool)} entries were dropped from net.group row {i}.")
    if verbose:
        if len(not_existing_et):
            logger.info(f"element_types {not_existing_et} are no dataframes in net and thus be "
                        "removed from net.group.")
        if len(empty_et):
            logger.info(f"net[*] are empty and thus be removed from net.group. "
                        f"* is placeholder for {empty_et}.")
    net.group = net.group.loc[keep]

    if verbose:
        gr_idx_after = set(net.group.index)
        removed_gr = gr_idx_before - gr_idx_after
        if len(removed_gr):
            logger.info(f"These groups are removed since no existing member remains: {removed_gr}.")


def ensure_lists_in_group_element_column(net, drop_empty_lines=True):
    """Ensure that all entries in net.group.element are of type list.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    drop_empty_lines : bool, optional
        This parameter decides whether empty entries should be removed (the complete row in
        net.group), by default True
    """
    keep = np.ones(net.group.shape[0], dtype=bool)
    for i in range(net.group.shape[0]):
        elm = net.group.element.iat[i]
        if hasattr(elm, "__iter__") and not isinstance(elm, str):
            net.group.element.iat[i] = list(elm)
            if not len(elm):
                keep[i] = False
        else:
            if elm is None or pd.isnull(elm):
                net.group.element.iat[i] = []
                keep[i] = False
            else:
                net.group.element.iat[i] = [elm]
    if drop_empty_lines:
        net.group = net.group.loc[keep]


def group_entries_exist_in_element_table(net, index, element_type):
    """Returns an array of booleans whether the entries in net.group.element exist in
    net[element_type], also considering reference_column
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        Index of the group
    element_type : str
        element type which entries should be checked, e.g. "bus"
    Returns
    -------
    array of booleans
        Whether the entries in net.group.element exist in net[element_type]
    """
    row = group_row(net, index, element_type)
    element = row.at["element"]
    reference_column = row.at["reference_column"]

    if not hasattr(element, "__iter__") or isinstance(element, str):
        raise ValueError("Entries in net.group.element should be lists. You can try "
                         "ensure_lists_in_group_element_column() to fix this.")

    if reference_column is None or pd.isnull(reference_column):
        return np.isin(np.array(element), net[element_type].index.values)
    else:
        return np.isin(np.array(element), net[element_type][reference_column].values)


# =================================================
# FURTHER GROUP FUNCTIONS
# =================================================


def set_group_in_service(net, index):
    """Sets all elements of the group in service.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    """
    set_value_to_group(net, index, True, "in_service", replace=True, append_column=False)


def set_group_out_of_service(net, index):
    """Sets all elements of the group out of service.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    """
    set_value_to_group(net, index, False, "in_service", replace=True, append_column=False)


def set_value_to_group(net, index, value, column, replace=True, append_column=True):
    """Sets the same value to the column of the element tables of all elements/members of the group.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    value : int/float/object
        value to be written to all group members into the element tables
    column : str
        column to be manipulated
    replace : bool, optional
        If False, value is only written to places where no value exist before (column doesn't exist
        before or value is nan), by default True
    append_column : bool, optional
        Decides whether the column should be added to element tables where this doesn't exist
        before, by default True
    """
    for et, elm, rc in zip(*group_element_lists(net, index)):
        if append_column or column in net[et].columns:
            if rc is None or pd.isnull(rc):
                if replace or column not in net[et].columns:
                    ix = elm
                else:
                    ix = net[et].loc[elm].index[net[et][column].loc[elm].isnull()]
            else:
                if replace or column not in net[et].columns:
                    ix = net[et].index[net[et][rc].isin(elm)]
                else:
                    ix = net[et].index[net[et][rc].isin(elm) & net[et][column].isnull()]
            net[et].loc[ix, column] = value


def _sum_powers(net, index, formula_character, unit):
    power = 0.
    missing_res_idx = list()
    no_power_column_found = list()
    no_res_table_found = list()
    for et in net.group.loc[[index], "element_type"].tolist():
        if et in ["switch", "measurement", "bus", "bus_geodata", "line_geodata"]:
            continue
        idx = group_element_index(net, index, et)
        res_et = "res_" + et
        if res_et not in net.keys():
            no_res_table_found.append(et)
            continue
        res_idx = net[res_et].index.intersection(idx)
        sign = 1 if et not in ["ext_grid", "gen", "sgen"] else -1
        if len(res_idx) != len(idx):
            missing_res_idx.append(et)
        col1 = "%s_%s" % (formula_character, unit)
        col2 = "%sl_%s" % (formula_character, unit)
        if col1 in net[res_et].columns:
            power += sign * net[res_et][col1].loc[res_idx].sum()
        elif col2 in net[res_et].columns:
            power += sign * net[res_et][col2].loc[res_idx].sum()
        else:
            no_power_column_found.append(et)

    if len(no_res_table_found):
        logger.warning(f"Result tables of this elements does not exist: {no_res_table_found}.")
    if len(missing_res_idx):
        logger.warning("The resulting power may be wrong since the result tables of these "
                        f"elements lack of indices: {missing_res_idx}.")
    if len(no_power_column_found):
        logger.warning("The resulting power may be wrong since the result tables of these "
                        f"elements have no requested power column: {no_power_column_found}.")
    return power


def group_res_p_mw(net, index):
    """Sums all result table values `p_mw` of group members.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    """
    return _sum_powers(net, index, "p", "mw")


def group_res_q_mvar(net, index):
    """Sums all result table values `q_mvar` of group members.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    """
    return _sum_powers(net, index, "q", "mvar")


def set_group_reference_column(net, index, reference_column, element_type=None):
    """Set a reference_column to the group of given index. The values in net.group.element get
    updated.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        Index of the group
    reference_column : str
        column in the elemt tables which should be used as reference to link the group members.
        If no column but the index should be used as the link (is the default), reference_column
        should be None.
    element_type : str, optional
        Type of element which should get a new column to reference. If None, all element types are
        considered, by default None
    Raises
    ------
    ValueError
        net[element_type][reference_column] has duplicated values.
    """
    if element_type is None:
        element_type = net.group.loc[[index], "element_type"].tolist()
    else:
        element_type = ensure_iterability(element_type)

    dupl_elements = list()
    for et in element_type:

        if reference_column is None:
            # determine duplicated indices which would corrupt Groups functionality
            if len(set(net[et].index)) != net[et].shape[0]:
                dupl_elements.append(et)

        else:
            # fill nan values in net[et][reference_column] with unique names
            if reference_column not in net[et].columns:
                net[et][reference_column] = pd.Series([None]*net[et].shape[0], dtype=object)
            if pd.api.types.is_object_dtype(net[et][reference_column]):
                idxs = net[et].index[net[et][reference_column].isnull()]
                net[et][reference_column].loc[idxs] = ["%s_%i_%s" % (et, idx, str(
                    uuid.uuid4())) for idx in idxs]

            # determine duplicated values which would corrupt Groups functionality
            if (net[et][reference_column].duplicated() | net[et][reference_column].isnull()).any():
                dupl_elements.append(et)

        # update net.group[["element", "reference_column"]] for element_type == et
        if not len(dupl_elements):
            pos_bool = ((net.group.index == index) & (net.group.element_type == et)).values
            if np.sum(pos_bool) > 1:
                raise ValueError(
                    f"Group of index {index} has multiple entries for element type '{et}'.")
            pos = np.arange(len(pos_bool), dtype=int)[pos_bool][0]

            if reference_column is None:
                net.group.element.iat[pos] = group_element_index(net, index, et).tolist()
                net.group.reference_column.iat[pos] = None
            else:
                net.group.element.iat[pos] = \
                    net[et].loc[group_element_index(net, index, et), reference_column].tolist()
                net.group.reference_column.iat[pos] = reference_column

    if len(dupl_elements):
        if reference_column is None:
            raise ValueError(f"In net[*].index have duplicated or nan values. "
                            f"* is placeholder for {dupl_elements}.")
        else:
            raise ValueError(f"In net[*].{reference_column} have duplicated or nan values. "
                            f"* is placeholder for {dupl_elements}.")


def return_group_as_net(net, index, keep_everything_else=False, verbose=True, **kwargs):
    """Returns a pandapower net consisting of the members of this group.
    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group
    keep_everything_else : bool, optional
        Decides whether other data than element tables are kept, by default False
    verbose : bool, optional
        Decides whether logging messages are triggered, by default True
    Returns
    -------
    pandapowerNet
        pandapower net, only with elements that are members of the group
    """
    if keep_everything_else:
        group_net = copy.deepcopy(net)
        for et in pp_elements():
            if group_net[et].shape[0]:
                keep = group_element_index(net, index, et)
                group_net[et].drop(group_net[et].index.difference(keep), inplace=True)
        if len(set(net.group.index)) > 1:
            if verbose:
                logger.warning("The returned net includes further groups. Not existing members get "
                               "dropped now.")
            remove_not_existing_group_members(net, verbose=verbose)
    else:
        group_net = create_empty_network(
            name=group_name(net, index), f_hz=net.f_hz, sn_mva=net.sn_mva,
            add_stdtypes=kwargs.get("add_stdtypes", True))
        group_net["group"] = net.group.loc[[index]]
        for et in net.group.loc[[index], "element_type"].tolist():
            idx = group_element_index(net, index, et)
            group_net[et] = net[et].loc[idx]
    return group_net


def elements_connected_to_group(net, index, element_types, find_buses_only_from_buses=False,
                                respect_switches=True, respect_in_service=False,
                                include_empty_lists=False):
    """Returns a dict of indices of elements that are connected to the group.
    Parameters
    ----------
    net : pandapowerNet
        the net of the group
    index : int
        index of the group
    element_types : iterable of element types
        element types of which connected elements are searched
    find_buses_only_from_buses : bool, optional
        if True, connected buses are not searched considering bus elements and branch elements of
        the group but only considering the buses of the group as done by
        pandapower.toolbox.get_connected_buses(). In that case it is ignored whether branch
        elements between the buses of the groups and the connected buses are in the group or not
    respect_switches : bool, optional
        True -> open switches will be respected,
        False -> open switches will be ignored,
        by default True
    respect_in_service : bool, optional
        True -> in_service status of connected lines will be respected,
        False -> in_service status will be ignored,
        by default False
    include_empty_lists : bool, optional
        if True, the output doesn't have values of empty lists but may lack of element types as
        keys, by default False
    Returns
    -------
    dict[str, pd.Index]
        elements that are connected to the group
    """
    def element_type_for_switch_et(element_type):
        return element_type[0] if element_type != "trafo3w" else "t3"

    # bus->bus (for find_buses_only_from_buses) and bus->other elements connections
    group_buses = group_element_index(net, index, "bus")
    if respect_in_service:
        group_buses = net.bus.loc[group_buses].index[net.bus.in_service.loc[group_buses]]
    connected = get_connected_elements_dict(
        net, group_buses, element_types=element_types, respect_switches=respect_switches,
        respect_in_service=respect_in_service, include_empty_lists=include_empty_lists)

    # switch -> branch connections
    group_sw = group_element_index(net, index, "bus")
    sw_bra_types = ["line", "trafo", "trafo3w"]
    for et in sw_bra_types:
        if et not in element_types:
            continue
        elms = net.switch.element.loc[group_sw].loc[net.switch.et.loc[group_sw] == element_type_for_switch_et(et)]
        if respect_in_service:
            elms = net[et].loc[elms].index[net[et].in_service.loc[elms]]
        connected[et] = set(connected[et]) | set(elms)

    # branch -> switch connections
    if "switch" in element_types:
        conn_sw = set(connected.get("switch", set()))
        for branch_type in sw_bra_types:
            if net[branch_type].shape[0]:
                group_branches = group_element_index(net, index, branch_type)
                if respect_in_service:
                    group_branches = group_branches.intersection(
                        net[branch_type].index[net[branch_type].in_service])
                conn_sw |= set(net.switch.index[
                    (net.switch.et == element_type_for_switch_et(branch_type)) &
                    net.switch.element.isin(group_branches)])
        connected["switch"] = conn_sw

    if not find_buses_only_from_buses:
        bed = branch_element_bus_dict(include_switch=True)
        bed["switch"].append("element")
        branch_types = list(bed.keys())
        conn_buses = set()
        bed.update({tpl[0]: [tpl[1]] for tpl in element_bus_tuples(branch_elements=False)})

        for row in net.group.loc[index].itertuples():
            et = row.element_type
            if et == "bus":
                continue
            for bus_col in bed[et]:
                if et == "switch" and bus_col == "element":
                    bed_buses = net[et][bus_col].loc[net.switch.index[
                        net.switch.et == "b"].intersection(row.element)]
                else:
                    bed_buses = net[et][bus_col].loc[row.element]
                if respect_in_service and "in_service" in net[et].columns:
                    bed_buses = bed_buses.loc[net[et].in_service.loc[row.element]]
                if respect_switches:
                    if et == "switch":
                        bed_buses = bed_buses.loc[net.switch.closed.loc[bed_buses.index]]
                    elif et in branch_types:
                        closed = np.ones(bed_buses.shape[0], dtype=bool)
                        switches = net.switch[["bus", "closed"]].loc[
                            (net.switch.et == element_type_for_switch_et(et)) &
                            net.switch.element.isin(row.element)]
                        if switches.shape[0]:
                            if switches.bus.duplicated().any():
                                raise ValueError(
                                    f"There are multiple {et} switches connecting the same "
                                    "element and bus. respect_switches is not possible due to "
                                    "multiple possible values.")
                            switches = switches.set_index("bus").closed
                            in_sw = bed_buses.isin(switches.index).values
                            closed[in_sw] = switches.loc[bed_buses.loc[in_sw]]
                            bed_buses = bed_buses.loc[closed]
                conn_buses |= set(bed_buses.values)
        if respect_in_service:
            conn_buses = pd.Index(conn_buses)
            conn_buses = conn_buses[net.bus.in_service.loc[conn_buses].values]
        connected["bus"] = conn_buses

    connected = {et: sorted(pd.Index(conn).difference(group_element_index(net, index, et))) for et,
                 conn in connected.items()}
    if include_empty_lists:
        return connected
    else:
        return {key: val for key, val in connected.items() if len(val)}


if __name__ == "__main__":
    import pandapower as pp

    net = create_empty_network()
    pp.create_buses(net, 3, 10)
    pp.create_gens(net, [0]*5, [10]*5)
    pp.create_group(net, ["bus", "gen"], [[2, 1], [1, 2]], name="hello")
    pp.create_group(net, "bus", [[0]], name="hello")
    print(net.group)
    print(pp.count_group_elements(net, 0))
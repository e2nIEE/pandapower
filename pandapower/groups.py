# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import numpy as np
import pandas as pd
import pandas.testing as pdt
import uuid

from pandapower.auxiliary import ensure_iterability, log_to_level
from pandapower.create import create_empty_network, _group_parameter_list, _set_multiple_entries, \
    _check_elements_existence, create_group
from pandapower.toolbox.power_factor import signing_system_value
from pandapower.toolbox.element_selection import branch_element_bus_dict, element_bus_tuples, pp_elements, get_connected_elements_dict
from pandapower.toolbox.result_info import res_power_columns

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

"""
Notes
-----
Using different reference_columns for the same group and element_type is not supported.
See check_unique_group_rows()
"""


# ====================================
# CREATE AND DELETE GROUPS
# ====================================

# pandapower.create.create_group


# pandapower.create.create_group_from_dict


def drop_group(net, index):
    """Drops the group of given index.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the group which should be dropped
    """
    net.group = net.group.drop(index)


def drop_group_and_elements(net, index):
    """
    Drops all elements of the group and in net.group the group itself.
    """
    # functions like drop_trafos, drop_lines, drop_buses are not considered since all elements
    # should be included in elements_dict
    for et in net.group.loc[[index], "element_type"].tolist():
        idx = group_element_index(net, index, et)
        net[et].drop(idx.intersection(net[et].index), inplace=True)
        res_et = "res_" + et
        if res_et in net.keys() and net[res_et].shape[0]:
            net[res_et].drop(net[res_et].index.intersection(idx), inplace=True)
    net.group = net.group.drop(index)


# ====================================
# ADAPT GROUP MEMBERS
# ====================================


def append_to_group(*args, **kwargs):
    msg = "The name of the function append_to_group() is deprecated with pp.version >= 2.12. " + \
        "Use attach_to_group() instead."
    raise DeprecationWarning(msg)


def attach_to_groups(net, index, element_types, elements, reference_columns=None):
    """Appends the groups by the elements given.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : list[int]
        index of the considered group
    element_types : str or list of strings
        defines, together with 'elements', which net elements belong to the group
    elements : list of list of indices
        defines, together with 'element_types', which net elements belong to the group
    reference_columns : string or list of strings, optional
        If given, the elements_dict should not refer to DataFrames index but to another column.
        It is highly relevant that the reference_column exists in all DataFrames of the grouped
        elements and have the same dtype, by default None

    See Also
    --------
    attach_to_group
    """
    for idx in index:
        attach_to_group(net, idx, element_types, elements, reference_columns=reference_columns)


def attach_to_group(net, index, element_types, elements, reference_columns=None,
                    take_existing_reference_columns=True):
    """Appends the group by the elements given.

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
        If given, the elements_dict should not refer to DataFrames index but to another column.
        It is highly relevant that the reference_column exists in all DataFrames of the grouped
        elements and have the same dtype, by default False
    take_existing_reference_columns : bool, optional
        Determines the behavior if the given reference_columns does not match the reference_column
        existing in net.group. If True, the existing reference_column is taken for both. If False,
        an error is raised.
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

            # handle different reference_column
            existing_rc = net.group.reference_column.loc[group_et].at[index]
            if existing_rc != rc:
                if take_existing_reference_columns:
                    temp_gr = create_group(net, [et], [elm], reference_columns=rc)
                    set_group_reference_column(net, temp_gr, existing_rc, element_type=et)
                    elm = net.group.element.at[temp_gr]
                    net.group = net.group.drop(temp_gr)
                else:
                    raise UserWarning(
                        f"The reference column of existing group {index} for element "
                        f"type '{et}' and of the elements to append differ. Use "
                        "set_group_reference_column() to change the reference column of net.group "
                        "before, or pass appropriate data to attach_to_group().")

            # append
            prev_elm = net.group.element.loc[group_et].at[index]
            prev_elm = [prev_elm] if isinstance(prev_elm, str) or not hasattr(
                prev_elm, "__iter__") else list(prev_elm)
            net.group.iat[np.arange(len(group_et), dtype=int)[group_et][0],
                          net.group.columns.get_loc("element")] = \
                prev_elm + list(pd.Index(elm).difference(pd.Index(prev_elm)))

        # --- prepare adding new rows to net.group (because no other elements of element type et
        # --- already belong to the group)
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


def drop_from_group(*args, **kwargs):
    msg = ("The name of the function drop_from_group() is deprecated with pp.version >= 2.12. "
           "Use detach_from_group() instead.")
    raise DeprecationWarning(msg)


def detach_from_group(net, index, element_type, element_index):
    """Detaches elements from the group with the given group index 'index'.
    No errors are raised if elements are passed to be drop from groups which alread don't have these
    elements as members.
    A reverse function is available -> pp.group.attach_to_group().

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        Index of the group from which the element should be dropped
    element_type : str
        The element type of which elements should be dropped from the group(s), e.g. "bus"
    element_index : int or list of integers
        indices of the elements which should be dropped from the group
    """
    detach_from_groups(net, element_type, element_index, index=index)


def drop_from_groups(*args, **kwargs):
    msg = ("The name of the function drop_from_groups() is deprecated with pp.version >= 2.12. "
           "Use detach_from_groups() instead.")
    raise DeprecationWarning(msg)


def detach_from_groups(net, element_type, element_index, index=None):
    """Detaches elements from one or multiple groups, defined by 'index'.
    No errors are raised if elements are passed to be dropped from groups which alread don't have
    these elements as members.
    A reverse function is available -> pp.group.attach_to_group().

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_type : str
        The element type of which elements should be dropped from the group(s), e.g. "bus"
    element_index : int or list of integers
        indices of the elements which should be dropped from the group
    index : int or list of integers, optional
        Indices of the group(s) from which the element should be dropped. If None, the elements are
        dropped from all groups, by default None
    """
    if index is None:
        index = net.group.index
    element_index = pd.Index(ensure_iterability(element_index), dtype=np.int64)

    to_check = np.isin(net.group.index.values, index)
    to_check &= net.group.element_type.values == element_type
    keep = np.ones(net.group.shape[0], dtype=bool)

    for i in np.arange(len(to_check), dtype=np.int64)[to_check]:
        rc = net.group.reference_column.iat[i]
        if rc is None or pd.isnull(rc):
            net.group.element.iat[i] = pd.Index(net.group.element.iat[i]).difference(
                element_index).tolist()
        else:
            net.group.element.iat[i] = pd.Index(net.group.element.iat[i]).difference(pd.Index(
                net[element_type][rc].loc[element_index.intersection(
                    net[element_type].index)])).tolist()

        if not len(net.group.element.iat[i]):
            keep[i] = False
    net.group = net.group.loc[keep]


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

    See also
    --------
    group_index : recursive function
    """
    names = net.group.name.loc[[index]]
    if len(set(names)) != 1 and not pd.isnull(names).all():
        raise ValueError(f"group {index} has different values in net.group.name.loc[index]")
    return names.values[0]


def group_index(net, name):
    """Returns the index of the group named as given

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    name : str
        name of the requested group

    See also
    --------
    group_name : recursive function
    """
    index = net.group.index[net.group.name == name]
    if len(set(index)) != 1:
        raise ValueError(f"group {name} has multiple indices: {index}")
    return index[0]


def group_element_index(net, index, element_type):
    """Returns the indices of the elements of the group in the element table net[element_type]. This
    function considers net.group.reference_column.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        Index of the group
    element_type : str
        name of the element table to which the returned indices of the elements of the group belong
        to

    Returns
    -------
    pd.Index
        indices of the elements of the group in the element table net[element_type]
    """
    if element_type not in net.group.loc[[index], "element_type"].values:
        return pd.Index([], dtype=np.int64)

    row = group_row(net, index, element_type)
    element = row.at["element"]
    reference_column = row.at["reference_column"]

    if reference_column is None or pd.isnull(reference_column):
        return pd.Index(element, dtype=np.int64)

    return net[element_type].index[net[element_type][reference_column].isin(element)]


def group_row(net, index, element_type):
    """Returns the row which consists the data of the requested group index and element type.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the group
    element_type : str
        element type (defines which row of the data of the group should be returned)

    Returns
    -------
    pandas.Series
        data of net.group, defined by the index of the group and the element type

    Raises
    ------
    KeyError
        Now row exist for the requested group and element type
    ValueError
        Multiple rows exist for the requested group and element type
    """
    group_df = net.group.loc[[index]].set_index("element_type")
    try:
        row = group_df.loc[element_type]
    except KeyError:
        raise KeyError(f"Group {index} has no {element_type}s.")
    if isinstance(row, pd.Series):
        return row
    elif isinstance(row, pd.DataFrame):
        raise ValueError(f"Multiple {element_type} rows for group {index}")
    else:
        raise ValueError(f"Returning row {element_type} for group {index} failed.")


def _single_element_index(element_index):
    if not hasattr(element_index, "__iter__"):
        return [element_index], True
    else:
        return list(element_index), False


def isin_group(net, element_type, element_index, index=None, drop_empty_lines=True):
    """Returns whether elements are in group(s).

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_type : str
        element type of the elements to be found in the groups, e.g. "gen" or "load"
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
    element_index, single_element = _single_element_index(element_index)
    if index is None:
        index = list(set(net.group.index))
    else:
        index = ensure_iterability(index)

    ensure_lists_in_group_element_column(net, drop_empty_lines=drop_empty_lines)

    member_idx = pd.Index([], dtype=np.int64)
    for idx in index:
        member_idx = member_idx.union(group_element_index(net, idx, element_type))

    isin = np.isin(element_index, member_idx)

    if single_element:
        return isin[0]
    else:
        return isin


def element_associated_groups(net, element_type, element_index, return_empties=True,
                              drop_empty_lines=True):
    """Returns to which groups given elements belong to.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_type : str
        element type of the elements to be found in the groups, e.g. "gen" or "load"
    element_index : str
        indices of the element table which should be found in the groups
    return_empties : bool, optional
        whether entries with an empty list of assiciated groups should be returned, by default True
    drop_empty_lines : bool, optional
        This parameter decides whether empty entries should be removed (the complete row in
        net.group), by default True

    Returns
    -------
    dict[int, list[int]]
        for each element index a list of associated group indices is returned as a dict
    """
    element_index, single_element = _single_element_index(element_index)
    index = list(set(net.group.index))
    ensure_lists_in_group_element_column(net, drop_empty_lines=drop_empty_lines)
    gr_et = net.group.loc[net.group.element_type == element_type]
    associated = pd.Series(dict.fromkeys(element_index, list()))
    for idx in gr_et.index:
        ass = pd.Index(element_index).intersection(group_element_index(net, idx, element_type))
        associated.loc[ass] = associated.loc[ass].apply(lambda x: x + [idx])
    if not return_empties:
        associated = associated.loc[associated.apply(len).astype(bool)]
    associated = associated.to_dict()
    if single_element:
        return associated[index[0]]
    else:
        return associated


def count_group_elements(net, index):
    """Returns a Series concluding the number of included elements in self.elements_dict

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group

    Returns
    -------
    pd.Series
        number of elements per element type existing in the group

    See also
    --------
    count_elements
    """
    return pd.Series({
        et: len(elm) if hasattr(elm, "__iter__") and not isinstance(elm, str) else 1 for
        et, elm in zip(*_get_lists_from_df(net.group.loc[[index]], ["element_type", "element"]))},
        dtype=np.int64)


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
            if len(group_element_index(net, index1, et).symmetric_difference(group_element_index(
                    net, index2, et))):
                return False
    return True

# =================================================
# FIX GROUP DATA
# =================================================


def check_unique_group_names(*args, **kwargs):
    msg = ("Function check_unique_group_names() is deprecated with pp.version >= 2.12. "
           "It is replaced by check_unique_group_rows() and the raise_ parameter defaults to True.")
    raise DeprecationWarning(msg)


def check_unique_group_rows(net, raise_error=True, log_level="warning"):
    """Checks whether all groups have unique names. raise_error decides whether duplicated names lead
    to error or log message.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    raise_error : bool, optional
        decides whether duplicated names lead to error or log message., by default False
    log_level : str, optional
        the level for logs, relevant if raise_error is False

    Notes
    -----
    Using different reference_columns for the same group and element_type is not supported.
    """
    df = net.group[["name", "element_type"]].reset_index()
    if df.duplicated().any():
        raise ValueError("There are multiple group rows with same index, name and element_type.")
    single_name_per_index = [len(names) == 1 for names in net.group.reset_index().groupby("index")[
        "name"].agg(set)]
    if not all(single_name_per_index):
        warn = "Groups with different names have the same index."
        if raise_error:
            raise UserWarning(warn)
        else:
            log_to_level(warn, logger, log_level)


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


def group_res_power_per_bus(net, index):
    """Returns active and reactive power consumptions of given group per bus

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        _description_

    Returns
    -------
    pd.DataFrame
        power consumption of the group per bus (index=bus, columns=["p_mw", "q_mvar"])

    Examples
    --------
    >>> import pandapower as pp
    >>> import pandapower.networks as nw
    >>> net = nw.create_cigre_network_mv(with_der="all")
    >>> pp.runpp(net)
    >>> gr_idx = pp.create_group(net, ["sgen", "line"], [[0, 1], [0, 1]], name="test group")
    >>> pp.group_res_power_per_bus(net, gr_idx)
                 p_mw        q_mvar
    bus
    1    2.953004e+00  1.328978e+00
    2   -3.641532e-14  4.618528e-14
    3   -2.875066e+00 -1.318864e+00
    4   -2.000000e-02  0.000000e+00
    """
    pq_sums = pd.DataFrame({'p_mw': float(), 'q_mvar': float()}, index=[])
    bra_ets = pp_elements(bus=False, bus_elements=False, other_elements=False)
    bra_ebd = branch_element_bus_dict()
    missing_res_idx = list()

    for et in net.group.loc[[index], "element_type"].tolist():
        if et in ["switch", "measurement", "bus", "bus_geodata", "line_geodata"]:
            continue
        gr_elm_idx = group_element_index(net, index, et)
        sides = [0] if et not in bra_ets else [0, 1] if et != "trafo3w" else [0, 1, 2]
        for side in sides:
            bus_col = "bus" if et not in bra_ets else bra_ebd[et][side]
            idx = net[f'res_{et}'].index.intersection(gr_elm_idx)
            if len(idx) != len(gr_elm_idx):
                missing_res_idx.append(et)
            cols = res_power_columns(et, side=side)
            pq_sum = pd.concat([net[et][bus_col].loc[idx], net[f'res_{et}'][cols].loc[idx]],
                               axis=1).groupby(bus_col).sum() * signing_system_value(et)
            pq_sum.index.name = "bus"  # needs to be set for branch elements
            if len(pq_sums.columns.difference(pq_sum.columns)):
                cols_repl = {old: "p_mw" if "p" in old else "q_mvar" for old in pq_sum.columns}
                pq_sum = pq_sum.rename(columns=cols_repl)
            pq_sums = pq_sums.add(pq_sum, fill_value=0)

    if len(missing_res_idx):
        logger.warning("The resulting power may be wrong since the result tables of these "
                        f"elements lack of indices: {missing_res_idx}.")
    return pq_sums


def group_res_p_mw(net, index):
    """Sums all result table values `p_mw` of group members.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    index : int
        index of the considered group

    Examples
    --------
    >>> import pandapower as pp
    >>> import pandapower.networks as nw
    >>> net = nw.create_cigre_network_mv(with_der="all")
    >>> pp.runpp(net)
    >>> gr_idx = pp.create_group(net, ["sgen", "line"], [[0, 1], [0, 1]], name="test group")
    >>> net.res_line.pl_mw.loc[[0, 1]].sum() - net.res_sgen.p_mw.loc[[0, 1]].sum()  # expected value
    0.057938
    >>> pp.group_res_p_mw(net, gr_idx)
    0.057938
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

    Examples
    --------
    >>> import pandapower as pp
    >>> import pandapower.networks as nw
    >>> net = nw.create_cigre_network_mv(with_der="all")
    >>> pp.runpp(net)
    >>> gr_idx = pp.create_group(net, ["sgen", "line"], [[0, 1], [0, 1]], name="test group")
    >>> net.res_line.ql_mvar.loc[[0, 1]].sum() - net.res_sgen.q_mvar.loc[[0, 1]].sum()  # expected value
    0.010114
    >>> pp.group_res_q_mvar(net, gr_idx)
    0.010114
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
            pos = np.arange(len(pos_bool), dtype=np.int64)[pos_bool][0]

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

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import numpy as np
import pandas as pd
import pandas.testing as pdt
import uuid

from pandapower.auxiliary import ensure_iterability
from pandapower.create import create_empty_network, _group_parameter_list, _set_multiple_entries
from pandapower.toolbox import pp_elements, group_element_index

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


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
    data = net.group.loc[index].set_index("element_type").at[element_type]
    element = data.at["element"]
    reference_column = data.at["reference_column"]

    if not hasattr(element, "__iter__") or isinstance(element, str):
        raise ValueError("Entries in net.group.element should be lists. You can try "
                         "ensure_lists_in_group_element_column() to fix this.")

    if reference_column is None:
        return np.isin(np.array(element), net[element_type].index.values)
    else:
        return np.isin(np.array(element), net[element_type][reference_column].values)


def isin_group(net, element_type, element_index, index=None):
    if not hasattr(index, "__iter__"):
        single_group = True
        index = [index]
    else:
        single_group = False

    TODO

    if single_group:
        return isin[0]
    else:
        return isin


def ensure_lists_in_group_element_column(net, drop_empty_lines=True):
    to_drop = np.zeros(net.group.shape[0], dtype=bool)
    for i in range(net.group.shape[0]):
        elm = net.group.element.iat[i]
        if hasattr(elm, "__iter__") and not isinstance(elm, str):
            net.group.element.iat[i] = list(elm)
            if not len(elm):
                to_drop[i] = True
        else:
            if elm is None:
                net.group.element.iat[i] = []
                to_drop[i] = True
            else:
                net.group.element.iat[i] = [elm]
    if drop_empty_lines:
        net.group.drop(to_drop, inplace=True)


def remove_not_existing_group_members(net, verbose=True):
    """Remove group members from net.group that do not exist in the elements tables.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    verbose : bool, optional
        Steers amount of logging messages, by default True
    """
    to_drop = np.zeros(net.group.shape[0], dtype=bool)
    not_existing_et = set()
    empty_et = set()
    for i in range(net.group.shape[0]):
        et = net.group.element_type.iat[i]
        if et not in net.keys() or not isinstance(net[et], pd.DataFrame):
            not_existing_et |= {et}
            to_drop[i] = True
        elif not net[et].shape[0]:
            empty_et |= {et}
            to_drop[i] = True
        else:
            not_exist_bool = ~group_entries_exist_in_element_table(
                net, net.group.index[i], net.group.element_type.iat[i])
            if np.all(not_exist_bool):
                to_drop[i] = True
                logger.info(f"net.group row {i} is dropped because no fitting elements exist in "
                            f"net[{net.group.element_type.iat[i]}].")
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
    net.group.drop(to_drop, inplace=True)


# def get_reference_column_type(net, index, element_type):
#     """Get the dtype of the reference column, which is basically:
#     net[element_type][reference_column].dtype

#     Parameters
#     ----------
#     net : pandapowerNet
#         pandapower net
#     index : int
#         Index of the group
#     element_type : str
#         The type of element for which the reference column type is requested, e.g. 'bus'

#     Returns
#     -------
#     Type of reference column, e.g. int, float or object
#     """
#     # Defines reference_column_type with respect to the net[elm].dtypes
#     reference_column = net.group.loc[index].set_index("element_type").at[
#         element_type, "reference_column"]

#     if reference_column is None:
#         return int
#     if reference_column not in net[element_type].columns:
#         raise ValueError(f"reference_column '{reference_column}' doesn't exist in net[{element_type}].")

#     if pd.api.types.is_integer_dtype(net[element_type][reference_column]):
#         return int
#     elif pd.api.types.is_numeric_dtype(net[element_type][reference_column]):
#         return float
#     else:  # pd.api.types.is_object_dtype(net[element_type][reference_column])
#         return object


def set_group_reference_column(net, index, reference_column, element_type=None):
    """
    Sets new reference_column value(s) to net.group and updates net.group.element.
    If self.elements_dict is not up-to-date, update_elements_dict() is needed before using this function.
    """
    if element_type is None:
        element_type = net.group.loc[index, "element_type"].tolist()
    else:
        element_type = ensure_iterability(element_type)

    dupl_elements = list()
    for et in element_type:

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
            pos_bool = (net.group.index == index) & (net.group.element_type == et)
            net.group.element.loc[pos_bool] = \
                pd.Index(net[et].loc[group_element_index(net, index, et), reference_column])
            net.group.reference_columns.loc[pos_bool] = reference_column

    if len(dupl_elements):
        raise ValueError(f"In net[*].{reference_column} are duplicated or nan values. "
                         f"* is placeholder for {dupl_elements}.")


def _get_dict_from_df(net, index, cols):
    return dict(zip(net.group.at[index, col].tolist() for col in cols))


def set_value_to_group(net, index, value, column, replace=True, append_column=True):
    """
    Sets the same value to the column of the element tables of all elements of the group.
    """
    for et, elm, rc in _get_dict_from_df(
            net, index, ["element_type", "element", "reference_columns"]).items():
        if append_column or column in net[et].columns:
            if rc is None:
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


def set_group_in_service(net, index):
    """
    Sets all elements of the group in service.
    """
    set_value_to_group(net, index, True, "in_service", replace=True, append_column=False)


def set_group_out_of_service(net, index):
    """
    Sets all elements of the group out of service.
    """
    set_value_to_group(net, index, False, "in_service", replace=True, append_column=False)


def return_group_as_net(net, index, keep_everything_else=False, **kwargs):
    """
    Returns a pandapower net consisting of the members of this group.
    """
    if keep_everything_else:
        group_net = copy.deepcopy(net)
        for et in pp_elements():
            if group_net[et].shape[0]:
                keep = group_element_index(net, index, et)
                group_net[et].drop(group_net[et].index.difference(keep), inplace=True)
        if len(set(net.group.index)) > 1:
            logger.warning("The returned net includes further groups which should probably be "
                            "updated.")
    else:
        group_net = create_empty_network(
            name=net.group.loc[index, "name"].values[0], f_hz=net.f_hz, sn_mva=net.sn_mva,
            add_stdtypes=kwargs.get("add_stdtypes", True))
        group_net["group"] = net.group.loc[index]
        for et in net.group.loc[index, "element_type"].tolist():
            idx = group_element_index(net, index, et)
            group_net[et] = net[et].loc[idx]
    return group_net


def append_to_group(net, index, element_types, elements, reference_columns=None):
    """
    Appends the group by the elements given as dict of indices in 'elements_dict_to_append'.
    If net is given, elements_dict_to_append is checked and updated by _update_elements_dict().
    """
    if index not in net.group.index:
        raise ValueError(
            f"{index} is not in net.group.index. Correct index or used create_group() instead.")

    name = net.group.name.loc[index].values[0]
    element_types, elements, reference_columns = _group_parameter_list(
        element_types, elements, reference_columns)

    complete_new = {col: list() for col in ["name", "element_type", "element", "reference_column"]}
    for et, elm, rc in zip(element_types, elements, reference_columns):

        if et in net.group.element_type.loc[index]:
            # element entry of existing is appended
            pass
        else:
            # prepare adding new rows to net.group (because no other elements of element type et
            # already belongs to the group)
            complete_new["name"].append(name)
            complete_new["element_type"].append(et)
            complete_new["element"].append(elm)
            complete_new["reference_column"].append(rc)

    if len(complete_new["name"]):
        # add new rows to net.group
        _set_multiple_entries(net, "group", index, **complete_new)
        # net.group = pd.concat([net.group, pd.DataFrame(complete_new)])
        net.group.sort_index(inplace=True)


def groups_equal(net, index1, index2, **kwargs):
    """ Returns a boolean whether both group are equal. """
    df1 = net.group.loc[index1].set_index("name")
    df2 = net.group.loc[index2].set_index("name")
    return pdt.assert_frame_equal(df1, df2, **kwargs)


def compare_group_elements(net, index1, index2):
    """ allow_cross_reference_column_comparison """
    ensure_lists_in_group_element_column(net)
    et1 = net.group.loc[index1, "element_type"].tolist()
    et2 = net.group.loc[index2, "element_type"].tolist()
    if len(et1) != len(set(et1)):
        raise ValueError(f"The Group of index {index1} has duplicated element_types.")
    if len(et2) != len(set(et2)):
        raise ValueError(f"The Group of index {index2} has duplicated element_types.")
    if set(et1) != set(et2):
        return False

    gr1 = net.group.loc[index1].set_index("element_type")
    gr2 = net.group.loc[index2].set_index("element_type")
    for et in et1:
        if gr1.reference_column.at[et] == gr2.reference_column.at[et]:
            if len(pd.Index(gr1.element.at[et]).symmetric_difference(gr2.element.at[et])):
                return False
        else:
            if len(group_element_index(net, index1, et).symmetric_difference(group_element_index(net, index2, et))):
                return False
    return True


def group_element_lists(net, index):
    return tuple([net.group.loc[index, col].tolist() for col in [
        "element_type", "element", "reference_column"]])


def count_group_elements(net, index):
    """ Returns a Series concluding the number of included elements in self.elements_dict """ # TODO
    return pd.Series({
        et: len(elm) if hasattr(elm, "__iter__") and not isinstance(elm, str) else 1 for
        et, elm in net.group[["element_type", "element"]].loc[index].todict().items()}, dtype=int)


def _sum_powers(net, index, formula_character, unit):
    power = 0.
    missing_res_idx = list()
    no_power_column_found = list()
    no_res_table_found = list()
    for et in net.group.loc[index, "element_type"].tolist():
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
    return _sum_powers(net, index, "p", "mw")


def group_res_q_mvar(net, index):
    return _sum_powers(net, index, "q", "mvar")


def check_unique_group_names(net, raise_=False):
    df = net.group[["name", "element_type"]].reset_index()
    if df.duplicated().any():
        raise ValueError("There are multiple groups with same index, name and element_type.")
    del df["element_type"]
    if df.duplicated().any():
        warn = "There are multiple groups with same index and name."
        if raise_:
            raise UserWarning(warn)
        else:
            logger.warning(warn)


if __name__ == "__main__":
    import pandapower as pp

    net = create_empty_network()
    pp.create_group(net, ["bus", "gen"], [[10, 12], [1, 2]], name="hello")
    pp.create_group(net, "bus", [[10]], name="hello")
    print(net.group)

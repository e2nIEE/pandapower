# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
import warnings
from typing import Iterable

import pandas as pd
from numpy import nan, isnan, arange, isin, any as np_any, all as np_all, float64, intersect1d, unique as uni, c_
import numpy.typing as npt
from pandas import isnull
from pandas.api.types import is_object_dtype

from pandapower.auxiliary import (
    ADict,
    pandapowerNet,
    get_free_id,
    _preserve_dtypes,
    ensure_iterability,
    empty_defaults_per_dtype,
)
from pandapower.plotting.geo import _is_valid_number
from pandapower.pp_types import Int
from pandapower.network_structure import get_structure_dict


logger = logging.getLogger(__name__)


def add_column_to_df(net: ADict, table_name: str, column_name: str) -> None:
    if table_name in net and column_name in net[table_name]:
        return
    # Add Table:
    net_struct_dict = get_structure_dict()
    if table_name not in net:
        if table_name not in net_struct_dict:
            raise ValueError(f"Table {table_name} has no definition in network structure.")
        net[table_name] = pd.DataFrame(columns=net_struct_dict[table_name].keys(), dtype=net_struct_dict[table_name].values())
    # Add Optional Column:
    dtype = get_structure_dict(False)[table_name][column_name]
    net[table_name][column_name] = pd.Series(dtype=dtype)


def _geodata_to_geo_series(data: Iterable[tuple[float, float]] | tuple[int, int], nr_buses: int) -> list[str]:
    geo = []
    for g in data:
        if isinstance(g, tuple):
            if len(g) != 2:
                raise ValueError("geodata tuples must be of length 2")
            elif not _is_valid_number(g[0]):
                raise UserWarning("geodata x must be a valid number")
            elif not _is_valid_number(g[1]):
                raise UserWarning("geodata y must be a valid number")
            else:
                x, y = g
                geo.append(f'{{"coordinates": [{x}, {y}], "type": "Point"}}')
        else:
            raise ValueError("geodata must be iterable of tuples of (x, y) coordinates")
    if len(geo) == 1:
        geo = [geo[0]] * nr_buses
    if len(geo) != nr_buses:
        raise ValueError("geodata must be a single point or have the same length as nr_buses")
    return geo


def _group_parameter_list(element_types, elements, reference_columns):
    """
    Ensures that element_types, elements and reference_columns are iterables with same lengths.
    """
    if isinstance(elements, str) or not hasattr(elements, "__iter__"):
        raise ValueError("'elements' should be a list of list of indices.")
    if any(isinstance(el, str) or not hasattr(el, "__iter__") for el in elements):
        raise ValueError("In 'elements' each item should be a list of element indices.")
    element_types = ensure_iterability(element_types, len_=len(elements))
    reference_columns = ensure_iterability(reference_columns, len_=len(elements))
    return element_types, elements, reference_columns


def _check_elements_existence(net, element_types, elements, reference_columns):
    """
    Raises UserWarnings if elements does not exist in net.
    """
    for et, elm, rc in zip(element_types, elements, reference_columns):
        if et not in net:
            raise UserWarning(f"Cannot create a group with elements of type '{et}', because net[{et}] does not exist.")
        if rc is None or pd.isnull(rc):
            diff = pd.Index(elm).difference(net[et].index)
        else:
            if rc not in net[et].columns:
                raise UserWarning(
                    f"Cannot create a group with reference column '{rc}' for elements"
                    f" of type '{et}', because net[{et}][{rc}] does not exist."
                )
            diff = pd.Index(elm).difference(pd.Index(net[et][rc]))
        if len(diff):
            raise UserWarning(f"Cannot create group with {et} members {diff}.")


def _get_index_with_check(net: pandapowerNet, table: str, index: Int | None, name: str | None = None) -> Int:
    if name is None:
        name = table
    if index is None:
        index = get_free_id(net[table])
    if index in net[table].index:
        raise UserWarning(f"A {name} with the id {index} already exists")
    return index


def _cost_existance_check(net, element, et, power_type=None):
    if power_type is None:
        return (
            bool(net.poly_cost.shape[0])
            and np_any((net.poly_cost.element == element).values & (net.poly_cost.et == et).values)
        ) or (
            bool(net.pwl_cost.shape[0])
            and np_any((net.pwl_cost.element == element).values & (net.pwl_cost.et == et).values)
        )
    else:
        return (
            bool(net.poly_cost.shape[0])
            and np_any((net.poly_cost.element == element).values & (net.poly_cost.et == et).values)
        ) or (
            bool(net.pwl_cost.shape[0])
            and np_any(
                (net.pwl_cost.element == element).values
                & (net.pwl_cost.et == et).values
                & (net.pwl_cost.power_type == power_type).values
            )
        )


def _costs_existance_check(net, elements, et, power_type=None):
    if isinstance(et, str) and (power_type is None or isinstance(power_type, str)):
        poly_exist = (net.poly_cost.element.isin(elements)).values & (net.poly_cost.et == et).values
        pwl_exist = (net.pwl_cost.element.isin(elements)).values & (net.pwl_cost.et == et).values
        if isinstance(power_type, str):
            pwl_exist &= (net.pwl_cost.power_type == power_type).values
        return sum(poly_exist) & sum(pwl_exist)

    else:
        cols = ["element", "et"]
        poly_df = pd.concat([net.poly_cost[cols], pd.DataFrame(c_[elements, et], columns=cols)])
        if power_type is None:
            pwl_df = pd.concat([net.pwl_cost[cols], pd.DataFrame(c_[elements, et], columns=cols)])
        else:
            cols.append("power_type")
            pwl_df = pd.concat(
                [net.pwl_cost[cols], pd.DataFrame(c_[elements, et, [power_type] * len(elements)], columns=cols)]
            )
        return poly_df.duplicated().sum() + pwl_df.duplicated().sum()


def _get_multiple_index_with_check(net, table, index, number, name=None):
    if index is None:
        bid = get_free_id(net[table])
        return arange(bid, bid + number, 1)
    u, c = uni(index, return_counts=True)
    if np_any(c > 1):
        raise UserWarning("Passed indexes %s exist multiple times" % (u[c > 1]))
    intersect = intersect1d(index, net[table].index.values)
    if len(intersect) > 0:
        if name is None:
            name = table.capitalize() + "s"
        raise UserWarning("%s with indexes %s already exist." % (name, intersect))
    return index


def _check_element(net, element_index, element="bus"):
    if element not in net:
        raise UserWarning(f"Node table {element} does not exist")
    if element_index not in net[element].index.values:
        raise UserWarning("Cannot attach to %s %s, %s does not exist" % (element, element_index, element_index))


def _check_multiple_elements(net, element_indices, element="bus", name="buses"):
    if element not in net:
        raise UserWarning(f"Node table {element} does not exist")
    if np_any(~isin(element_indices, net[element].index.values)):
        node_not_exist = set(element_indices) - set(net[element].index.values)
        raise UserWarning(f"Cannot attach to {name} {node_not_exist}, they do not exist")


def _check_branch_element(net, element_name, index, from_node, to_node, node_name="bus", plural="es"):
    if node_name not in net:
        raise UserWarning(f"Node table {node_name} does not exist")
    missing_nodes = {from_node, to_node} - set(net[node_name].index.values)
    if len(missing_nodes) > 0:
        raise UserWarning(
            "%s %d tries to attach to non-existing %s(%s) %s"
            % (element_name.capitalize(), index, node_name, plural, missing_nodes)
        )


def _check_multiple_branch_elements(net, from_nodes, to_nodes, element_name, node_name="bus", plural="es"):
    if node_name not in net:
        raise UserWarning(f"Node table {node_name} does not exist")
    all_nodes = set(from_nodes) | set(to_nodes)
    node_not_exist = all_nodes - set(net[node_name].index)
    if len(node_not_exist) > 0:
        raise UserWarning(
            "%s trying to attach to non existing %s%s %s" % (element_name, node_name, plural, node_not_exist)
        )


def _not_nan(value, all_=True):
    if isinstance(value, str):
        return True
    elif hasattr(value, "__iter__"):
        if all_:
            if is_object_dtype(value):
                return not all(isnull(value))
            return not all(isnan(value))
        else:
            if is_object_dtype(value):
                return not any(isnull(value))
            return not any(isnan(value))
    else:
        try:
            return pd.notna(value)
        except TypeError:
            return True


def _try_astype(df, column, dtyp):
    try:
        df[column] = df[column].astype(dtyp)
    except TypeError:
        pass


def _set_value_if_not_nan(net, index, value, column, element_type, dtype=float64, default_val=nan):
    """Sets the given value to the dataframe net[element_type]. If the value is nan, default_val
    is assumed if this is not nan.
    If the value is not nan and the column does not exist already, the column is created and filled
    by default_val.

    Parameters
    ----------
    net : pp.pandapowerNet
        pp net
    index : int
        index of the element to get a value
    value : Any
        value to be set
    column : str
        name of column
    element_type : str
        element_type type, e.g. "gen"
    dtype : Any, optional
        e.g. float64, "Int64", bool_, ..., by default float64
    default_val : Any, optional
        default value to be set if the column exists and value is nan and if the column does not
        exist and the value is not nan, by default nan

    See Also
    --------
    _add_to_entries_if_not_nan
    """
    column_exists = column in net[element_type].columns
    dtype = get_structure_dict(required_only=False)[element_type][column]
    if _not_nan(value):
        if not column_exists:
            net[element_type].loc[:, column] = pd.Series(data=default_val, index=net[element_type].index)
        _try_astype(net[element_type], column, dtype)
        net[element_type].at[index, column] = value
    elif column_exists:
        if _not_nan(default_val):
            net[element_type].at[index, column] = default_val
        _try_astype(net[element_type], column, dtype)


def _add_to_entries_if_not_nan(net, element_type, entries, index, column, values, dtype=float64, default_val=nan):
    """

    See Also
    --------
    _set_value_if_not_nan
    """
    column_exists = column in net[element_type].columns
    dtype = get_structure_dict(required_only=False)[element_type][column]
    if _not_nan(values):
        entries[column] = pd.Series(values, index=index)
        if _not_nan(default_val):
            entries[column] = entries[column].fillna(default_val)
        _try_astype(entries, column, dtype)
    elif column_exists:
        entries[column] = pd.Series(data=default_val, index=index)
        _try_astype(entries, column, dtype)


def _branch_geodata(geodata: Iterable[list[float] | tuple[float, float]]) -> list[list[float]]:
    geo: list[list[float]] = []
    for x, y in geodata:
        if (not _is_valid_number(x)) | (not _is_valid_number(y)):
            raise ValueError("geodata contains invalid values")
        geo.append([x, y])
    return geo


def _add_branch_geodata(net: pandapowerNet, geodata, index, table="line"):
    if geodata:
        if not isinstance(geodata, (list, tuple)):
            raise ValueError("geodata needs to be list or tuple")
        geodata = f'{{"coordinates": {_branch_geodata(geodata)}, "type": "LineString"}}'
    else:
        geodata = pd.NA
    net[table].loc[index, "geo"] = geodata
    net[table]["geo"] = net[table]["geo"].astype(get_structure_dict(required_only=False)[table]["geo"])


def _add_multiple_branch_geodata(net, geodata, index, table="line"):
    dtype = get_structure_dict(required_only=False)[table]["geo"]
    if not geodata:
        net[table].loc[index, "geo"] = pd.Series(data=[pd.NA]*len(net[table]), index=net[table].index, dtype=dtype)
        return
    dtypes = net[table].dtypes
    if hasattr(geodata, "__iter__") and all(isinstance(g, tuple) and len(g) == 2 for g in geodata):
        # geodata is a single Iterable of coordinate tuples
        geo = [[x, y] for x, y in geodata]
        series = [f'{{"coordinates": {geo}, "type": "LineString"}}'] * len(index)
    elif hasattr(geodata, "__iter__") and all(isinstance(g, Iterable) for g in geodata):
        # geodata is Iterable of coordinate tuples
        geo = [[[x, y] for x, y in g] for g in geodata]
        series = [f'{{"coordinates": {g}, "type": "LineString"}}' for g in geo]
    else:
        raise ValueError(
            "geodata must be an Iterable of Iterable of coordinate tuples or an Iterable of coordinate tuples"
        )

    net[table].loc[index, "geo"] = pd.Series(series, index=index, dtype=dtype)

    _preserve_dtypes(net[table], dtypes)


def _set_entries(net, table, index, preserve_dtypes=True, entries: dict | None = None):
    if entries is None:
        return

    dtypes = None
    if preserve_dtypes:
        # only get dtypes of columns that are set and that are already present in the table
        dtypes = net[table][intersect1d(net[table].columns, list(entries))].dtypes

    for col, val in entries.items():
        if not pd.isna(val):  # TODO: questionable
            net[table].at[index, col] = val
            try:
                dtype = get_structure_dict(required_only=False)[table][col]
                if dtype == bool and net[table][col].isna().any(): # default value for bool entries # TODO: check if wanted behaviour
                    net[table][col] = net[table][col].astype(pd.BooleanDtype()).fillna(False)
                net[table][col] = net[table][col].astype(dtype)
            except KeyError:
                pass

    # and preserve dtypes
    if preserve_dtypes:
        _preserve_dtypes(net[table], dtypes)


def _check_entry(val, index):
    if isinstance(val, pd.Series) and not np_all(isin(val.index, index)):
        return val.values
    elif isinstance(val, set) and len(val) == len(index):
        return list(val)
    return val


def _set_multiple_entries(
    net: pandapowerNet,
    table: str,
    index: npt.ArrayLike | list[Int] | pd.Index,
    preserve_dtypes: bool | None = True,
    defaults_to_fill: list[tuple] | None = None,
    entries: dict | None = None,
):
    if entries is None:
        return

    dtypes = None
    if preserve_dtypes:
        # store dtypes
        dtypes = net[table].dtypes

    entries = {k: _check_entry(v, index) for k, v in entries.items()}

    dd = pd.DataFrame(index=index, columns=net[table].columns)
    dd = dd.assign(**entries)

    dtype_dict = get_structure_dict(required_only=False)[table]

    # defaults_to_fill needed due to pandas bug https://github.com/pandas-dev/pandas/issues/46662:
    # concat adds new bool columns as object dtype -> fix it by setting default value to net[table]
    if defaults_to_fill is not None:
        for col, val in defaults_to_fill:
            if col in dd.columns and col not in net[table].columns:
                net[table][col] = val
                try:
                    net[table][col] = net[table][col].astype(dtype_dict[col])
                except KeyError:
                    pass

    # set correct dtypes
    for col in dd.columns:
        if col in dtype_dict:
            dd[col] = dd[col].astype(dtype_dict[col])

    # extend the table by the frame we just created
    if len(net[table]):
        net[table] = pd.concat([net[table], dd[dd.columns[~dd.isnull().all()]]], sort=False)
    else:
        dd_columns = dd.columns[~dd.isnull().all()]
        complete_columns = list(net[table].columns) + list(dd_columns.difference(net[table].columns))
        empty_dict = {
            key: empty_defaults_per_dtype(dtype)
            for key, dtype in net[table][net[table].columns.difference(dd_columns)].dtypes.to_dict().items()
        }
        net[table] = dd[dd_columns].assign(**empty_dict)[complete_columns]

    # and preserve dtypes
    if preserve_dtypes:
        _preserve_dtypes(net[table], dtypes)


def _set_const_percent_values(const_percent_values_list, kwargs_input):
    const_percent_values_default_initials = all(value == 0 for value in const_percent_values_list)
    if (
        "const_z_percent" in kwargs_input and "const_i_percent" in kwargs_input
    ) and const_percent_values_default_initials:
        const_z_p_percent = kwargs_input["const_z_percent"]
        const_z_q_percent = kwargs_input["const_z_percent"]
        const_i_p_percent = kwargs_input["const_i_percent"]
        const_i_q_percent = kwargs_input["const_i_percent"]
        del kwargs_input["const_z_percent"]
        del kwargs_input["const_i_percent"]
        msg = (
            "Parameters const_z_percent and const_i_percent will be deprecated in further "
            "pandapower version. For now the values were transfered in "
            "const_z_p_percent and const_i_p_percent for you."
        )
        warnings.warn(msg, DeprecationWarning)
        return const_z_p_percent, const_i_p_percent, const_z_q_percent, const_i_q_percent, kwargs_input
    elif ("const_z_percent" in kwargs_input or "const_i_percent" in kwargs_input) and (
        const_percent_values_default_initials == False
    ):
        raise UserWarning("Definition of voltage dependecies is faulty, please check the parameters again.")
    elif ("const_z_percent" in kwargs_input or "const_i_percent" not in kwargs_input) or (
        "const_z_percent" not in kwargs_input or "const_i_percent" in kwargs_input
    ):
        raise UserWarning("Definition of voltage dependecies is faulty, please check the parameters again.")

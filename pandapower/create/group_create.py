# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging

import numpy as np

from pandapower import pandapowerNet
from pandapower.create._utils import (
    _check_elements_existence,
    _get_index_with_check,
    _group_parameter_list,
    _set_multiple_entries,
)

logger = logging.getLogger(__name__)


def create_group(
    net: pandapowerNet,
    element_types,
    element_indices,
    name: str = "",
    reference_columns=None,
    index: int | None = None,
    **kwargs,
):
    """Add a new group to net['group'] dataframe.

    Attention
    ::

        If you declare a group but forget to declare all connected elements although
        you wants to (e.g. declaring lines but forgetting to mention the connected switches),
        you may get problems after using drop_elements_and_group() or other functions.
        There are different pandapower toolbox functions which may help you to define
        'elements_dict', such as get_connecting_branches(),
        get_inner_branches(), get_connecting_elements_dict().

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_types : str or list of strings
        defines, together with 'elements', which net elements belong to the group
    element_indices : list of list of indices
        defines, together with 'element_types', which net elements belong to the group
    name : str, optional
        name of the group, by default ""
    reference_columns : string or list of strings, optional
        If given, the elements_dict should
        not refer to DataFrames index but to another column. It is highly relevant that the
        reference_column exists in all DataFrames of the grouped elements and have the same dtype,
        by default None
    index : int, optional
        index for the dataframe net.group, by default None

    EXAMPLES:
        >>> create_group(net, ["bus", "gen"], [[10, 12], [1, 2]])
        >>> create_group(net, ["bus", "gen"], [["Berlin", "Paris"], ["Wind_1", "Nuclear1"]], reference_columns="name")
    """
    element_types, element_indices, reference_columns = _group_parameter_list(
        element_types, element_indices, reference_columns
    )

    _check_elements_existence(net, element_types, element_indices, reference_columns)

    index = np.array([_get_index_with_check(net, "group", index)] * len(element_types), dtype=np.int64)

    entries = {
        "name": name,
        "element_type": element_types,
        "element_index": element_indices,
        "reference_column": reference_columns,
        **kwargs,
    }
    _set_multiple_entries(net, "group", index, entries=entries)
    net.group.loc[net.group.reference_column == "", "reference_column"] = None  # overwrite

    return index[0]


def create_group_from_dict(
    net, elements_dict, name: str = "", reference_column=None, index: int | None = None, **kwargs
):
    """Wrapper function of create_group()."""
    return create_group(
        net,
        elements_dict.keys(),
        elements_dict.values(),
        name=name,
        reference_columns=reference_column,
        index=index,
        **kwargs,
    )

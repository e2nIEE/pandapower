# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

from pandapower.create import _get_multiple_index_with_check
from pandapower.create import _set_multiple_entries

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def select_group(net, group_name):
    return net.group.query("name==@group_name").groupby("element").agg(
        {"element_index": lambda x: x.values,
         "element_index_column": "first"})


def set_value(net, group_name, variable, value, replace=True, append_column=True,
              missing_column_error=True):
    group_tab = select_group(net, group_name)
    for element, element_index, element_index_column in group_tab.itertuples():
        if pd.isnull(element_index_column):
            if not append_column and variable not in net[element]:
                if missing_column_error:
                    raise UserWarning(f"{variable} not in net.{element}")
                else:
                    continue
            if not replace and np.any(~pd.isnull(net[element].loc[element_index, variable])):
                raise UserWarning(f"values already in net.{element}.{variable}")
            net[element].loc[element_index, variable] = value


def set_in_service(net, group_name):
    set_value(net, group_name=group_name, variable="in_service", value=True, replace=True,
              append_column=False, missing_column_error=False)


def set_out_of_service(net, group_name):
    set_value(net, group_name=group_name, variable="in_service", value=False, replace=True,
              append_column=False, missing_column_error=False)


def add_elements(net, group_name, element, element_index=None, element_index_column=np.nan):
    if element_index is None:
        element_index = net[element].index.values \
            if pd.isnull(element_index_column) else net[element][element_index_column].values

    index = _get_multiple_index_with_check(net, "group", None, len(element_index))

    entries = {"name": group_name, "element": element, "element_index": element_index,
               "element_index_column": element_index_column}

    _set_multiple_entries(net, "group", index, **entries)

    return index
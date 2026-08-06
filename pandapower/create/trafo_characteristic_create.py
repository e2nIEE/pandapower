# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandas as pd

from pandapower.network import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create.utils import add_column_to_df, _get_index_with_check


def _create_trafo_characteristic_table(net: pandapowerNet):
    if "trafo_characteristic_table" in net:
        return
    net["trafo_characteristic_table"] = pd.DataFrame({
        'id_characteristic': pd.Series(dtype='int64'),
        'step': pd.Series(dtype='int64'),
        'voltage_ratio': pd.Series(dtype='float64'),
        'angle_deg': pd.Series(dtype='float64'),
        'vk_percent': pd.Series(dtype='float64'),
        'vkr_percent': pd.Series(dtype='float64'),
        'vk_hv_percent': pd.Series(dtype='float64'),
        'vkr_hv_percent': pd.Series(dtype='float64'),
        'vk_mv_percent': pd.Series(dtype='float64'),
        'vkr_mv_percent': pd.Series(dtype='float64'),
        'vk_lv_percent': pd.Series(dtype='float64'),
        'vkr_lv_percent': pd.Series(dtype='float64'),
    }).set_index(['id_characteristic', 'step'])


def create_trafo_characteristic(net: pandapowerNet, characteristic_values: dict, index: Int | None = None) -> Int:
    """
    Add a complete set of step‑rows for a single tap characteristic.
    for a trafo characteristic

    Parameters:
        net: the pandapower network where the trafo_characteristic_table entry is to be created.
        index: The index that will be common to all rows being inserted.
        characteristic_values: Dictionary whose keys are column names (including ``'step'``) and values are equal‑length
            lists/arrays.

    Returns:
        The index added to the trafo_characteristic_table
    """
    # add_column_to_df(net, "trafo_characteristic_table", "voltage_ratio")  # ensure table is created #TODO: create pandera schema for trafo_characteristic_table
    if "trafo_characteristic_table" not in net:
        _create_trafo_characteristic_table(net)
    index = _get_index_with_check(net, "trafo_characteristic_table", index)

    # TODO: ensure that trafo and trafo3w values are not mixed (vk_percent not together with vk_hv_percent)

    # ensure step, voltage_ratio, angle_deg are present in characteristic_values
    if ('step' not in characteristic_values or
            'voltage_ratio' not in characteristic_values or
            'angle_deg' not in characteristic_values):
        raise ValueError("create_tap_dependency is missing some values")

    # Turn the dict of lists into a DataFrame
    new_rows = pd.DataFrame(characteristic_values)  # shape: (n_steps, n_cols)
    # Tag every row with the index
    new_rows['id_characteristic'] = index
    # Set the MultiIndex that matches `df`
    new_rows = new_rows.set_index(['id_characteristic', 'step'])
    # Make sure we have the same columns as the target DF (order irrelevant here)
    for col in net.trafo_characteristic_table.columns:
        if col not in new_rows.columns:
            new_rows[col] = float('nan')
    # Keep only the columns that exist in the target (re‑order to match)
    new_rows = new_rows[net.trafo_characteristic_table.columns]
    # if angle_deg is nan it should be set to 0
    new_rows.loc[:, "angle_deg"] = new_rows.angle_deg.fillna(0.0)
    # Append (concat) to the original DataFrame
    net.trafo_characteristic_table = pd.concat([net.trafo_characteristic_table, new_rows])

    return index

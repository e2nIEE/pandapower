# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Creation of buses and lines from the Lines and Tielines sheets of the JAO excel file."""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from pandapower.converter.jao._common import _drop_duplicates_and_join_TSO, _get_bus_idx, _raise_for_missing_buses
from pandapower.create import create_buses, create_lines_from_parameters
from pandapower.network import pandapowerNet

logger = logging.getLogger(__name__)


def _create_buses_from_line_data(net: pandapowerNet, data: dict[str, pd.DataFrame]) -> None:
    """Creates buses to the pandapower net using information from the lines and tielines sheets
    (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame data: data provided by the excel file which will be corrected
    """
    bus_df_empty = pd.DataFrame({"name": "", "vn_kv": 0.0, "TSO": ""}, index=[])
    bus_df = deepcopy(bus_df_empty)
    for key in ["Lines", "Tielines"]:
        for subst in ["Substation_1", "Substation_2"]:
            data_col_tuples = [(subst, "Full_name"), (None, "Voltage_level(kV)"), (None, "TSO")]
            to_add = data[key].loc[:, data_col_tuples].set_axis(bus_df.columns, axis="columns")
            if len(bus_df):
                bus_df = pd.concat([bus_df, to_add])
            else:
                bus_df = to_add
    bus_df = _drop_duplicates_and_join_TSO(bus_df)
    new_bus_idx = create_buses(net, len(bus_df), vn_kv=bus_df.vn_kv, name=bus_df.name, zone=bus_df.TSO)
    if any(new_bus_idx != bus_df.index):  # type: ignore[arg-type]
        raise AssertionError("Not all values of now_bus_idx are identical to bus_df.index")


def _create_lines(net: pandapowerNet, data: dict[str, pd.DataFrame], max_i_ka_fillna: float) -> None:
    """
    Creates lines to the pandapower net using information from the lines and tielines sheets (excel file).

    :param pandapowerNet net: net to be filled by buses
    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    :param float|int max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.
    """

    bus_idx = _get_bus_idx(net)

    for key in ["Lines", "Tielines"]:
        length_km = data[key][("Electrical Parameters", "Length_(km)")].to_numpy(dtype=float, copy=True)
        zero_length = np.isclose(length_km, 0)
        no_length = np.isnan(length_km)
        if sum(zero_length) or sum(no_length):  # type: ignore[call-overload]
            logger.warning(
                f"According to given data, {sum(zero_length)} {key.lower()} have zero "  # type: ignore[call-overload]
                f"length and {sum(zero_length)} {key.lower()} have no length data. "  # type: ignore[call-overload]
                "Both types of wrong data are replaced by 1 km."
            )
            length_km[zero_length | no_length] = 1
        vn_kvs = data[key].loc[:, (None, "Voltage_level(kV)")].values
        keys_1 = list(zip(data[key].loc[:, ("Substation_1", "Full_name")].values, vn_kvs))  # type: ignore[arg-type]
        keys_2 = list(zip(data[key].loc[:, ("Substation_2", "Full_name")].values, vn_kvs))  # type: ignore[arg-type]
        _raise_for_missing_buses(bus_idx, keys_1 + keys_2, key)

        _ = create_lines_from_parameters(
            net,
            bus_idx.loc[keys_1].values,
            bus_idx.loc[keys_2].values,
            length_km,
            data[key][("Electrical Parameters", "Resistance_R(Ω)")].values / length_km,  # type: ignore[operator]
            data[key][("Electrical Parameters", "Reactance_X(Ω)")].values / length_km,  # type: ignore[operator]
            data[key][("Electrical Parameters", "Susceptance_B(μS)")].values / length_km,  # type: ignore[operator]
            data[key][("Maximum Current Imax (A)", "Fixed")].fillna(max_i_ka_fillna * 1e3).values / 1e3,  # type: ignore[operator]
            name=data[key].xs("NE_name", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            EIC_Code=data[key].xs("EIC_Code", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            TSO=data[key].xs("TSO", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            Comment=data[key].xs("Comment", level=1, axis=1).values[:, 0],  # type: ignore[call-overload]
            Tieline=key == "Tielines",
        )

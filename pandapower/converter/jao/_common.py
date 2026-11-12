# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Helpers shared by several parts of the JAO converter."""

import logging

import pandas as pd

from pandapower.network import pandapowerNet

logger = logging.getLogger(__name__)


def _get_bus_idx(net: pandapowerNet) -> pd.Series:
    """Returns a Series mapping (bus name, vn_kv) to the bus index."""
    return net.bus[["name", "vn_kv"]].rename_axis("index").reset_index().set_index(["name", "vn_kv"])["index"]


def _drop_duplicates_and_join_TSO(bus_df: pd.DataFrame) -> pd.DataFrame:
    """Keeps one row per (name, vn_kv) and joins the TSO strings of duplicated rows."""
    bus_df = bus_df.drop_duplicates(ignore_index=True)
    # just keep one bus per name and vn_kv. If there are multiple buses of different TSOs, join the
    # TSO strings:
    bus_df = bus_df.groupby(["name", "vn_kv"], as_index=False).agg({"TSO": lambda x: "/".join(x)})  # type: ignore[misc]
    if bus_df.duplicated(["name", "vn_kv"]).any():
        raise AssertionError("bus_df contains duplicate names with identical vn_kv")
    return bus_df


def _get_float_column(df, col_tuple, fill=0):
    """Returns a float column, replacing non-breaking spaces and missing values by ``fill``."""
    series = df.loc[:, col_tuple].copy()
    series.loc[series == "\xa0"] = fill
    return series.astype(float).fillna(fill)


def _raise_for_missing_buses(bus_idx: pd.Series, endpoints: list[tuple], sheet: str) -> None:
    """Raises a descriptive error if line endpoints do not match any bus (name, vn_kv)."""
    missing = sorted({endpoint for endpoint in endpoints if endpoint not in bus_idx.index}, key=str)
    if missing:
        raise ValueError(
            f"{len(missing)} endpoint(s) of sheet {sheet!r} have no matching bus (name, vn_kv): "
            + ", ".join(f"{name!r} at {vn} kV" for name, vn in missing)
            + ". Rows with missing substation name or voltage level cannot be converted."
        )

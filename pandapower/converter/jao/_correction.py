# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Correction of the raw data read from the JAO excel file."""

import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from pandapower.converter.jao._locnames import (
    _locname_display_names,
    _multi_str_repl,
    _replace_locnames,
    _replace_substrings,
)

logger = logging.getLogger(__name__)


def _data_correction(data: dict[str, pd.DataFrame], html_str: str | None, max_i_ka_fillna: float) -> str | None:
    """
    Corrects input data in particular with regard to obvious weaknesses in the data provided,
    such as inconsistent spellings and missing necessary information

    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    :param str|None html_str: data provided by the html file which will be corrected
    :param float|int max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.

    :return: corrected html_str
    :rtype: str
    """
    # --- Line and Tieline data ---------------------------
    for key in ["Lines", "Tielines"]:
        # --- correct column names
        cols = data[key].columns.to_frame().reset_index(drop=True)
        cols.loc[cols[1] == "Voltage_level(kV)", 0] = None
        cols.loc[cols[1] == "Comment", 0] = None
        cols.loc[cols[0].str.startswith("Unnamed:").astype(bool), 0] = None
        cols.loc[cols[1] == "Length_(km)", 0] = "Electrical Parameters"  # might be wrong in
        # Tielines otherwise
        data[key].columns = pd.MultiIndex.from_arrays(cols.values.T.tolist())

        # --- correct comma separation and cast to floats
        data[key][("Maximum Current Imax (A)", "Fixed")] = (
            data[key][("Maximum Current Imax (A)", "Fixed")]
            .replace("\xa0", max_i_ka_fillna * 1e3)
            .replace("-", max_i_ka_fillna * 1e3)
            .replace(" ", max_i_ka_fillna * 1e3)
            .infer_objects(copy=False)
        )
        col_names = [
            ("Electrical Parameters", col_level1)
            for col_level1 in ["Length_(km)", "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)", "Length_(km)"]
        ] + [("Maximum Current Imax (A)", "Fixed")]
        _float_col_comma_correction(data, key, col_names)

    # --- repair incomplete line data
    _repair_incomplete_line_data(data)

    # --- consolidate inconsistent location name spellings
    # Build the raw -> display name mapping from the bus names in the Lines and Tielines sheets.
    # Names that only differ in case, spacing, punctuation, accents or special characters are
    # merged onto a single display spelling.
    bus_name_series: list[pd.Series] = []
    for key in ["Lines", "Tielines"]:
        for subst in ["Substation_1", "Substation_2"]:
            bus_name_series.append(data[key][(subst, "Full_name")])
    bus_names: pd.Series = pd.concat(bus_name_series, ignore_index=True)
    locname_mapping = _locname_display_names(bus_names)
    rename_locnames = sorted(
        ((raw, disp) for raw, disp in locname_mapping.items() if raw != disp), key=lambda x: len(x[0]), reverse=True
    )

    for key in ["Lines", "Tielines"]:
        data[key][(None, "NE_name")] = _replace_substrings(data[key][(None, "NE_name")], rename_locnames)
        for loc_name in [("Substation_1", "Full_name"), ("Substation_2", "Full_name")]:
            data[key][loc_name] = _replace_locnames(data[key][loc_name], locname_mapping)
    html_str = _multi_str_repl(html_str, rename_locnames)  # type: ignore[arg-type]

    # --- Transformer data --------------------------------
    key = "Transformers"

    # --- correct current data (some files hold non-breaking spaces) and drop rows without any
    # --- electrical data, which cannot be converted
    current_cols = [("Maximum Current Imax (A) primary", col) for col in ("Min", "Max", "Fixed")]
    for col in current_cols:
        data[key][col] = pd.to_numeric(data[key][col].replace("\xa0", np.nan), errors="coerce")
    vn_cols = [("Voltage_level(kV)", "Primary"), ("Voltage_level(kV)", "Secondary")]
    no_data = data[key][vn_cols].isna().all(axis=1) & data[key][current_cols].isna().all(axis=1)
    if no_data.any():
        logger.warning(
            f"Transformers: dropping {no_data.sum()} row(s) without electrical data: {list(data[key].index[no_data])}"
        )
        data[key] = data[key].drop(index=data[key].index[no_data])

    # --- fix Locations
    loc_name = ("Location", "Full Name")
    data[key][loc_name] = _replace_substrings(data[key][loc_name], rename_locnames)

    # --- fix data in nonnull_taps
    taps = (
        data[key].loc[:, ("Phase Shifting Properties", "Taps used for RAO")].fillna("").astype(str).str.replace(" ", "")
    )
    nonnull = taps.apply(len).astype(bool)
    nonnull_taps = taps.loc[nonnull]
    surrounded = nonnull_taps.str.startswith("<") & nonnull_taps.str.endswith(">")
    nonnull_taps.loc[surrounded] = nonnull_taps.loc[surrounded].str[1:-1]
    slash_sep = (~nonnull_taps.str.contains(";")) & nonnull_taps.str.contains("/")
    nonnull_taps.loc[slash_sep] = nonnull_taps.loc[slash_sep].str.replace("/", ";")
    nonnull_taps.loc[nonnull_taps == "0"] = "0;0"
    data[key].loc[nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = nonnull_taps
    data[key].loc[~nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = "0;0"

    # --- phase shifter with double info
    cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]  # type: ignore[assignment]
    for col in cols:
        if is_object_dtype(data[key].loc[:, ("Phase Shifting Properties", col)]):
            tr_double = data[key].index[
                data[key].loc[:, ("Phase Shifting Properties", col)].str.contains("/").fillna(0).astype(bool)
            ]
            data[key].loc[tr_double, ("Phase Shifting Properties", col)] = (
                data[key]
                .loc[tr_double, ("Phase Shifting Properties", col)]
                .str.split("/", expand=True)[1]
                .str.replace(",", ".")
                .astype(float)
                .values
            )  # take second info and correct
            # separation: , -> .

    return html_str


def _float_col_comma_correction(data: dict[str, pd.DataFrame], key: str, col_names: list):
    """Casts columns to floats, interpreting commas as decimal separators."""
    for col_name in col_names:
        data[key][col_name] = pd.to_numeric(data[key][col_name].astype(str).str.replace(",", "."), errors="coerce")


def _repair_incomplete_line_data(data: dict[str, pd.DataFrame]) -> None:
    """Drops Lines/Tielines rows without any data and fills missing substation names and voltage
    levels from unambiguous information of the same file.

    Some JAO releases contain malformed rows, e.g. rows that only repeat an NE_name without any
    substation or electrical data, lines with one missing substation name, or lines with a missing
    voltage level. Rows without any information are dropped, the other gaps are filled if the
    information is unambiguous.

    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    """
    s1, s2 = ("Substation_1", "Full_name"), ("Substation_2", "Full_name")
    vn_col, len_col, ne_col = (None, "Voltage_level(kV)"), ("Electrical Parameters", "Length_(km)"), (None, "NE_name")

    # --- names and voltage levels that are known from complete rows of all sheets
    known_names: set[str] = set()
    name_vns: dict[str, set] = {}
    for key in ("Lines", "Tielines"):
        for subst in (s1, s2):
            pairs = data[key][[subst, vn_col]].dropna()
            known_names |= set(pairs[subst].astype(str))
            for name, vn in pairs.itertuples(index=False):
                name_vns.setdefault(name, set()).add(vn)

    for key in ("Lines", "Tielines"):
        df = data[key].copy()

        empty = df[[s1, s2, vn_col, len_col]].isna().all(axis=1)
        if empty.any():
            logger.warning(f"{key}: dropping {empty.sum()} row(s) without any data: {list(df.index[empty])}")
            df = df.drop(index=df.index[empty])

        for idx in df.index[df[s1].isna() | df[s2].isna()]:
            missing = s1 if pd.isna(df.at[idx, s1]) else s2
            other = df.at[idx, s2] if missing == s1 else df.at[idx, s1]
            tokens = [token.strip().upper() for token in re.sub(r"\s*\(.*?\)", "", str(df.at[idx, ne_col])).split("-")]
            candidates = [
                name
                for name in known_names
                if name != other
                and any(token == name.upper() or token.startswith(name.upper() + " ") for token in tokens)
            ]
            if len(candidates) == 1:
                df.at[idx, missing] = candidates[0]
                logger.warning(
                    f"{key} row {idx}: missing {missing[0]} name set to {candidates[0]!r} "
                    f"(derived from NE_name {df.at[idx, ne_col]!r})"
                )

        for idx in df.index[df[vn_col].isna()]:
            vns = {
                next(iter(name_vns[name]))
                for name in (df.at[idx, s1], df.at[idx, s2])
                if len(name_vns.get(name, ())) == 1
            }
            if len(vns) == 1:
                df.at[idx, vn_col] = vns.pop()
                logger.warning(f"{key} row {idx}: missing Voltage_level(kV) set to {df.at[idx, vn_col]}")

        data[key] = df

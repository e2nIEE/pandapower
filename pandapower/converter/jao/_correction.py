# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Correction of the raw data read from the JAO excel file."""

import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from pandapower.converter.jao._locnames import (
    _locname_display_names,
    _multi_str_repl,
    _normalize_locname,
    _replace_locnames,
    _replace_substrings,
)

logger = logging.getLogger(__name__)

# documented second-level header variants (see doc/converter/jao-converter.csv) -> canonical label
_vn_header_variants = {"Voltage_level (kV)": "Voltage_level(kV)", "Voltage_level [kV]": "Voltage_level(kV)"}


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
        cols[1] = cols[1].replace(_vn_header_variants)
        cols.loc[cols[1] == "Voltage_level(kV)", 0] = None
        cols.loc[cols[1] == "Comment", 0] = None
        cols.loc[cols[0].str.startswith("Unnamed:").astype(bool), 0] = None
        cols.loc[cols[1] == "Length_(km)", 0] = "Electrical Parameters"  # might be wrong in
        # Tielines otherwise
        data[key].columns = pd.MultiIndex.from_arrays(cols.values.T.tolist())

        # --- correct comma separation and cast to floats
        col_names = [
            ("Electrical Parameters", col_level1)
            for col_level1 in ["Length_(km)", "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)", "Length_(km)"]
        ] + [("Maximum Current Imax (A)", "Fixed")]
        _float_col_comma_correction(data, key, col_names)
        imax_col = ("Maximum Current Imax (A)", "Fixed")
        data[key][imax_col] = data[key][imax_col].fillna(max_i_ka_fillna * 1e3)

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
    taps_col = ("Phase Shifting Properties", "Taps used for RAO")
    taps = data[key].loc[:, taps_col].fillna("").astype(str).str.replace(" ", "")
    blank = taps == ""
    surrounded = taps.str.startswith("<") & taps.str.endswith(">")
    taps.loc[surrounded] = taps.loc[surrounded].str[1:-1]
    slash_sep = (~taps.str.contains(";")) & taps.str.contains("/")
    taps.loc[slash_sep] = taps.loc[slash_sep].str.replace("/", ";")
    taps.loc[taps == "0"] = "0;0"
    data[key][taps_col] = taps.mask(blank, "0;0")

    # --- phase shifter with double info
    cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]  # type: ignore[assignment]
    for col in cols:
        col_key = ("Phase Shifting Properties", col)
        series = data[key].loc[:, col_key]
        if is_object_dtype(series) or is_string_dtype(series):
            tr_double = series.index[series.str.contains("/", na=False)]
            if len(tr_double) > 0:
                # take second info and correct the decimal separator; the column is set positionally
                # so that also a string dtype column can take the numeric values
                corrected = series.astype(object).copy()
                second_info = series.loc[tr_double].astype(str).str.split("/", expand=True)[1]
                corrected.loc[tr_double] = second_info.str.replace(",", ".").astype(float).values
                data[key].isetitem(data[key].columns.get_loc(col_key), corrected)  # type: ignore[arg-type]

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
    voltage level. Rows without any substation, voltage level or electrical data are dropped, the
    other gaps are filled if the information is unambiguous.

    :param dict[str, pd.DataFrame] data: data provided by the excel file which will be corrected
    """
    s1, s2 = ("Substation_1", "Full_name"), ("Substation_2", "Full_name")
    vn_col, ne_col = (None, "Voltage_level(kV)"), (None, "NE_name")
    electrical_cols = [
        ("Electrical Parameters", col)
        for col in ("Length_(km)", "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)")
    ]

    # --- names and voltage levels that are known from complete rows of all sheets
    known_names: set[str] = set()
    name_vns: dict[str, set] = {}
    for key in ("Lines", "Tielines"):
        for subst in (s1, s2):
            pairs = data[key][[subst, vn_col]].dropna()
            known_names |= set(pairs[subst].astype(str))
            for name, vn in pairs.itertuples(index=False):
                name_vns.setdefault(_normalize_locname(name), set()).add(vn)
    known_keys = {_normalize_locname(name): name for name in sorted(known_names)}

    for key in ("Lines", "Tielines"):
        df = data[key].copy()

        empty = df[[s1, s2, vn_col, *electrical_cols]].isna().all(axis=1)
        if empty.any():
            logger.warning(f"{key}: dropping {empty.sum()} row(s) without any data: {list(df.index[empty])}")
            df = df.drop(index=df.index[empty])

        for idx in df.index[df[s1].isna() | df[s2].isna()]:
            missing = s1 if pd.isna(df.at[idx, s1]) else s2
            other = df.at[idx, s2] if missing == s1 else df.at[idx, s1]
            other_location = _normalize_locname(other)
            ne_key = _normalize_locname(df.at[idx, ne_col])
            # match complete known names as whole words; hyphens and parentheses inside a name are not separators
            candidate_names = {
                name_key: known_keys[name_key]
                for name_key in known_keys
                if name_key
                and name_key != other_location
                and re.search(rf"(?<!\S){re.escape(name_key)}(?!\S)", ne_key) is not None
            }
            if len(candidate_names) == 1:
                candidate = next(iter(candidate_names.values()))
                if not (is_object_dtype(df[missing]) or is_string_dtype(df[missing])):
                    df[missing] = df[missing].astype(object)
                df.at[idx, missing] = candidate
                logger.warning(
                    f"{key} row {idx}: missing {missing[0]} name set to {candidate!r} "
                    f"(derived from NE_name {df.at[idx, ne_col]!r})"
                )

        for idx in df.index[df[vn_col].isna()]:
            lookup_keys = [_normalize_locname(name) for name in (df.at[idx, s1], df.at[idx, s2]) if not pd.isna(name)]
            vns = {next(iter(name_vns[key])) for key in lookup_keys if len(name_vns.get(key, ())) == 1}
            if len(vns) == 1:
                df.at[idx, vn_col] = vns.pop()
                logger.warning(f"{key} row {idx}: missing Voltage_level(kV) set to {df.at[idx, vn_col]}")

        data[key] = df

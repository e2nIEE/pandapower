# -*- coding: utf-8 -*

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from copy import deepcopy
import os
import json
from functools import reduce
import numpy as np
import re
import difflib
import unicodedata
import pandas as pd
from pandas.api.types import is_integer_dtype, is_object_dtype
from pandapower.io_utils import pandapowerNet
from pandapower.create import create_empty_network, create_buses, create_lines_from_parameters, \
    create_transformers_from_parameters
from pandapower.topology import create_nxgraph, connected_components
from pandapower.plotting import set_line_geodata_from_bus_geodata
from pandapower.toolbox import drop_buses, fuse_buses
from pandapower.converter.jao.utils import ColumnFuzzyMatchingUtils, NameNormalizationUtils, MiscUtils
import logging

logger = logging.getLogger(__name__)
# String constants
ELECTRICAL_PARAMETER_STR = 'Electrical Parameters'
PHASE_SHIFT_PROPERTIES_STR = 'Phase Shifting Properties'
VOLTAGE_LEVEL_STR = 'voltage level kv'
LENGTH_STR = 'Length_(km)'
RESISTANCE_STR = 'Resistance_R(Ω)'
FULL_NAME_STR = 'Full Name'
TAPS_STR = 'Taps used for RAO'
REACTANCE_STR = 'Reactance_X(Ω)'
TSO_1_STR = 'TSO 1'
TSO_2_STR = 'TSO 2'


def from_jao(excel_file_path: str,
             html_file_path: str | None,
             extend_data_for_grid_group_connections: bool,
             drop_grid_groups_islands: bool = False,
             apply_data_correction: bool = True,
             max_i_ka_fillna: float | int = 999,
             **kwargs) -> pandapowerNet:
    """
    Convert a JAO Core EHV static grid model into a pandapowerNet.

    Overview:
      - Reads Excel multi-sheet data (Lines, Tielines, Transformers).
      - Optionally reads HTML to extract geodata for lines.
      - Optionally applies robust data correction and name normalization.
      - Creates buses, lines, and transformers in pandapower.
      - Optionally invents synthetic connections between grid groups to reduce islanding.
      - Optionally drops islanded grid groups based on size or supply condition.
      - Attaches geodata to buses and lines when possible.


    Note
    ----
    This module deliberately includes robust fallback heuristics and fuzzy matching logic to
    handle real-world inconsistencies in published data and thus may trade strictness for
    practical usability.

    Parameters
    ----------
    excel_file_path : str
        Path to the Excel file (typically contains sheets: "Lines", "Tielines", "Transformers").
        A MultiIndex header (2 levels) is expected; variations are matched via fuzzy logic.
    html_file_path : str | None
        Optional path to an HTML file that contains embedded map/geodata for lines.
        Pass None to skip geodata extraction.
    extend_data_for_grid_group_connections : bool
        If True, attempts to connect islanded grid groups by:
          - Inserting representative transformers between same-location buses of different voltage levels.
          - Fusing buses with the same base name and same voltage but in different groups.
          - Fusing some special-case close buses.
    drop_grid_groups_islands : bool, optional
        If True, drops islanded grid groups determined by `min_bus_number` in kwargs (default 6).
        Special modes: min_bus_number can be "max" (keep only the largest group) or "unsupplied"
        (drop groups without slack generation).
    apply_data_correction : bool, optional
        If True, apply correction routines:
          - Comprehensive rename normalization across sheets.
          - Numeric conversions and column harmonization.
          - Minor cleanup for tap settings and shifter columns.
    max_i_ka_fillna : float | int, optional
        Fallback (in kA) for missing/invalid Imax data (lines/transformers).
        Use np.nan to avoid filling. Default is 999 (treated as 999 kA).

    Additional Parameters (via kwargs)
    ----------------------------------
    minimal_trafo_invention : bool, optional
        When connecting grid groups, if True, stop adding transformers as soon as no islands remain.
        Note: Not applied for release version 5 or 6 (value ignored).
    min_bus_number : Union[int, str], optional
        For drop_grid_groups_islands:
          - int: drop groups smaller than this
          - "max": keep only the largest group
          - "unsupplied": drop groups without any slack generator/element
    rel_deviation_threshold_for_trafo_bus_creation : float, optional
        When matching transformer voltage to buses by location, if the nearest bus voltage
        deviates more than this fraction (default 0.2), create a new bus instead.
    log_rel_vn_deviation : float, optional
        Log-warning threshold for voltage deviation (default 0.12).
    sn_mva : float, optional
        System base apparent power (MVA) for the pandapower net.

    Returns
    -------
    pandapowerNet
        The constructed pandapower network with buses, lines, transformers, and possibly geodata.

    Raises
    ------
    KeyError
        If essential columns cannot be found even via fuzzy matching.
    ValueError
        If transformer locations cannot be matched to buses (after robust normalization).
    json.JSONDecodeError
        If the HTML geodata JSON part cannot be parsed (caught and logged; conversion proceeds).

    Examples
    --------
    >>> from pathlib import Path
    >>> import os
    >>> from pandapower.converter import from_jao
    >>> net = pp.converter.from_jao()
    >>> home = str(Path.home())
    >>> excel_file_path = os.path.join(home, "desktop", "202409_Core Static Grid Mode_6th release")
    >>> html_file_path = os.path.join(home, "desktop", "2024-09-13_Core_SGM_publication.html")
    >>> net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
    """

    # --- read data
    data = pd.read_excel(excel_file_path, sheet_name=None, header=[0, 1])
    _ = NameNormalizationUtils.report_problematic_names_after_normalization(data)
    if html_file_path is not None:
        with open(html_file_path, mode='r', encoding=kwargs.get("encoding", "utf-8")) as f:
            html_str = f.read()
    else:
        html_str = ""

    # --- manipulate data / data corrections
    if apply_data_correction:
        html_str = _data_correction(data, html_str, max_i_ka_fillna)

    # --- parse html_str to line_geo_data
    line_geo_data = None
    if html_str:
        try:
            line_geo_data = _parse_html_str(html_str)
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            logger.error(f"html data were ignored due to this error:\n{e}")

    # --- create the pandapower net
    net = create_empty_network(name=os.path.splitext(os.path.basename(excel_file_path))[0],
                               **{key: val for key, val in kwargs.items() if key == "sn_mva"})
    _create_buses_from_line_data(net, data)
    _create_lines(net, data, max_i_ka_fillna)
    _create_transformers_and_buses(net, data, **kwargs)

    # --- invent connections between grid groups
    if extend_data_for_grid_group_connections:
        _invent_connections_between_grid_groups(net, **kwargs)

    # --- drop islanded grid groups
    if drop_grid_groups_islands:
        drop_islanded_grid_groups(net, kwargs.get("min_bus_number", 6))

    # --- add geo data to buses and lines
    if line_geo_data is not None:
        _add_bus_geo(net, line_geo_data)
        set_line_geodata_from_bus_geodata(net)

    return net


# ==================================================================================================
# Data correction pipeline
# ==================================================================================================

def _data_correction(
        data: dict[str, pd.DataFrame],
        html_str: str | None,
        max_i_ka_fillna: float | int) -> str | None:
    """
    Apply corrections and normalizations to Excel and HTML data before building the network.

    Corrections include:
      - Compute rename rules from cross-sheet analysis and apply them to Lines/Tielines/Transformers
        location columns and to HTML geodata (string replace).
      - Harmonize known column naming variants under consistent top-level and level-1 labels
        (e.g., unify "Full Name" -> "Full_name", "Voltage_level [kV]" -> "Voltage_level(kV)").
      - Ensure a (None, "TSO") column in Lines/Tielines when only side-specific TSO columns exist.
      - Numeric coercions for key electrical parameters (Length, R, X, B) and Imax.
      - Clean up tap strings and duplicate shifter data.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Excel sheets dict; modified in-place.
    html_str : str | None
        HTML geodata raw text; rename rules are applied to embedded names/EICs via find/replace.
    max_i_ka_fillna : float | int
        Value to fill missing or invalid maximum currents (Imax) in lines and transformers.
        Pass np.nan to disable filling.

    Returns
    -------
    str | None
        Corrected HTML string (same value if None was provided).

    Notes
    -----
    - Additional filtering is applied to rename rules based on string similarity >= 0.8 and to
      avoid mapping between two names that already exist in bus set.
    - Uses robust fuzzy matching for detection of R/X/B columns as well as TSO columns where needed.
    """
    # old name -> new name
    combined = NameNormalizationUtils.report_problematic_names_after_normalization(data)
    rename_locnames = NameNormalizationUtils.generate_rename_locnames_from_combined(data, combined)
    filtered_rename_locnames = []
    bus_location_names = NameNormalizationUtils.collect_bus_location_names(data)
    for old, new in rename_locnames:
        if old in bus_location_names and new in bus_location_names:
            continue
        similarity = difflib.SequenceMatcher(None, old.lower(), new.lower()).ratio()
        if similarity < 0.8:
            continue
        filtered_rename_locnames.append((old, new))
    rename_locnames = filtered_rename_locnames

    # --- Lines/Tielines: Spaltenrobustheit + Datentyp-Korrekturen
    for key in ["Lines", "Tielines"]:
        df = data[key]
        voltage_str = 'Voltage_level(kV)'
        # MultiIndex-Korrektur: bekannte Varianten unter einen Top-Level (None) heben
        cols = df.columns.to_frame(index=False)
        # harmonisiere zweite Ebene (Spaltennamen)
        replace_map = {
            FULL_NAME_STR: "Full_name",
            "Short Name": "Short_name",
            "Susceptance_B (µS)": "Susceptance_B(μS)",
            "Voltage_level (kV)": voltage_str,
            "Voltage_level [kV]": voltage_str,
        }
        cols.iloc[:, 1] = cols.iloc[:, 1].replace(replace_map)
        # setze Top-Level = None für diese Felder
        cols.loc[cols.iloc[:, 1].isin([voltage_str, "Comment"]), cols.columns[0]] = None
        cols.loc[cols.iloc[:, 0].astype(str).str.startswith("Unnamed:"), cols.columns[0]] = None
        # Länge unter "Electrical Parameters" sicherstellen
        cols.loc[cols.iloc[:, 1] == LENGTH_STR, cols.columns[0]] = ELECTRICAL_PARAMETER_STR
        # rekonstruieren
        df.columns = pd.MultiIndex.from_frame(cols)
        # Stelle (None, "TSO") bereit, falls TSO 1/TSO 2-Struktur verwendet wird
        ColumnFuzzyMatchingUtils.ensure_line_tso_column(df)

        # Imax-Festwert säubern (falls vorhanden)
        imax_fixed = ("Maximum Current Imax (A)", "Fixed")
        if imax_fixed in df.columns:
            df[imax_fixed] = (
                df[imax_fixed]
                .replace({"\xa0": max_i_ka_fillna * 1e3, "-": max_i_ka_fillna * 1e3, " ": max_i_ka_fillna * 1e3})
                .astype(str).str.replace(",", "."))
            df[imax_fixed] = pd.to_numeric(df[imax_fixed], errors="coerce")

        # --- numerische Konvertierung für Basis-Spalten (falls exakt vorhanden)
        static_cols = [(ELECTRICAL_PARAMETER_STR, LENGTH_STR),
                        (ELECTRICAL_PARAMETER_STR, RESISTANCE_STR),
                        (ELECTRICAL_PARAMETER_STR, REACTANCE_STR),]
        for col in static_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        # --- Fuzzy-Matching für R/X/B (Level-1), inkl. Susceptance_B (µ/μS)
        R_col = ColumnFuzzyMatchingUtils.best_resistance_col_lines_fuzzy(df)
        X_col = ColumnFuzzyMatchingUtils.best_reactance_col_lines_fuzzy(df)
        B_col = ColumnFuzzyMatchingUtils.best_susceptance_col_lines_fuzzy(df)
        for fuzzy_col in [R_col, X_col, B_col]:
            if fuzzy_col is not None:
                pos = ColumnFuzzyMatchingUtils.get_col_pos(df, fuzzy_col)  # positionsbasiert, robust gg. (None/NaN)-Top-Level
                df.iloc[:, pos] = pd.to_numeric(df.iloc[:, pos].astype(str).str.replace(",", "."), errors="coerce")
        # Namensnormierung anwenden (NE_name, Full_name, ...)
        for loc_name in [(None, "NE_name"),
                         ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, "Substation_1"),
                         ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, "Substation_2")]:
            if loc_name is not None and loc_name in df.columns:
                df.loc[:, loc_name] = df.loc[:, loc_name].astype(str).str.strip().apply(MiscUtils.multi_str_repl,
                                                                                        repl=rename_locnames)
    html_str = MiscUtils.multi_str_repl(html_str, rename_locnames)
    # --- Transformer-Daten: nur kleinere Anpassungen, Rest bleibt wie im Original
    key = "Transformers"
    if key in data:
        df = data[key]
        # Location vereinheitlichen
        loc_name = ("Location", FULL_NAME_STR)
        if loc_name in df.columns:
            df.loc[:, loc_name] = df.loc[:, loc_name].astype(str).str.strip().apply(MiscUtils.multi_str_repl,
                                                                                    repl=rename_locnames)
        # Tap-String-corrections
        taps = df.loc[:, (PHASE_SHIFT_PROPERTIES_STR, TAPS_STR)].fillna("").astype(str).str.replace(" ", "")
        nonnull = taps.apply(len).astype(bool)
        nonnull_taps = taps.loc[nonnull]
        surrounded = nonnull_taps.str.startswith("<") & nonnull_taps.str.endswith(">")
        nonnull_taps.loc[surrounded] = nonnull_taps.loc[surrounded].str[1:-1]
        slash_sep = (~nonnull_taps.str.contains(";")) & nonnull_taps.str.contains("/")
        nonnull_taps.loc[slash_sep] = nonnull_taps.loc[slash_sep].str.replace("/", ";")
        nonnull_taps.loc[nonnull_taps == "0"] = "0;0"
        df.loc[nonnull, (PHASE_SHIFT_PROPERTIES_STR, TAPS_STR)] = nonnull_taps
        df.loc[~nonnull, (PHASE_SHIFT_PROPERTIES_STR, TAPS_STR)] = "0;0"
        # Phase Shifter Doppelinfos
        cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]
        for col in cols:
            tup = (PHASE_SHIFT_PROPERTIES_STR, col)
            if tup in df.columns and is_object_dtype(df.loc[:, tup]):
                tr_double = df.index[df.loc[:, tup].str.contains("/").fillna(0).astype(bool)]
                df.loc[tr_double, tup] = df.loc[tr_double, tup].str.split("/", expand=True)[1].str.replace(",",
                                                                                            ".").astype(float).values
    return html_str


# ==================================================================================================
# HTML geodata parsing
# ==================================================================================================

def _parse_html_str(html_str: str) -> pd.DataFrame:
    """
    Parse embedded JSON geodata from an HTML map file and return line-endpoint coordinates.

    Expected map widget structure: a specific htmlwidget with a JSON script tag.
    Extracts polylines and associated tooltips to recover:
      - EIC code per line
      - NE name (line name)
      - Coordinates (lng/lat) for 'from' and 'to' endpoints

    Parameters
    ----------
    html_str : str
        Full HTML content as a string.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
          ["EIC_Code", "name", "bus", "geo_dim", "value"]
        Where 'bus' is ["from", "to"], 'geo_dim' is ["lng", "lat"], and 'value' is numeric.

    Raises
    ------
    AssertionError
        If EIC list length does not match polyline list length.
    KeyError, json.JSONDecodeError
        If the internal widget structure is not found or JSON is malformed.
    """
    def _filter_name(st: str) -> str:
        name_start = "<b>NE name: "
        name_end = "</b>"
        pos0 = st.find(name_start) + len(name_start)
        pos1 = st.find(name_end, pos0)
        assert pos0 >= 0
        assert pos1 >= len(name_start)
        return st[pos0:pos1]

    json_start_str = '<script type="application/json" data-for="htmlwidget-216030e6806f328c00fb">'
    json_start_pos = html_str.find(json_start_str) + len(json_start_str)
    json_end_pos = html_str[json_start_pos:].find('</script>')
    json_str = html_str[json_start_pos:(json_start_pos + json_end_pos)]
    geo_data = json.loads(json_str)
    geo_data = geo_data["x"]["calls"]
    methods_pos = pd.Series({item["method"]: i for i, item in enumerate(geo_data)})
    polylines = geo_data[methods_pos.at["addPolylines"]]["args"]
    EIC_start = "EIC Code:<b> "
    if len(polylines[6]) != len(polylines[0]):
        raise AssertionError("The lists of EIC Code data and geo data are not of the same length.")
    line_EIC = [polylines[6][i][polylines[6][i].find(EIC_start) + len(EIC_start):] for i in range(
        len(polylines[6]))]
    line_name = [_filter_name(polylines[6][i]) for i in range(len(polylines[6]))]
    line_geo_data = pd.concat([_lng_lat_to_df(polylines[0][i][0][0], line_EIC[i], line_name[i]) for
                               i in range(len(polylines[0]))], ignore_index=True)

    # remove trailing whitespaces
    for col in ["EIC_Code", "name"]:
        line_geo_data[col] = line_geo_data[col].str.strip()

    return line_geo_data


def _lng_lat_to_df(dict_: dict, line_EIC: str, line_name: str) -> pd.DataFrame:
    """
    Helper: convert a small lng/lat dict from map JSON into a tidy 4-row DataFrame
    for 'from'/'to' endpoints.

    Parameters
    ----------
    dict_ : dict
        Dict with 'lng' and 'lat' lists (each size 2).
    line_EIC : str
        EIC code string for the line.
    line_name : str
        NE name for the line.

    Returns
    -------
    pd.DataFrame
        Rows with columns ["EIC_Code", "name", "bus", "geo_dim", "value"].
    """
    return pd.DataFrame([
        [line_EIC, line_name, "from", "lng", dict_["lng"][0]],
        [line_EIC, line_name, "to", "lng", dict_["lng"][1]],
        [line_EIC, line_name, "from", "lat", dict_["lat"][0]],
        [line_EIC, line_name, "to", "lat", dict_["lat"][1]],
    ], columns=["EIC_Code", "name", "bus", "geo_dim", "value"])


# ==================================================================================================
# Element creation: buses, lines, transformers
# ==================================================================================================

def _create_buses_from_line_data(net: pandapowerNet, data: dict[str, pd.DataFrame]) -> None:
    """
    Create pandapower buses from Lines and Tielines data.

    Logic:
      - Fuzzy-detect voltage (kV) and substation "Full_name" columns.
      - Build a DataFrame of (name, vn_kv, TSO) for Substation_1 and Substation_2.
      - Drop duplicates (join TSO labels when multiple), then create buses in pandapower.

    Parameters
    ----------
    net : pandapowerNet
        Target pandapower network (modified in-place).
    data : dict[str, pd.DataFrame]
        Excel sheets dict.
    """
    bus_df_empty = pd.DataFrame({"name": str(), "vn_kv": float(), "TSO": str()}, index=[])
    bus_df = deepcopy(bus_df_empty)

    for key in ["Lines", "Tielines"]:
        if key not in data:
            continue
        df = data[key]

        vn_tuple = ColumnFuzzyMatchingUtils.get_voltage_tuple(df)
        if vn_tuple is None:
            raise KeyError(f"{key}: Keine Voltage_level-Spalte gefunden (fuzzy).")

        # Substation 1
        s1_full = ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, "Substation_1")
        if s1_full is not None:
            to_add1 = pd.DataFrame({
                "name": df.loc[:, s1_full].astype(str).str.strip().values,
                "vn_kv": pd.to_numeric(df.loc[:, vn_tuple].astype(str).str.replace(",", "."), errors="coerce").values,
                "TSO": ColumnFuzzyMatchingUtils.get_tso_series_for_side_fuzzy(df, "Substation_1").values
            })
            bus_df = pd.concat([bus_df, to_add1], ignore_index=True) if len(bus_df) else to_add1

        # Substation 2
        s2_full = ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, "Substation_2")
        if s2_full is not None:
            to_add2 = pd.DataFrame({
                "name": df.loc[:, s2_full].astype(str).str.strip().values,
                "vn_kv": pd.to_numeric(df.loc[:, vn_tuple].astype(str).str.replace(",", "."), errors="coerce").values,
                "TSO": ColumnFuzzyMatchingUtils.get_tso_series_for_side_fuzzy(df, "Substation_2").values
            })
            bus_df = pd.concat([bus_df, to_add2], ignore_index=True) if len(bus_df) else to_add2

    bus_df = _drop_duplicates_and_join_TSO(bus_df)
    new_bus_idx = create_buses(net, len(bus_df), vn_kv=bus_df.vn_kv, name=bus_df.name, zone=bus_df.TSO)
    assert np.allclose(new_bus_idx, bus_df.index)


def _create_lines(
        net: pandapowerNet,
        data: dict[str, pd.DataFrame],
        max_i_ka_fillna: float | int) -> None:
    """
    Create pandapower lines from Lines/Tielines data.

    Steps:
      - Validate/repair length (km), set zero/NaN lengths to 1 km with a warning.
      - Determine from/to buses via (Substation, Full_name) and voltage (vn_kv).
      - Fuzzy-detect R, X, B columns, compute per-km values.
      - Read Imax (kA) from Fixed column if present, else use fallback.
      - Attach metadata (name/EIC/TSO/comment) via fuzzy detection.
      - Create lines in pandapower; set Tieline=True for tielines.

    Parameters
    ----------
    net : pandapowerNet
        Target pandapower network (modified in-place).
    data : dict[str, pd.DataFrame]
        Excel sheets dict.
    max_i_ka_fillna : float | int
        Fallback Imax (kA) if missing; use np.nan to avoid filling.

    Raises
    ------
    KeyError
        If voltage or substation name columns cannot be found via fuzzy matching.
    """
    bus_idx = _get_bus_idx(net)

    for key in ["Lines", "Tielines"]:
        if key not in data:
            continue
        df = data[key]

        # VN
        vn_tuple = ColumnFuzzyMatchingUtils.get_voltage_tuple(df)
        if vn_tuple is None:
            raise KeyError(f"{key}: Voltage_level (fuzzy) not found.")
        vn_kvs = df.loc[:, vn_tuple].values

        # Substation names
        s1_full = ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, "Substation_1")
        s2_full = ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, "Substation_2")
        if s1_full is None or s2_full is None:
            raise KeyError(f"{key}: Substation_1/2 Full_name (fuzzy) not found.")

        valid_mask = (
                ~pd.isna(vn_kvs) &
                (df[s1_full].astype(str).str.strip() != "") &
                (df[s1_full].astype(str).str.strip() != "NAN") &
                (df[s2_full].astype(str).str.strip() != "") &
                (df[s2_full].astype(str).str.strip() != "NAN")
        )

        # Logge entfernte ungültige Zeilen
        invalid_count = len(df) - valid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"{invalid_count} {key.lower()} wurden aufgrund fehlender oder ungültiger Daten entfernt")

        # Filtere ungültige Zeilen heraus
        df = df[valid_mask].copy()
        vn_kvs = vn_kvs[valid_mask]

        # Length
        length_km = df[(ELECTRICAL_PARAMETER_STR, LENGTH_STR)].values
        zero_length = np.isclose(length_km, 0)
        no_length = np.isnan(length_km)
        if sum(zero_length) or sum(no_length):
            logger.warning(
                f"Nach den Daten haben {sum(zero_length)} {key.lower()} 0 km Länge und {sum(no_length)} ohne Länge; beide auf 1 km gesetzt.")
            length_km[zero_length | no_length] = 1

        # Bus indices
        from_bus = bus_idx.loc[list(zip(df.loc[:, s1_full].astype(str).values, vn_kvs))].values
        to_bus = bus_idx.loc[list(zip(df.loc[:, s2_full].astype(str).values, vn_kvs))].values

        # Per unit-length R/X/B (fuzzy, with fallback)
        R_col = ColumnFuzzyMatchingUtils.best_resistance_col_lines_fuzzy(df)
        X_col = ColumnFuzzyMatchingUtils.best_reactance_col_lines_fuzzy(df)
        B_col = ColumnFuzzyMatchingUtils.best_susceptance_col_lines_fuzzy(df)

        if R_col is not None:
            R_vals = df.iloc[:, ColumnFuzzyMatchingUtils.get_col_pos(df, R_col)].values
        elif (ELECTRICAL_PARAMETER_STR, RESISTANCE_STR) in df.columns:
            R_vals = df[(ELECTRICAL_PARAMETER_STR, RESISTANCE_STR)].values
        else:
            R_vals = np.zeros(len(df))
        if X_col is not None:
            X_vals = df.iloc[:, ColumnFuzzyMatchingUtils.get_col_pos(df, X_col)].values
        elif (ELECTRICAL_PARAMETER_STR, REACTANCE_STR) in df.columns:
            X_vals = df[(ELECTRICAL_PARAMETER_STR, REACTANCE_STR)].values
        else:
            X_vals = np.zeros(len(df))
        if B_col is not None:
            B_vals = df.iloc[:, ColumnFuzzyMatchingUtils.get_col_pos(df, B_col)].values
        else:
            B_vals = np.zeros(len(df))

        R = R_vals / length_km
        X = X_vals / length_km
        B = B_vals / length_km

        # Imax
        imax_fixed = ("Maximum Current Imax (A)", "Fixed")
        I_ka = df[imax_fixed].fillna(max_i_ka_fillna * 1e3).values / 1e3 if imax_fixed in df.columns else np.full(
            len(df), max_i_ka_fillna)

        # Metadata (fuzzy)
        name_vals = ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy(df, "NE name", tokens=["ne", "name"], default=None)
        eic_vals = ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy(df, "EIC code", tokens=["eic", "code"], default=None)
        comment_vals = ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy(df, "comment", tokens=["comment"], default="")
        tso_vals = ColumnFuzzyMatchingUtils.get_line_tso_array_fuzzy(df)

        _ = create_lines_from_parameters(
            net,
            from_bus,
            to_bus,
            length_km,
            R, X, B,
            I_ka,
            name=name_vals,
            EIC_Code=eic_vals,
            TSO=tso_vals,
            Comment=comment_vals,
            Tieline=(key == "Tielines"),
        )


def _create_transformers_and_buses(
        net: pandapowerNet, data: dict[str, pd.DataFrame], **kwargs) -> None:
    """
    Create transformers from the Transformers sheet and ensure valid bus connections.

    Flow:
      - Determine transformer HV/LV nominal voltages (fuzzy).
      - Match each transformer to a bus pair at its location; create buses when:
          * no bus exists at the required voltage and best existing voltage deviates more than
            rel_deviation_threshold_for_trafo_bus_creation, or
          * HV and LV sides would connect to the same bus (duplicate LV bus with suffix " (2)").
      - Compute transformer parameters (vk%, vkr%, pfe, i0, etc.) from R/X/B/G and base values.
      - Add tap/phase-shifter settings.
      - Create pandapower transformers.

    Parameters
    ----------
    net : pandapowerNet
        Target pandapower network (modified in-place).
    data : dict[str, pd.DataFrame]
        Excel sheets dict.
    kwargs :
        - rel_deviation_threshold_for_trafo_bus_creation: float (default 0.2)
        - log_rel_vn_deviation: float (default 0.12)
    """
    key = "Transformers"
    dfT = data[key]

    # VN & allocation
    bus_idx = _get_bus_idx(net)
    vn_hv_kv, vn_lv_kv = _get_transformer_voltages(data, bus_idx)
    trafo_connections = _allocate_trafos_to_buses_and_create_buses(
        net, data, bus_idx, vn_hv_kv, vn_lv_kv, **kwargs)
    # Imax primary
    max_fixed = pd.to_numeric(data[key].loc[:, ("Maximum Current Imax (A) primary", "Fixed")], errors="coerce")
    max_max = pd.to_numeric(data[key].loc[:, ("Maximum Current Imax (A) primary", "Max")], errors="coerce")
    max_i_a = np.asarray(max_fixed.fillna(max_max), dtype=float)
    # Base quantities
    vn_hv_arr = vn_hv_kv.astype(float)
    vn_lv_arr = vn_lv_kv.astype(float)
    sn_mva = (np.sqrt(3.0) * max_i_a * vn_hv_arr) / 1e3
    z_pu = (vn_lv_arr ** 2) / sn_mva
    # Transformerparameter (R/X/B/G als floats)
    R_ohm = np.asarray(pd.to_numeric(
        ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy_numeric(
            dfT, "resistance r ohm", tokens=["resistance"], default=0.0
        ), errors="coerce"), dtype=float)
    X_ohm = np.asarray(pd.to_numeric(
        ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy_numeric(
            dfT, "reactance x ohm", tokens=["reactance"], default=0.0
        ), errors="coerce"), dtype=float)
    B_uS = np.asarray(pd.to_numeric(
        ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy_numeric(
            dfT, "susceptance b us", tokens=["susceptance", "b", "us"], default=0.0
        ), errors="coerce"), dtype=float)
    G_uS = np.asarray(pd.to_numeric(
        ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy_numeric(
            dfT, "conductance g us", tokens=["conductance", "g", "us"], default=0.0
        ), errors="coerce"), dtype=float)

    rk = R_ohm / z_pu
    xk = X_ohm / z_pu
    b0 = (B_uS * 1e-6) * z_pu
    g0 = (G_uS * 1e-6) * z_pu
    zk = np.sqrt(rk ** 2 + xk ** 2)
    vk_percent = np.sign(xk) * zk * 100
    vkr_percent = rk * 100
    pfe_kw = g0 * sn_mva * 1e3
    i0_percent = 100 * np.sqrt(b0 ** 2 + g0 ** 2) * net.sn_mva / sn_mva

    # Tap/shifter
    taps = data[key].loc[:, (PHASE_SHIFT_PROPERTIES_STR, TAPS_STR)].str.split(";", expand=True).astype(
        int).set_axis(["tap_min", "tap_max"], axis=1)
    du = _get_float_column(data[key], (PHASE_SHIFT_PROPERTIES_STR, "Phase Regulation δu (%)"))
    dphi = _get_float_column(data[key], (PHASE_SHIFT_PROPERTIES_STR, "Angle Regulation δu (%)"))
    phase_shifter = np.isclose(du, 0) & (~np.isclose(dphi, 0))

    # Name/TSO/EIC/Comment (fuzzy)
    name_series_tr = ColumnFuzzyMatchingUtils.get_transformer_location_fullname_series_fuzzy(dfT)
    tso_series_tr = ColumnFuzzyMatchingUtils.get_transformer_tso_series_fuzzy(dfT)
    eic_vals = ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy(dfT, "eic code", tokens=["eic", "code"], default=None)
    comment_vals = pd.Series(ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy(dfT, "comment", tokens=["comment"], default="")).replace("\xa0",
                                                                                                            "").values
    theta_vals = ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy_numeric(dfT, "theta degree", tokens=["theta"], default=0.0)

    _ = create_transformers_from_parameters(
        net,
        trafo_connections.hv_bus.values,
        trafo_connections.lv_bus.values,
        sn_mva,
        vn_hv_kv,
        vn_lv_kv,
        vkr_percent,
        vk_percent,
        pfe_kw,
        i0_percent,
        shift_degree=theta_vals,
        tap_pos=0,
        tap_neutral=0,
        tap_side="lv",
        tap_min=taps["tap_min"].values,
        tap_max=taps["tap_max"].values,
        tap_phase_shifter=phase_shifter,
        tap_step_percent=du,
        tap_step_degree=dphi,
        name=name_series_tr.values,
        EIC_Code=eic_vals,
        TSO=tso_series_tr.values,
        Comment=comment_vals,
    )


def _get_transformer_voltages(
        data: dict[str, pd.DataFrame], bus_idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    key = "Transformers"
    df = data[key]

    col_p, col_s = ColumnFuzzyMatchingUtils.find_voltage_cols_in_transformers_fuzzy(df)
    if col_p is None or col_s is None:
        raise KeyError("Transformers: Voltage_level (Primary/Secondary) per Fuzzy-Matching nicht gefunden.")

    vn_p = pd.to_numeric(df.loc[:, col_p].astype(str).str.replace(",", "."), errors="coerce").values
    vn_s = pd.to_numeric(df.loc[:, col_s].astype(str).str.replace(",", "."), errors="coerce").values
    vn_hv_kv = np.maximum(vn_p, vn_s)
    vn_lv_kv = np.minimum(vn_p, vn_s)

    try:
        if is_integer_dtype(list(bus_idx.index.dtypes)[1]):
            vn_hv_kv = vn_hv_kv.astype(int)
            vn_lv_kv = vn_lv_kv.astype(int)
    except Exception:
        pass

    return vn_hv_kv, vn_lv_kv


def _allocate_trafos_to_buses_and_create_buses(
        net: pandapowerNet, data: dict[str, pd.DataFrame], bus_idx: pd.Series,
        vn_hv_kv: np.ndarray, vn_lv_kv: np.ndarray,
        rel_deviation_threshold_for_trafo_bus_creation: float = 0.2,
        log_rel_vn_deviation: float = 0.12, **kwargs) -> pd.DataFrame:
    """
    Allocate transformers to bus pairs by matching location names and voltages and create buses
    when needed.

    For each transformer:
      - Determine its location name (fuzzy).
      - Try mapping to an existing bus (same name, same voltage).
      - If no exact match, pick nearest available voltage at the location; compute relative
        deviation. If deviation > threshold, create a new bus at transformer's VN.
      - If HV and LV map to the same bus, duplicate the LV bus (" (2)") to avoid same-bus trafos.
      - Log warnings for moderate deviations (> log_rel_vn_deviation).
      - Return a DataFrame of allocations and deviations.

    Parameters
    ----------
    net : pandapowerNet
        Target pandapower network; modified when new buses must be created.
    data : dict[str, pd.DataFrame]
        Excel sheets dict.
    bus_idx : pd.Series
        Mapping (name, vn_kv) -> bus index.
    vn_hv_kv : np.ndarray
        HV nominal voltages for each transformer.
    vn_lv_kv : np.ndarray
        LV nominal voltages for each transformer.
    rel_deviation_threshold_for_trafo_bus_creation : float, optional
        Threshold for creating new buses when nearest existing voltage deviates too much.
    log_rel_vn_deviation : float, optional
        Warning threshold for voltage deviations when not creating a bus.

    Returns
    -------
    pd.DataFrame
        Allocation info with columns:
        ["name", "hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv",
         "vn_hv_kv_next_bus", "vn_lv_kv_next_bus",
         "hv_rel_deviation", "lv_rel_deviation",
         "trafo_hv_to_bus_deviation", "trafo_lv_to_bus_deviation"]

    Raises
    ------
    ValueError
        If transformer locations cannot be resolved to any bus/location after robust normalization.
    """
    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violates the ineqation {rel_deviation_threshold_for_trafo_bus_creation=} >= {log_rel_vn_deviation=}. Therefore, rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} is assumed.")
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    key = "Transformers"
    dfT = data[key]
    bus_location_names = set(net.bus.name)

    # Standortnamen (fuzzy)
    trafo_bus_names = ColumnFuzzyMatchingUtils.get_transformer_location_fullname_series_fuzzy(dfT)
    trafo_location_names = _find_trafo_locations(trafo_bus_names, bus_location_names)

    empties = -1 * np.ones(len(vn_hv_kv), dtype=int)
    trafo_connections = pd.DataFrame({
        "name": trafo_location_names,
        "hv_bus": empties,
        "lv_bus": empties,
        "vn_hv_kv": vn_hv_kv,
        "vn_lv_kv": vn_lv_kv,
        "vn_hv_kv_next_bus": vn_hv_kv,
        "vn_lv_kv_next_bus": vn_lv_kv,
        "hv_rel_deviation": np.zeros(len(vn_hv_kv)),
        "lv_rel_deviation": np.zeros(len(vn_hv_kv)),
    })
    trafo_connections[["hv_bus", "lv_bus"]] = trafo_connections[["hv_bus", "lv_bus"]].astype(np.int64)

    for side in ["hv", "lv"]:
        bus_col, trafo_vn_col, next_col, rel_dev_col, has_dev_col = \
            f"{side}_bus", f"vn_{side}_kv", f"vn_{side}_kv_next_bus", f"{side}_rel_deviation", f"trafo_{side}_to_bus_deviation"
        name_vn_series = pd.Series(tuple(zip(trafo_location_names, trafo_connections[trafo_vn_col])))
        isin = name_vn_series.isin(bus_idx.index)
        trafo_connections[has_dev_col] = ~isin
        trafo_connections.loc[isin, bus_col] = bus_idx.loc[name_vn_series.loc[isin]].values

        next_vn = np.array([bus_idx.loc[tln.name].index.values[
                                (pd.Series(bus_idx.loc[tln.name].index) - getattr(tln, trafo_vn_col)).abs().idxmin()
                            ] for tln in trafo_connections.loc[~isin, ["name", trafo_vn_col]].itertuples()])
        trafo_connections.loc[~isin, next_col] = next_vn
        rel_dev = np.abs(next_vn - trafo_connections.loc[~isin, trafo_vn_col].values) / next_vn
        trafo_connections.loc[~isin, rel_dev_col] = rel_dev
        trafo_connections.loc[~isin, bus_col] = bus_idx.loc[list(zip(trafo_connections.loc[~isin, "name"],
            trafo_connections.loc[~isin, next_col]))].values

        need_bus_creation = trafo_connections[rel_dev_col] > rel_deviation_threshold_for_trafo_bus_creation
        if need_bus_creation.any():
            tso_series_tr = ColumnFuzzyMatchingUtils.get_transformer_tso_series_fuzzy(dfT)
            new_bus_data = pd.DataFrame({
                "vn_kv": trafo_connections.loc[need_bus_creation, trafo_vn_col].values,
                "name": trafo_connections.loc[need_bus_creation, "name"].values,
                "TSO": tso_series_tr.loc[need_bus_creation].values
            })
            new_bus_data_dd = _drop_duplicates_and_join_TSO(new_bus_data)
            new_bus_idx = create_buses(net, len(new_bus_data_dd),
                                       vn_kv=new_bus_data_dd.vn_kv,
                                       name=new_bus_data_dd.name,
                                       zone=new_bus_data_dd.TSO)
            trafo_connections.loc[need_bus_creation, bus_col] = \
            net.bus.loc[new_bus_idx, ["name", "vn_kv"]].reset_index().set_index(["name", "vn_kv"]).loc[
                list(new_bus_data[["name", "vn_kv"]].itertuples(index=False, name=None))
            ].values
            trafo_connections.loc[need_bus_creation, next_col] = trafo_connections.loc[
                need_bus_creation, trafo_vn_col].values
            trafo_connections.loc[need_bus_creation, rel_dev_col] = 0
            trafo_connections.loc[need_bus_creation, has_dev_col] = False

    same_bus_connection = trafo_connections.hv_bus == trafo_connections.lv_bus
    duplicated_buses = net.bus.loc[trafo_connections.loc[same_bus_connection, "lv_bus"]].copy()
    duplicated_buses["name"] += " (2)"
    duplicated_buses.index = list(range(net.bus.index.max() + 1, net.bus.index.max() + 1 + len(duplicated_buses)))
    trafo_connections.loc[same_bus_connection, "lv_bus"] = duplicated_buses.index
    net.bus = pd.concat([net.bus, duplicated_buses])
    if n_add_buses := len(duplicated_buses):
        tr_names = trafo_location_names.loc[same_bus_connection]
        are_PSTs = tr_names.str.contains("PST")
        logger.info(
            f"{n_add_buses} additional buses created to avoid same-bus trafos. Of {len(tr_names)} trafos, {sum(are_PSTs)} contain 'PST'.")

    for side in ["hv", "lv"]:
        need_logging = trafo_connections.loc[trafo_connections[has_dev_col], rel_dev_col] > log_rel_vn_deviation
        if n_need_logging := sum(need_logging):
            max_dev = trafo_connections[rel_dev_col].max()
            idx_max_dev = trafo_connections[rel_dev_col].idxmax()
            logger.warning(
                f"For {n_need_logging} Transformers ({side} side), only locations with relative deviation > {log_rel_vn_deviation} were found. Max deviation {max_dev} at "
                f"{trafo_connections.at[idx_max_dev, trafo_vn_col]} kV vs bus {trafo_connections.at[idx_max_dev, next_col]} kV.")

    assert (trafo_connections.hv_bus > -1).all()
    assert (trafo_connections.lv_bus > -1).all()
    assert (trafo_connections.hv_bus != trafo_connections.lv_bus).all()

    return trafo_connections


def _find_trafo_locations(trafo_bus_names, bus_location_names):
    """
    Resolve transformer location strings to existing bus location names via normalization.
    The procedure:
      - Split original names into tokens on spaces and patterns like '-A\\d+', '-TD\\d+', '-PF\\d+', and '/'.
      - Remove tokens that are stopwords (tr, pst, trafo, kv), empties, or contain digits.
      - Compose two candidates:
          * joined string of remaining tokens
          * longest single token
      - Try exact matches against bus names; if failed, try a close match (cutoff=0.8).
      - If still unmatched, raise a ValueError for the count of unmatched transformers.

    Parameters
    ----------
    trafo_bus_names : pd.Series
        Series of transformer location strings (raw).
    bus_location_names : set[str]
        Known bus location names from Lines/Tielines.

    Returns
    -------
    pd.Series
        Best-matched bus location names for each transformer entry.

    Raises
    ------
    ValueError
        If after robust tries some transformers remain unresolved.
    """
    # Convert bus_location_names to a list for easier searching
    bus_names_list = list(bus_location_names)

    # --- split (original and lower case) strings at " " separators to remove impeding parts for
    # identifying the location names
    trafo_bus_names_expended = trafo_bus_names.str.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+",
                                                         expand=True).fillna("").replace(" ", "")
    trafo_bus_names_expended_lower = trafo_bus_names.str.lower().str.split(
        r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+", expand=True).fillna("").replace(" ", "")

    # --- identify impeding parts
    contains_number = trafo_bus_names_expended.map(lambda x: any(char.isdigit() for char in x))
    to_drop = (trafo_bus_names_expended_lower == "tr") | (trafo_bus_names_expended_lower == "pst") \
              | (trafo_bus_names_expended == "") | (trafo_bus_names_expended == "/") | (
                      trafo_bus_names_expended == "LIPST") | (trafo_bus_names_expended == "EHPST") | (
                      trafo_bus_names_expended == "TFO") | (trafo_bus_names_expended_lower == "trafo") | (
                      trafo_bus_names_expended_lower == "kv") | contains_number
    trafo_bus_names_expended[to_drop] = ""

    # --- reconstruct name strings for identification
    trafo_bus_names_joined = trafo_bus_names_expended.where(~to_drop).fillna('').agg(
        ' '.join, axis=1).str.strip()
    trafo_bus_names_longest_part = trafo_bus_names_expended.apply(
        lambda row: max(row, key=len), axis=1)

    # --- Check for exact matches first (more conservative approach)
    joined_in_buses = trafo_bus_names_joined.isin(bus_location_names)
    longest_part_in_buses = trafo_bus_names_longest_part.isin(bus_location_names)

    # --- For entries that don't have exact matches, try very close matches before giving up
    still_missing = ~(joined_in_buses | longest_part_in_buses)

    if still_missing.any():
        # Try difflib for close matches (but be more restrictive)
        for i in still_missing[still_missing].index:
            cand1 = trafo_bus_names_joined.iat[i]
            cand2 = trafo_bus_names_longest_part.iat[i]

            query = cand1 if cand1 else cand2
            if query:
                matches = difflib.get_close_matches(query, bus_names_list, n=1, cutoff=0.8)
                if matches:
                    joined_in_buses.iat[i] = True
                    trafo_bus_names_joined.iat[i] = matches[0]
                    still_missing.iat[i] = False

    # --- Final check - raise error only for truly unmatched transformers
    if still_missing.any():
        n_bus_names_not_found = sum(still_missing)
        raise ValueError(
            f"For {n_bus_names_not_found} Transformers, no suitable bus location names were found. "
            f"This may indicate missing buses or incorrect naming conventions.")

    # --- set the trafo location names and trafo bus indices respectively
    trafo_location_names = trafo_bus_names_longest_part.copy()
    trafo_location_names.loc[joined_in_buses] = trafo_bus_names_joined.loc[joined_in_buses]

    return trafo_location_names


def _drop_duplicates_and_join_TSO(bus_df: pd.DataFrame) -> pd.DataFrame:
    bus_df = bus_df.drop_duplicates(ignore_index=True)
    bus_df = bus_df.groupby(["name", "vn_kv"], as_index=False).agg({"TSO": lambda x: '/'.join(x)})
    assert not bus_df.duplicated(["name", "vn_kv"]).any()
    return bus_df


def _get_float_column(df, col_tuple, fill=0):
    series = df.loc[:, col_tuple]
    series.loc[series == "\xa0"] = fill
    return series.astype(float).fillna(fill)


def _get_bus_idx(net: pandapowerNet) -> pd.Series:
    return net.bus[["name", "vn_kv"]].rename_axis("index").reset_index().set_index([
        "name", "vn_kv"])["index"]


# ==================================================================================================
# Grid grouping, augmentation and cleanup
# ==================================================================================================

def get_grid_groups(net: pandapowerNet, **kwargs) -> pd.DataFrame:
    notravbuses_dict = {} if "notravbuses" not in kwargs.keys() else {
        "notravbuses": kwargs.pop("notravbuses")}
    grid_group_buses = list(connected_components(create_nxgraph(net, **kwargs), **notravbuses_dict))
    grid_groups = pd.DataFrame({"buses": grid_group_buses})
    grid_groups["n_buses"] = grid_groups["buses"].apply(len)
    return grid_groups


def _invent_connections_between_grid_groups(
        net: pandapowerNet, minimal_trafo_invention: bool = False, **kwargs) -> None:
    """
    Connect islanded grid groups through synthetic links to improve network connectivity.

    Three mechanisms:
      1) Add representative transformers between equally named buses in different groups
         (same location, different voltage levels) using parameters copied from existing trafos
         that connect the same voltage pair (prefer same TSO).
      2) Fuse buses with same base name and same voltage level that belong to different groups.
      3) Fuse specific known close-by bus pairs (hardcoded list).

    Parameters
    ----------
    net : pandapowerNet
        Network to modify in-place.
    minimal_trafo_invention : bool, optional
        If True, stop adding transformers as soon as the grid is no longer islanded. Not applied
        for certain published releases (value may be ignored).
    kwargs : dict
        Passed through; not used currently.

    Notes
    -----
    - Replaces "Wuergau (2)" with "Wuergau" for base-name equality before matching.
    - After each synthetic connection, group assignments are updated.
    - Emits info logs when transformer data are copied.
    """
    grid_groups = get_grid_groups(net)
    bus_idx = _get_bus_idx(net)
    bus_grid_groups = pd.concat([pd.Series(group, index=buses) for group, buses in zip(
        grid_groups.index, grid_groups.buses)]).sort_index()

    # treat for example "Wuergau" equally as "Wuergau (2)":
    location_names = pd.Series(bus_idx.index.get_level_values(0))
    location_names = location_names.str.replace(r"(.) \([0-9]+\)", r"\1", regex=True)
    bus_idx.index = pd.MultiIndex.from_arrays(
        [location_names.values, bus_idx.index.get_level_values(1).to_numpy()],
        names=bus_idx.index.names)

    # --- 1) add Transformers between equally named buses that have different voltage level and lay in different groups
    connected_vn_kvs_by_trafos = pd.DataFrame({
        "hv": net.bus.vn_kv.loc[net.trafo.hv_bus.values].values,
        "lv": net.bus.vn_kv.loc[net.trafo.lv_bus.values].values,
        "index": net.trafo.index}).set_index(["hv", "lv"]).sort_index()
    dupl_location_names = location_names[location_names.duplicated()]

    for location_name in dupl_location_names:
        if minimal_trafo_invention and len(bus_grid_groups.unique()) <= 1:
            break
        grid_groups_at_location = bus_grid_groups.loc[bus_idx.loc[location_name].values]
        grid_groups_at_location = grid_groups_at_location.drop_duplicates()
        if len(grid_groups_at_location) < 2:
            continue
        elif len(grid_groups_at_location) > 2:
            raise NotImplementedError("Code is not provided to invent Transformer connections "
                                      "between locations with more than two grid groups, i.e. "
                                      "voltage levels.")
        TSO = net.bus.zone.at[grid_groups_at_location.index[0]]
        vn_kvs = net.bus.vn_kv.loc[grid_groups_at_location.index].sort_values(ascending=False)
        try:
            trafos_connecting_same_voltage_levels = \
                connected_vn_kvs_by_trafos.loc[tuple(vn_kvs)]
        except KeyError:
            logger.info(f"For location {location_name}, no transformer data can be reused since "
                        f"no transformer connects {vn_kvs.sort_values(ascending=False).iat[0]} kV "
                        f"and {vn_kvs.sort_values(ascending=False).iat[1]} kV.")
            continue
        trafos_of_same_TSO = trafos_connecting_same_voltage_levels.loc[(net.bus.zone.loc[
                                                                            net.trafo.hv_bus.loc[
                                                                                trafos_connecting_same_voltage_levels.values.flatten(
                                                                                )].values] == TSO).values].values.flatten()

        # choose trafo to copy parameters from
        tr_to_be_copied = trafos_of_same_TSO[0] if len(trafos_of_same_TSO) else \
            trafos_connecting_same_voltage_levels.values.flatten()[0]

        # duplicate transformer row
        duplicated_row = net.trafo.loc[[tr_to_be_copied]].copy()
        duplicated_row.index = [net.trafo.index.max() + 1]
        duplicated_row.hv_bus = vn_kvs.index[0]
        duplicated_row.lv_bus = vn_kvs.index[1]
        duplicated_row.name = "additional transformer to connect the grid"
        net.trafo = pd.concat([net.trafo, duplicated_row])

        bus_grid_groups.loc[bus_grid_groups == grid_groups_at_location.iat[1]] = \
            grid_groups_at_location.iat[0]

    # --- 2) merge buses of same voltage level, different grid groups and equal name base
    bus_name_splits = net.bus.name.str.split(r"[ -/]+", expand=True)
    buses_with_single_base = net.bus.name.loc[(~bus_name_splits.isnull()).sum(axis=1) == 1]
    for idx, name_base in buses_with_single_base.items():
        same_name_base = net.bus.drop(idx).name.str.contains(name_base)
        if not any(same_name_base):
            continue
        other_group = bus_grid_groups.drop(idx) != bus_grid_groups.at[idx]
        same_vn = net.bus.drop(idx).vn_kv == net.bus.vn_kv.at[idx]
        is_fuse_candidate = same_name_base & other_group & same_vn
        if not any(is_fuse_candidate):
            continue
        to_fuse = bus_grid_groups.drop(idx).loc[is_fuse_candidate].drop_duplicates()
        fuse_buses(net, idx, set(to_fuse.index))

        bus_grid_groups.loc[bus_grid_groups.isin(bus_grid_groups.drop(idx).loc[
                                                     is_fuse_candidate].unique())] = grid_groups_at_location.iat[0]
        bus_grid_groups = bus_grid_groups.drop(to_fuse.index)

    # --- 3) fuse buses that are close to each other (known cases)
    for name1, name2 in [("CROISIERE", "BOLLENE (POSTE RESEAU)"),
                         ("CAEN", "DRONNIERE (LA)"),
                         ("TRINITE-VICTOR", "MENTON/TRINITE VICTOR")]:
        b1 = net.bus.index[net.bus.name == name1]
        b2 = net.bus.index[net.bus.name == name2]
        if len(b1) == 1 and len(b2) >= 1:
            fuse_buses(net, b1[0], set(b2))
            bus_grid_groups = bus_grid_groups.drop(b2)
        else:
            logger.info("Buses of the following names were intended to be fused but were not found."
                        f"\n'{name1}' and '{name2}'")


def drop_islanded_grid_groups(
        net: pandapowerNet,
        min_bus_number: int | str,
        **kwargs) -> None:
    """
    Drop islanded grid groups based on group size or supply condition.

    Modes:
      - Integer (e.g., 6): drop groups with number of buses < min_bus_number.
      - "max": keep only the largest group (drop all others).
      - "unsupplied": drop groups that do not contain any slack element (ext_grid or slack gen).

    Parameters
    ----------
    net : pandapowerNet
        Network to clean up (modified in-place).
    min_bus_number : int | str
        Threshold or special mode ("max", "unsupplied").
    kwargs : dict
        Additional parameters passed to get_grid_groups.

    Raises
    ------
    NotImplementedError
        If 'min_bus_number' is neither an int nor one of the special strings.

    Notes
    -----
    Logs the number of dropped groups and total buses dropped.
    """

    def _grid_groups_to_drop_by_min_bus_number():
        return grid_groups.loc[grid_groups["n_buses"] < min_bus_number]

    grid_groups = get_grid_groups(net, **kwargs)

    if min_bus_number == "unsupplied":
        slack_buses = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"]) | \
                      set(net.gen.loc[net.gen.in_service & net.gen.slack, "bus"])
        grid_groups_to_drop = grid_groups.loc[~grid_groups.buses.apply(
            lambda x: not x.isdisjoint(slack_buses))]

    elif min_bus_number == "max":
        min_bus_number = grid_groups["n_buses"].max()
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    elif isinstance(min_bus_number, int):
        grid_groups_to_drop = _grid_groups_to_drop_by_min_bus_number()

    else:
        raise NotImplementedError(
            f"{min_bus_number=} is not implemented. Use an int, 'max', or 'unsupplied' instead.")

    buses_to_drop = reduce(set.union, grid_groups_to_drop.buses)
    drop_buses(net, buses_to_drop)
    logger.info(f"drop_islanded_grid_groups() drops {len(grid_groups_to_drop)} grid groups with a "
                f"total of {grid_groups_to_drop.n_buses.sum()} buses.")


# ==================================================================================================
# Geodata integration
# ==================================================================================================

def _add_bus_geo(net: pandapowerNet, line_geo_data: pd.DataFrame) -> None:
    """
    Add geodata to buses using line endpoint geodata from the HTML-exported map.

    Method:
      - Build two pivot tables of geodata: by EIC_Code/bus and by name/bus.
      - For each bus, inspect connected lines and decide whether to use EIC_Code or name
        as the primary lookup key (based on duplicates/availability).
      - If multiple candidate coordinates remain, reduce by rounding and pick the most
        frequently occurring coordinate across lines.
      - Write GeoJSON-like strings into net.bus.geo.
      - Lines receive geodata via set_line_geodata_from_bus_geodata elsewhere.

    Parameters
    ----------
    net : pandapowerNet
        Target network (modified in-place).
    line_geo_data : pd.DataFrame
        Tidy DataFrame from _parse_html_str().

    Notes
    -----
    For ambiguous geodata (all EIC/Name keys duplicated), the function falls back to any
    available (non-missing) value and logs info.
    """
    iSl = pd.IndexSlice
    lgd_EIC_bus = line_geo_data.pivot_table(values="value", index=["EIC_Code", "bus"],
                                            columns="geo_dim")
    lgd_name_bus = line_geo_data.pivot_table(values="value", index=["name", "bus"],
                                             columns="geo_dim")
    lgd_EIC_bus_idx_extended = pd.MultiIndex.from_frame(lgd_EIC_bus.index.to_frame().assign(col_name="EIC_Code")
                                .rename(columns={"EIC_Code": "identifier"}).loc[:, ["col_name", "identifier", "bus"]])
    lgd_name_bus_idx_extended = pd.MultiIndex.from_frame(lgd_name_bus.index.to_frame().assign(col_name="name")
        .rename(columns={"name": "identifier"}).loc[:, ["col_name", "identifier", "bus"]])
    lgd_bus = pd.concat([lgd_EIC_bus.set_axis(lgd_EIC_bus_idx_extended),
                         lgd_name_bus.set_axis(lgd_name_bus_idx_extended)])
    dupl_EICs = net.line.EIC_Code.loc[net.line.EIC_Code.duplicated()]
    dupl_names = net.line.name.loc[net.line.name.duplicated()]

    def _geo_json_str(this_bus_geo: pd.Series) -> str:
        return f'{{"coordinates": [{this_bus_geo.at["lng"]}, {this_bus_geo.at["lat"]}], "type": "Point"}}'

    def _add_bus_geo_inner(bus: int) -> str | None:
        from_bus_line_excerpt = net.line.loc[net.line.from_bus ==
                                             bus, ["EIC_Code", "name", "Tieline"]]
        to_bus_line_excerpt = net.line.loc[net.line.to_bus == bus, ["EIC_Code", "name", "Tieline"]]
        line_excerpt = pd.concat([from_bus_line_excerpt, to_bus_line_excerpt])
        n_connected_line_ends = len(line_excerpt)
        if n_connected_line_ends == 0:
            logger.error(
                f"Bus {bus} (name {net.bus.at[bus, 'name']}) is not found in line_geo_data.")
            return None
        is_dupl = pd.concat([
            pd.DataFrame({"EIC": from_bus_line_excerpt.EIC_Code.isin(dupl_EICs).values,
                          "name": from_bus_line_excerpt.name.isin(dupl_names).values},
                         index=pd.MultiIndex.from_product([["from"], from_bus_line_excerpt.index],
                                                          names=["bus", "line_index"])),
            pd.DataFrame({"EIC": to_bus_line_excerpt.EIC_Code.isin(dupl_EICs).values,
                          "name": to_bus_line_excerpt.name.isin(dupl_names).values},
                         index=pd.MultiIndex.from_product([["to"], to_bus_line_excerpt.index],
                                                          names=["bus", "line_index"]))
        ])
        is_missing = pd.DataFrame({"EIC": ~line_excerpt.EIC_Code.isin(
                lgd_bus.loc["EIC_Code"].index.get_level_values("identifier")),"name": ~line_excerpt.name.isin(
                lgd_bus.loc["name"].index.get_level_values("identifier"))}).set_axis(is_dupl.index, axis=0)
        is_tieline = pd.Series(net.line.loc[is_dupl.index.get_level_values("line_index"),
        "Tieline"].values, index=is_dupl.index)

        # construct access_vals: default use EIC_Code, but switch to name if EIC duplicated/missing
        access_vals = pd.DataFrame({
            "col_name": "EIC_Code",
            "identifier": line_excerpt.EIC_Code.values,
            "bus": is_dupl.index.get_level_values("bus").values
        })
        take_from_name = ((is_dupl.EIC | is_missing.EIC) & (
                ~is_dupl.name & ~is_missing.name)).values
        access_vals.loc[take_from_name, "col_name"] = "name"
        access_vals.loc[take_from_name, "identifier"] = line_excerpt.name.loc[take_from_name].values
        keep = (~(is_dupl | is_missing)).any(axis=1).values
        if np.all(is_missing):
            log_msg = (f"For bus {bus} (name {net.bus.at[bus, 'name']}), {n_connected_line_ends} "
                       "were found but no EIC_Codes or names of corresponding lines were found ."
                       "in the geo data from the html file.")
            if is_tieline.all():
                logger.debug(log_msg)
            else:
                logger.warning(log_msg)
            return None
        elif sum(keep) == 0:
            logger.info(f"For {bus=}, all EIC_Codes and names of connected lines are ambiguous. "
                        "No geo data is dropped at this point.")
            keep[(~is_missing).any(axis=1)] = True
        access_vals = access_vals.loc[keep]
        # get geodata entries
        this_bus_geo = lgd_bus.loc[iSl[access_vals.col_name, access_vals.identifier, access_vals.bus], :]
        if len(this_bus_geo) > 1:
            # reduce similar/equal lines
            this_bus_geo = this_bus_geo.loc[this_bus_geo.round(2).drop_duplicates().index]
        # resolve to single coordinate
        len_this_bus_geo = len(this_bus_geo)
        if len_this_bus_geo == 1:
            return _geo_json_str(this_bus_geo.iloc[0])
        elif len_this_bus_geo == 2:
            how_often = pd.Series(
                [sum(np.isclose(lgd_EIC_bus["lat"], this_bus_geo["lat"].iat[i]) &
                     np.isclose(lgd_EIC_bus["lng"], this_bus_geo["lng"].iat[i])) for i in
                 range(len_this_bus_geo)], index=this_bus_geo.index)
            if how_often.at[how_often.idxmax()] >= 1:
                logger.warning(f"Bus {bus} (name {net.bus.at[bus, 'name']}) was found multiple times"
                               " in line_geo_data. No value exists more often than others. "
                               "The first of most used geo positions is used.")
            return _geo_json_str(this_bus_geo.loc[how_often.idxmax()])

    net.bus.geo = [_add_bus_geo_inner(bus) for bus in net.bus.index]


def _fill_geo_at_one_sided_branches_without_geo_extent(net: pandapowerNet):
    """
    Propagate bus geodata across branches when only one end has geodata.

    Iteratively:
      - Find lines/transformers where one end bus has geo and the other has not.
      - Copy the available geodata to the missing side.
      - Repeat until no more one-sided geo branches exist.
      - Finally, compute line geodata from bus geodata.

    Parameters
    ----------
    net : pandapowerNet
        Network with potentially partial geodata.

    Notes
    -----
    Intended as a post-processing helper to fill gaps if initial geodata coverage is sparse.
    """

    def _check_geo_availablitiy(net: pandapowerNet) -> dict[str, pd.Index | int]:
        av = {}  # availablitiy of geodata
        av["bus_with_geo"] = net.bus.index[~net.bus.geo.isnull()]
        av["lines_fbw_tbwo"] = net.line.index[net.line.from_bus.isin(av["bus_with_geo"]) &
                                              (~net.line.to_bus.isin(av["bus_with_geo"]))]
        av["lines_fbwo_tbw"] = net.line.index[(~net.line.from_bus.isin(av["bus_with_geo"])) &
                                              net.line.to_bus.isin(av["bus_with_geo"])]
        av["trafos_hvbw_lvbwo"] = net.trafo.index[net.trafo.hv_bus.isin(av["bus_with_geo"]) &
                                                  (~net.trafo.lv_bus.isin(av["bus_with_geo"]))]
        av["trafos_hvbwo_lvbw"] = net.trafo.index[(~net.trafo.hv_bus.isin(av["bus_with_geo"])) &
                                                  net.trafo.lv_bus.isin(av["bus_with_geo"])]
        av["n_lines_one_side_geo"] = len(av["lines_fbw_tbwo"]) + len(av["lines_fbwo_tbw"])
        return av
    geo_avail = _check_geo_availablitiy(net)
    while geo_avail["n_lines_one_side_geo"]:
        # copy available geodata to the other end of branches where geodata are missing
        for et, bus_w_geo, bus_wo_geo, idx_key in zip(
                ["line", "line", "trafo", "trafo"],
                ["to_bus", "from_bus", "lv_bus", "hv_bus"],
                ["from_bus", "to_bus", "hv_bus", "lv_bus"],
                ["lines_fbwo_tbw", "lines_fbw_tbwo", "trafos_hvbwo_lvbw", "trafos_hvbw_lvbwo"]):
            net.bus.loc[net[et].loc[geo_avail[idx_key], bus_wo_geo].values, "geo"] = \
                net.bus.loc[net[et].loc[geo_avail[idx_key], bus_w_geo].values, "geo"].values
        geo_avail = _check_geo_availablitiy(net)
    set_line_geodata_from_bus_geodata(net)


# ==================================================================================================
# __main__ demo
# ==================================================================================================
if __name__ == "__main__":
    from pathlib import Path
    import os
    from pandapower.file_io import from_json, to_json

    home = str(Path.home())
    jao_data_folder = os.path.join(home, "Documents", "JAO Static Grid Model")

    release5 = os.path.join(jao_data_folder, "20240329_Core Static Grid Model – 5th release")
    excel_file_path = os.path.join(release5, "20240329_Core Static Grid Model_public.xlsx")
    html_file_path = os.path.join(release5, "20240329_Core Static Grid Model Map_public",
                                  "2024-03-18_Core_SGM_publication.html")

    release6 = os.path.join(jao_data_folder, "202409_Core Static Grid Mode_6th release")
    excel_file_path = os.path.join(release6, "20240916_Core Static Grid Model_for publication.xlsx")
    html_file_path = os.path.join(release6, "2024-09-13_Core_SGM_publication_files",
                                  "2024-09-13_Core_SGM_publication.html")

    pp_net_json_file = os.path.join(home, "desktop", "jao_grid.json")

    if 1:  # read from original data
        net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
        to_json(net, pp_net_json_file)
    else:  # load net from already converted and stored net
        net = from_json(pp_net_json_file)
    print(net)
    grid_groups = get_grid_groups(net)
    print(grid_groups)

    _fill_geo_at_one_sided_branches_without_geo_extent(net)

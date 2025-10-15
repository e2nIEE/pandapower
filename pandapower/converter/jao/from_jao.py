# -*- coding: utf-8 -*-nt

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from copy import deepcopy
import os
import json
from functools import reduce
from typing import Optional, Union, Tuple
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
import logging

logger = logging.getLogger(__name__)


def from_jao(excel_file_path: str,
             html_file_path: Optional[str],
             extend_data_for_grid_group_connections: bool,
             drop_grid_groups_islands: bool = False,
             apply_data_correction: bool = True,
             max_i_ka_fillna: Union[float, int] = 999,
             **kwargs) -> pandapowerNet:
    """Converts European (Core) EHV grid data provided by JAO (Joint Allocation Office), the
        "Single Allocation Platform (SAP) for all European Transmission System Operators (TSOs) that
        operate in accordance to EU legislation".

        High-level pipeline:
          1) Read and stage Excel + optional HTML geodata.
          2) Identify and fix inconsistent or incomplete data (name normalization, numeric coercions).
          3) Build buses, lines, transformers based on cleaned data.
          4) Optionally add synthetic connections to merge islanded grid groups.
          5) Optionally drop very small unsupplied groups (islands).
          6) Add geodata to buses and lines if available, resolving ambiguity carefully.

        Parameters
        ----------
        excel_file_path : str
            Path to Excel with multi-sheet data (Lines, Tielines, Transformers).
        html_file_path : str | None
            Path to HTML file with geodata embedded; pass None to skip geodata.
        extend_data_for_grid_group_connections : bool
            If True, add synthetic transformers and bus fusions to connect islanded groups.
        drop_grid_groups_islands : bool, optional
            If True, drop small islanded groups based on min_bus_number in kwargs (default 6).
        apply_data_correction : bool, optional
            If True, run correction routines (rename normalization, numeric conversion, etc.).
        max_i_ka_fillna : float | int, optional
            Fallback value for missing/invalid maximum current (Imax) in kA for lines/transformers.
            Use np.nan to avoid filling; default 999 (treated as 999 kA).

        Returns
        -------
        pandapowerNet
            pandapower network created from JAO data.

        Additional Parameters (via kwargs)
        ----------------------------------
        minimal_trafo_invention : bool, optional
            If True, stop adding synthetic transformers once no islands remain.
            Note: Not applied for release version 5 or 6 (value ignored).
        min_bus_number : Union[int,str], optional
            Threshold for dropping islanded grid groups; can be 'max' or 'unsupplied' (special modes).
        rel_deviation_threshold_for_trafo_bus_creation : float, optional
            VN deviation threshold above which new buses are created for transformer sides (default 0.2).
        log_rel_vn_deviation : float, optional
            VN deviation threshold to log warnings (default 0.12).
        sn_mva : float, optional
            System base apparent power (MVA) for pandapower net.

        Notes
        -----
        - The converter intentionally uses heuristics to match transformer location names to bus names,
          favoring valid topology over strict literal string matching.
        - HTML geodata parsing relies on specific widget structure; robust error handling ensures
          the conversion proceeds if parsing fails.

    >>> from pathlib import Path
    >>> import os
    >>> import pandapower as pp
    >>> net = pp.converter.from_jao()
    >>> home = str(Path.home())
    >>> # assume that the files are located at your desktop:
    >>> excel_file_path = os.path.join(home, "desktop", "202409_Core Static Grid Mode_6th release")
    >>> html_file_path = os.path.join(home, "desktop", "2024-09-13_Core_SGM_publication.html")
    >>> net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
    """

    # --- read data
    data = pd.read_excel(excel_file_path, sheet_name=None, header=[0, 1])
    _ = report_problematic_names_after_normalization(data)
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


# --- secondary functions --------------------------------------------------------------------------

def _simplify_name(name: str) -> str:
    """
    robust canonical rename of location name (without kV).
    Goal: produce a normalized 'base' token for equivalence-class grouping of names.
    - Removes trailing marker variants like '(2)', directional suffixes (/W, -West).
    - Unifies delimiters to spaces, strips voltage-level texts ('220 kV').
    - Uppercases and collapses whitespace.
    """
    _RE_TRAILING = re.compile(r"""
        (?:\s*\([0-9]+\)\s*$)                 |  #  (2)  (3) …
        (?:\s*(?:/|[- ])\s*[EWNS](?:EST)?\s*$)   #  /W   -West  _E  …
    """, re.I | re.X)
    if not isinstance(name, str):
        name = str(name)
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode()
    s = _RE_TRAILING.sub("", s)
    s = re.sub(r"[/_.-]", " ", s)
    s = re.sub(r"\b[0-9]{2,4}\s*k?V\b", "", s, flags=re.I)
    return " ".join(s.split()).upper()


def _extra_variants_from_busnames(bus_names: set[str]) -> list[tuple[str, str]]:
    """
    Generate (old,new) pairs using heuristic 'name without suffixes'.
    E.g. 'Doerpen/W/W'  -> 'DOERPEN'
         'Stade/W/W (2)'-> 'STADE'
    This provides conservative normalization proposals from raw bus names.
    """
    mapping = {}
    for n in bus_names:
        base = _simplify_name(n)
        if base and n != base:
            mapping.setdefault(n, base)
    return list(mapping.items())


def generate_rename_locnames_from_combined(
    data: dict[str, pd.DataFrame],
    combined: Optional[pd.DataFrame] = None
) -> list[tuple[str, str]]:
    """
    Generate a robust list of (old_name, new_name) tuples to normalize naming across sheets.
    Sources:
      - 'same_canonical_variant' entries: unify variants that differ only by formatting/case.
      - 'no_match_after_normalization' entries: propose mapping transformer locations to bus names.
      - Trafo location case harmonization: direct lowercase/uppercase mapping to existing bus names.
      - Prefix heuristics: expand compact 'PSTXYZ'/'TRXYZ' to spaced 'PST XYZ'/'TR XYZ'.

    Safety:
      - Resolve suggestions to existing bus names if possible (case-insensitive).
      - Filter ambiguous mappings: avoid old_name mapping to multiple new names.
      - Deduplicate while preserving insertion order.

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of rename rules (old -> new).
    """
    if combined is None:
        combined = report_problematic_names_after_normalization(data)

    # Bus-Standorte einsammeln (Case-sensitiv + Case-insensitiv Map)
    bus_location_names = _collect_bus_location_names(data)
    bus_location_names_lower = {b.lower(): b for b in bus_location_names}

    def _resolve_to_existing_bus_name(name: str) -> str:
        """Map 'name' case-insensitively to an existing bus name (original form if present)."""
        if not name:
            return name
        return bus_location_names_lower.get(name.lower(), name)

    renames: list[tuple[str, str]] = []

    def _add(old: str, new: str):
        """Append a rename rule if it leads to an existing bus name and is not a no-op."""
        if not old or not new:
            return
        target = _resolve_to_existing_bus_name(new)
        if old == target:
            return
        renames.append((old, target))

    # 1) Variants from Busdata (same_canonical_variant): original -> suggested
    if "reason" in combined.columns:
        sv = combined.loc[combined["reason"] == "same_canonical_variant"]
        for _, row in sv.iterrows():
            _add(str(row["original"]).strip(), str(row["suggested"]).strip())

    # 2) Unmatched Trafo-cases (no_match_after_normalization): original -> suggested/Fallback
    nm = combined.loc[combined["reason"] == "no_match_after_normalization"]
    for _, row in nm.iterrows():
        orig = str(row.get("original", "")).strip()
        joined = str(row.get("joined", "")).strip()
        longest = str(row.get("longest", "")).strip()
        sugg = str(row.get("suggested", "")).strip()

        if sugg and (sugg in bus_location_names or sugg.lower() in bus_location_names_lower):
            _add(orig, sugg)
            if joined and joined != sugg:
                _add(joined, sugg)
            if longest and longest != sugg:
                _add(longest, sugg)
        else:
            fallback = None
            for val in (joined, longest):
                if val and (val in bus_location_names or val.lower() in bus_location_names_lower):
                    fallback = _resolve_to_existing_bus_name(val)
                    break
            if fallback:
                _add(orig, fallback)

        # Präfix-Heuristik basierend auf Vorschlag (wenn vorhanden)
        if sugg:
            target_base = _resolve_to_existing_bus_name(sugg)
            for prefix in ("PST", "TR"):
                compact_lower = f"{prefix}{target_base.replace(' ', '')}"
                compact_upper = f"{prefix}{target_base.upper().replace(' ', '')}"
                spaced_lower = f"{prefix} {target_base}"
                spaced_upper = f"{prefix} {target_base.upper()}"
                # Beide Kompaktvarianten auf beide Spaced-Varianten mappen,
                # damit Groß-/Kleinvarianten abgedeckt sind:
                _add(compact_lower, spaced_lower)
                _add(compact_upper, spaced_upper)

        # Zusätzliche Sicherheit: Tokens aus joined/longest auf Vorschlag mappen
        for token in (joined, longest):
            if token and sugg and token not in bus_location_names:
                _add(token, sugg)

    # 3) Case-Harmonisierung direkt aus allen Transformer-Standorten
    if "Transformers" in data and ("Location", "Full Name") in data["Transformers"].columns:
        trafo_names = data["Transformers"].loc[:, ("Location", "Full Name")].astype(str).str.strip()
        for original in trafo_names:
            joined, longest = _normalize_transformer_name_for_matching(original)
            for tok in {joined, longest}:
                if tok and tok.lower() in bus_location_names_lower:
                    target = bus_location_names_lower[tok.lower()]
                    if tok != target:
                        _add(tok, target)
                    # Präfix-Heuristiken auch hier (für Fälle wie 'PSTMIKULOWA'):
                    for prefix in ("PST", "TR"):
                        compact_lower = f"{prefix}{target.replace(' ', '')}"
                        compact_upper = f"{prefix}{target.upper().replace(' ', '')}"
                        spaced_lower = f"{prefix} {target}"
                        spaced_upper = f"{prefix} {target.upper()}"
                        _add(compact_lower, spaced_lower)
                        _add(compact_upper, spaced_upper)

    # 4) Sicherheitsfilter: Entferne mehrdeutige Ersetzungen (ein alter Name -> mehrere Ziele)
    renames.extend(_extra_variants_from_busnames(bus_location_names))
    from collections import defaultdict
    targets_by_old = defaultdict(set)
    for old, new in renames:
        targets_by_old[old].add(new)

    filtered = [(old, new) for (old, new) in renames if len(targets_by_old[old]) == 1]

    # 5) Dedup bei Reihenfolge-Erhalt
    out: list[tuple[str, str]] = []
    seen = set()
    for pair in filtered:
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    # Sammle alle Zielnamen und deren Quellnamen
    target_to_sources = {}
    for old, new in renames:
        if new not in target_to_sources:
            target_to_sources[new] = []
        target_to_sources[new].append(old)

    # Entferne Regeln, bei denen verschiedene Quellnamen zum selben Zielnamen führen
    # aber behalte diejenigen, die aus demselben kanonischen Namen stammen
    filtered_renames = []
    for old, new in renames:
        sources = target_to_sources[new]
        if len(sources) > 1:
            # Prüfe ob alle Quellen zum selben kanonischen Namen führen würden
            canonical_sources = [_canonical_bus_key(src) for src in sources]
            current_canonical = _canonical_bus_key(old)
            # Behalte nur, wenn diese Gruppe konsistent ist
            if canonical_sources.count(current_canonical) == len(canonical_sources):
                filtered_renames.append((old, new))
            # Alternative: Entferne komplett widersprüchliche Gruppen
            else:
                # Nur behalten, wenn es keine echten Konflikte gibt
                continue
        else:
            filtered_renames.append((old, new))

    # Ersetze renames mit gefilterter Liste
    renames = filtered_renames
    print(out)  # zur Kontrolle
    return out

def _collect_bus_location_names(data: dict[str, pd.DataFrame]) -> set[str]:
    names = []
    for key in [k for k in ["Lines", "Tielines"] if k in data]:
        df = data[key]
        for subst in ["Substation_1", "Substation_2"]:
            col = _get_fullname_tuple(df, subst)
            if col is not None:
                s = df.loc[:, col].astype(str).str.strip()
                names.append(s)
    if not names:
        return set()
    bus_series = pd.concat(names, ignore_index=True)
    return set(bus_series.dropna().tolist())


def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _canonical_bus_key(s: str) -> str:
    # Vereinheitlichung: Casefold, Akzentabbau, Trennzeichen raus, Whitespaces raus
    s = str(s)
    s = _strip_accents(s.casefold())
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"[-_/.,;:]+", " ", s)
    s = re.sub(r"\s+", "", s)
    return s.upper()


def _normalize_transformer_name_for_matching(name: str) -> tuple[str, str]:
    """
    Repliziert die Logik aus _find_trafo_locations (ohne rename_locnames),
    um 'joined' und 'longest' zu erzeugen.
    """
    if not isinstance(name, str):
        name = str(name)
    s = name.strip()

    # Wie in _find_trafo_locations: an Leerzeichen und an '-A\d+', '-TD\d+', '-PF\d+' trennen, auch '/'
    parts_orig = re.split(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+|/", s)
    parts_orig = [p.strip().replace(" ", "") for p in parts_orig if p is not None]

    stopwords_lower = {"tr", "pst", "trafo", "kv"}
    block_exact = {"", "LIPST", "EHPST", "TFO"}

    def _keep_token(tok: str) -> bool:
        if tok in block_exact:
            return False
        if tok.lower() in stopwords_lower:
            return False
        if any(ch.isdigit() for ch in tok):
            return False
        return True

    filtered = [p for p in parts_orig if _keep_token(p)]
    joined = " ".join([p for p in filtered if p]).strip()
    longest = max(filtered, key=len) if filtered else ""
    return joined, longest


def _suggest_closest(q: str, candidates: list[str], n: int = 1) -> str:
    if not q:
        return ""
    # Case-insensitive Vorschlag aus dem Original-Kandidatenraum
    # Wir nutzen difflib für ungefähre Übereinstimmung
    pool = candidates
    try:
        matches = difflib.get_close_matches(q, pool, n=n, cutoff=0.6)
        if matches:
            return matches[0]
    except Exception:
        pass
    # fallback: low case match
    lower_map = {}
    for c in candidates:
        lower_map.setdefault(c.lower(), []).append(c)
    if q.lower() in lower_map:
        return lower_map[q.lower()][0]
    return ""


def find_unmatched_transformer_locations_extended(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Liefert für Trafostandorte (Transformers['Location','Full Name']) alle Einträge,
    die nach der Normalisierung weder als 'joined' noch als 'longest' in den Bus-Standorten
    (aus Lines/Tielines Substation_* Full_name) gefunden werden.
    Gibt DataFrame mit Spalten ['original','joined','longest','suggested','reason'] zurück.
    """
    rows = []
    if "Transformers" not in data:
        return pd.DataFrame(columns=["original", "joined", "longest", "suggested", "reason"])

    bus_location_names = _collect_bus_location_names(data)
    bus_location_names_lower = {b.lower(): b for b in bus_location_names}

    trafo_names = data["Transformers"].loc[:, ("Location", "Full Name")].astype(str).str.strip()

    for original in trafo_names:
        joined, longest = _normalize_transformer_name_for_matching(original)

        # Direkte Übereinstimmung
        if (joined and joined in bus_location_names) or (longest and longest in bus_location_names):
            continue

        # Case-insensitive Übereinstimmung (dann nicht als "unmatched" zählen)
        if (joined and joined.lower() in bus_location_names_lower) or \
           (longest and longest.lower() in bus_location_names_lower):
            continue

        # Keine Übereinstimmung: Vorschlag suchen
        suggest = _suggest_closest(joined if joined else longest, list(bus_location_names))
        rows.append({
            "original": original,
            "joined": joined,
            "longest": longest,
            "suggested": suggest,
            "reason": "no_match_after_normalization"
        })

    df = pd.DataFrame(rows, columns=["original", "joined", "longest", "suggested", "reason"])
    return df


def find_problematic_bus_name_variants(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Ermittelt Busnamen-Varianten aus Lines/Tielines (Substation_* Full_name), die sich nur durch
    Groß-/Kleinschreibung, Sonderzeichen, Bindestriche/Leerzeichen unterscheiden.
    Gibt DataFrame mit Spalten ['original','suggested','reason'] zurück. (joined/longest leer)
    """
    all_names = []
    for key in [k for k in ["Lines", "Tielines"] if k in data]:
        for subst in ["Substation_1", "Substation_2"]:
            if (subst, "Full_name") in data[key].columns:
                all_names.append(data[key].loc[:, (subst, "Full_name")].astype(str).str.strip())
    if not all_names:
        return pd.DataFrame(columns=["original", "suggested", "reason"])

    s = pd.concat(all_names, ignore_index=True).dropna()
    if s.empty:
        return pd.DataFrame(columns=["original", "suggested", "reason"])

    df = pd.DataFrame({"original": s})
    df["canonical"] = df["original"].map(_canonical_bus_key)

    # Häufigkeiten je Original-Name (für sinnvolle 'suggested'-Wahl)
    freq = df["original"].value_counts()

    out_rows = []
    for canon, sub in df.groupby("canonical"):
        uniques = sub["original"].unique()
        if len(uniques) <= 1:
            continue
        # Wähle als "suggested" den häufigsten Namen; bei Gleichstand den längsten
        uniques_sorted = sorted(uniques, key=lambda x: (-freq.get(x, 0), -len(x), x))
        suggested = uniques_sorted[0]
        for orig in uniques:
            if orig == suggested:
                continue
            out_rows.append({
                "original": orig,
                "suggested": suggested,
                "reason": "same_canonical_variant"
            })

    return pd.DataFrame(out_rows, columns=["original", "suggested", "reason"])


def report_problematic_names_after_normalization(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Wrapper wie gewünscht: ruft die beiden Analysefunktionen auf, harmonisiert Spalten,
    printed die kombinierten Ergebnisse und gibt sie zurück.
    """
    trafo = find_unmatched_transformer_locations_extended(data)
    bus_vars = find_problematic_bus_name_variants(data)

    if len(bus_vars):
        bus_vars = bus_vars.assign(joined="", longest="")

    cols = ["original", "joined", "longest", "suggested", "reason"]
    # reindex für bus_vars (die hat joined/longest leer)
    bus_vars = bus_vars.reindex(columns=["original", "joined", "longest", "suggested", "reason"])

    combined = pd.concat([trafo.reindex(columns=cols), bus_vars.reindex(columns=cols)],
                         ignore_index=True).drop_duplicates()
    print(combined)
    return combined


def _first_present_tuple(df: pd.DataFrame, candidates: list[tuple]) -> Optional[tuple]:
    for t in candidates:
        if t in df.columns:
            return t
    return None

# def _get_voltage_tuple(df: pd.DataFrame) -> Optional[tuple]:
#     return _best_voltage_col_lines(df)

# def _get_fullname_tuple(df: pd.DataFrame, subst: str) -> Optional[tuple]:
#     return _first_present_tuple(df, [
#         (subst, "Full_name"),
#         (subst, "Full Name"),
#         (subst, "Fullname"),
#     ])

def _get_tso_tuple(df: pd.DataFrame, subst: str) -> Optional[tuple]:
    if subst == "Substation_1":
        return _first_present_tuple(df, [(None, "TSO 1"), (None, "TSO1"), (None, "TSO")])
    elif subst == "Substation_2":
        return _first_present_tuple(df, [(None, "TSO 2"), (None, "TSO2"), (None, "TSO")])
    else:
        return _first_present_tuple(df, [(None, "TSO")])

def _ensure_line_tso_column(df: pd.DataFrame) -> None:
    # Erzeuge (None, "TSO"), falls nicht vorhanden, aus TSO 1/TSO 2
    if (None, "TSO") not in df.columns:
        t1 = _first_present_tuple(df, [(None, "TSO 1"), (None, "TSO1")])
        t2 = _first_present_tuple(df, [(None, "TSO 2"), (None, "TSO2")])
        if t1 and t2:
            df[(None, "TSO")] = df.loc[:, t1].astype(str).str.strip() + "/" + df.loc[:, t2].astype(str).str.strip()
        elif t1:
            df[(None, "TSO")] = df.loc[:, t1]
        elif t2:
            df[(None, "TSO")] = df.loc[:, t2]
        # sonst bleibt (None, "TSO") ungesetzt – wird später robust behandelt
def _find_first_present_lvl1(df: pd.DataFrame, variants: list[str]):
    lvl1 = df.columns.get_level_values(1)
    for lab in variants:
        if lab in lvl1:
            for col in df.columns:
                if col[1] == lab:
                    return col
    return None

# def _get_voltage_tuple(df: pd.DataFrame) -> Optional[tuple]:
#     # sucht beliebige "Voltage_level..."-Varianten über Level 1
#     return _find_first_present_lvl1(df, [
#         "Voltage_level(kV)",
#         "Voltage_level [kV]",
#         "Voltage_level (kV)",
#         "Voltage level [kV]",
#         "Voltage level (kV)",
#     ])

def _get_tso_col_for_subst(df: pd.DataFrame, subst: str) -> Optional[tuple]:
    # Bevorzugt TSO 1 / TSO 2 je nach Substation, sonst generisches "TSO"
    if subst == "Substation_1":
        col = _find_first_present_lvl1(df, ["TSO 1", "TSO1"])
        if col is not None:
            return col
    elif subst == "Substation_2":
        col = _find_first_present_lvl1(df, ["TSO 2", "TSO2"])
        if col is not None:
            return col
    return _find_first_present_lvl1(df, ["TSO"])

def _get_tso_series_for_side(df: pd.DataFrame, subst: str) -> pd.Series:
    col = _get_tso_col_for_subst(df, subst)
    if col is not None:
        return df.loc[:, col].astype(str).str.strip()
    # Fallback: generische TSO-Spalte?
    col_generic = _find_first_present_lvl1(df, ["TSO"])
    if col_generic is not None:
        return df.loc[:, col_generic].astype(str).str.strip()
    # letzter Fallback: leere Strings
    return pd.Series([""] * len(df), index=df.index)

def _series_by_lvl1(df: pd.DataFrame, label: str) -> Optional[pd.Series]:
    # positionsbasiert statt labelbasiert → kein (None,'TSO')-KeyError, keine PerformanceWarning
    lvl1 = df.columns.get_level_values(1)
    pos = np.flatnonzero(lvl1 == label)
    if pos.size:
        return df.iloc[:, pos[0]]
    return None

def _values_by_lvl1(df: pd.DataFrame, label: str, default="") -> np.ndarray:
    s = _series_by_lvl1(df, label)
    if s is None:
        return np.array([default] * len(df))
    return s.astype(str).str.strip().values

def _get_line_tso_array(df: pd.DataFrame) -> np.ndarray:
    # 1) Generische "TSO"
    s = _series_by_lvl1(df, "TSO")
    if s is not None:
        return s.astype(str).str.strip().values
    # 2) Kombination "TSO 1/TSO 2"
    s1 = _series_by_lvl1(df, "TSO 1") or _series_by_lvl1(df, "TSO1")
    s2 = _series_by_lvl1(df, "TSO 2") or _series_by_lvl1(df, "TSO2")
    if s1 is not None and s2 is not None:
        return (s1.astype(str).str.strip() + "/" + s2.astype(str).str.strip()).values
    if s1 is not None:
        return s1.astype(str).str.strip().values
    if s2 is not None:
        return s2.astype(str).str.strip().values
    return np.array([""] * len(df))

def _canon_label(s: str) -> str:
    s = str(s or "").strip()
    s = s.replace("µ", "u").replace("μ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def _get_col_pos(df: pd.DataFrame, col: Tuple) -> Optional[int]:
    try:
        return list(df.columns).index(col)
    except ValueError:
        return None

def _best_col_by_lvl1_similarity(df: pd.DataFrame,
                                 target_label: str,
                                 min_ratio: float = 0.55,
                                 required_tokens: Optional[list[str]] = None) -> Optional[Tuple]:
    tgt = _canon_label(target_label)
    req = set(required_tokens or [])
    best, best_score = None, -1.0
    for col in df.columns:
        lvl1 = _canon_label(col[1])
        score = _sim(lvl1, tgt)
        if req and not all(tok in lvl1 for tok in req):
            score -= 0.15
        if score >= min_ratio and score > best_score:
            best, best_score = col, score
    return best

def _best_voltage_col_lines(df: pd.DataFrame) -> Optional[Tuple]:
    # verlangt "volt" und "kv" im Level-1-Label
    return _best_col_by_lvl1_similarity(df, target_label="voltage level kv",
                                        min_ratio=0.45, required_tokens=["volt", "kv"])

def _find_voltage_cols_in_transformers_fuzzy(df: pd.DataFrame) -> tuple[Optional[Tuple], Optional[Tuple]]:
    tgt_top = _canon_label("voltage level kv")
    prim_list, sec_list = [], []
    for col in df.columns:
        top_c = _canon_label(col[0])
        lvl1_c = _canon_label(col[1])
        top_ok = (_sim(top_c, tgt_top) >= 0.45) and ("volt" in top_c and "kv" in top_c)
        if top_ok and _sim(lvl1_c, _canon_label("primary")) >= 0.7:
            prim_list.append(col)
        if top_ok and _sim(lvl1_c, _canon_label("secondary")) >= 0.7:
            sec_list.append(col)
    # Paare mit identischem Top-Level bevorzugen
    for p in prim_list:
        matches = [s for s in sec_list if _canon_label(s[0]) == _canon_label(p[0])]
        if matches:
            return p, matches[0]
    prim_best = max(prim_list, key=lambda c: _sim(_canon_label(c[0]), tgt_top), default=None)
    sec_best  = max(sec_list,  key=lambda c: _sim(_canon_label(c[0]), tgt_top), default=None)
    return prim_best, sec_best

def _best_fullname_tuple_fuzzy(df: pd.DataFrame, subst: str) -> Optional[Tuple]:
    tgt0 = _canon_label(subst)      # "substation1"/"substation2"
    tgt1 = _canon_label("full name")
    best, best_score = None, -1.0
    for col in df.columns:
        top_c = _canon_label(col[0])
        lvl1_c = _canon_label(col[1])
        s0 = _sim(top_c, tgt0)
        s1 = _sim(lvl1_c, tgt1)
        if "substation" not in top_c:
            s0 -= 0.1
        if "fullname" not in lvl1_c and not ("full" in lvl1_c and "name" in lvl1_c):
            s1 -= 0.1
        score = 0.5 * (s0 + s1)
        if s0 >= 0.5 and s1 >= 0.6 and score > best_score:
            best, best_score = col, score
    return best

def _best_susceptance_col_lines_fuzzy(df: pd.DataFrame) -> Optional[Tuple]:
    return _best_col_by_lvl1_similarity(df, "susceptance b us",
                                        min_ratio=0.5, required_tokens=["susceptance", "b", "us"])

def _best_resistance_col_lines_fuzzy(df: pd.DataFrame) -> Optional[Tuple]:
    return _best_col_by_lvl1_similarity(df, "resistance r ohm",
                                        min_ratio=0.5, required_tokens=["resistance"])

def _best_reactance_col_lines_fuzzy(df: pd.DataFrame) -> Optional[Tuple]:
    return _best_col_by_lvl1_similarity(df, "reactance x ohm",
                                        min_ratio=0.5, required_tokens=["reactance"])

def _best_transformer_location_fullname_col_fuzzy(df: pd.DataFrame) -> Optional[Tuple]:
    tgt0 = _canon_label("location")
    tgt1 = _canon_label("full name")
    best, best_score = None, -1.0
    for col in df.columns:
        top_c = _canon_label(col[0])
        lvl1_c = _canon_label(col[1])
        s0 = _sim(top_c, tgt0)
        s1 = _sim(lvl1_c, tgt1)
        if "location" not in top_c:
            s0 -= 0.1
        if "fullname" not in lvl1_c and not ("full" in lvl1_c and "name" in lvl1_c):
            s1 -= 0.1
        score = 0.5 * (s0 + s1)
        if s0 >= 0.5 and s1 >= 0.6 and score > best_score:
            best, best_score = col, score
    return best

def _get_transformer_location_fullname_series_fuzzy(df: pd.DataFrame) -> pd.Series:
    col = _best_transformer_location_fullname_col_fuzzy(df)
    if col is None:
        raise KeyError("Transformers: Location / Full Name per Fuzzy-Matching nicht gefunden.")
    pos = _get_col_pos(df, col)
    return df.iloc[:, pos].astype(str).str.strip()

def _values_by_lvl1_fuzzy(df: pd.DataFrame, target_label: str,
                          tokens: Optional[list[str]] = None,
                          default="") -> np.ndarray:
    col = _best_col_by_lvl1_similarity(df, target_label, min_ratio=0.5, required_tokens=tokens)
    if col is None:
        return np.array([default] * len(df))
    pos = _get_col_pos(df, col)
    return df.iloc[:, pos].astype(str).str.strip().values

def _values_by_lvl1_fuzzy_numeric(df: pd.DataFrame, target_label: str,
                                  tokens: Optional[list[str]] = None,
                                  default=0.0) -> np.ndarray:
    vals = _values_by_lvl1_fuzzy(df, target_label, tokens=tokens, default=str(default))
    return pd.to_numeric(pd.Series(vals).str.replace(",", "."), errors="coerce").fillna(default).values

def _get_tso_series_for_side_fuzzy(df: pd.DataFrame, subst: str) -> pd.Series:
    target = "TSO 1" if subst == "Substation_1" else "TSO 2"
    col = _best_col_by_lvl1_similarity(df, target, min_ratio=0.6, required_tokens=["tso"])
    if col is not None:
        pos = _get_col_pos(df, col)
        return df.iloc[:, pos].astype(str).str.strip()
    col = _best_col_by_lvl1_similarity(df, "TSO", min_ratio=0.6, required_tokens=["tso"])
    if col is not None:
        pos = _get_col_pos(df, col)
        return df.iloc[:, pos].astype(str).str.strip()
    return pd.Series([""] * len(df), index=df.index)

def _get_line_tso_array_fuzzy(df: pd.DataFrame) -> np.ndarray:
    col = _best_col_by_lvl1_similarity(df, "TSO", min_ratio=0.6, required_tokens=["tso"])
    if col is not None:
        pos = _get_col_pos(df, col)
        return df.iloc[:, pos].astype(str).str.strip().values
    c1 = _best_col_by_lvl1_similarity(df, "TSO 1", min_ratio=0.6, required_tokens=["tso"])
    c2 = _best_col_by_lvl1_similarity(df, "TSO 2", min_ratio=0.6, required_tokens=["tso"])
    if c1 is not None and c2 is not None:
        p1 = _get_col_pos(df, c1); p2 = _get_col_pos(df, c2)
        return (df.iloc[:, p1].astype(str).str.strip() + "/" + df.iloc[:, p2].astype(str).str.strip()).values
    if c1 is not None:
        return df.iloc[:, _get_col_pos(df, c1)].astype(str).str.strip().values
    if c2 is not None:
        return df.iloc[:, _get_col_pos(df, c2)].astype(str).str.strip().values
    return np.array([""] * len(df))

def _get_transformer_tso_series_fuzzy(df: pd.DataFrame) -> pd.Series:
    best, best_score = None, -1.0
    for col in df.columns:
        if "tso" in _canon_label(col[1]):
            score = 0.0
            if "location" in _canon_label(col[0]):
                score += 0.2
            score += _sim(_canon_label(col[1]), _canon_label("tso"))
            if score > best_score:
                best, best_score = col, score
    if best is None:
        return pd.Series([""] * len(df), index=df.index)
    pos = _get_col_pos(df, best)
    return df.iloc[:, pos].astype(str).str.strip()

# Ersetzt: _get_voltage_tuple (Lines/Tielines)
def _get_voltage_tuple(df: pd.DataFrame) -> Optional[Tuple]:
    return _best_voltage_col_lines(df)

# Ersetzt: Substation Full_name – nutzt Fuzzy
def _get_fullname_tuple(df: pd.DataFrame, subst: str) -> Optional[Tuple]:
    return _best_fullname_tuple_fuzzy(df, subst)


def _data_correction(
        data: dict[str, pd.DataFrame],
        html_str: Optional[str],
        max_i_ka_fillna: Union[float, int]) -> Optional[str]:
    """Corrects input data in particular with regard to obvious weaknesses in the data provided,
    such as inconsistent spellings and missing necessary information

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
    html_str : str | None
        data provided by the html file which will be corrected
    max_i_ka_fillna : float | int
        value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.

    Returns
    -------
    str
        corrected html_str
    """
    # old name -> new name
    combined = report_problematic_names_after_normalization(data)
    rename_locnames = generate_rename_locnames_from_combined(data, combined)

    filtered_rename_locnames = []
    bus_location_names = _collect_bus_location_names(data)
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

        # MultiIndex-Korrektur: bekannte Varianten unter einen Top-Level (None) heben
        cols = df.columns.to_frame(index=False)
        # harmonisiere zweite Ebene (Spaltennamen)
        replace_map = {
            "Full Name": "Full_name",
            "Short Name": "Short_name",
            "Susceptance_B (µS)": "Susceptance_B(μS)",
            "Voltage_level (kV)": "Voltage_level(kV)",
            "Voltage_level [kV]": "Voltage_level(kV)",
        }
        cols.iloc[:, 1] = cols.iloc[:, 1].replace(replace_map)

        # setze Top-Level = None für diese Felder
        cols.loc[cols.iloc[:, 1].isin(["Voltage_level(kV)", "Comment"]), cols.columns[0]] = None
        cols.loc[cols.iloc[:, 0].astype(str).str.startswith("Unnamed:"), cols.columns[0]] = None
        # Länge unter "Electrical Parameters" sicherstellen
        cols.loc[cols.iloc[:, 1] == "Length_(km)", cols.columns[0]] = "Electrical Parameters"

        # rekonstruieren
        df.columns = pd.MultiIndex.from_frame(cols)

        # Stelle (None, "TSO") bereit, falls TSO 1/TSO 2-Struktur verwendet wird
        _ensure_line_tso_column(df)

        # Imax-Festwert säubern (falls vorhanden)
        imax_fixed = ("Maximum Current Imax (A)", "Fixed")
        if imax_fixed in df.columns:
            df[imax_fixed] = (
                df[imax_fixed]
                .replace({"\xa0": max_i_ka_fillna * 1e3, "-": max_i_ka_fillna * 1e3, " ": max_i_ka_fillna * 1e3})
                .astype(str).str.replace(",", ".")
            )
            df[imax_fixed] = pd.to_numeric(df[imax_fixed], errors="coerce")

        # --- numerische Konvertierung für Basis-Spalten (falls exakt vorhanden)
        static_cols = [
            ("Electrical Parameters", "Length_(km)"),
            ("Electrical Parameters", "Resistance_R(Ω)"),
            ("Electrical Parameters", "Reactance_X(Ω)"),
        ]
        for col in static_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")

        # --- Fuzzy-Matching für R/X/B (Level-1), inkl. Susceptance_B (µ/μS)
        R_col = _best_resistance_col_lines_fuzzy(df)  # findet z. B. "Resistance_R(Ω)"
        X_col = _best_reactance_col_lines_fuzzy(df)  # findet z. B. "Reactance_X(Ω)"
        B_col = _best_susceptance_col_lines_fuzzy(df)  # findet z. B. "Susceptance_B (µS)/(μS)"

        for fuzzy_col in [R_col, X_col, B_col]:
            if fuzzy_col is not None:
                pos = _get_col_pos(df, fuzzy_col)  # positionsbasiert, robust gg. (None/NaN)-Top-Level
                df.iloc[:, pos] = pd.to_numeric(df.iloc[:, pos].astype(str).str.replace(",", "."), errors="coerce")
        # Namensnormierung anwenden (NE_name, Full_name, ...)
        for loc_name in [(None, "NE_name"),
                         _get_fullname_tuple(df, "Substation_1"),
                         _get_fullname_tuple(df, "Substation_2")]:
            if loc_name is not None and loc_name in df.columns:
                df.loc[:, loc_name] = df.loc[:, loc_name].astype(str).str.strip().apply(_multi_str_repl,
                                                                                        repl=rename_locnames)

    html_str = _multi_str_repl(html_str, rename_locnames)

    # --- Transformer-Daten: nur kleinere Anpassungen, Rest bleibt wie im Original
    key = "Transformers"
    if key in data:
        df = data[key]
        # Location vereinheitlichen
        loc_name = ("Location", "Full Name")
        if loc_name in df.columns:
            df.loc[:, loc_name] = df.loc[:, loc_name].astype(str).str.strip().apply(_multi_str_repl,
                                                                                    repl=rename_locnames)

        # Tap-String-Korrekturen (wie gehabt)
        taps = df.loc[:, ("Phase Shifting Properties", "Taps used for RAO")].fillna("").astype(str).str.replace(" ", "")
        nonnull = taps.apply(len).astype(bool)
        nonnull_taps = taps.loc[nonnull]
        surrounded = nonnull_taps.str.startswith("<") & nonnull_taps.str.endswith(">")
        nonnull_taps.loc[surrounded] = nonnull_taps.loc[surrounded].str[1:-1]
        slash_sep = (~nonnull_taps.str.contains(";")) & nonnull_taps.str.contains("/")
        nonnull_taps.loc[slash_sep] = nonnull_taps.loc[slash_sep].str.replace("/", ";")
        nonnull_taps.loc[nonnull_taps == "0"] = "0;0"
        df.loc[nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = nonnull_taps
        df.loc[~nonnull, ("Phase Shifting Properties", "Taps used for RAO")] = "0;0"

        # Phase Shifter Doppelinfos (wie gehabt)
        cols = ["Phase Regulation δu (%)", "Angle Regulation δu (%)"]
        for col in cols:
            tup = ("Phase Shifting Properties", col)
            if tup in df.columns and is_object_dtype(df.loc[:, tup]):
                tr_double = df.index[df.loc[:, tup].str.contains("/").fillna(0).astype(bool)]
                df.loc[tr_double, tup] = df.loc[tr_double, tup].str.split("/", expand=True)[1].str.replace(",",
                                                                                                           ".").astype(
                    float).values

    return html_str


def _parse_html_str(html_str: str) -> pd.DataFrame:
    """Converts ths geodata from the html file (information hidden in the string), from Lines in
    particular, to a DataFrame that can be used later in _add_bus_geo()

    Parameters
    ----------
    html_str : str
        html file that includes geodata information

    Returns
    -------
    pd.DataFrame
        extracted geodata for a later and easy use
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


def _create_buses_from_line_data(net: pandapowerNet, data: dict[str, pd.DataFrame]) -> None:
    """Creates buses to the pandapower net using information from the lines and tielines sheets
    (excel file).

    Parameters
    ----------
    net : pandapowerNet
        net to be filled by buses
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
    """
    bus_df_empty = pd.DataFrame({"name": str(), "vn_kv": float(), "TSO": str()}, index=[])
    bus_df = deepcopy(bus_df_empty)

    for key in ["Lines", "Tielines"]:
        if key not in data:
            continue
        df = data[key]

        vn_tuple = _get_voltage_tuple(df)
        if vn_tuple is None:
            raise KeyError(f"{key}: Keine Voltage_level-Spalte gefunden (fuzzy).")

        # Substation 1
        s1_full = _get_fullname_tuple(df, "Substation_1")
        if s1_full is not None:
            to_add1 = pd.DataFrame({
                "name": df.loc[:, s1_full].astype(str).str.strip().values,
                "vn_kv": pd.to_numeric(df.loc[:, vn_tuple].astype(str).str.replace(",", "."), errors="coerce").values,
                "TSO": _get_tso_series_for_side_fuzzy(df, "Substation_1").values
            })
            bus_df = pd.concat([bus_df, to_add1], ignore_index=True) if len(bus_df) else to_add1

        # Substation 2
        s2_full = _get_fullname_tuple(df, "Substation_2")
        if s2_full is not None:
            to_add2 = pd.DataFrame({
                "name": df.loc[:, s2_full].astype(str).str.strip().values,
                "vn_kv": pd.to_numeric(df.loc[:, vn_tuple].astype(str).str.replace(",", "."), errors="coerce").values,
                "TSO": _get_tso_series_for_side_fuzzy(df, "Substation_2").values
            })
            bus_df = pd.concat([bus_df, to_add2], ignore_index=True) if len(bus_df) else to_add2

    bus_df = _drop_duplicates_and_join_TSO(bus_df)
    new_bus_idx = create_buses(net, len(bus_df), vn_kv=bus_df.vn_kv, name=bus_df.name, zone=bus_df.TSO)
    assert np.allclose(new_bus_idx, bus_df.index)


def _create_lines(
        net: pandapowerNet,
        data: dict[str, pd.DataFrame],
        max_i_ka_fillna: Union[float, int]) -> None:
    """Creates lines to the pandapower net using information from the lines and tielines sheets
    (excel file).

    Parameters
    ----------
    net : pandapowerNet
        net to be filled by buses
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
    max_i_ka_fillna : float | int
        value to fill missing values or data of false type in max_i_ka of lines and transformers.
        If no value should be set, you can also pass np.nan.
    """

    bus_idx = _get_bus_idx(net)

    for key in ["Lines", "Tielines"]:
        if key not in data:
            continue
        df = data[key]

        # Länge
        length_km = df[("Electrical Parameters", "Length_(km)")].values
        zero_length = np.isclose(length_km, 0)
        no_length = np.isnan(length_km)
        if sum(zero_length) or sum(no_length):
            logger.warning(
                f"Nach den Daten haben {sum(zero_length)} {key.lower()} 0 km Länge und {sum(no_length)} ohne Länge; beide auf 1 km gesetzt.")
            length_km[zero_length | no_length] = 1

        # VN
        vn_tuple = _get_voltage_tuple(df)
        if vn_tuple is None:
            raise KeyError(f"{key}: Voltage_level (fuzzy) nicht gefunden.")
        vn_kvs = df.loc[:, vn_tuple].values

        # Substation-Namen
        s1_full = _get_fullname_tuple(df, "Substation_1")
        s2_full = _get_fullname_tuple(df, "Substation_2")
        if s1_full is None or s2_full is None:
            raise KeyError(f"{key}: Substation_1/2 Full_name (fuzzy) nicht gefunden.")

        # Bus-Indizes
        from_bus = bus_idx.loc[list(tuple(zip(df.loc[:, s1_full].astype(str).values, vn_kvs)))].values
        to_bus = bus_idx.loc[list(tuple(zip(df.loc[:, s2_full].astype(str).values, vn_kvs)))].values

        # Leitungsparameter je km (fuzzy, mit Fallback)
        R_col = _best_resistance_col_lines_fuzzy(df)
        X_col = _best_reactance_col_lines_fuzzy(df)
        B_col = _best_susceptance_col_lines_fuzzy(df)

        if R_col is not None:
            R_vals = df.iloc[:, _get_col_pos(df, R_col)].values
        elif ("Electrical Parameters", "Resistance_R(Ω)") in df.columns:
            R_vals = df[("Electrical Parameters", "Resistance_R(Ω)")].values
        else:
            R_vals = np.zeros(len(df))
        if X_col is not None:
            X_vals = df.iloc[:, _get_col_pos(df, X_col)].values
        elif ("Electrical Parameters", "Reactance_X(Ω)") in df.columns:
            X_vals = df[("Electrical Parameters", "Reactance_X(Ω)")].values
        else:
            X_vals = np.zeros(len(df))
        if B_col is not None:
            B_vals = df.iloc[:, _get_col_pos(df, B_col)].values
        else:
            # Fallback-Wert 0 wenn nicht vorhanden
            B_vals = np.zeros(len(df))

        R = R_vals / length_km
        X = X_vals / length_km
        B = B_vals / length_km

        # Imax
        imax_fixed = ("Maximum Current Imax (A)", "Fixed")
        I_ka = df[imax_fixed].fillna(max_i_ka_fillna * 1e3).values / 1e3 if imax_fixed in df.columns else np.full(
            len(df), max_i_ka_fillna)

        # Metadaten (fuzzy)
        name_vals = _values_by_lvl1_fuzzy(df, "NE name", tokens=["ne", "name"], default=None)
        eic_vals = _values_by_lvl1_fuzzy(df, "EIC code", tokens=["eic", "code"], default=None)
        comment_vals = _values_by_lvl1_fuzzy(df, "comment", tokens=["comment"], default="")
        tso_vals = _get_line_tso_array_fuzzy(df)

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
    """Creates transformers to the pandapower net using information from the transformers sheet
    (excel file).

    Parameters
    ----------
    net : pandapowerNet
        net to be filled by buses
    data : dict[str, pd.DataFrame]
        data provided by the excel file which will be corrected
    """

    # --- data preparations
    key = "Transformers"
    dfT = data[key]

    # VN & Zuteilung
    bus_idx = _get_bus_idx(net)
    vn_hv_kv, vn_lv_kv = _get_transformer_voltages(data, bus_idx)
    trafo_connections = _allocate_trafos_to_buses_and_create_buses(
        net, data, bus_idx, vn_hv_kv, vn_lv_kv, **kwargs)

    # Imax primary
    max_i_a = data[key].loc[:, ("Maximum Current Imax (A) primary", "Fixed")]
    empty_i_idx = max_i_a.index[max_i_a.isnull()]
    max_i_a.loc[empty_i_idx] = data[key].loc[empty_i_idx, ("Maximum Current Imax (A) primary", "Max")].values

    # Basisgrößen
    sn_mva = np.sqrt(3) * max_i_a * vn_hv_kv / 1e3
    z_pu = vn_lv_kv ** 2 / sn_mva

    # Trafoparameter (fuzzy, Level-1)
    R_ohm = _values_by_lvl1_fuzzy_numeric(dfT, "resistance r ohm", tokens=["resistance"], default=0.0)
    X_ohm = _values_by_lvl1_fuzzy_numeric(dfT, "reactance x ohm", tokens=["reactance"], default=0.0)
    B_uS = _values_by_lvl1_fuzzy_numeric(dfT, "susceptance b us", tokens=["susceptance", "b", "us"], default=0.0)
    G_uS = _values_by_lvl1_fuzzy_numeric(dfT, "conductance g us", tokens=["conductance", "g", "us"], default=0.0)

    rk = R_ohm / z_pu
    xk = X_ohm / z_pu
    b0 = B_uS * 1e-6 * z_pu
    g0 = G_uS * 1e-6 * z_pu
    zk = np.sqrt(rk ** 2 + xk ** 2)
    vk_percent = np.sign(xk) * zk * 100
    vkr_percent = rk * 100
    pfe_kw = g0 * sn_mva * 1e3
    i0_percent = 100 * np.sqrt(b0 ** 2 + g0 ** 2) * net.sn_mva / sn_mva

    # Taps/Phasensteller (wie bisher über feste Labels)
    taps = data[key].loc[:, ("Phase Shifting Properties", "Taps used for RAO")].str.split(";", expand=True).astype(
        int).set_axis(["tap_min", "tap_max"], axis=1)
    du = _get_float_column(data[key], ("Phase Shifting Properties", "Phase Regulation δu (%)"))
    dphi = _get_float_column(data[key], ("Phase Shifting Properties", "Angle Regulation δu (%)"))
    phase_shifter = np.isclose(du, 0) & (~np.isclose(dphi, 0))

    # Name/TSO/EIC/Comment (fuzzy)
    name_series_tr = _get_transformer_location_fullname_series_fuzzy(dfT)
    tso_series_tr = _get_transformer_tso_series_fuzzy(dfT)
    eic_vals = _values_by_lvl1_fuzzy(dfT, "eic code", tokens=["eic", "code"], default=None)
    comment_vals = pd.Series(_values_by_lvl1_fuzzy(dfT, "comment", tokens=["comment"], default="")).replace("\xa0",
                                                                                                            "").values
    theta_vals = _values_by_lvl1_fuzzy_numeric(dfT, "theta degree", tokens=["theta"], default=0.0)

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


def _invent_connections_between_grid_groups(
        net: pandapowerNet, minimal_trafo_invention: bool = False, **kwargs) -> None:
    """Adds connections between islanded grid groups via:

    - adding transformers between equally named buses that have different voltage level and lay in different groups
    - merge buses of same voltage level, different grid groups and equal name base
    - fuse buses that are close to each other

    Parameters
    ----------
    net : pandapowerNet
        net to be manipulated
    minimal_trafo_invention : bool, optional
        if True, adding transformers stops when no grid groups is islanded anymore (does not apply
        for release version 5 or 6, i.e. it does not care what value is passed to
        minimal_trafo_invention). If False, all equally named buses that have different voltage
        level and lay in different groups will be connected via additional transformers,
        by default False
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

    # --- add Transformers between equally named buses that have different voltage level and lay in
    # --- different groups
    connected_vn_kvs_by_trafos = pd.DataFrame({
        "hv": net.bus.vn_kv.loc[net.trafo.hv_bus.values].values,
        "lv": net.bus.vn_kv.loc[net.trafo.lv_bus.values].values,
        "index": net.trafo.index}).set_index(["hv", "lv"]).sort_index()
    dupl_location_names = location_names[location_names.duplicated()]

    for location_name in dupl_location_names:
        if minimal_trafo_invention and len(bus_grid_groups.unique()) <= 1:
            break  # break with regard to minimal_trafo_invention
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

        # from which trafo parameters are copied:
        tr_to_be_copied = trafos_of_same_TSO[0] if len(trafos_of_same_TSO) else \
            trafos_connecting_same_voltage_levels.values.flatten()[0]

        # copy transformer data
        duplicated_row = net.trafo.loc[[tr_to_be_copied]].copy()
        duplicated_row.index = [net.trafo.index.max() + 1]  # adjust index
        duplicated_row.hv_bus = vn_kvs.index[0]  # adjust hv_bus, lv_bus
        duplicated_row.lv_bus = vn_kvs.index[1]  # adjust hv_bus, lv_bus
        duplicated_row.name = "additional transformer to connect the grid"
        net.trafo = pd.concat([net.trafo, duplicated_row])

        bus_grid_groups.loc[bus_grid_groups == grid_groups_at_location.iat[1]] = \
            grid_groups_at_location.iat[0]

    # --- merge buses of same voltage level, different grid groups and equal name base
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

    # --- fuse buses that are close to each other
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
        min_bus_number: Union[int, str],
        **kwargs) -> None:
    """Drops grid groups that are islanded and include a number of buses below min_bus_number.

    Parameters
    ----------
    net : pandapowerNet
        net in which islanded grid groups will be dropped
    min_bus_number : int | str, optional
        Threshold value to decide which small grid groups should be dropped and which large grid
        groups should be kept. If all islanded grid groups should be dropped except of the one
        largest, set "max". If all grid groups that do not contain a slack element should be
        dropped, set "unsupplied".
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


def _add_bus_geo(net: pandapowerNet, line_geo_data: pd.DataFrame) -> None:
    """Adds geodata to the buses. The function needs to handle cases where line_geo_data does not
    include no or multiple geodata per bus. Primarly, the geodata are allocate via EIC Code names,
    if ambigous, names are considered.

    Parameters
    ----------
    net : pandapowerNet
        net in which geodata are added to the buses
    line_geo_data : pd.DataFrame
        Converted geodata from the html file
    """
    iSl = pd.IndexSlice
    lgd_EIC_bus = line_geo_data.pivot_table(values="value", index=["EIC_Code", "bus"],
                                            columns="geo_dim")
    lgd_name_bus = line_geo_data.pivot_table(values="value", index=["name", "bus"],
                                             columns="geo_dim")
    lgd_EIC_bus_idx_extended = pd.MultiIndex.from_frame(lgd_EIC_bus.index.to_frame().assign(
        **dict(col_name="EIC_Code")).rename(columns=dict(EIC_Code="identifier")).loc[
                                                        :, ["col_name", "identifier", "bus"]])
    lgd_name_bus_idx_extended = pd.MultiIndex.from_frame(lgd_name_bus.index.to_frame().assign(
        **dict(col_name="name")).rename(columns=dict(name="identifier")).loc[
                                                         :, ["col_name", "identifier", "bus"]])
    lgd_bus = pd.concat([lgd_EIC_bus.set_axis(lgd_EIC_bus_idx_extended),
                         lgd_name_bus.set_axis(lgd_name_bus_idx_extended)])
    dupl_EICs = net.line.EIC_Code.loc[net.line.EIC_Code.duplicated()]
    dupl_names = net.line.name.loc[net.line.name.duplicated()]

    def _geo_json_str(this_bus_geo: pd.Series) -> str:
        return f'{{"coordinates": [{this_bus_geo.at["lng"]}, {this_bus_geo.at["lat"]}], "type": "Point"}}'

    def _add_bus_geo_inner(bus: int) -> Optional[str]:
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
        is_missing = pd.DataFrame({
            "EIC": ~line_excerpt.EIC_Code.isin(
                lgd_bus.loc["EIC_Code"].index.get_level_values("identifier")),
            "name": ~line_excerpt.name.isin(
                lgd_bus.loc["name"].index.get_level_values("identifier"))
        }).set_axis(is_dupl.index)
        is_tieline = pd.Series(net.line.loc[is_dupl.index.get_level_values("line_index"),
        "Tieline"].values, index=is_dupl.index)

        # --- construct access_vals, i.e. values to take line geo data from lgd_bus
        # --- if not duplicated, take "EIC_Code". Otherwise and if not dupl, take "name".
        # --- Otherwise ignore. Do it for both from and to bus
        access_vals = pd.DataFrame({
            "col_name": "EIC_Code",
            "identifier": line_excerpt.EIC_Code.values,
            "bus": is_dupl.index.get_level_values("bus").values
        })  # default is EIC_Code
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

        # --- get this_bus_geo from EIC_Code or name with regard to access_vals
        this_bus_geo = lgd_bus.loc[iSl[
                                       access_vals.col_name, access_vals.identifier, access_vals.bus], :]

        if len(this_bus_geo) > 1:
            # reduce similar/equal lines
            this_bus_geo = this_bus_geo.loc[this_bus_geo.round(2).drop_duplicates().index]

        # --- return geo_json_str
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


# --- tertiary functions ---------------------------------------------------------------------------

def _float_col_comma_correction(data: dict[str, pd.DataFrame], key: str, col_names: list):
    for col_name in col_names:
        data[key][col_name] = pd.to_numeric(data[key][col_name].astype(str).str.replace(
            ",", "."), errors="coerce")


def _get_transformer_voltages(
        data: dict[str, pd.DataFrame], bus_idx: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    key = "Transformers"
    df = data[key]

    col_p, col_s = _find_voltage_cols_in_transformers_fuzzy(df)
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
    """Provides a DataFrame of data to allocate transformers to the buses according to their
    location names. If locations of transformers do not exist due to the data of the lines and
    tielines sheets, additional buses are created. If locations exist but have a far different
    voltage level than the transformer, either a warning is logged or additional buses are created
    according to rel_deviation_threshold_for_trafo_bus_creation and log_rel_vn_deviation.

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    data : dict[str, pd.DataFrame]
        _description_
    bus_idx : pd.Series
        Series of indices and corresponding location names and voltage levels in the MultiIndex of
        the Series
    vn_hv_kv : np.ndarray
        nominal voltages of the hv side of the transformers
    vn_lv_kv : np.ndarray
        Nominal voltages of the lv side of the transformers
    rel_deviation_threshold_for_trafo_bus_creation : float, optional
        If the voltage level of transformer locations is far different than the transformer data,
        additional buses are created. rel_deviation_threshold_for_trafo_bus_creation defines the
        tolerance in which no additional buses are created. By default 0.2
    log_rel_vn_deviation : float, optional
        This parameter allows a range below rel_deviation_threshold_for_trafo_bus_creation in which
        a warning is logged instead of a creating additional buses. By default 0.12

    Returns
    -------
    pd.DataFrame
        information to which bus the trafos should be connected to. Columns are
        ["name", "hv_bus", "lv_bus", "vn_hv_kv", "vn_lv_kv", ...]
    """

    if rel_deviation_threshold_for_trafo_bus_creation < log_rel_vn_deviation:
        logger.warning(
            f"Given parameters violates the ineqation {rel_deviation_threshold_for_trafo_bus_creation=} >= {log_rel_vn_deviation=}. Therefore, rel_deviation_threshold_for_trafo_bus_creation={log_rel_vn_deviation} is assumed.")
        rel_deviation_threshold_for_trafo_bus_creation = log_rel_vn_deviation

    key = "Transformers"
    dfT = data[key]
    bus_location_names = set(net.bus.name)

    # Standortnamen (fuzzy)
    trafo_bus_names = _get_transformer_location_fullname_series_fuzzy(dfT)
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
        trafo_connections.loc[~isin, bus_col] = bus_idx.loc[list(tuple(zip(
            trafo_connections.loc[~isin, "name"],
            trafo_connections.loc[~isin, next_col]
        )))].values

        need_bus_creation = trafo_connections[rel_dev_col] > rel_deviation_threshold_for_trafo_bus_creation
        if need_bus_creation.any():
            tso_series_tr = _get_transformer_tso_series_fuzzy(dfT)
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
    if (n_add_buses := len(duplicated_buses)):
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

            # Try both candidates with higher cutoff (0.8 instead of 0.6)
            query = cand1 if cand1 else cand2
            if query:
                matches = difflib.get_close_matches(query, bus_names_list, n=1, cutoff=0.8)
                if matches:
                    # Accept close match only if it's very similar
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
    # just keep one bus per name and vn_kv. If there are multiple buses of different TSOs, join the
    # TSO strings:
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


def get_grid_groups(net: pandapowerNet, **kwargs) -> pd.DataFrame:
    notravbuses_dict = dict() if "notravbuses" not in kwargs.keys() else {
        "notravbuses": kwargs.pop("notravbuses")}
    grid_group_buses = [set_ for set_ in connected_components(create_nxgraph(net, **kwargs),
                                                              **notravbuses_dict)]
    grid_groups = pd.DataFrame({"buses": grid_group_buses})
    grid_groups["n_buses"] = grid_groups["buses"].apply(len)
    return grid_groups


def _lng_lat_to_df(dict_: dict, line_EIC: str, line_name: str) -> pd.DataFrame:
    return pd.DataFrame([
        [line_EIC, line_name, "from", "lng", dict_["lng"][0]],
        [line_EIC, line_name, "to", "lng", dict_["lng"][1]],
        [line_EIC, line_name, "from", "lat", dict_["lat"][0]],
        [line_EIC, line_name, "to", "lat", dict_["lat"][1]],
    ], columns=["EIC_Code", "name", "bus", "geo_dim", "value"])


def _fill_geo_at_one_sided_branches_without_geo_extent(net: pandapowerNet):
    def _check_geo_availablitiy(net: pandapowerNet) -> dict[str, Union[pd.Index, int]]:
        av = dict()  # availablitiy of geodata
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


def _multi_str_repl(st: str, repl: list[tuple]) -> str:
    for (old, new) in repl:
        st = st.replace(old, new)
    return st





if __name__ == "__main__":
    from pathlib import Path
    import os
    import pandapower as pp

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
        pp.to_json(net, pp_net_json_file)
    else:  # load net from already converted and stored net
        net = pp.from_json(pp_net_json_file)

    print(net)
    grid_groups = get_grid_groups(net)
    print(grid_groups)

    _fill_geo_at_one_sided_branches_without_geo_extent(net)

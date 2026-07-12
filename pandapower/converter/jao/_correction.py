# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Data correction and name normalization for the JAO static grid model.

The published data is maintained by hand and contains many inconsistencies: differing
capitalisation and spelling of the same substation across sheets, decimal commas, ``\\xa0``
placeholders, unicode variants of Ω/µ, tap strings wrapped in ``<...>`` etc. This module

  * harmonises the column headers and coerces the electrical parameters to floats
    (:func:`data_correction`), and
  * derives a list of ``(old_name -> new_name)`` rename rules that unify the many spellings of
    the same location before the network is built (:func:`generate_rename_locnames`).
"""

import re
import difflib
import logging
import unicodedata
from collections import defaultdict

import pandas as pd
from pandas.api.types import is_object_dtype

from pandapower.converter.jao import _schema
from pandapower.converter.jao._schema import SheetSchema

logger = logging.getLogger(__name__)

# Tokens that must be dropped from a transformer location before matching it to a bus name.
_TRAFO_STOPWORDS = {"tr", "pst", "trafo", "kv"}
_TRAFO_BLOCK_EXACT = {"", "LIPST", "EHPST", "TFO"}
# Split a transformer location on whitespace and the "-A<n>"/"-TD<n>"/"-PF<n>" unit tags only.
# Note: we deliberately do NOT split on "/", so that a slashed location such as "Hamburg/Nord"
# stays intact and can match the identically named bus.
_TRAFO_SPLIT = re.compile(r"[ ]+|-A[0-9]+|-TD[0-9]+|-PF[0-9]+")


def multi_str_repl(text: str, replacements: list[tuple[str, str]]) -> str:
    """Apply a list of ``(old, new)`` string replacements in order."""
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ==================================================================================================
# Name normalization / matching helpers
# ==================================================================================================

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _simplify_name(name: object) -> str:
    """Reduce a location name to a canonical uppercase base token.

    Removes trailing markers like ``(2)`` and directional suffixes (``/W``, ``-West``, ...),
    voltage texts (``220 kV``) and unifies delimiters, e.g. ``"Doerpen/W/W (2)" -> "DOERPEN"``.
    """
    trailing = re.compile(r"(?ix)(?:\s*\(\d+\) | (?:[/\-]\s*|\s+)(?:W(?:EST)?|E|N|S))\s*$")
    volt = re.compile(r"\b\d{2,4}\s*k?V\b", re.IGNORECASE)
    delims = re.compile(r"[/_.-]")

    s = _strip_accents(str(name)).encode("ascii", "ignore").decode()
    s = volt.sub("", s).rstrip()
    while True:  # remove trailing markers/directions until stable
        s2 = trailing.sub("", s)
        if s2 == s:
            break
        s = s2
    s = delims.sub(" ", s)
    return " ".join(s.split()).upper()


def _canonical_bus_key(s: object) -> str:
    """Canonical key used to detect bus names that differ only in case/accents/punctuation."""
    s = _strip_accents(str(s).casefold())
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"[-_/.,;:]+", " ", s)
    return re.sub(r"\s+", "", s).upper()


def normalize_trafo_name(name: object) -> tuple[str, str]:
    """Tokenize a transformer location into ``(joined, longest)`` matching candidates.

    Drops stopwords, blocked tokens and tokens that contain digits, then returns the remaining
    tokens joined by spaces and the single longest token. ``"/"`` is preserved inside tokens so
    a slashed location survives (see :data:`_TRAFO_SPLIT`).
    """
    parts = [p.strip().replace(" ", "") for p in _TRAFO_SPLIT.split(str(name).strip())]

    def keep(tok: str) -> bool:
        return (tok not in _TRAFO_BLOCK_EXACT
                and tok.lower() not in _TRAFO_STOPWORDS
                and not any(ch.isdigit() for ch in tok))

    filtered = [p for p in parts if keep(p)]
    joined = " ".join(filtered).strip()
    longest = max(filtered, key=len) if filtered else ""
    return joined, longest


def _suggest_closest(query: str, candidates: list[str]) -> str:
    if not query:
        return ""
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else ""


def collect_bus_location_names(data: dict[str, pd.DataFrame]) -> set[str]:
    """All unique substation ``Full_name`` strings found in the Lines/Tielines sheets."""
    names = []
    for key in ("Lines", "Tielines"):
        if key not in data:
            continue
        schema = SheetSchema(data[key])
        for side in ("Substation_1", "Substation_2"):
            col = schema.fullname_col(side)
            if col is not None:
                names.append(schema.series(col))
    if not names:
        return set()
    return set(pd.concat(names, ignore_index=True).dropna())


def _find_problematic_bus_name_variants(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bus names that are equal up to case/accents/punctuation, mapped to one representative.

    The representative is the most frequent spelling (ties broken by length, then lexically).
    Returns rows ``[original, suggested, reason="same_canonical_variant"]``.
    """
    cols = ["original", "suggested", "reason"]
    all_names = []
    for key in ("Lines", "Tielines"):
        if key not in data:
            continue
        schema = SheetSchema(data[key])
        for side in ("Substation_1", "Substation_2"):
            col = schema.fullname_col(side)
            if col is not None:
                all_names.append(schema.series(col))
    if not all_names:
        return pd.DataFrame(columns=cols)
    s = pd.concat(all_names, ignore_index=True).dropna()
    if s.empty:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame({"original": s})
    df["canonical"] = df["original"].map(_canonical_bus_key)
    freq = df["original"].value_counts()

    rows = []
    for _, sub in df.groupby("canonical"):
        uniques = sub["original"].unique()
        if len(uniques) <= 1:
            continue
        representative = sorted(uniques, key=lambda x: (-freq.get(x, 0), -len(x), x))[0]
        for orig in uniques:
            if orig != representative:
                rows.append({"original": orig, "suggested": representative,
                             "reason": "same_canonical_variant"})
    return pd.DataFrame(rows, columns=cols)


def _find_unmatched_transformer_locations(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Transformer locations that do not match any bus name after normalization.

    Returns rows ``[original, joined, longest, suggested, reason]``.
    """
    cols = ["original", "joined", "longest", "suggested", "reason"]
    if "Transformers" not in data:
        return pd.DataFrame(columns=cols)

    bus_names = collect_bus_location_names(data)
    bus_names_lower = {b.lower(): b for b in bus_names}
    trafo_names = SheetSchema(data["Transformers"]).transformer_location_series()

    rows = []
    for original in trafo_names:
        joined, longest = normalize_trafo_name(original)
        if (joined in bus_names or longest in bus_names
                or joined.lower() in bus_names_lower or longest.lower() in bus_names_lower):
            continue
        rows.append({
            "original": original,
            "joined": joined,
            "longest": longest,
            "suggested": _suggest_closest(joined or longest, list(bus_names)),
            "reason": "no_match_after_normalization",
        })
    return pd.DataFrame(rows, columns=cols)


def report_problematic_names(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the transformer-location and bus-variant reports into one DataFrame."""
    cols = ["original", "joined", "longest", "suggested", "reason"]
    trafo = _find_unmatched_transformer_locations(data)
    bus_vars = _find_problematic_bus_name_variants(data).reindex(columns=cols)
    combined = pd.concat([trafo.reindex(columns=cols), bus_vars],
                         ignore_index=True).drop_duplicates()
    if not combined.empty:
        logger.debug("Problematic JAO names after normalization:\n%s", combined.to_string())
    return combined


def generate_rename_locnames(data: dict[str, pd.DataFrame],
                             combined: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """Build ``(old -> new)`` rename rules that unify the many spellings of each location.

    Sources: bus-name variants that differ only in formatting, unmatched transformer locations
    (mapped to their closest existing bus name, with PST/TR prefix heuristics) and conservative
    suffix-removal proposals derived from the bus names themselves. Ambiguous rules (one source
    mapping to several targets) are dropped, and the result is de-duplicated while preserving
    order.
    """
    if combined is None:
        combined = report_problematic_names(data)

    bus_names = collect_bus_location_names(data)
    bus_names_lower = {b.lower(): b for b in bus_names}

    def resolve_to_bus(name: str) -> str:
        return bus_names_lower.get(name.lower(), name) if name else name

    renames: list[tuple[str, str]] = []

    def add(old: str, new: str) -> None:
        if not old or not new:
            return
        target = resolve_to_bus(new)
        if old != target:
            renames.append((old, target))

    def add_prefix_variants(target_base: str) -> None:
        for prefix in ("PST", "TR"):
            add(f"{prefix}{target_base.replace(' ', '')}", f"{prefix} {target_base}")
            add(f"{prefix}{target_base.upper().replace(' ', '')}",
                f"{prefix} {target_base.upper()}")

    if "reason" in combined.columns:
        # 1) bus-name formatting variants: original -> representative
        for _, row in combined.loc[combined["reason"] == "same_canonical_variant"].iterrows():
            add(str(row["original"]).strip(), str(row["suggested"]).strip())

        # 2) unmatched transformer locations -> suggestion / fallback
        for _, row in combined.loc[
                combined["reason"] == "no_match_after_normalization"].iterrows():
            orig = str(row.get("original", "")).strip()
            joined = str(row.get("joined", "")).strip()
            longest = str(row.get("longest", "")).strip()
            sugg = str(row.get("suggested", "")).strip()

            if sugg and (sugg in bus_names or sugg.lower() in bus_names_lower):
                add(orig, sugg)
                if joined and joined != sugg:
                    add(joined, sugg)
                if longest and longest != sugg:
                    add(longest, sugg)
            else:
                for val in (joined, longest):
                    if val and (val in bus_names or val.lower() in bus_names_lower):
                        add(orig, resolve_to_bus(val))
                        break

            if sugg:
                add_prefix_variants(resolve_to_bus(sugg))
            for token in (joined, longest):
                if token and sugg and token not in bus_names:
                    add(token, sugg)

    # 3) case harmonization straight from the transformer locations
    if "Transformers" in data:
        schema = SheetSchema(data["Transformers"])
        if schema.fullname_col(None) is not None:
            for original in schema.transformer_location_series():
                for tok in set(normalize_trafo_name(original)):
                    if tok and tok.lower() in bus_names_lower:
                        target = bus_names_lower[tok.lower()]
                        if tok != target:
                            add(tok, target)
                        add_prefix_variants(target)

    # 4) conservative suffix-removal proposals from the bus names themselves
    for name in bus_names:
        base = _simplify_name(name)
        if base and name != base:
            add(name, base)

    # drop ambiguous rules (one source -> several targets) and de-duplicate, keeping order
    targets_by_old = defaultdict(set)
    for old, new in renames:
        targets_by_old[old].add(new)

    out: list[tuple[str, str]] = []
    seen = set()
    for pair in renames:
        if len(targets_by_old[pair[0]]) == 1 and pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


# ==================================================================================================
# Data correction
# ==================================================================================================

def _correct_line_columns(df: pd.DataFrame) -> None:
    """Harmonise known Lines/Tielines header variants in place."""
    voltage = "Voltage_level(kV)"
    replace_map = {
        "Full Name": "Full_name",
        "Short Name": "Short_name",
        "Susceptance_B (µS)": "Susceptance_B(μS)",
        "Voltage_level (kV)": voltage,
        "Voltage_level [kV]": voltage,
    }
    cols = df.columns.to_frame(index=False)
    cols.iloc[:, 1] = cols.iloc[:, 1].replace(replace_map)
    cols.loc[cols.iloc[:, 1].isin([voltage, "Comment"]), cols.columns[0]] = None
    cols.loc[cols.iloc[:, 0].astype(str).str.startswith("Unnamed:"), cols.columns[0]] = None
    cols.loc[cols.iloc[:, 1] == _schema.LENGTH, cols.columns[0]] = _schema.ELECTRICAL_PARAMETERS
    df.columns = pd.MultiIndex.from_frame(cols)


def _correct_line_numerics(df: pd.DataFrame, schema: SheetSchema,
                           max_i_ka_fillna: float) -> None:
    """Coerce Imax and the R/X/B/length columns of a Lines/Tielines sheet to floats."""
    if _schema.LINE_IMAX in df.columns:
        fill = max_i_ka_fillna * 1e3
        df[_schema.LINE_IMAX] = pd.to_numeric(
            df[_schema.LINE_IMAX]
            .replace({"\xa0": fill, "-": fill, " ": fill})
            .astype(str).str.replace(",", ".", regex=False),
            errors="coerce")

    for col in ((_schema.ELECTRICAL_PARAMETERS, _schema.LENGTH),
                (_schema.ELECTRICAL_PARAMETERS, _schema.RESISTANCE),
                (_schema.ELECTRICAL_PARAMETERS, _schema.REACTANCE)):
        if col in df.columns:
            df[col] = _schema.numeric_column(df, col)

    for col in (schema.resistance_col(), schema.reactance_col(), schema.susceptance_col()):
        if col is not None:
            df.iloc[:, list(df.columns).index(col)] = _schema.numeric_column(df, col)


def _correct_transformer_taps(df: pd.DataFrame) -> None:
    """Clean the tap and phase-shifter columns of the Transformers sheet in place."""
    taps = df.loc[:, _schema.TAPS].fillna("").astype(str).str.replace(" ", "", regex=False)
    nonnull = taps.apply(len).astype(bool)
    nonnull_taps = taps.loc[nonnull]
    surrounded = nonnull_taps.str.startswith("<") & nonnull_taps.str.endswith(">")
    nonnull_taps.loc[surrounded] = nonnull_taps.loc[surrounded].str[1:-1]
    slash_sep = (~nonnull_taps.str.contains(";")) & nonnull_taps.str.contains("/")
    nonnull_taps.loc[slash_sep] = nonnull_taps.loc[slash_sep].str.replace("/", ";", regex=False)
    nonnull_taps.loc[nonnull_taps == "0"] = "0;0"
    df.loc[nonnull, _schema.TAPS] = nonnull_taps
    df.loc[~nonnull, _schema.TAPS] = "0;0"

    # phase shifters sometimes carry two "/"-separated values; keep the second one
    for col in ("Phase Regulation δu (%)", "Angle Regulation δu (%)"):
        tup = (_schema.PHASE_SHIFT_PROPERTIES, col)
        if tup in df.columns and is_object_dtype(df.loc[:, tup]):
            double = df.index[df.loc[:, tup].str.contains("/").fillna(False).astype(bool)]
            df.loc[double, tup] = df.loc[double, tup].str.split("/", expand=True)[1].str.replace(
                ",", ".", regex=False).astype(float).values


def data_correction(data: dict[str, pd.DataFrame], html_str: str | None,
                    max_i_ka_fillna: float) -> str | None:
    """Correct the Excel sheets in place and apply the rename rules to the HTML string.

    Returns the (possibly corrected) HTML string.
    """
    combined = report_problematic_names(data)
    rename_locnames = generate_rename_locnames(data, combined)

    # keep only high-similarity renames that do not map between two already existing bus names
    bus_names = collect_bus_location_names(data)
    filtered = []
    for old, new in rename_locnames:
        if old in bus_names and new in bus_names:
            continue
        if difflib.SequenceMatcher(None, old.lower(), new.lower()).ratio() < 0.8:
            continue
        filtered.append((old, new))
    rename_locnames = filtered

    for key in ("Lines", "Tielines"):
        if key not in data:
            continue
        df = data[key]
        _correct_line_columns(df)
        # the schema must be built AFTER the column headers were corrected above
        schema = SheetSchema(df)
        _ensure_line_tso_column(df)
        _correct_line_numerics(df, schema, max_i_ka_fillna)

        # unify capitalisation/spelling of the location names
        loc_cols = [(None, "NE_name"),
                    schema.fullname_col("Substation_1"),
                    schema.fullname_col("Substation_2")]
        for col in loc_cols:
            if col is not None and col in df.columns:
                df.loc[:, col] = df.loc[:, col].astype(str).str.strip().apply(
                    multi_str_repl, replacements=rename_locnames)

    html_str = multi_str_repl(html_str, rename_locnames)

    if "Transformers" in data:
        df = data["Transformers"]
        loc_name = ("Location", "Full Name")
        if loc_name in df.columns:
            df.loc[:, loc_name] = df.loc[:, loc_name].astype(str).str.strip().apply(
                multi_str_repl, replacements=rename_locnames)
        _correct_transformer_taps(df)

    return html_str


def _ensure_line_tso_column(df: pd.DataFrame) -> None:
    """Ensure a generic ``(None, "TSO")`` column exists, joining ``TSO 1``/``TSO 2`` if needed."""
    if (None, "TSO") in df.columns:
        return

    def first_present(candidates: list[tuple]) -> tuple | None:
        return next((t for t in candidates if t in df.columns), None)

    t1 = first_present([(None, "TSO 1"), (None, "TSO1")])
    t2 = first_present([(None, "TSO 2"), (None, "TSO2")])
    if t1 and t2:
        df[(None, "TSO")] = df.loc[:, t1].astype(str).str.strip() + "/" + \
            df.loc[:, t2].astype(str).str.strip()
    elif t1:
        df[(None, "TSO")] = df.loc[:, t1]
    elif t2:
        df[(None, "TSO")] = df.loc[:, t2]

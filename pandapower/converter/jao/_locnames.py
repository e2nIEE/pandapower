# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Normalization and replacement of inconsistent JAO location names."""

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_locname(name: object) -> str:
    """Returns a comparison key for a location name.

    The key is insensitive to case, accents, spacing, punctuation and special characters
    (e.g. "Rohrsdorf" and "Röhrsdorf", "T HAL-BYC" and "T HAL-BYĆ", "BAULER" and "Bauler" all map
    to the same key).
    """
    if not isinstance(name, str):
        return ""
    st = name.replace("\xa0", " ").replace("\u00ad", "")
    st = st.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
    st = st.replace("\u00df", "ss").replace("\u1e9e", "SS")
    st = unicodedata.normalize("NFKD", st)
    st = "".join(char for char in st if not unicodedata.combining(char))
    return re.sub(r"[^0-9a-z]+", " ", st.casefold()).strip()


def _parse_voltage_kv(name: object) -> int | None:
    """Returns the voltage level (kV) explicitly written in a name, e.g. "TR Ens 380kV TR 404" -> 380.

    Only voltages in the range 60...800 kV are considered, so that transformer numbers such as "441"
    are not mistaken for a voltage level.
    """
    for match in re.findall(r"(\d{2,3})\s*kV", str(name), flags=re.IGNORECASE):
        if 60 <= int(match) <= 800:
            return int(match)
    return None


def _split_name_voltage(normalized_name: str) -> tuple[str, int | None]:
    """Splits a trailing voltage level off a normalized name, e.g. "eemshaven 380" -> ("eemshaven", 380)."""
    tokens = normalized_name.split()
    if tokens and tokens[-1].isdigit():
        return " ".join(tokens[:-1]), int(tokens[-1])
    return normalized_name, None


def _locname_display_names(names: pd.Series) -> dict[str, str]:
    """Maps each raw location name spelling to the display spelling of its normalized group.

    For each group of spellings that share a normalization key, the display name is the most
    frequent raw spelling, preferring mixed-case over ALL-CAPS, then alphabetical order.
    """
    names = names.dropna().astype(str).str.strip()
    counts = names.value_counts()
    groups: dict[str, list[str]] = {}
    for name in counts.index:
        groups.setdefault(_normalize_locname(name), []).append(name)

    mapping: dict[str, str] = {}
    for raws in groups.values():
        mixed_case = [raw for raw in raws if raw != raw.upper()] or raws
        display = min(mixed_case, key=lambda raw: (-counts[raw], raw))
        for raw in raws:
            mapping[raw] = display

    merged = {key: raws for key, raws in groups.items() if len(raws) > 1}
    if merged:
        logger.info(f"Merged {len(merged)} groups of inconsistent location name spellings:")
        for raws in merged.values():
            logger.info(f"  {mapping[raws[0]]!r} <- {sorted(raws)}")

    return mapping


def _replace_locnames(names: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Applies the raw -> display name mapping to a Series, keeping missing values untouched."""
    stripped = names.where(names.isna(), names.astype(str).str.strip())
    return stripped.map(lambda name: mapping.get(name, name) if isinstance(name, str) else name)


def _replace_substrings(names: pd.Series, repl: list[tuple]) -> pd.Series:
    """Strips and applies substring replacements to all non-missing string values of a Series."""
    return names.map(lambda st: _multi_str_repl(st.strip(), repl) if isinstance(st, str) else st)


def _multi_str_repl(st: str, repl: list[tuple]) -> str:
    """Applies a list of (old, new) substring replacements to a string."""
    for old, new in repl:
        st = st.replace(old, new)
    return st

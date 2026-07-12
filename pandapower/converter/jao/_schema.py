# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""
Column resolution for the JAO static grid model sheets.

The JAO Excel files are maintained largely by hand, so the two-level column headers drift
between releases (a voltage column may live under ``Substation_2`` in one release and under an
``Unnamed:`` top level in another, ``TSO`` may be split into ``TSO 1``/``TSO 2``, unicode Ω/µ/μ
vary, etc.). Instead of hard-coding column tuples, the builders ask a :class:`SheetSchema` for
the column they need; the schema resolves it with a single fuzzy matcher based on a canonical,
accent- and punctuation-insensitive form of the label.
"""

import re
import difflib
import unicodedata

import numpy as np
import pandas as pd


# --- Level-1 (column) labels the converter looks for -----------------------------------------
FULL_NAME = "Full Name"
VOLTAGE_LEVEL = "voltage level kv"
LENGTH = "Length_(km)"
RESISTANCE = "Resistance_R(Ω)"
REACTANCE = "Reactance_X(Ω)"

# --- Level-0 (group) labels ------------------------------------------------------------------
ELECTRICAL_PARAMETERS = "Electrical Parameters"
PHASE_SHIFT_PROPERTIES = "Phase Shifting Properties"

# --- Imax column tuples ----------------------------------------------------------------------
LINE_IMAX = ("Maximum Current Imax (A)", "Fixed")
TAPS = (PHASE_SHIFT_PROPERTIES, "Taps used for RAO")


def canon(label: object) -> str:
    """Return a canonical form of a header label for fuzzy comparison.

    Lower-cases, strips accents, unifies the micro sign (µ/μ -> u) and drops every character
    that is not a letter or digit, so that e.g. ``"Susceptance_B (µS)"`` and
    ``"Susceptance_B(μS)"`` collapse to the same token.
    """
    s = str(label or "").strip()
    s = s.replace("µ", "u").replace("μ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    return re.sub(r"[^0-9a-z]+", "", s)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def to_numeric(values) -> np.ndarray:
    """Coerce string cells with decimal commas / stray whitespace to floats (NaN on failure)."""
    return pd.to_numeric(
        pd.Series(values).astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    ).to_numpy()


def numeric_column(df: pd.DataFrame, col: tuple) -> np.ndarray:
    """Numeric values of a MultiIndex column, robust to duplicate labels (position based)."""
    pos = list(df.columns).index(col)
    return to_numeric(df.iloc[:, pos])


class SheetSchema:
    """Resolve the columns a single JAO sheet exposes via fuzzy label matching.

    A schema is built once per sheet (``Lines``, ``Tielines`` or ``Transformers``) and shared by
    the correction and element-building steps. The generic :meth:`resolve` does the matching;
    the named accessors express the converter's intent (voltage column, full-name column, R/X/B
    columns, TSO series, ...).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # --- generic matcher ---------------------------------------------------------------------

    def resolve(self, target_label: str, min_ratio: float = 0.55,
                required_tokens: list[str] | None = None) -> tuple | None:
        """Return the MultiIndex column whose level-1 label best matches ``target_label``.

        ``required_tokens`` (already canonical substrings) are penalised when absent, which
        avoids accidental matches between similar labels. Returns ``None`` if nothing clears
        ``min_ratio``.
        """
        target = canon(target_label)
        required = required_tokens or []
        best, best_score = None, -1.0
        for col in self.df.columns:
            lvl1 = canon(col[1])
            score = _similarity(lvl1, target)
            if required and not all(tok in lvl1 for tok in required):
                score -= 0.15
            if score >= min_ratio and score > best_score:
                best, best_score = col, score
        return best

    def _position(self, col: tuple) -> int:
        return list(self.df.columns).index(col)

    def series(self, col: tuple) -> pd.Series:
        """Stripped string series of a column, robust to duplicate labels."""
        return self.df.iloc[:, self._position(col)].astype(str).str.strip()

    # --- Lines / Tielines --------------------------------------------------------------------

    def voltage_col(self) -> tuple | None:
        return self.resolve(VOLTAGE_LEVEL, min_ratio=0.45, required_tokens=["volt", "kv"])

    def fullname_col(self, side: str | None) -> tuple | None:
        """``("Substation_1"|"Substation_2", "Full name")`` for lines, ``("Location", ...)`` for
        transformers (``side=None``)."""
        if side is not None:
            top_target, top_element = canon(side), "substation"
        else:
            top_target, top_element = canon("location"), "location"
        lvl1_target = canon("full name")
        best, best_score = None, -1.0
        for col in self.df.columns:
            top_c, lvl1_c = canon(col[0]), canon(col[1])
            s0 = _similarity(top_c, top_target)
            s1 = _similarity(lvl1_c, lvl1_target)
            if top_element not in top_c:
                s0 -= 0.1
            if "fullname" not in lvl1_c and not ("full" in lvl1_c and "name" in lvl1_c):
                s1 -= 0.1
            score = 0.5 * (s0 + s1)
            if s0 >= 0.5 and s1 >= 0.6 and score > best_score:
                best, best_score = col, score
        return best

    def resistance_col(self) -> tuple | None:
        return self.resolve("resistance r ohm", min_ratio=0.5, required_tokens=["resistance"])

    def reactance_col(self) -> tuple | None:
        return self.resolve("reactance x ohm", min_ratio=0.5, required_tokens=["reactance"])

    def susceptance_col(self) -> tuple | None:
        return self.resolve("susceptance b us", min_ratio=0.5,
                            required_tokens=["susceptance", "b", "us"])

    def line_tso_array(self) -> np.ndarray:
        """Line-wise TSO string: prefer a generic ``TSO`` column, else join ``TSO 1``/``TSO 2``."""
        col = self.resolve("TSO", min_ratio=0.6, required_tokens=["tso"])
        if col is not None:
            return self.series(col).to_numpy()
        c1 = self.resolve("TSO 1", min_ratio=0.6, required_tokens=["tso"])
        c2 = self.resolve("TSO 2", min_ratio=0.6, required_tokens=["tso"])
        if c1 is not None and c2 is not None:
            return (self.series(c1) + "/" + self.series(c2)).to_numpy()
        if c1 is not None:
            return self.series(c1).to_numpy()
        if c2 is not None:
            return self.series(c2).to_numpy()
        return np.array([""] * len(self.df))

    def tso_series_for_side(self, side: str) -> pd.Series:
        """TSO series for one substation side, falling back to a generic ``TSO`` column."""
        target = "TSO 1" if side == "Substation_1" else "TSO 2"
        col = self.resolve(target, min_ratio=0.6, required_tokens=["tso"])
        if col is None:
            col = self.resolve("TSO", min_ratio=0.6, required_tokens=["tso"])
        if col is not None:
            return self.series(col)
        return pd.Series([""] * len(self.df), index=self.df.index)

    def string_values(self, target_label: str, tokens: list[str] | None = None,
                      default: object = "") -> np.ndarray:
        col = self.resolve(target_label, min_ratio=0.5, required_tokens=tokens)
        if col is None:
            return np.array([default] * len(self.df))
        return self.series(col).to_numpy()

    def numeric_values(self, target_label: str, tokens: list[str] | None = None,
                       default: float = 0.0) -> np.ndarray:
        vals = self.string_values(target_label, tokens=tokens, default=str(default))
        return pd.to_numeric(
            pd.Series(vals).str.replace(",", ".", regex=False), errors="coerce"
        ).fillna(default).to_numpy()

    # --- Transformers ------------------------------------------------------------------------

    def transformer_voltage_cols(self) -> tuple[tuple | None, tuple | None]:
        """Primary and secondary voltage columns, preferring a pair under the same top level."""
        top_target = canon(VOLTAGE_LEVEL)
        primaries, secondaries = [], []
        for col in self.df.columns:
            top_c, lvl1_c = canon(col[0]), canon(col[1])
            top_ok = _similarity(top_c, top_target) >= 0.45 and "volt" in top_c and "kv" in top_c
            if not top_ok:
                continue
            if _similarity(lvl1_c, canon("primary")) >= 0.7:
                primaries.append(col)
            if _similarity(lvl1_c, canon("secondary")) >= 0.7:
                secondaries.append(col)
        for p in primaries:
            same_top = [s for s in secondaries if canon(s[0]) == canon(p[0])]
            if same_top:
                return p, same_top[0]
        prim = max(primaries, key=lambda c: _similarity(canon(c[0]), top_target), default=None)
        sec = max(secondaries, key=lambda c: _similarity(canon(c[0]), top_target), default=None)
        return prim, sec

    def transformer_location_series(self) -> pd.Series:
        col = self.fullname_col(None)
        if col is None:
            raise KeyError("Transformers: could not fuzzy-match the Location / Full Name column.")
        return self.series(col)

    def transformer_tso_series(self) -> pd.Series:
        """TSO column for transformers, preferring one under a ``Location`` top level."""
        best, best_score = None, -1.0
        for col in self.df.columns:
            if "tso" not in canon(col[1]):
                continue
            score = _similarity(canon(col[1]), canon("tso"))
            if "location" in canon(col[0]):
                score += 0.2
            if score > best_score:
                best, best_score = col, score
        if best is None:
            return pd.Series([""] * len(self.df), index=self.df.index)
        return self.series(best)

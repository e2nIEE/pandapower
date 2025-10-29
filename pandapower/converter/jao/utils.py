
# Utility classes for:

# 1) Name normalization and location matching

# 2) Column fuzzy matching and schema handling

import re
import difflib
from typing import Any

import unicodedata
import pandas as pd
import numpy as np


class NameNormalizationUtils:
    """
    Utility class for:

      - Name normalization
      - Location matching

      - Cross-sheet rename rule generation

    """

    # String constants used by some methods (expected to match original module constants)
    FULL_NAME_STR = 'Full Name'

    @staticmethod
    def _simplify_name(name: str) -> str:
        """
        Normalize a location name into a canonical, comparable base token.

        Operations:

          - Convert to ASCII and strip accents.
          - Remove trailing markers like "(2)", directional suffixes (/W, -West, etc.).

          - Unify delimiters to spaces.
          - Remove voltage-level texts (e.g. "220 kV").

          - Collapse repeated whitespace and uppercase.

        Parameters
        ----------
        name : str
            Original location name as found in data.

        Returns
        -------
        str
            Canonical uppercase base token (without voltage markers), usable as key for grouping.

        Examples
        --------
        "Doerpen/W/W (2)" -> "DOERPEN"
        "Stade - West 220 kV" -> "STADE"
        """
        _RE_TRAILING_SAFE = re.compile(r"(?ix)(?:\s*\(\d+\) | (?:[/\-]\s*|\s+)(?:W(?:EST)?|E|N|S))\s*$")
        _RE_VOLT = re.compile(r"\b\d{2,4}\s*k?V\b", re.IGNORECASE)
        _RE_DELIMS = re.compile(r"[/_.-]")
        if not isinstance(name, str):
            name = str(name)
        s = unicodedata.normalize("NFKD", name)
        s = s.encode("ascii", "ignore").decode()
        s = _RE_VOLT.sub("", s)
        s = s.rstrip()
        # Remove trailing markers/directions until stable
        while True:
            s2 = _RE_TRAILING_SAFE.sub("", s)
            if s2 == s:
                break
            s = s2
        s = _RE_DELIMS.sub(" ", s)
        return " ".join(s.split()).upper()

    @staticmethod
    def _extra_variants_from_busnames(bus_names: set[str]) -> list[tuple[str, str]]:
        """
        Generate conservative (old, new) normalization pairs directly from raw bus names.

        This provides proposals like mapping 'Doerpen/W/W' -> 'DOERPEN' or 'Stade/W/W (2)' -> 'STADE'.

        Parameters
        ----------
        bus_names : set[str]
            Collection of bus location names seen in Lines/Tielines.

        Returns
        -------
        list[tuple[str, str]]
            List of renames from observed variants to canonical base tokens.
        """
        mapping = {}
        for n in bus_names:
            base = NameNormalizationUtils._simplify_name(n)
            if base and n != base:
                mapping.setdefault(n, base)
        return list(mapping.items())

    @staticmethod
    def collect_bus_location_names(data: dict[str, pd.DataFrame]) -> set[str]:
        """
        Collect all unique bus "Full_name" strings from Lines/Tielines sheets.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Excel sheets dict, expected to include "Lines", "Tielines" (if present).

        Returns
        -------
        set[str]
            Set of unique bus location names discovered.

        Notes
        -----
        Uses fuzzy detection of the ("Substation_*", "Full_name") columns.
        """
        names = []
        for key in [k for k in ["Lines", "Tielines"] if k in data]:
            df = data[key]
            for subst in ["Substation_1", "Substation_2"]:
                # Reuse schema util if available externally; else expect exact tuple present
                col = None
                # Try exact tuple first
                if (subst, "Full_name") in df.columns:
                    col = (subst, "Full_name")
                if col is not None:
                    s = df.loc[:, col].astype(str).str.strip()
                    names.append(s)
        if not names:
            return set()
        bus_series = pd.concat(names, ignore_index=True)
        return set(bus_series.dropna().tolist())

    @staticmethod
    def _strip_accents(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        nfkd = unicodedata.normalize("NFKD", s)
        return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

    @staticmethod
    def _canonical_bus_key(s: str) -> str:
        s = str(s)
        s = NameNormalizationUtils._strip_accents(s.casefold())
        s = re.sub(r"[()\[\]{}]", " ", s)
        s = re.sub(r"[-_/.,;:]+", " ", s)
        s = re.sub(r"\s+", "", s)
        return s.upper()

    @staticmethod
    def _normalize_transformer_name_for_matching(name: str) -> tuple[str, str]:
        """
        Normalize a transformer 'Location/Full Name' into tokens helpful for matching buses.

        The algorithm replicates the tokenization used in _find_trafo_locations:

          - Split by spaces and special patterns '-A\\d+', '-TD\\d+', '-PF\\d+', and '/'.
          - Filter tokens: drop known stopwords (tr, pst, trafo, kv), empty tokens, tokens with digits.

          - Construct two candidates:
            - joined: join remaining tokens with spaces

            - longest: single longest token

        Parameters
        ----------
        name : str
            Original transformer location string.

        Returns
        -------
        tuple[str, str]
            (joined, longest) tokens for matching.
        """
        if not isinstance(name, str):
            name = str(name)
        s = name.strip()

        parts_orig = re.split(r" +|-A\d+|-TD\d+|-PF\d+|/", s)
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

    @staticmethod
    def _suggest_closest(q: str, candidates: list[str], n: int = 1) -> str | None | Any:
        """
        Suggest the closest match from a candidate list using difflib.

        Parameters
        ----------
        q : str
            Query string.
        candidates : list[str]
            List of candidate strings to match against.
        n : int, optional
            Number of matches to consider (default 1).

        Returns
        -------
        str
            The best candidate or empty string if none found.
        """
        if not q:
            return ""
        try:
            matches = difflib.get_close_matches(q, candidates, n=n, cutoff=0.6)
            if matches:
                return matches[0]
        except Exception:
            lower_map = {}
            for c in candidates:
                lower_map.setdefault(c.lower(), []).append(c)
            if q.lower() in lower_map:
                return lower_map[q.lower()][0]
            return ""

    @staticmethod
    def generate_rename_locnames_from_combined(
        data: dict[str, pd.DataFrame],
        combined: pd.DataFrame | None = None
    ) -> list[tuple[str, str]]:
        """
        Generate a comprehensive list of (old_name -> new_name) rename rules to normalize naming.

        Sources (in order of construction):
          1) same_canonical_variant (bus variants that differ only in formatting/case).
          2) no_match_after_normalization (trafo locations that don't match any bus after normalization):

             - prefer suggestions that resolve to existing bus names.
             - fallback to joined/longest forms if they directly exist as bus names.

             - generate prefix heuristics for PST/TR like "PSTMIKULOWA" -> "PST MIKULOWA".
             - map tokens from joined/longest onto the suggested target.

          3) Case harmonization across all transformer location tokens.
          4) Conservative normalization proposals from bus names themselves (suffix removal).

        Safety measures:

          - Resolve suggested targets to existing bus names when possible (case-insensitive).
          - Filter out ambiguous mappings (one source to multiple targets).

          - Deduplicate while preserving insertion order.
          - Additional filter in _data_correction applies a similarity >= 0.8 and removes

            rules where both old and new are already present bus names.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Excel sheets dict.
        combined : pd.DataFrame, optional
            Pre-computed combination of problematic-name reports (from
            report_problematic_names_after_normalization). If None, recomputed.

        Returns
        -------
        list[tuple[str, str]]
            Ordered list of (old -> new) rename rules suitable for applying across data.
        """
        if combined is None:
            combined = NameNormalizationUtils.report_problematic_names_after_normalization(data)

        # Bus-Standorte einsammeln (Case-sensitiv + Case-insensitiv Map)
        bus_location_names = NameNormalizationUtils.collect_bus_location_names(data)
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
                    _add(compact_lower, spaced_lower)
                    _add(compact_upper, spaced_upper)

            # Zusätzliche Sicherheit: Tokens aus joined/longest auf Vorschlag mappen
            for token in (joined, longest):
                if token and sugg and token not in bus_location_names:
                    _add(token, sugg)

        # 3) Case-Harmonisierung direkt aus allen Transformer-Standorten
        if "Transformers" in data and ("Location", NameNormalizationUtils.FULL_NAME_STR) in data["Transformers"].columns:
            trafo_names = data["Transformers"].loc[:, ("Location", NameNormalizationUtils.FULL_NAME_STR)].astype(str).str.strip()
            for original in trafo_names:
                joined, longest = NameNormalizationUtils._normalize_transformer_name_for_matching(original)
                for tok in {joined, longest}:
                    if tok and tok.lower() in bus_location_names_lower:
                        target = bus_location_names_lower[tok.lower()]
                        if tok != target:
                            _add(tok, target)
                        for prefix in ("PST", "TR"):
                            compact_lower = f"{prefix}{target.replace(' ', '')}"
                            compact_upper = f"{prefix}{target.upper().replace(' ', '')}"
                            spaced_lower = f"{prefix} {target}"
                            spaced_upper = f"{prefix} {target.upper()}"
                            _add(compact_lower, spaced_lower)
                            _add(compact_upper, spaced_upper)

        # 4) Sicherheitsfilter: Entferne mehrdeutige Ersetzungen (ein alter Name -> mehrere Ziele)
        renames.extend(NameNormalizationUtils._extra_variants_from_busnames(bus_location_names))
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
                canonical_sources = [NameNormalizationUtils._canonical_bus_key(src) for src in sources]
                current_canonical = NameNormalizationUtils._canonical_bus_key(old)
                if canonical_sources.count(current_canonical) == len(canonical_sources):
                    filtered_renames.append((old, new))
                else:
                    continue
            else:
                filtered_renames.append((old, new))

        renames = filtered_renames
        return out

    @staticmethod
    def find_unmatched_transformer_locations_extended(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Identify transformer locations that fail to match any bus location name after normalization.

        For each Transformers['Location','Full Name'] entry:

          - Tokenize/normalize to (joined, longest) candidates (see _normalize_transformer_name_for_matching).
          - Check for exact or case-insensitive matches against Lines/Tielines bus names.

          - If not matched, suggest the closest bus name using difflib.
          - Return all unmatched entries with columns:

            ['original', 'joined', 'longest', 'suggested', 'reason']

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Excel sheets dict including a "Transformers" sheet.

        Returns
        -------
        pd.DataFrame
            Rows for unmatched transformer locations. Columns:

            - original: original location string
            - joined: joined token candidate

            - longest: longest token candidate
            - suggested: closest bus name suggestion (may be empty)

            - reason: "no_match_after_normalization"

        """
        rows = []
        if "Transformers" not in data:
            return pd.DataFrame(columns=["original", "joined", "longest", "suggested", "reason"])

        bus_location_names = NameNormalizationUtils.collect_bus_location_names(data)
        bus_location_names_lower = {b.lower(): b for b in bus_location_names}

        trafo_names = data["Transformers"].loc[:, ("Location", NameNormalizationUtils.FULL_NAME_STR)].astype(str).str.strip()

        for original in trafo_names:
            joined, longest = NameNormalizationUtils._normalize_transformer_name_for_matching(original)

            # Direct match
            if (joined and joined in bus_location_names) or (longest and longest in bus_location_names):
                continue

            # Case-insensitive match
            if (joined and joined.lower() in bus_location_names_lower) or \
               (longest and longest.lower() in bus_location_names_lower):
                continue

            # No match: suggest
            suggest = NameNormalizationUtils._suggest_closest(joined if joined else longest, list(bus_location_names))
            rows.append({
                "original": original,
                "joined": joined,
                "longest": longest,
                "suggested": suggest,
                "reason": "no_match_after_normalization"
            })

        df = pd.DataFrame(rows, columns=["original", "joined", "longest", "suggested", "reason"])
        return df

    @staticmethod
    def find_problematic_bus_name_variants(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Detect bus-name variants across Lines/Tielines that are equivalent except for case/diacritics/
        punctuation/spacing, and propose a canonical representative.

        Strategy:

          - Compute a canonical key per name via _canonical_bus_key.
          - For each canonical key that maps to multiple originals, choose the representative as:

            - The most frequent original; ties broken by longest length, then lexicographic order.
          - Emit rename rows: 'original' -> 'suggested', reason='same_canonical_variant'.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Excel sheets dict.

        Returns
        -------
        pd.DataFrame
            Columns ['original', 'suggested', 'reason'] where reason == "same_canonical_variant".
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
        df["canonical"] = df["original"].map(NameNormalizationUtils._canonical_bus_key)
        freq = df["original"].value_counts()

        out_rows = []
        for _, sub in df.groupby("canonical"):
            uniques = sub["original"].unique()
            if len(uniques) <= 1:
                continue
            uniques_sorted = sorted(uniques, key=lambda x: (-freq.get(x, 0), -len(x), x))
            suggested = uniques_sorted[0]
            for orig in uniques:
                if orig == suggested:
                    continue
                out_rows.append({"original": orig, "suggested": suggested, "reason": "same_canonical_variant"})
        return pd.DataFrame(out_rows, columns=["original", "suggested", "reason"])

    @staticmethod
    def report_problematic_names_after_normalization(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate name issues across sheets to inform normalization rules.

        Combines:

          - find_unmatched_transformer_locations_extended: transformer locations that fail to match.
          - find_problematic_bus_name_variants: bus naming variants that are essentially the same.

        Returns a combined DataFrame with harmonized columns:
        ['original', 'joined', 'longest', 'suggested', 'reason'] and prints it for debugging.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Excel sheets dict.

        Returns
        -------
        pd.DataFrame
            Combined report of problematic names across sources.
        """
        trafo = NameNormalizationUtils.find_unmatched_transformer_locations_extended(data)
        bus_vars = NameNormalizationUtils.find_problematic_bus_name_variants(data)
        if len(bus_vars):
            bus_vars = bus_vars.assign(joined="", longest="")
        cols = ["original", "joined", "longest", "suggested", "reason"]
        bus_vars = bus_vars.reindex(columns=["original", "joined", "longest", "suggested", "reason"])
        combined = pd.concat([trafo.reindex(columns=cols), bus_vars.reindex(columns=cols)],
                             ignore_index=True).drop_duplicates()
        print(combined)
        return combined


class ColumnFuzzyMatchingUtils:
    """
    Utility class for:

      - Column fuzzy matching
      - Schema handling across MultiIndex headers

      - TSO/Voltage column detection

    """
    # Constants used by several methods (expected to match original module constants)
    VOLTAGE_LEVEL_STR = 'voltage level kv'
    ELECTRICAL_PARAMETER_STR = 'Electrical Parameters'
    LENGTH_STR = 'Length_(km)'
    RESISTANCE_STR = 'Resistance_R(Ω)'
    REACTANCE_STR = 'Reactance_X(Ω)'
    FULL_NAME_STR = 'Full Name'
    TSO_1_STR = 'TSO 1'
    TSO_2_STR = 'TSO 2'

    @staticmethod
    def _canon_label(s: str) -> str:
        s = str(s or "").strip()
        s = s.replace("µ", "u").replace("μ", "u")
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.casefold()
        s = re.sub(r"[^0-9a-z]+", "", s)
        return s

    @staticmethod
    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def get_col_pos(df: pd.DataFrame, col: tuple) -> int | None:
        """
        Get the positional index of a given MultiIndex column tuple.
        """
        try:
            return list(df.columns).index(col)
        except ValueError:
            return None

    @staticmethod
    def _best_col_by_lvl1_similarity(df: pd.DataFrame,
                                     target_label: str,
                                     min_ratio: float = 0.55,
                                     required_tokens: list[str] | None = None) -> tuple | None:
        """
        Find the best-matching column by fuzzy similarity on level-1 label only.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex columns (2 levels).
        target_label : str
            Desired label to match (e.g. "voltage level kv").
        min_ratio : float, optional
            Minimal similarity ratio for acceptance.
        required_tokens : list[str], optional
            Tokens that must appear (in canonicalized label); reduces accidental matches.

        Returns
        -------
        tuple | None
            The best-matching MultiIndex column tuple (top-level, level-1), or None if not found.
        """
        tgt = ColumnFuzzyMatchingUtils._canon_label(target_label)
        req = set(required_tokens or [])
        best, best_score = None, -1.0
        for col in df.columns:
            lvl1 = ColumnFuzzyMatchingUtils._canon_label(col[1])
            score = ColumnFuzzyMatchingUtils._sim(lvl1, tgt)
            if req and not all(tok in lvl1 for tok in req):
                score -= 0.15
            if score >= min_ratio and score > best_score:
                best, best_score = col, score
        return best


    @staticmethod
    def find_voltage_cols_in_transformers_fuzzy(df: pd.DataFrame) -> tuple[list | None, list | None]:
        """
        Fuzzy-find primary and secondary voltage level columns in Transformers sheet.

        Prefers pairs that share the same top-level label. If multiple candidates exist,
        chooses the best pair based on similarity to the target top-level label.

        Parameters
        ----------
        df : pd.DataFrame
            Transformers DataFrame.

        Returns
        -------
        tuple[tuple | None, tuple | None]
            (primary_col, secondary_col), each may be None if not found.
        """
        tgt_top = ColumnFuzzyMatchingUtils._canon_label(ColumnFuzzyMatchingUtils.VOLTAGE_LEVEL_STR)
        prim_list, sec_list = [], []
        for col in df.columns:
            top_c = ColumnFuzzyMatchingUtils._canon_label(col[0])
            lvl1_c = ColumnFuzzyMatchingUtils._canon_label(col[1])
            top_ok = (ColumnFuzzyMatchingUtils._sim(top_c, tgt_top) >= 0.45) and ("volt" in top_c and "kv" in top_c)
            if top_ok and ColumnFuzzyMatchingUtils._sim(lvl1_c, ColumnFuzzyMatchingUtils._canon_label("primary")) >= 0.7:
                prim_list.append(col)
            if top_ok and ColumnFuzzyMatchingUtils._sim(lvl1_c, ColumnFuzzyMatchingUtils._canon_label("secondary")) >= 0.7:
                sec_list.append(col)
        # Prefer pairs with same top-level
        for p in prim_list:
            matches = [s for s in sec_list if ColumnFuzzyMatchingUtils._canon_label(s[0]) == ColumnFuzzyMatchingUtils._canon_label(p[0])]
            if matches:
                return p, matches[0]
        prim_best = max(prim_list, key=lambda c: ColumnFuzzyMatchingUtils._sim(ColumnFuzzyMatchingUtils._canon_label(c[0]), tgt_top), default=None)
        sec_best = max(sec_list,  key=lambda c: ColumnFuzzyMatchingUtils._sim(ColumnFuzzyMatchingUtils._canon_label(c[0]), tgt_top), default=None)
        return prim_best, sec_best

    @staticmethod
    def best_fullname_tuple_fuzzy(df: pd.DataFrame, subst: str | None = None) -> tuple | None:
        """
        Fuzzy-find ("Substation_*", "Full Name") column for Lines/Tielines or
        ("Location", "Full Name") column for Transformers.

        Parameters
        ----------
        df : pd.DataFrame
            Lines, Tielines or Transformers DataFrame.
        subst : str | None
            Either "Substation_1", "Substation_2" for Lines/Tielines, or None for Transformers.

        Returns
        -------
        tuple | None
            Matching column tuple if found, else None.
        """
        def _col_fuzzy_helper(element_str: str, top_c: str, s0: float, s1: float, lvl1_c: str, best_score: float, col: any, best: None) -> tuple | None:
            if element_str not in top_c:
                s0 -= 0.1
            if "fullname" not in lvl1_c and not ("full" in lvl1_c and "name" in lvl1_c):
                s1 -= 0.1
            score = 0.5 * (s0 + s1)
            if s0 >= 0.5 and s1 >= 0.6 and score > best_score:
                best, best_score = col, score
            return s0, s1, best, best_score

        if subst is not None:
            # Behavior for Lines/Tielines (Substation_1 or Substation_2)
            tgt0 = ColumnFuzzyMatchingUtils._canon_label(subst)  # "substation1"/"substation2"
            element_str = 'substation'
        else:
            # Behavior for Transformers (Location / Full Name)
            tgt0 = ColumnFuzzyMatchingUtils._canon_label("location")
            element_str = 'location'
        tgt1 = ColumnFuzzyMatchingUtils._canon_label("full name")
        best, best_score = None, -1.0
        for col in df.columns:
            top_c = ColumnFuzzyMatchingUtils._canon_label(col[0])
            lvl1_c = ColumnFuzzyMatchingUtils._canon_label(col[1])
            s0 = ColumnFuzzyMatchingUtils._sim(top_c, tgt0)
            s1 = ColumnFuzzyMatchingUtils._sim(lvl1_c, tgt1)
            s0, s1, best, best_score = _col_fuzzy_helper(element_str, top_c, s0, s1, lvl1_c, best_score, col, best)
        return best

    @staticmethod
    def best_susceptance_col_lines_fuzzy(df: pd.DataFrame) -> tuple | None:
        return ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(
            df, "susceptance b us",
            min_ratio=0.5, required_tokens=["susceptance", "b", "us"]
        )

    @staticmethod
    def best_resistance_col_lines_fuzzy(df: pd.DataFrame) -> tuple | None:
        return ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(
            df, "resistance r ohm",
            min_ratio=0.5, required_tokens=["resistance"]
        )

    @staticmethod
    def best_reactance_col_lines_fuzzy(df: pd.DataFrame) -> tuple | None:
        return ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(
            df, "reactance x ohm",
            min_ratio=0.5, required_tokens=["reactance"]
        )

    @staticmethod
    def get_transformer_location_fullname_series_fuzzy(df: pd.DataFrame) -> pd.Series:
        col = ColumnFuzzyMatchingUtils.best_fullname_tuple_fuzzy(df, None)
        if col is None:
            raise KeyError("Transformers: Location / Full Name per Fuzzy-Matching nicht gefunden.")
        pos = ColumnFuzzyMatchingUtils.get_col_pos(df, col)
        return df.iloc[:, pos].astype(str).str.strip()

    @staticmethod
    def values_by_lvl1_fuzzy(df: pd.DataFrame, target_label: str,
                              tokens: list[str] | None = None,
                              default="") -> np.ndarray:
        col = ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(df, target_label, min_ratio=0.5, required_tokens=tokens)
        if col is None:
            return np.array([default] * len(df))
        pos = ColumnFuzzyMatchingUtils.get_col_pos(df, col)
        return df.iloc[:, pos].astype(str).str.strip().values

    @staticmethod
    def values_by_lvl1_fuzzy_numeric(df: pd.DataFrame, target_label: str,
                                      tokens: list[str] | None = None,
                                      default=0.0) -> np.ndarray:
        vals = ColumnFuzzyMatchingUtils.values_by_lvl1_fuzzy(df, target_label, tokens=tokens, default=str(default))
        return pd.to_numeric(pd.Series(vals).str.replace(",", "."), errors="coerce").fillna(default).values

    @staticmethod
    def _find_first_present_lvl1(df: pd.DataFrame, variants: list[str]):
        lvl1 = df.columns.get_level_values(1)
        for lab in variants:
            if lab in lvl1:
                for col in df.columns:
                    if col[1] == lab:
                        return col
        return None

    @staticmethod
    def _series_by_lvl1(df: pd.DataFrame, label: str) -> pd.Series | None:
        lvl1 = df.columns.get_level_values(1)
        pos = np.flatnonzero(lvl1 == label)
        if pos.size:
            return df.iloc[:, pos[0]]
        return None

    @staticmethod
    def _values_by_lvl1(df: pd.DataFrame, label: str, default="") -> np.ndarray:
        s = ColumnFuzzyMatchingUtils._series_by_lvl1(df, label)
        if s is None:
            return np.array([default] * len(df))
        return s.astype(str).str.strip().values

    @staticmethod
    def get_line_tso_array(df: pd.DataFrame) -> np.ndarray:
        s = ColumnFuzzyMatchingUtils._series_by_lvl1(df, "TSO")
        if s is not None:
            return s.astype(str).str.strip().values
        s1 = ColumnFuzzyMatchingUtils._series_by_lvl1(df, ColumnFuzzyMatchingUtils.TSO_1_STR) or ColumnFuzzyMatchingUtils._series_by_lvl1(df, "TSO1")
        s2 = ColumnFuzzyMatchingUtils._series_by_lvl1(df, ColumnFuzzyMatchingUtils.TSO_2_STR) or ColumnFuzzyMatchingUtils._series_by_lvl1(df, "TSO2")
        if s1 is not None and s2 is not None:
            return (s1.astype(str).str.strip() + "/" + s2.astype(str).str.strip()).values
        if s1 is not None:
            return s1.astype(str).str.strip().values
        if s2 is not None:
            return s2.astype(str).str.strip().values
        return np.array([""] * len(df))

    @staticmethod
    def get_tso_col_for_subst(df: pd.DataFrame, subst: str) -> tuple | None:
        """
        Return a TSO column for a given substation side if available, else a generic TSO column.

        Parameters
        ----------
        df : pd.DataFrame
            Lines/Tielines DataFrame.
        subst : str
            "Substation_1" or "Substation_2".

        Returns
        -------
        tuple | None
            Column tuple if found, else None.
        """
        if subst == "Substation_1":
            col = ColumnFuzzyMatchingUtils._find_first_present_lvl1(df, [ColumnFuzzyMatchingUtils.TSO_1_STR, "TSO1"])
            if col is not None:
                return col
        elif subst == "Substation_2":
            col = ColumnFuzzyMatchingUtils._find_first_present_lvl1(df, [ColumnFuzzyMatchingUtils.TSO_2_STR, "TSO2"])
            if col is not None:
                return col
        return ColumnFuzzyMatchingUtils._find_first_present_lvl1(df, ["TSO"])

    @staticmethod
    def get_tso_series_for_side(df: pd.DataFrame, subst: str) -> pd.Series:
        """
        Retrieve the TSO series for a substation side (strict variant).

        If no side-specific TSO column is present, tries a generic "TSO". Else returns empty series.

        Parameters
        ----------
        df : pd.DataFrame
            Lines/Tielines DataFrame.
        subst : str
            "Substation_1" or "Substation_2".

        Returns
        -------
        pd.Series
            Series of TSO strings for each row.
        """
        col = ColumnFuzzyMatchingUtils.get_tso_col_for_subst(df, subst)
        if col is not None:
            return df.loc[:, col].astype(str).str.strip()
        col_generic = ColumnFuzzyMatchingUtils._find_first_present_lvl1(df, ["TSO"])
        if col_generic is not None:
            return df.loc[:, col_generic].astype(str).str.strip()
        return pd.Series([""] * len(df), index=df.index)

    @staticmethod
    def get_tso_series_for_side_fuzzy(df: pd.DataFrame, subst: str) -> pd.Series:
        """
        Retrieve the TSO series for a substation side via fuzzy matching.

        Parameters
        ----------
        df : pd.DataFrame
            Lines/Tielines DataFrame.
        subst : str
            "Substation_1" or "Substation_2".

        Returns
        -------
        pd.Series
            Series of TSO strings for each row; empty strings if not found.
        """
        target = ColumnFuzzyMatchingUtils.TSO_1_STR if subst == "Substation_1" else ColumnFuzzyMatchingUtils.TSO_2_STR
        col = ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(df, target, min_ratio=0.6, required_tokens=["tso"])
        if col is not None:
            pos = ColumnFuzzyMatchingUtils.get_col_pos(df, col)
            return df.iloc[:, pos].astype(str).str.strip()
        col = ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(df, "TSO", min_ratio=0.6, required_tokens=["tso"])
        if col is not None:
            pos = ColumnFuzzyMatchingUtils.get_col_pos(df, col)
            return df.iloc[:, pos].astype(str).str.strip()
        return pd.Series([""] * len(df), index=df.index)

    @staticmethod
    def get_line_tso_array_fuzzy(df: pd.DataFrame) -> np.ndarray:
        """
        Build a line-wise TSO string via fuzzy matching:

          - Try generic "TSO"
          - Else combine "TSO 1/TSO 2"

          - Else fall back to single-available side.

        Parameters
        ----------
        df : pd.DataFrame
            Lines/Tielines DataFrame.

        Returns
        -------
        np.ndarray
            Array of TSO strings.
        """
        col = ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(df, "TSO", min_ratio=0.6, required_tokens=["tso"])
        if col is not None:
            pos = ColumnFuzzyMatchingUtils.get_col_pos(df, col)
            return df.iloc[:, pos].astype(str).str.strip().values
        c1 = ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(df, ColumnFuzzyMatchingUtils.TSO_1_STR, min_ratio=0.6, required_tokens=["tso"])
        c2 = ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(df, ColumnFuzzyMatchingUtils.TSO_2_STR, min_ratio=0.6, required_tokens=["tso"])
        if c1 is not None and c2 is not None:
            p1 = ColumnFuzzyMatchingUtils.get_col_pos(df, c1); p2 = ColumnFuzzyMatchingUtils.get_col_pos(df, c2)
            return (df.iloc[:, p1].astype(str).str.strip() + "/" + df.iloc[:, p2].astype(str).str.strip()).values
        if c1 is not None:
            return df.iloc[:, ColumnFuzzyMatchingUtils.get_col_pos(df, c1)].astype(str).str.strip().values
        if c2 is not None:
            return df.iloc[:, ColumnFuzzyMatchingUtils.get_col_pos(df, c2)].astype(str).str.strip().values
        return np.array([""] * len(df))

    @staticmethod
    def get_transformer_tso_series_fuzzy(df: pd.DataFrame) -> pd.Series:
        """
        Retrieve a TSO column for Transformers via fuzzy scanning.

        Heuristic:

          - Prefer columns with level-1 label containing 'tso'.
          - Add slight score for top-level labels containing 'location'.

        Parameters
        ----------
        df : pd.DataFrame
            Transformers DataFrame.

        Returns
        -------
        pd.Series
            TSO strings per transformer (empty if not found).
        """
        best, best_score = None, -1.0
        for col in df.columns:
            if "tso" in ColumnFuzzyMatchingUtils._canon_label(col[1]):
                score = 0.0
                if "location" in ColumnFuzzyMatchingUtils._canon_label(col[0]):
                    score += 0.2
                score += ColumnFuzzyMatchingUtils._sim(ColumnFuzzyMatchingUtils._canon_label(col[1]), ColumnFuzzyMatchingUtils._canon_label("tso"))
                if score > best_score:
                    best, best_score = col, score
        if best is None:
            return pd.Series([""] * len(df), index=df.index)
        pos = ColumnFuzzyMatchingUtils.get_col_pos(df, best)
        return df.iloc[:, pos].astype(str).str.strip()

    @staticmethod
    def get_voltage_tuple(df: pd.DataFrame) -> tuple | None:
        return ColumnFuzzyMatchingUtils._best_col_by_lvl1_similarity(
            df, target_label=ColumnFuzzyMatchingUtils.VOLTAGE_LEVEL_STR,
            min_ratio=0.45, required_tokens=["volt", "kv"]
        )

    @staticmethod
    def ensure_line_tso_column(df: pd.DataFrame) -> None:
        """
        Ensure a generic (None, "TSO") column exists on Lines/Tielines.

        If only "TSO 1"/"TSO 2" are present, creates a combined "TSO" column by concatenation.
        This helps unify downstream handling for TSO metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Lines or Tielines DataFrame with MultiIndex columns.

        Notes
        -----
        Column is created in-place when needed.
        """
        def _first_present_tuple(df: pd.DataFrame, candidates: list[tuple]) -> tuple | None:
            for t in candidates:
                if t in df.columns:
                    return t
            return None

        if (None, "TSO") not in df.columns:
            t1 = _first_present_tuple(df, [(None, ColumnFuzzyMatchingUtils.TSO_1_STR), (None, "TSO1")])
            t2 = _first_present_tuple(df, [(None, ColumnFuzzyMatchingUtils.TSO_2_STR), (None, "TSO2")])
            if t1 and t2:
                df[(None, "TSO")] = df.loc[:, t1].astype(str).str.strip() + "/" + df.loc[:, t2].astype(str).str.strip()
            elif t1:
                df[(None, "TSO")] = df.loc[:, t1]
            elif t2:
                df[(None, "TSO")] = df.loc[:, t2]
            # else: leave unset; downstream is robust

    @staticmethod
    def _first_present_tuple(df: pd.DataFrame, candidates: list[tuple]) -> tuple | None:
        for t in candidates:
            if t in df.columns:
                return t
        return None

class MiscUtils:
    @staticmethod
    def float_col_comma_correction(data: dict[str, pd.DataFrame], key: str, col_names: list):
        for col_name in col_names:
            data[key][col_name] = pd.to_numeric(data[key][col_name].astype(str).str.replace(
                ",", "."), errors="coerce")

    @staticmethod
    def multi_str_repl(st: str, repl: list[tuple]) -> str:
        for (old, new) in repl:
            st = st.replace(old, new)
        return st
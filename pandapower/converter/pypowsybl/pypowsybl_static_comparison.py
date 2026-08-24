# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pandapower.converter.pypowsybl.pypowsybl_comparison_utils import (
    PyPowSyBlComparisonUtilsMixin,
)

logger = logging.getLogger(__name__)


class PyPowSyBlStaticComparisonMixin(PyPowSyBlComparisonUtilsMixin):
    """
    Provide static transfer comparisons for converted networks.

    This mixin builds and logs comparison tables between values expected from the
    powsybl source network and values created in pandapower network. It checks
    buses, generators, lines, transformers, loads, switches, and converted impedance
    elements.

    """

    def _log_static_comparison_outputs(self) -> None:
        """
        Build and log the static transfer comparison output.

        This is a convenience wrapper around the transfer table builder and logger.

        """
        self.transfer_table = self._build_transfer_table()
        self._log_transfer_table()

    def _build_transfer_table(self) -> pd.DataFrame:
        """
        Build the complete static powsybl-to-pandapower transfer table.

        Returns:
            DataFrame with one row per checked element parameter.

        """
        rows: list[dict[str, Any]] = []

        self._build_bus_transfer_rows(rows)
        self._build_generator_transfer_rows(rows)
        self._build_line_transfer_rows(rows)
        self._build_2w_trafo_transfer_rows(rows)
        self._build_3w_trafo_transfer_rows(rows)
        self._build_load_transfer_rows(rows)
        self._build_switch_transfer_rows(rows)

        columns = [
            "element_type",
            "element",
            "parameter",
            "unit",
            "powsybl_expected",
            "pandapower_created",
            "delta",
            "status",
        ]

        return pd.DataFrame(rows, columns=columns)

    def _build_transfer_dataframes(
        self, only_errors=False
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Split the transfer table into grouped DataFrames.

        Args:
            only_errors: If ``True``, keep only rows whose status is not ``OK``.

        Returns:
            Tuple containing grouped transfer DataFrames and transfer summary.

        """
        if self.transfer_table is None:
            self.transfer_table = self._build_transfer_table()

        table = self.transfer_table.copy()

        if only_errors:
            table = table[table["status"] != "OK"].copy()

        self.transfer_dataframes = {}

        if not table.empty:
            for element_type, dataframe in table.groupby("element_type", sort=False):
                self.transfer_dataframes[str(element_type)] = dataframe.reset_index(
                    drop=True
                ).copy()

        self.transfer_summary = self._build_transfer_summary_dataframe()

        return self.transfer_dataframes, self.transfer_summary

    def _build_transfer_summary_dataframe(self) -> pd.DataFrame:
        """
        Build a status-count summary for the static transfer table.

        Returns:
            DataFrame containing counts per status and a total error count.

        """
        if self.transfer_table is None or self.transfer_table.empty:
            return pd.DataFrame(
                [{"status": "TOTAL_ERRORS", "count": 0}], columns=["status", "count"]
            )

        summary = (
            self.transfer_table["status"]
            .value_counts(dropna=False)
            .rename_axis("status")
            .reset_index(name="count")
        )

        error_count = int((self.transfer_table["status"] != "OK").sum())
        error_summary = pd.DataFrame([{"status": "TOTAL_ERRORS", "count": error_count}])

        return pd.concat([summary, error_summary], ignore_index=True)

    def _log_transfer_table(self, only_errors=False) -> None:
        """
        Log grouped static transfer comparison tables.

        Args:
            only_errors: If ``True``, log only failing comparison rows.

        """
        self.transfer_dataframes, self.transfer_summary = (
            self._build_transfer_dataframes(only_errors=only_errors)
        )

        logger.info("\nSTATIC COMPARISON POWSYBL -> PANDAPOWER")
        logger.info(
            "Note: IDs and names are used only for mapping, not as value comparisons."
        )
        logger.info(
            "Expected pandapower values derived from powsybl are compared against the "
            "actually created values."
        )

        if not self.transfer_dataframes:
            logger.info("\nNo comparison values available.")
            logger.info("\nCOMPARISON_SUMMARY\n%s", self.transfer_summary)
            return

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            240,
            "display.max_colwidth",
            80,
            "display.float_format",
            "{:.12g}".format,
        ):
            for element_type, dataframe in self.transfer_dataframes.items():
                logger.info(
                    "\nCOMPARISON_DATAFRAME_%s\n%s",
                    str(element_type).upper(),
                    dataframe,
                )

            logger.info("\nCOMPARISON_SUMMARY\n%s", self.transfer_summary)

    def _build_bus_transfer_rows(self, rows) -> None:
        """Append static bus nominal-voltage comparison rows."""
        pyp_buses = self.pyp_net.get_bus_breaker_view_buses()
        pyp_voltage_levels = self.pyp_net.get_voltage_levels()
        pp_buses = self.pandap_net.bus

        for bus_name, row in pyp_buses.iterrows():
            pp_row = self._comparison_get_pp_match(pp_buses, bus_name)

            if pp_row is None:
                self._add_missing_pp_element(rows, "bus", bus_name)
                continue

            voltage_level_id = row.get("voltage_level_id")

            if voltage_level_id in pyp_voltage_levels.index:
                expected_vn_kv = self._comparison_to_float(
                    pyp_voltage_levels.at[voltage_level_id, "nominal_v"], np.nan
                )
            else:
                expected_vn_kv = np.nan

            self._add_transfer_check(
                rows,
                "bus",
                bus_name,
                "vn_kv",
                expected_vn_kv,
                pp_row.get("vn_kv"),
                "kV",
            )

    def _build_generator_transfer_rows(self, rows) -> None:
        """
        Append static generator parameter comparison rows.

        The method checks active power, voltage magnitude, limits, slack state,
        and service state.

        """
        pyp_gens = self.pyp_net.get_generators(all_attributes=True)
        pp_gens = self.pandap_net.gen
        slack_generator_id = self._select_slack_generator_id(pyp_gens)

        for gen_id, row in pyp_gens.iterrows():
            pp_row = self._comparison_get_pp_match(pp_gens, gen_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "gen", gen_id)
                continue

            pp_bus_idx = pp_row.get("bus")
            bus_vn_kv = np.nan

            if pp_bus_idx in self.pandap_net.bus.index:
                bus_vn_kv = self._comparison_to_float(
                    self.pandap_net.bus.at[pp_bus_idx, "vn_kv"], np.nan
                )

            target_v_kv = self._comparison_to_float(row.get("target_v"), np.nan)

            if np.isfinite(target_v_kv) and np.isfinite(bus_vn_kv) and bus_vn_kv > 0:
                expected_vm_pu = target_v_kv / bus_vn_kv
            else:
                expected_vm_pu = self.default_vm_pu

            checks = [
                (
                    "p_mw",
                    self._comparison_to_float(
                        row.get("target_p"), self.default_active_power_mw
                    ),
                    pp_row.get("p_mw"),
                    "MW",
                ),
                ("vm_pu", expected_vm_pu, pp_row.get("vm_pu"), "pu"),
                (
                    "sn_mva",
                    self._comparison_to_float(row.get("rated_s"), np.nan),
                    pp_row.get("sn_mva"),
                    "MVA",
                ),
                (
                    "min_p_mw",
                    self._comparison_to_float(row.get("min_p"), np.nan),
                    pp_row.get("min_p_mw"),
                    "MW",
                ),
                (
                    "max_p_mw",
                    self._comparison_to_float(row.get("max_p"), np.nan),
                    pp_row.get("max_p_mw"),
                    "MW",
                ),
                (
                    "slack",
                    str(gen_id) == str(slack_generator_id),
                    pp_row.get("slack"),
                    "",
                ),
                (
                    "in_service",
                    self._to_bool(row.get("connected"), True),
                    pp_row.get("in_service"),
                    "",
                ),
            ]

            for parameter, expected, actual, unit in checks:
                self._add_transfer_check(
                    rows, "gen", gen_id, parameter, expected, actual, unit
                )

    def _build_line_transfer_rows(self, rows) -> None:
        """Append static line parameter comparison rows."""
        pyp_lines = self.pyp_net.get_lines(all_attributes=True)
        pp_lines = self.pandap_net.line
        pp_impedances = getattr(self.pandap_net, "impedance", pd.DataFrame())

        for line_id, row in pyp_lines.iterrows():
            pp_row = self._comparison_get_pp_match(pp_lines, line_id)

            if pp_row is None:
                pp_impedance_row = self._comparison_get_pp_match(pp_impedances, line_id)

                if (
                    pp_impedance_row is None
                    and not pp_impedances.empty
                    and "powsybl_id" in pp_impedances.columns
                ):
                    match = pp_impedances[
                        pp_impedances["powsybl_id"].astype(str) == str(line_id)
                    ]

                    if not match.empty:
                        pp_impedance_row = match.iloc[0]

                if pp_impedance_row is None:
                    self._add_missing_pp_element(rows, "line", line_id)
                    continue

                expected_impedance = (
                    self._comparison_expected_cross_voltage_impedance_values(
                        row, pp_impedance_row
                    )
                )

                for parameter, unit in (
                    ("rft_pu", "pu"),
                    ("xft_pu", "pu"),
                    ("rtf_pu", "pu"),
                    ("xtf_pu", "pu"),
                    ("sn_mva", "MVA"),
                    ("powsybl_g_total_s", "S"),
                    ("powsybl_b_total_s", "S"),
                    ("powsybl_element_type", ""),
                    ("in_service", ""),
                ):
                    self._add_transfer_check(
                        rows,
                        "impedance",
                        line_id,
                        parameter,
                        expected_impedance.get(parameter),
                        pp_impedance_row.get(parameter),
                        unit,
                    )

                continue

            expected = self._comparison_expected_line_values(row)

            for parameter, unit in (
                ("length_km", "km"),
                ("r_ohm_per_km", "Ohm/km"),
                ("x_ohm_per_km", "Ohm/km"),
                ("c_nf_per_km", "nF/km"),
                ("g_us_per_km", "uS/km"),
                ("parallel", ""),
                ("in_service", ""),
            ):
                self._add_transfer_check(
                    rows,
                    "line",
                    line_id,
                    parameter,
                    expected.get(parameter),
                    pp_row.get(parameter),
                    unit,
                )

    def _comparison_expected_line_values(self, row) -> dict[str, Any]:
        """
        Calculate expected pandapower line parameters from a powsybl line row.

        Returns:
            Dictionary containing length, impedance, shunt, parallel, and service-state
            values in pandapower's expected units.

        """
        r_ohm = self._to_float(row.get("r"), self.default_impedance_ohm)
        x_ohm = self._to_float(row.get("x"), self.default_impedance_ohm)

        g1_s = self._to_float(row.get("g1"), self.default_admittance_s)
        g2_s = self._to_float(row.get("g2"), g1_s)

        b1_s = self._to_float(row.get("b1"), self.default_admittance_s)
        b2_s = self._to_float(row.get("b2"), b1_s)

        g_total_s = g1_s + g2_s
        b_total_s = b1_s + b2_s

        parallel = self._comparison_to_float(row.get("parallel"), np.nan)
        if not np.isfinite(parallel) or parallel <= 0:
            parallel = self.default_parallel

        length_km = self._comparison_to_float(row.get("length_km"), np.nan)

        if not np.isfinite(length_km):
            length_km = self._comparison_to_float(row.get("length"), np.nan)

        if not np.isfinite(length_km) or length_km <= 0:
            length_km = self.default_length_km

        r_ohm_per_km = r_ohm * parallel / length_km
        x_ohm_per_km = x_ohm * parallel / length_km

        c_nf_per_km = (b_total_s * 1e9) / (
            2.0 * np.pi * self.pandap_net.f_hz * length_km * parallel
        )

        g_us_per_km = (g_total_s * 1e6) / (length_km * parallel)

        in_service = self._to_bool(row.get("connected1"), True) and self._to_bool(
            row.get("connected2"), True
        )

        return {
            "length_km": length_km,
            "r_ohm_per_km": r_ohm_per_km,
            "x_ohm_per_km": x_ohm_per_km,
            "c_nf_per_km": c_nf_per_km,
            "g_us_per_km": g_us_per_km,
            "parallel": parallel,
            "in_service": in_service,
        }

    def _comparison_expected_cross_voltage_impedance_values(
        self, row, pp_impedance_row
    ) -> dict[str, Any]:
        """
        Calculate expected pandapower impedance values for cross-voltage powsybl lines.

        Returns:
            Expected pandapower impedance values and stored powsybl metadata.

        """
        r_ohm = self._to_float(row.get("r"), self.default_impedance_ohm)
        x_ohm = self._to_float(row.get("x"), self.default_impedance_ohm)

        if abs(r_ohm) < 1e-12 and abs(x_ohm) < 1e-12:
            x_ohm = 1e-6

        g1_s = self._to_float(row.get("g1"), self.default_admittance_s)
        g2_s = self._to_float(row.get("g2"), g1_s)
        b1_s = self._to_float(row.get("b1"), self.default_admittance_s)
        b2_s = self._to_float(row.get("b2"), b1_s)

        g_total_s = g1_s + g2_s
        b_total_s = b1_s + b2_s

        from_bus = pp_impedance_row.get("from_bus")
        to_bus = pp_impedance_row.get("to_bus")

        from_vn_kv = self._get_bus_vn_kv(from_bus)
        to_vn_kv = self._get_bus_vn_kv(to_bus)

        sn_mva = self._to_float(
            pp_impedance_row.get("sn_mva"), self.default_base_sn_mva
        )

        if not np.isfinite(sn_mva) or sn_mva <= 0:
            sn_mva = self.default_base_sn_mva

        in_service = self._to_bool(row.get("connected1"), True) and self._to_bool(
            row.get("connected2"), True
        )

        return {
            "rft_pu": self._ohm_to_pu(r_ohm, from_vn_kv, sn_mva),
            "xft_pu": self._ohm_to_pu(x_ohm, from_vn_kv, sn_mva),
            "rtf_pu": self._ohm_to_pu(r_ohm, to_vn_kv, sn_mva),
            "xtf_pu": self._ohm_to_pu(x_ohm, to_vn_kv, sn_mva),
            "sn_mva": sn_mva,
            "powsybl_g_total_s": g_total_s,
            "powsybl_b_total_s": b_total_s,
            "powsybl_element_type": "cross_voltage_line",
            "in_service": in_service,
        }

    def _build_2w_trafo_transfer_rows(self, rows) -> None:
        """Append static two-windings transformer comparison rows."""
        pyp_2w_trafos = self.pyp_net.get_2_windings_transformers(all_attributes=True)
        pp_trafos = self.pandap_net.trafo

        for trafo_id, row in pyp_2w_trafos.iterrows():
            pp_row = self._comparison_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "trafo_2w", trafo_id)
                continue

            expected = self._comparison_expected_2w_trafo_values(row)

            if expected is None:
                self._add_missing_pp_element(rows, "trafo_2w", trafo_id)
                continue

            for parameter, unit in (
                ("sn_mva", "MVA"),
                ("vn_hv_kv", "kV"),
                ("vn_lv_kv", "kV"),
                ("vk_percent", "%"),
                ("vkr_percent", "%"),
                ("pfe_kw", "kW"),
                ("i0_percent", "%"),
                ("shift_degree", "deg"),
                ("in_service", ""),
            ):
                self._add_transfer_check(
                    rows,
                    "trafo_2w",
                    trafo_id,
                    parameter,
                    expected.get(parameter),
                    pp_row.get(parameter),
                    unit,
                )

    def _comparison_expected_2w_trafo_values(self, row) -> dict[str, Any] | None:
        """
        Calculate expected pandapower values for a two-winding transformer.

        Returns:
            Dictionary with pandapower transformer parameters or ``None`` when required
            rated-voltage data is missing or invalid.

        """
        r_ohm = self._to_float(row.get("r"), self.default_impedance_ohm)
        x_ohm = self._to_float(row.get("x"), self.default_impedance_ohm)
        g_s = self._to_float(row.get("g"), self.default_admittance_s)
        b_s = self._to_float(row.get("b"), self.default_admittance_s)

        rated_u1 = self._comparison_to_float(row.get("rated_u1"), np.nan)
        rated_u2 = self._comparison_to_float(row.get("rated_u2"), np.nan)
        rated_s = self._comparison_to_float(row.get("rated_s"), np.nan)

        if not np.isfinite(rated_u1) or rated_u1 <= 0:
            return None

        if not np.isfinite(rated_u2) or rated_u2 <= 0:
            return None

        if not np.isfinite(rated_s) or rated_s <= 0:
            rated_s = self.default_trafo_sn_mva

        if rated_u1 >= rated_u2:
            vn_hv_kv = rated_u1
            vn_lv_kv = rated_u2
        else:
            vn_hv_kv = rated_u2
            vn_lv_kv = rated_u1

        z_base_side2 = (rated_u2**2) / rated_s

        r_pu = r_ohm / z_base_side2
        x_pu = x_ohm / z_base_side2

        vk_percent = 100.0 * np.hypot(r_pu, x_pu)
        vkr_percent = 100.0 * r_pu

        if not np.isfinite(vk_percent) or vk_percent <= 0:
            vk_percent = self.default_trafo_vk_percent

        if not np.isfinite(vkr_percent) or vkr_percent <= 0:
            vkr_percent = 0.0

        y_pu = complex(g_s, b_s) * z_base_side2
        i0_percent = 100.0 * abs(y_pu)

        pfe_kw = (rated_u2**2) * g_s * 1000.0

        expected_alpha = self._comparison_to_float(
            row.get("alpha"), self.default_shift_degree
        )

        in_service = self._to_bool(row.get("connected1"), True) and self._to_bool(
            row.get("connected2"), True
        )

        return {
            "sn_mva": rated_s,
            "vn_hv_kv": vn_hv_kv,
            "vn_lv_kv": vn_lv_kv,
            "vk_percent": vk_percent,
            "vkr_percent": vkr_percent,
            "pfe_kw": pfe_kw,
            "i0_percent": i0_percent,
            "shift_degree": expected_alpha,
            "in_service": in_service,
        }

    def _build_3w_trafo_transfer_rows(self, rows) -> None:
        """Append static three-winding transfer comparison rows."""
        pyp_3w_trafos = self.pyp_net.get_3_windings_transformers(all_attributes=True)
        pp_trafos = self.pandap_net.trafo3w

        for trafo_id, row in pyp_3w_trafos.iterrows():
            pp_row = self._comparison_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "trafo_3w", trafo_id)
                continue

            expected = self._comparison_expected_3w_trafo_values(row)

            if expected is None:
                self._add_missing_pp_element(rows, "trafo_3w", trafo_id)
                continue

            for parameter, unit in (
                ("sn_hv_mva", "MVA"),
                ("sn_mv_mva", "MVA"),
                ("sn_lv_mva", "MVA"),
                ("vn_hv_kv", "kV"),
                ("vn_mv_kv", "kV"),
                ("vn_lv_kv", "kV"),
                ("vk_hv_percent", "%"),
                ("vk_mv_percent", "%"),
                ("vk_lv_percent", "%"),
                ("vkr_hv_percent", "%"),
                ("vkr_mv_percent", "%"),
                ("vkr_lv_percent", "%"),
                ("pfe_kw", "kW"),
                ("i0_percent", "%"),
                ("shift_mv_degree", "deg"),
                ("shift_lv_degree", "deg"),
                ("in_service", ""),
            ):
                self._add_transfer_check(
                    rows,
                    "trafo_3w",
                    trafo_id,
                    parameter,
                    expected.get(parameter),
                    pp_row.get(parameter),
                    unit,
                )

    def _comparison_expected_3w_trafo_values(self, row) -> dict[str, Any] | None:
        """
        Calculate expected pandapower values for a three-winding transformer.

        The method mirrors the creation logic used during conversion so that the debug
        table can verify voltage-side ordering, pairwise impedance values, losses,
        phase shifts, and service state.

        Returns:
            Dictionary with expected pandapower parameters or ``None`` when required
            data is missing.

        """
        legs: list[dict[str, Any]] = []

        for leg_no in (1, 2, 3):
            bus_ref = self._get_first_non_missing_value(
                row, f"bus{leg_no}_id", f"bus_breaker_bus{leg_no}_id", f"bus{leg_no}"
            )

            connected = self._to_bool(
                self._get_first_non_missing_value(
                    row, f"connected{leg_no}", default=not self._is_missing(bus_ref)
                ),
                default=not self._is_missing(bus_ref),
            )

            if self._is_missing(bus_ref):
                connected = False

            rated_u = self._to_float(
                self._get_first_non_missing_value(
                    row, f"rated_u{leg_no}", f"ratedU{leg_no}", default=np.nan
                ),
                np.nan,
            )

            rated_s = self._to_float(
                self._get_first_non_missing_value(
                    row, f"rated_s{leg_no}", f"ratedS{leg_no}", default=np.nan
                ),
                np.nan,
            )

            if not np.isfinite(rated_u) or rated_u <= 0:
                return None

            if not np.isfinite(rated_s) or rated_s <= 0:
                rated_s = self.default_trafo_sn_mva

            r_ohm = self._to_float(
                self._get_first_non_missing_value(
                    row,
                    f"r{leg_no}_at_current_tap",
                    f"r{leg_no}",
                    default=self.default_impedance_ohm,
                ),
                self.default_impedance_ohm,
            )

            x_ohm = self._to_float(
                self._get_first_non_missing_value(
                    row,
                    f"x{leg_no}_at_current_tap",
                    f"x{leg_no}",
                    default=self.default_impedance_ohm,
                ),
                self.default_impedance_ohm,
            )

            g_s = self._to_float(
                self._get_first_non_missing_value(
                    row,
                    f"g{leg_no}_at_current_tap",
                    f"g{leg_no}",
                    default=self.default_admittance_s,
                ),
                self.default_admittance_s,
            )

            b_s = self._to_float(
                self._get_first_non_missing_value(
                    row,
                    f"b{leg_no}_at_current_tap",
                    f"b{leg_no}",
                    default=self.default_admittance_s,
                ),
                self.default_admittance_s,
            )

            alpha_degree = self._to_float(
                self._get_first_non_missing_value(
                    row, f"alpha{leg_no}", default=self.default_shift_degree
                ),
                self.default_shift_degree,
            )

            z_ohm = complex(r_ohm, x_ohm)

            z_base_ohm = (rated_u**2) / rated_s
            y_relative = complex(g_s, b_s) * z_base_ohm

            i0_percent = 100.0 * abs(y_relative)
            pfe_kw = (rated_u**2) * g_s * 1000.0

            legs.append(
                {
                    "leg_no": leg_no,
                    "rated_u": rated_u,
                    "rated_s": rated_s,
                    "z_ohm": z_ohm,
                    "i0_percent": i0_percent,
                    "pfe_kw": pfe_kw,
                    "connected": connected,
                    "alpha_degree": alpha_degree,
                }
            )

        if len(legs) != 3:
            return None

        legs.sort(key=lambda leg: leg["rated_u"], reverse=True)

        hv_leg = legs[0]
        mv_leg = legs[1]
        lv_leg = legs[2]
        reference_u_kv = hv_leg["rated_u"]

        vk_hv_percent, vkr_hv_percent = self._calculate_3w_pair_short_circuit_values(
            hv_leg, mv_leg, reference_u_kv
        )

        vk_mv_percent, vkr_mv_percent = self._calculate_3w_pair_short_circuit_values(
            mv_leg, lv_leg, reference_u_kv
        )

        vk_lv_percent, vkr_lv_percent = self._calculate_3w_pair_short_circuit_values(
            hv_leg, lv_leg, reference_u_kv
        )

        pfe_kw = hv_leg["pfe_kw"] + mv_leg["pfe_kw"] + lv_leg["pfe_kw"]

        i0_percent = hv_leg["i0_percent"] + mv_leg["i0_percent"] + lv_leg["i0_percent"]

        shift_mv_degree = mv_leg["alpha_degree"] - hv_leg["alpha_degree"]
        shift_lv_degree = lv_leg["alpha_degree"] - hv_leg["alpha_degree"]

        in_service = hv_leg["connected"] and mv_leg["connected"] and lv_leg["connected"]

        return {
            "vn_hv_kv": hv_leg["rated_u"],
            "vn_mv_kv": mv_leg["rated_u"],
            "vn_lv_kv": lv_leg["rated_u"],
            "sn_hv_mva": hv_leg["rated_s"],
            "sn_mv_mva": mv_leg["rated_s"],
            "sn_lv_mva": lv_leg["rated_s"],
            "vk_hv_percent": vk_hv_percent,
            "vk_mv_percent": vk_mv_percent,
            "vk_lv_percent": vk_lv_percent,
            "vkr_hv_percent": vkr_hv_percent,
            "vkr_mv_percent": vkr_mv_percent,
            "vkr_lv_percent": vkr_lv_percent,
            "pfe_kw": pfe_kw,
            "i0_percent": i0_percent,
            "shift_mv_degree": shift_mv_degree,
            "shift_lv_degree": shift_lv_degree,
            "in_service": in_service,
        }

    def _build_load_transfer_rows(self, rows) -> None:
        """Append static load parameter comparison rows."""
        pyp_loads = self.pyp_net.get_loads(all_attributes=True)
        pp_loads = self.pandap_net.load

        for load_id, row in pyp_loads.iterrows():
            pp_row = self._comparison_get_pp_match(pp_loads, load_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "load", load_id)
                continue

            self._add_transfer_check(
                rows,
                "load",
                load_id,
                "p_mw",
                self._comparison_to_float(row.get("p0"), self.default_active_power_mw),
                pp_row.get("p_mw"),
                "MW",
            )

            self._add_transfer_check(
                rows,
                "load",
                load_id,
                "q_mvar",
                self._comparison_to_float(
                    row.get("q0"), self.default_reactive_power_mvar
                ),
                pp_row.get("q_mvar"),
                "MVar",
            )

            self._add_transfer_check(
                rows,
                "load",
                load_id,
                "in_service",
                self._to_bool(row.get("connected"), True),
                pp_row.get("in_service"),
            )

    def _build_switch_transfer_rows(self, rows) -> None:
        """
        Append static switch parameter comparison rows.

        The method checks the closed state, switch type, and switch impedance.

        """
        pyp_switches = self.pyp_net.get_switches(all_attributes=True)
        pp_switches = self.pandap_net.switch

        for switch_id, row in pyp_switches.iterrows():
            pp_row = self._comparison_get_pp_match(pp_switches, switch_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "switch", switch_id)
                continue

            expected_closed = not self._to_bool(row.get("open"), False)
            expected_type = self._get_pp_switch_type(row)
            expected_z_ohm = self.default_switch_z_ohm

            self._add_transfer_check(
                rows,
                "switch",
                switch_id,
                "closed",
                expected_closed,
                pp_row.get("closed"),
            )

            self._add_transfer_check(
                rows, "switch", switch_id, "type", expected_type, pp_row.get("type")
            )

            self._add_transfer_check(
                rows,
                "switch",
                switch_id,
                "z_ohm",
                expected_z_ohm,
                pp_row.get("z_ohm"),
                "Ohm",
            )
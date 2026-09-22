# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandapower.auxiliary import pandapowerNet


class PyPowSyBlComparisonUtilsMixin:
    """
    Provide shared helper methods for powsybl and pandapower comparisons.

    This mixin contains utility methods for tolerant value comparison, missing-value
    handling in comparison tables, pandapower result lookup, delta calculation, and
    shared transfer-table row creation.
    
    """

    if TYPE_CHECKING:
        # Converter state and helpers that are provided by ``PyPowSyBlConverter``.
        pyp_net: Any
        pandap_net: pandapowerNet
        transfer_table: pd.DataFrame | None
        transfer_dataframes: dict[str, pd.DataFrame]
        transfer_summary: pd.DataFrame
        loadflow_table: pd.DataFrame | None
        loadflow_dataframes: dict[str, pd.DataFrame]
        loadflow_summary: pd.DataFrame
        powsybl_loadflow_result: Any
        pandapower_loadflow_result: dict[str, bool] | None
        default_vm_pu: float
        default_parallel: float
        default_length_km: float
        default_shift_degree: float
        default_base_sn_mva: float
        default_impedance_ohm: float
        default_admittance_s: float
        default_active_power_mw: float
        default_reactive_power_mvar: float
        default_trafo_sn_mva: float
        default_trafo_vk_percent: float
        default_switch_z_ohm: float

        def _to_float(self, value, default=np.nan) -> float: ...

        def _to_bool(self, value, default=False) -> bool: ...

        def _is_missing(self, value) -> bool: ...

        def _get_first_non_missing_value(self, row, *keys, default=None) -> Any: ...

        def _get_bus_vn_kv(self, pp_bus_idx) -> float: ...

        def _get_pp_switch_type(self, switch_row, default=None) -> str: ...

        def _ohm_to_pu(self, z_ohm, vn_kv, sn_mva) -> float: ...

        def _select_slack_generator_id(self, generators) -> Any | None: ...

        def _calculate_3w_pair_short_circuit_values(
            self, leg_a, leg_b, reference_u_kv
        ) -> tuple[float, float]: ...

    def _comparison_to_float(self, value, default=np.nan) -> float:
        """
        Convert a comparison value to ``float`` while rejecting invalid values.

        Boolean values are intentionally not converted to floats because they represent
        states rather than numeric quantities in the comparison tables.

        """
        try:
            if self._is_missing(value):
                return default

            if hasattr(value, "item"):
                value = value.item()

            if isinstance(value, (bool, np.bool_)):
                return default

            value = float(value)

            if np.isnan(value):
                return default

            return value

        except (TypeError, ValueError):
            return default

    def _comparison_get_pp_match(self, pp_table, element_id) -> pd.Series | None:
        """
        Find a pandapower element row by its ``name`` column.

        The comparison first tries an exact match and then falls back to string-based
        matching to handle ids with different scalar types.

        """
        if pp_table is None or pp_table.empty:
            return None

        if "name" not in pp_table.columns:
            return None

        match = pp_table[pp_table["name"] == element_id]

        if match.empty:
            match = pp_table[pp_table["name"].astype(str) == str(element_id)]

        if match.empty:
            return None

        return match.iloc[0]

    def _comparison_get_pp_impedance_match(self, element_id) -> pd.Series | None:
        """Find a pandapower impedance row by name or stored powsybl id."""
        pp_impedances = getattr(self.pandap_net, "impedance", pd.DataFrame())

        pp_row = self._comparison_get_pp_match(pp_impedances, element_id)

        if pp_row is not None:
            return pp_row

        if (
            pp_impedances is not None
            and not pp_impedances.empty
            and "powsybl_id" in pp_impedances.columns
        ):
            match = pp_impedances[
                pp_impedances["powsybl_id"].astype(str) == str(element_id)
            ]

            if not match.empty:
                return match.iloc[0]

        return None

    def _comparison_values_equal(
        self, expected, actual, abs_tol=1e-6, rel_tol=1e-6
    ) -> bool:
        """
        Compare two comparison values with tolerant numerical handling.

        Numeric values are compared with ``numpy.isclose``. Missing values compare equal
        only when both sides are missing. Non-numeric values are compared as stripped
        strings, while booleans are compared as booleans.

        """
        expected_missing = self._is_missing(expected)
        actual_missing = self._is_missing(actual)

        if expected_missing and actual_missing:
            return True

        if expected_missing != actual_missing:
            return False

        if isinstance(expected, (bool, np.bool_)) or isinstance(
            actual, (bool, np.bool_)
        ):
            return bool(expected) == bool(actual)

        expected_float = self._comparison_to_float(expected, np.nan)
        actual_float = self._comparison_to_float(actual, np.nan)

        if np.isfinite(expected_float) and np.isfinite(actual_float):
            return bool(
                np.isclose(actual_float, expected_float, rtol=rel_tol, atol=abs_tol)
            )

        return str(expected).strip() == str(actual).strip()

    def _comparison_delta(self, expected, actual) -> float | str:
        """Return the numerical delta ``actual - expected`` when possible."""
        expected_float = self._comparison_to_float(expected, np.nan)
        actual_float = self._comparison_to_float(actual, np.nan)

        if np.isfinite(expected_float) and np.isfinite(actual_float):
            return actual_float - expected_float

        return ""

    def _comparison_get_result_values(self, row, *keys, default=np.nan) -> Any:
        """
        Return the first available result value from a powsybl result row.

        Args:
            row: Source row that may contain different naming variants.
            *keys: Candidate column names to check in order.
            default: Value returned if no usable result is found.

        Returns:
            First available non-missing result value or ``default``.

        """
        if row is None:
            return default

        for key in keys:
            if key in row.index:
                value = row.get(key)
                if not self._is_missing(value):
                    return value

        return default

    def _comparison_ampere_to_kiloampere(self, value) -> float:
        """Convert a current value from ampere to kiloampere for comparison."""
        value = self._comparison_to_float(value, np.nan)

        if np.isfinite(value):
            return value / 1000.0

        return np.nan

    def _comparison_get_pp_result_row(
        self, result_table, element_index
    ) -> pd.Series | None:
        """
        Return a pandapower result row by element index.

        Returns:
            The matching pandas row or ``None`` if the result table is missing, empty,
            or does not contain the requested index.

        """
        if result_table is None or result_table.empty:
            return None

        if element_index in result_table.index:
            return result_table.loc[element_index]

        return None

    def _add_transfer_check(
        self,
        rows,
        element_type,
        element,
        parameter,
        powsybl_expected,
        pandapower_created,
        unit="",
        abs_tol=1e-6,
        rel_tol=1e-6,
    ) -> None:
        """
        Append one static conversion row.

        The method compares the expected value derived from powsybl with the value that
        was created in pandapower and stores the result status in the transfer table.

        """
        is_equal = self._comparison_values_equal(
            powsybl_expected, pandapower_created, abs_tol=abs_tol, rel_tol=rel_tol
        )

        rows.append(
            {
                "element_type": element_type,
                "element": element,
                "parameter": parameter,
                "unit": unit,
                "powsybl_expected": powsybl_expected,
                "pandapower_created": pandapower_created,
                "delta": self._comparison_delta(powsybl_expected, pandapower_created),
                "status": "OK" if is_equal else "ERROR",
            }
        )

    def _add_missing_pp_element(self, rows, element_type, element) -> None:
        """Append a transfer-table row for a missing pandapower element."""
        rows.append(
            {
                "element_type": element_type,
                "element": element,
                "parameter": "element_present",
                "unit": "",
                "powsybl_expected": True,
                "pandapower_created": False,
                "delta": "",
                "status": "ERROR",
            }
        )
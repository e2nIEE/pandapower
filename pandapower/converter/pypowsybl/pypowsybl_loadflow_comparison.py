# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandapower import runpp
from pandapower.auxiliary import soft_dependency_error
from pandapower.converter.pypowsybl.pypowsybl_comparison_utils import (
    PyPowSyBlComparisonUtilsMixin,
)

logger = logging.getLogger(__name__)


def _require_pypowsybl_loadflow(function_name: str) -> Any:
    """Import pypowsybl.loadflow only when load-flow comparison is actually used."""
    try:
        import pypowsybl.loadflow as pyp_lf
    except ImportError:
        soft_dependency_error(function_name, "pypowsybl")
        raise

    return pyp_lf


class PyPowSyBlLoadflowComparisonMixin(PyPowSyBlComparisonUtilsMixin):
    """
    Provide load-flow comparison methods for converted networks.

    This mixin runs AC load-flow calculations in powsybl and pandapower, builds
    grouped result comparison tables, and compares voltages, powers, currents,
    convergence status, and supported element results.

    """

    def _log_loadflow_comparison_table(self, only_errors=False) -> None:
        """
        Log grouped AC load-flow comparison tables.

        Args:
            only_errors: If ``True``, log only failing load-flow comparison rows.

        """
        self.loadflow_table = self._build_loadflow_table()
        self.loadflow_dataframes, self.loadflow_summary = (
            self._build_loadflow_dataframes(only_errors=only_errors)
        )

        logger.info("\nLOAD FLOW COMPARISON POWSYBL -> PANDAPOWER")
        logger.info(
            "Note: After conversion, an AC load flow is calculated in both networks."
        )
        logger.info(
            "Result values from powsybl are compared against the pandapower res_* "
            "tables."
        )

        if not self.loadflow_dataframes:
            logger.info("\nNo load flow comparison values available.")
            logger.info("\nLOADFLOW_COMPARISON_SUMMARY\n%s", self.loadflow_summary)
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
            for element_type, dataframe in self.loadflow_dataframes.items():
                logger.info(
                    "\nLOADFLOW_COMPARISON_DATAFRAME_%s\n%s",
                    str(element_type).upper(),
                    dataframe,
                )

            logger.info("\nLOADFLOW_COMPARISON_SUMMARY\n%s", self.loadflow_summary)

    def _build_loadflow_table(self) -> pd.DataFrame:
        """
        Run load-flow calculations and build the result comparison table.

        The method first executes powsybl and pandapower load flows. If both converge,
        it compares result values for supported element types. Otherwise, it records a
        status row explaining why the numerical comparison was skipped.

        Returns:
            DataFrame with load-flow static rows and result comparison rows.

        """
        rows: list[dict[str, Any]] = []

        pyp_status = self._run_powsybl_loadflow_for_comparison()
        pp_status = self._run_pandapower_loadflow_for_comparison()

        self._add_loadflow_execution_rows(rows, pyp_status, pp_status)

        if pyp_status.get("success") and pp_status.get("success"):
            self._build_loadflow_bus_rows(rows)
            self._build_loadflow_generator_rows(rows)
            self._build_loadflow_line_rows(rows)
            self._build_loadflow_2w_trafo_rows(rows)
            self._build_loadflow_3w_trafo_rows(rows)
            self._build_loadflow_load_rows(rows)
        else:
            rows.append(
                {
                    "element_type": "loadflow_status",
                    "element": "comparison",
                    "parameter": "comparison_executed",
                    "unit": "",
                    "powsybl_result": pyp_status.get("message", ""),
                    "pandapower_result": pp_status.get("message", ""),
                    "delta": "",
                    "status": "ERROR",
                }
            )

        columns = [
            "element_type",
            "element",
            "parameter",
            "unit",
            "powsybl_result",
            "pandapower_result",
            "delta",
            "status",
        ]

        return pd.DataFrame(rows, columns=columns)

    def _build_loadflow_dataframes(
        self, only_errors=False
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Split the load-flow into grouped DataFrames.

        Args:
            only_errors: If ``True``, keep only rows whose status is not ``OK``.

        Returns:
            Tuple containing grouped load-flow DataFrames and the load-flow summary.

        """
        if self.loadflow_table is None:
            self.loadflow_table = self._build_loadflow_table()

        table = self.loadflow_table.copy()

        if only_errors:
            table = table[table["status"] != "OK"].copy()

        self.loadflow_dataframes = {}

        if not table.empty:
            for element_type, dataframe in table.groupby("element_type", sort=False):
                self.loadflow_dataframes[str(element_type)] = dataframe.reset_index(
                    drop=True
                ).copy()

        self.loadflow_summary = self._build_loadflow_summary_dataframe()

        return self.loadflow_dataframes, self.loadflow_summary

    def _build_loadflow_summary_dataframe(self) -> pd.DataFrame:
        """
        Build a status-count summary for the load-flow comparison table.

        Returns:
            DataFrame containing counts per status and a total error count.

        """
        if self.loadflow_table is None or self.loadflow_table.empty:
            return pd.DataFrame(
                [{"status": "TOTAL_ERRORS", "count": 0}], columns=["status", "count"]
            )

        summary = (
            self.loadflow_table["status"]
            .value_counts(dropna=False)
            .rename_axis("status")
            .reset_index(name="count")
        )

        error_count = int((self.loadflow_table["status"] != "OK").sum())
        error_summary = pd.DataFrame([{"status": "TOTAL_ERRORS", "count": error_count}])

        return pd.concat([summary, error_summary], ignore_index=True)

    def _run_powsybl_loadflow_for_comparison(self) -> dict[str, Any]:
        """
        Run a powsybl AC load-flow for comparison.

        Returns:
            Dictionary containing success flag, a readable status message, and the
            raw powsybl component results.

        """
        pyp_lf = _require_pypowsybl_loadflow("_run_powsybl_loadflow_for_comparison()")

        try:
            self.powsybl_loadflow_result = pyp_lf.run_ac(self.pyp_net)
            results = self.powsybl_loadflow_result or []

            statuses = []
            for index, result in enumerate(results):
                status = getattr(result, "status", None)
                statuses.append(f"component_{index}={status}")

            success = bool(results) and all(
                self._comparison_loadflow_status_is_converged(
                    getattr(result, "status", "OK")
                )
                for result in results
            )

            if not results:
                success = False
                message = "No powsybl ComponentResults received."
            else:
                message = "; ".join(statuses)

            return {"success": success, "message": message, "results": results}

        except Exception as exc:
            self.powsybl_loadflow_result = None
            return {
                "success": False,
                "message": f"powsybl load flow failed: {exc}",
                "results": [],
            }

    def _run_pandapower_loadflow_for_comparison(self) -> dict[str, Any]:
        """
        Run a pandapower AC load-flow for comparison.

        Returns:
            Dictionary containing a success flag and a readable convergence message.

        """
        try:
            runpp(
                self.pandap_net,
                algorithm="nr",
                calculate_voltage_angles=True,
                init="auto",
            )

            converged = bool(getattr(self.pandap_net, "converged", False))
            self.pandapower_loadflow_result = {"converged": converged}

            return {
                "success": converged,
                "message": "converged"
                if converged
                else "pandapower load flow did not converge",
            }

        except Exception as exc:
            self.pandapower_loadflow_result = None
            return {"success": False, "message": f"pandapower load flow failed: {exc}"}

    def _comparison_loadflow_status_is_converged(self, status) -> bool:
        """Return whether a powsybl load-flow status indicates convergence."""
        status_text = str(getattr(status, "name", status)).upper()
        return status_text == "CONVERGED" or status_text.endswith(".CONVERGED")

    def _add_loadflow_execution_rows(self, rows, pyp_status, pp_status) -> None:
        """
        Append load-flow execution status rows to the comparison table.

        Args:
            rows: Mutable list that receives result dictionaries.
            pyp_status: Status dictionary returned by the powsybl load-flow runner.
            pp_status: Status dictionary returned by the pandapower load-flow runner.

        """
        pyp_results = pyp_status.get("results", []) or []

        if pyp_results:
            for index, result in enumerate(pyp_results):
                status = getattr(result, "status", None)
                rows.append(
                    {
                        "element_type": "loadflow_status",
                        "element": f"powsybl_component_{index}",
                        "parameter": "status",
                        "unit": "",
                        "powsybl_result": status,
                        "pandapower_result": "",
                        "delta": "",
                        "status": "OK"
                        if self._comparison_loadflow_status_is_converged(status)
                        else "ERROR",
                    }
                )

        else:
            rows.append(
                {
                    "element_type": "loadflow_status",
                    "element": "powsybl",
                    "parameter": "status",
                    "unit": "",
                    "powsybl_result": pyp_status.get("message", ""),
                    "pandapower_result": "",
                    "delta": "",
                    "status": "OK" if pyp_status.get("success") else "ERROR",
                }
            )

        rows.append(
            {
                "element_type": "loadflow_status",
                "element": "pandapower",
                "parameter": "converged",
                "unit": "",
                "powsybl_result": "",
                "pandapower_result": pp_status.get("message", ""),
                "delta": "",
                "status": "OK" if pp_status.get("success") else "ERROR",
            }
        )

    def _add_loadflow_check(
        self,
        rows,
        element_type,
        element,
        parameter,
        powsybl_result,
        pandapower_result,
        unit="",
        abs_tol=1e-5,
        rel_tol=1e-5,
    ) -> None:
        """
        Append one numerical load-flow comparison row.

        The method compares the powsybl result value against the pandapower result
        value, calculates the delta where possible, and stores an ``OK`` or ``ERROR``
        status according to the given tolerances.

        """
        is_equal = self._comparison_values_equal(
            powsybl_result, pandapower_result, abs_tol=abs_tol, rel_tol=rel_tol
        )

        rows.append(
            {
                "element_type": element_type,
                "element": element,
                "parameter": parameter,
                "unit": unit,
                "powsybl_result": powsybl_result,
                "pandapower_result": pandapower_result,
                "delta": self._comparison_delta(powsybl_result, pandapower_result),
                "status": "OK" if is_equal else "ERROR",
            }
        )

    def _build_loadflow_bus_rows(self, rows) -> None:
        """
        Append bus voltage magnitude and angle comparison rows.

        The powsybl voltage magnitude is compared in kilovolts. The pandapower result
        is converted from per-unit to kilovolts using the nominal bus voltage.

        """
        pyp_bb_buses = self.pyp_net.get_bus_breaker_view_buses()
        pyp_buses = self.pyp_net.get_buses(all_attributes=True)
        pp_buses = self.pandap_net.bus
        pp_res_buses = self.pandap_net.res_bus

        for bus_name, row in pyp_bb_buses.iterrows():
            pp_row = self._comparison_get_pp_match(pp_buses, bus_name)

            if pp_row is None:
                continue

            pp_result_row = self._comparison_get_pp_result_row(
                pp_res_buses, pp_row.name
            )

            if pp_result_row is None:
                continue

            pyp_bus_id = row.get("bus_id")
            pyp_row = None

            if pyp_bus_id in pyp_buses.index:
                pyp_row = pyp_buses.loc[pyp_bus_id]
            else:
                pyp_row = row

            pp_vn_kv = self._comparison_to_float(pp_row.get("vn_kv"), np.nan)
            pp_vm_pu = self._comparison_to_float(pp_result_row.get("vm_pu"), np.nan)
            pp_vm_kv = (
                pp_vm_pu * pp_vn_kv
                if np.isfinite(pp_vm_pu) and np.isfinite(pp_vn_kv)
                else np.nan
            )

            self._add_loadflow_check(
                rows,
                "bus",
                bus_name,
                "v_mag",
                self._comparison_get_result_values(
                    pyp_row, "v_mag", "voltage", "voltage_kv"
                ),
                pp_vm_kv,
                "kV",
                abs_tol=1e-3,
                rel_tol=1e-5,
            )

            self._add_loadflow_check(
                rows,
                "bus",
                bus_name,
                "v_angle",
                self._comparison_get_result_values(
                    pyp_row, "v_angle", "angle", "va_degree"
                ),
                pp_result_row.get("va_degree"),
                "deg",
                abs_tol=1e-2,
                rel_tol=1e-5,
            )

    def _build_loadflow_generator_rows(self, rows) -> None:
        """Append generator active and reactive power comparison rows."""
        pyp_gens = self.pyp_net.get_generators(all_attributes=True)
        pp_gens = self.pandap_net.gen
        pp_res_gens = self.pandap_net.res_gen

        for gen_id, row in pyp_gens.iterrows():
            pp_row = self._comparison_get_pp_match(pp_gens, gen_id)

            if pp_row is None:
                continue

            pp_result_row = self._comparison_get_pp_result_row(pp_res_gens, pp_row.name)

            if pp_result_row is None:
                continue

            pyp_p_terminal = self._comparison_get_result_values(row, "p")
            pyp_q_terminal = self._comparison_get_result_values(row, "q")

            if not self._is_missing(pyp_p_terminal):
                pyp_p = -self._comparison_to_float(pyp_p_terminal, np.nan)
            else:
                pyp_p = self._comparison_to_float(
                    self._comparison_get_result_values(row, "p_mw", "target_p"), np.nan
                )

            if not self._is_missing(pyp_q_terminal):
                pyp_q = -self._comparison_to_float(pyp_q_terminal, np.nan)
            else:
                pyp_q = self._comparison_to_float(
                    self._comparison_get_result_values(row, "q_mvar", "target_q"),
                    np.nan,
                )

            self._add_loadflow_check(
                rows,
                "gen",
                gen_id,
                "p_mw",
                pyp_p,
                pp_result_row.get("p_mw"),
                "MW",
                abs_tol=5e-2,
                rel_tol=5e-3,
            )

            self._add_loadflow_check(
                rows,
                "gen",
                gen_id,
                "q_mvar",
                pyp_q,
                pp_result_row.get("q_mvar"),
                "MVar",
                abs_tol=5e-2,
                rel_tol=5e-3,
            )

    def _build_loadflow_line_rows(self, rows) -> None:
        """
        Append line or impedance power and current comparison rows.

        The method compares active power, reactive power and current for both terminals.
        Cross-voltage powsybl lines converted as pandapower impedances are handled here
        as impedance elements.

        """
        pyp_lines = self.pyp_net.get_lines(all_attributes=True)
        pp_lines = self.pandap_net.line
        pp_res_line = self.pandap_net.res_line

        pp_res_impedance = getattr(self.pandap_net, "res_impedance", pd.DataFrame())

        for line_id, row in pyp_lines.iterrows():
            pp_row = self._comparison_get_pp_match(pp_lines, line_id)

            if pp_row is not None:
                pp_result_row = self._comparison_get_pp_result_row(
                    pp_res_line, pp_row.name
                )

                element_type = "line"
            else:
                pp_row = self._comparison_get_pp_impedance_match(line_id)

                if pp_row is None:
                    continue

                pp_result_row = self._comparison_get_pp_result_row(
                    pp_res_impedance, pp_row.name
                )

                element_type = "impedance"

            if pp_result_row is None:
                continue

            for pyp_parameter, pp_parameter, unit in (
                ("p1", "p_from_mw", "MW"),
                ("q1", "q_from_mvar", "MVar"),
                ("p2", "p_to_mw", "MW"),
                ("q2", "q_to_mvar", "MVar"),
                ("i1", "i_from_ka", "kA"),
                ("i2", "i_to_ka", "kA"),
            ):
                pyp_value = self._comparison_get_result_values(row, pyp_parameter)

                if pyp_parameter.startswith("i"):
                    pyp_value = self._comparison_ampere_to_kiloampere(pyp_value)

                if pyp_parameter.startswith("p"):
                    abs_tol = 5e-3
                    rel_tol = 3e-2
                elif pyp_parameter.startswith("q"):
                    abs_tol = 1e-3
                    rel_tol = 3e-2
                else:
                    abs_tol = 5e-3
                    rel_tol = 3e-2

                self._add_loadflow_check(
                    rows,
                    element_type,
                    line_id,
                    pyp_parameter,
                    pyp_value,
                    pp_result_row.get(pp_parameter),
                    unit,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                )

    def _build_loadflow_2w_trafo_rows(self, rows) -> None:
        """
        Append two-windings transformer load-flow comparison rows.

        The method maps powsybl side 1 and side 2 to pandapower high- and low-voltage
        result columns according to the rated voltages.

        """
        pyp_trafos = self.pyp_net.get_2_windings_transformers(all_attributes=True)
        pp_trafos = self.pandap_net.trafo
        pp_res_trafo = self.pandap_net.res_trafo

        for trafo_id, row in pyp_trafos.iterrows():
            pp_row = self._comparison_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                continue

            pp_result_row = self._comparison_get_pp_result_row(
                pp_res_trafo, pp_row.name
            )

            if pp_result_row is None:
                continue

            rated_u1 = self._comparison_to_float(row.get("rated_u1"), np.nan)
            rated_u2 = self._comparison_to_float(row.get("rated_u2"), np.nan)

            if np.isfinite(rated_u1) and np.isfinite(rated_u2) and rated_u1 >= rated_u2:
                side_mapping = {
                    "1": ("p_hv_mw", "q_hv_mvar", "i_hv_ka"),
                    "2": ("p_lv_mw", "q_lv_mvar", "i_lv_ka"),
                }
            else:
                side_mapping = {
                    "1": ("p_lv_mw", "q_lv_mvar", "i_lv_ka"),
                    "2": ("p_hv_mw", "q_hv_mvar", "i_hv_ka"),
                }

            for pyp_side, pp_parameters in side_mapping.items():
                for pyp_prefix, pp_parameter, unit in (
                    ("p", pp_parameters[0], "MW"),
                    ("q", pp_parameters[1], "MVar"),
                    ("i", pp_parameters[2], "kA"),
                ):
                    pyp_parameter = f"{pyp_prefix}{pyp_side}"
                    pyp_value = self._comparison_get_result_values(row, pyp_parameter)

                    if pyp_prefix == "i":
                        pyp_value = self._comparison_ampere_to_kiloampere(pyp_value)

                    if pyp_prefix.startswith("p"):
                        abs_tol = 5e-3
                        rel_tol = 3e-2
                    elif pyp_prefix.startswith("q"):
                        abs_tol = 5e-3
                        rel_tol = 3e-2
                    else:
                        abs_tol = 1e-4
                        rel_tol = 1e-5

                    self._add_loadflow_check(
                        rows,
                        "trafo_2w",
                        trafo_id,
                        pyp_parameter,
                        pyp_value,
                        pp_result_row.get(pp_parameter),
                        unit,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                    )

    def _build_loadflow_3w_trafo_rows(self, rows) -> None:
        """
        Append three-windings transformer load-flow comparison rows.

        Transformer legs are sorted by rated voltage before they are compared to
        pandapower high-, medium-, and low-voltage result columns.

        """
        pyp_trafos = self.pyp_net.get_3_windings_transformers(all_attributes=True)
        pp_trafos = self.pandap_net.trafo3w
        pp_res_trafo = self.pandap_net.res_trafo3w

        for trafo_id, row in pyp_trafos.iterrows():
            pp_row = self._comparison_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                continue

            pp_result_row = self._comparison_get_pp_result_row(
                pp_res_trafo, pp_row.name
            )

            if pp_result_row is None:
                continue

            legs = []
            for leg_no in (1, 2, 3):
                rated_u = self._comparison_to_float(
                    self._get_first_non_missing_value(
                        row, f"rated_u{leg_no}", f"ratedU{leg_no}", default=np.nan
                    ),
                    np.nan,
                )

                if np.isfinite(rated_u):
                    legs.append({"leg_no": leg_no, "rated_u": rated_u})

            if len(legs) != 3:
                continue

            legs.sort(key=lambda leg: leg["rated_u"], reverse=True)
            side_names = ["hv", "mv", "lv"]

            for leg, side_name in zip(legs, side_names):
                leg_no = leg["leg_no"]

                for pyp_prefix, pp_parameter, unit in (
                    ("p", f"p_{side_name}_mw", "MW"),
                    ("q", f"q_{side_name}_mvar", "MVar"),
                    ("i", f"i_{side_name}_ka", "kA"),
                ):
                    pyp_parameter = f"{pyp_prefix}{leg_no}"
                    pyp_value = self._comparison_get_result_values(row, pyp_parameter)

                    if pyp_prefix == "i":
                        pyp_value = self._comparison_ampere_to_kiloampere(pyp_value)

                    if pyp_prefix in ("p", "q"):
                        abs_tol = 5e-2
                        rel_tol = 5e-3
                    else:
                        abs_tol = 1e-4
                        rel_tol = 1e-5

                    self._add_loadflow_check(
                        rows,
                        "trafo_3w",
                        trafo_id,
                        pyp_parameter,
                        pyp_value,
                        pp_result_row.get(pp_parameter),
                        unit,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                    )

    def _build_loadflow_load_rows(self, rows) -> None:
        """Append load active and reactive power comparison rows."""
        pyp_loads = self.pyp_net.get_loads(all_attributes=True)
        pp_loads = self.pandap_net.load
        pp_res_loads = self.pandap_net.res_load

        for load_id, row in pyp_loads.iterrows():
            pp_row = self._comparison_get_pp_match(pp_loads, load_id)

            if pp_row is None:
                continue

            pp_result_row = self._comparison_get_pp_result_row(
                pp_res_loads, pp_row.name
            )

            if pp_result_row is None:
                continue

            self._add_loadflow_check(
                rows,
                "load",
                load_id,
                "p_mw",
                self._comparison_get_result_values(row, "p", "p0", "p_mw"),
                pp_result_row.get("p_mw"),
                "MW",
                abs_tol=1e-4,
                rel_tol=1e-5,
            )

            self._add_loadflow_check(
                rows,
                "load",
                load_id,
                "q_mvar",
                self._comparison_get_result_values(row, "q", "q0", "q_mvar"),
                pp_result_row.get("q_mvar"),
                "MVar",
                abs_tol=1e-4,
                rel_tol=1e-5,
            )
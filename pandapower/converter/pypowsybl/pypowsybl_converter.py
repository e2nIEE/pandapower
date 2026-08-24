# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""
Power grid conversion utilities for powsybl and pandapower.

This module contains the ``PyPowSyBlConverter`` class, a conversion helper that
loads a powsybl ``.xiidm`` network, creates an equivalent pandapower network,
exports the result as JSON and optionally logs detailed transfer and load-flow
comparison tables.

The comparison methods are designed to make numerical deviations visible without
changing the converted network itself.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from pandapower import (
    create_buses,
    create_empty_network,
    create_gen,
    create_impedance,
    create_line_from_parameters,
    create_load,
    create_switch,
    create_transformer3w_from_parameters,
    create_transformer_from_parameters,
    to_json,
)
import pandas as pd
from pandapower.auxiliary import soft_dependency_error
from pandapower.pp_types import SwitchType
from pandapower.converter.pypowsybl.pypowsybl_static_comparison import (
    PyPowSyBlStaticComparisonMixin,
)
from pandapower.converter.pypowsybl.pypowsybl_loadflow_comparison import (
    PyPowSyBlLoadflowComparisonMixin,
)
from pandapower.converter.pypowsybl.pypowsybl_comparison_utils import (
    PyPowSyBlComparisonUtilsMixin,
)

logger = logging.getLogger(__name__)


def _require_pypowsybl(function_name: str) -> Any:
    """Import pypowsybl only when the converter is actually used."""
    try:
        import pypowsybl as pyp
    except ImportError:
        soft_dependency_error(function_name, "pypowsybl")
        raise

    return pyp


class PyPowSyBlConverter(
    PyPowSyBlStaticComparisonMixin,
    PyPowSyBlLoadflowComparisonMixin,
    PyPowSyBlComparisonUtilsMixin,
):

    """Convert powsybl network models into pandapower networks.

    The class stores both the source powsybl network and the generated pandapower
    network on the instance. During conversion it creates buses, generators,
    lines, transformers, loads and switches, then writes the pandapower network to
    a JSON file.

    Several comparison helpers build tables between the expected values from
    powsybl and the values actually created in pandapower. A second comparison mode can
    run AC load-flow calculations in both toolchains and compare the resulting
    voltages, powers, currents and element statuses.

    Attributes:
        pandap_net: Generated pandapower network.
        pyp_net: Loaded powsybl network.
        transfer_table: DataFrame containing static conversion checks.
        loadflow_table: DataFrame containing load-flow result checks.
    """

    def __init__(self) -> None:
        """
        Initialize converter state and default conversion parameters.

        The constructor defines fallback values that are used whenever the source data
        is missing, invalid, or not directly available in powsybl. It also initializes
        the comparison tables and load-flow result placeholders.

        No file is loaded and no network element is created at construction time.

        """
        self.default_shift_degree = 0.0
        self.default_length_km = 1.0
        self.default_frequency_hz = 50.0
        self.default_base_sn_mva = 100.0
        self.default_vm_pu = 1.0
        self.default_parallel = 1.0
        self.default_impedance_ohm = 0.0
        self.default_admittance_s = 0.0
        self.default_active_power_mw = 0.0
        self.default_reactive_power_mvar = 0.0
        self.default_line_max_i_ka = 9999.0
        self.default_trafo_sn_mva = 9999.0
        self.default_trafo_vk_percent = 0.001
        self.default_switch_type: SwitchType = "CB"
        self.switch_type_mapping: dict[str, SwitchType] = {
            "BREAKER": "CB",
            "DISCONNECTOR": "DS",
            "LOAD_BREAK_SWITCH": "LBS",
        }
        self.default_switch_z_ohm = 0.0

        self.transfer_table = None
        self.transfer_dataframes = {}
        self.transfer_summary = pd.DataFrame()

        self.loadflow_table = None
        self.loadflow_dataframes = {}
        self.loadflow_summary = pd.DataFrame()
        self.powsybl_loadflow_result = None
        self.pandapower_loadflow_result = None

    def _powsybl_to_pandapower(
        self,
        filename: str,
        log_static_comparison: bool = False,
        log_loadflow_comparison: bool = False,
        return_loadflow_table: bool = False,
        default_shift_degree: float = 0.0,
        default_length_km: float = 1.0,
    ) -> tuple[Any, Any, str] | tuple[Any, Any, str, pd.DataFrame]:
        """
        Convert a powsybl ``.xiidm`` file into a pandapower network.

        The method is the main conversion workflow. It loads the powsybl network,
        initializes an empty pandapower network, collects voltage levels and bus
        references, creates all supported pandapower elements, exports the result as a
        JSON file, and optionally logs comparison tables.

        Args:
            filename: Path to powsybl ``.xiidm`` file.
            log_static_comparison: If ``True``, build and log a static transfer
                comparison table.
            log_loadflow_comparison: If ``True``, run AC load-flow calculations
                in both frameworks and log a result comparison table.
            return_loadflow_table: If ``True``, return the load-flow comparison
                table as a fourth return value.
            default_shift_degree: Fallback phase-shift angle in degrees.
            default_length_km: Fallback line length in kilometres.

        Returns:
            A tuple containing the generated pandapower network, the loaded powsybl
            network, and the path of the exported pandapower JSON file. If
            ``return_loadflow_table`` is ``True``, the load-flow comparison table is
            returned as a fourth value.

        """
        self.default_shift_degree = self._to_float(
            default_shift_degree, self.default_shift_degree
        )

        self.default_length_km = self._to_float(
            default_length_km, self.default_length_km
        )
        if not np.isfinite(self.default_length_km) or self.default_length_km <= 0:
            self.default_length_km = 1.0

        self._load_powsybl_network(filename)
        self._init_pandapower_network()

        voltage_level_map = self._collect_voltage_levels()
        elements_map, bus_id_to_bus_names, bus_map = self._collect_buses()

        self._create_buses(bus_map, voltage_level_map, elements_map)
        self._create_generators(elements_map, bus_id_to_bus_names)
        self._create_lines(elements_map, bus_id_to_bus_names)
        self._create_2_windings_transformers(elements_map, bus_id_to_bus_names)
        self._create_3_windings_transformers(elements_map, bus_id_to_bus_names)
        self._create_loads(elements_map, bus_id_to_bus_names)
        self._create_switches(elements_map, bus_id_to_bus_names)

        json_filename = self._export_to_json(filename)

        if log_static_comparison:
            self.transfer_table = self._build_transfer_table()
            self._log_transfer_table()

        if log_loadflow_comparison:
            self._log_loadflow_comparison_table()

        if return_loadflow_table:
            if self.loadflow_table is None:
                self.loadflow_table = self._build_loadflow_table()

            return self.pandap_net, self.pyp_net, json_filename, self.loadflow_table

        return self.pandap_net, self.pyp_net, json_filename

    # ======================================================
    # Helper
    # ======================================================

    def _to_scalar(self, value):
        """Return scalar value for NumPy/Pandas scalar-like objects."""
        if not hasattr(value, "item"):
            return value

        try:
            return value.item()
        except (AttributeError, TypeError, ValueError):
            return value

    def _to_float(self, value, default=np.nan) -> float:
        """
        Convert a value to ``float`` while applying a safe fallback.

        Args:
            value: Input value that may be numeric, nullable, or a NumPy scalar.
            default: Value returned when conversion is impossible or the result in NaN.

        Returns:
            The converted floating-point value or ``default``.

        """
        try:
            if value is None:
                return default
            if hasattr(value, "item"):
                value = value.item()
            value = float(value)
            if np.isnan(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    def _is_missing(self, value) -> bool:
        """
        Return whether a value should be treated as missing.

        The check handles Python ``None``, NumPy/Pandas null values, and column textual
        representations such as ``nan``, ``none``, and ``<na>``.

        """
        if value is None:
            return True

        value = self._to_scalar(value)

        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)):
                return bool(missing)
        except (TypeError, ValueError):
            return False

        value_str = str(value).strip().lower()
        return value_str in ("", "nan", "none", "<na>")

    def _to_bool(self, value, default=False) -> bool:
        """Convert a value to ``bool`` with support for textual states."""
        if self._is_missing(value):
            return default

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        value_str = str(value).strip().lower()

        if value_str in ("true", "1", "yes", "y"):
            return True

        if value_str in ("false", "0", "no", "n"):
            return False

        return bool(value)

    def _get_first_non_missing_value(self, row, *keys, default=None) -> Any:
        """Return the first non-missing value for requested row keys."""
        for key in keys:
            if key in row.index:
                value = row.get(key)
                if not self._is_missing(value):
                    return value

        return default

    def _get_pp_bus_names(
        self, elements_map, bus_id_to_bus_names, pyp_bus_ref
    ) -> list[str]:
        """
        Resolve a powsybl bus reference to pandapower bus names.

        The source reference may already be a bus-breaker bus name or only a logical
        powsybl bus id. In the latter case the method uses the collected mapping from
        powsybl bus ids to pandapower bus names.

        Returns:
            A list of matching pandapower bus names. The list is empty if no match is
            available.

        """
        if pyp_bus_ref is None:
            return []

        if pyp_bus_ref in elements_map:
            return [pyp_bus_ref]

        return list(bus_id_to_bus_names.get(pyp_bus_ref, []))

    def _get_pp_bus_indices(
        self, elements_map, bus_id_to_bus_names, pyp_bus_ref
    ) -> list[int]:
        """
        Resolve a powsybl bus reference to pandapower bus indices.

        Args:
            elements_map: Mapping that stores created pandapower indices per element.
            bus_id_to_bus_names: Mapping from powsybl bus ids to bus-breaker names.
            pyp_bus_ref: Bus reference from a powsybl element row.

        Returns:
            List of matching pandapower bus indices.

        """
        bus_indices = []

        for bus_name in self._get_pp_bus_names(
            elements_map, bus_id_to_bus_names, pyp_bus_ref
        ):
            pp_bus_idx = elements_map.get(bus_name, {}).get("pandap_bus_idx")
            if pp_bus_idx is not None:
                bus_indices.append(pp_bus_idx)

        return bus_indices

    def _get_pp_bus_idx(
        self, elements_map, bus_id_to_bus_names, pyp_bus_ref
    ) -> int | None:
        """
        Return the first pandapower bus index for a powsybl bus reference.

        This helper is used by element creation methods that require exactly one bus
        index. If the reference cannot be resolved, ``None`` is returned and the caller
        usually skips the affected element.

        """
        bus_indices = self._get_pp_bus_indices(
            elements_map, bus_id_to_bus_names, pyp_bus_ref
        )
        if not bus_indices:
            return None

        return bus_indices[0]

    def _select_slack_generator_id(self, generators) -> Any | None:
        """
        Select exactly one powsybl generator as pandapower slack generator.

        powsybl generators should remain pandapower generators.
        Only one of them receives slack = True so pandapower can run a load flow.

        """
        if generators is None or generators.empty:
            return None

        try:
            pyp_buses = self.pyp_net.get_buses(all_attributes=True)
        except (ArithmeticError, TypeError, ValueError) as exc:
            logger.debug(
                "Could not read powsybl buses while selecting slack generator: %s",
                exc,
            )
            pyp_buses = pd.DataFrame()

        candidates = []

        for gen_id, row in generators.iterrows():
            connected = self._to_bool(row.get("connected"), True)

            if not connected:
                continue

            bus_id = row.get("bus_id")

            if bus_id in pyp_buses.index and "v_angle" in pyp_buses.columns:
                angle_score = abs(
                    self._to_float(pyp_buses.at[bus_id, "v_angle"], np.inf)
                )
            else:
                angle_score = np.inf

            target_p_score = abs(
                self._to_float(row.get("target_p"), self.default_active_power_mw)
            )

            candidates.append((angle_score, -target_p_score, str(gen_id), gen_id))

        if not candidates:
            return generators.index[0]

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        return candidates[0][3]

    def _normalize_switch_kind(self, value) -> str | None:
        """
        Normalize a powsybl switch kind to a comparable uppercase token.

        Empty strings, missing values, textual ``nan`` values, and ``None``-like values
        are treated as missing and return ``None``.

        """
        if self._is_missing(value):
            return None

        if hasattr(value, "item"):
            value = value.item()

        return str(value).strip().upper()

    def _get_pp_switch_type(self, switch_row, default=None) -> SwitchType:
        """
        Map a powsybl switch kind to pandapower switch type.

        Args:
            switch_row: Row from the powsybl switch table.
            default: Optional fallback pandapower switch type.

        Returns:
            The mapped pandapower switch type, for example ``CB``, ``DS``, or ``LBS``.

        """
        if default is None:
            default = self.default_switch_type

        pyp_switch_kind = self._normalize_switch_kind(switch_row.get("kind"))

        if pyp_switch_kind is None:
            return default

        return self.switch_type_mapping.get(pyp_switch_kind, default)

    def _ohm_to_pu(self, z_ohm, vn_kv, sn_mva) -> float:
        """
        Convert an impedance from ohms to per-unit.

        Args:
            z_ohm: Physical impedance in ohms.
            vn_kv: Nominal voltage in kilovolts.
            sn_mva: Base apparent power in megavolt-amperes.

        Returns:
            Impedance in per-unit on the specified voltage and power base.

        """
        z_base = (vn_kv**2) / sn_mva
        return z_ohm / z_base

    # ======================================================
    # Initialization
    # ======================================================

    def _load_powsybl_network(self, filename: str) -> None:
        """
        Load and validate the powsybl source network.

        Args:
            filename: Path to source network. The file must use the ``.xiidm``
                extension.

        Raises:
            ValueError: If the file extension is unsupported or the network uses
                per-unit values.
            RuntimeError: If powsybl fails to load the file.

        """
        if not filename.endswith(".xiidm"):
            raise ValueError("pypowsybl-files must use the .xiidm extension.")

        pyp = _require_pypowsybl("_load_powsybl_network()")

        try:
            self.pyp_net = pyp.network.load(filename)
        except Exception as exc:
            raise RuntimeError(f"Failed to load powsybl file: {exc}") from exc

        if getattr(self.pyp_net, "per_unit", False):
            raise ValueError(
                "This converter mapping expects physical powsybl data "
                "(per_unit == False)."
            )

    def _init_pandapower_network(self) -> None:
        """
        Create an empty pandapower network using the source base power.

        The method reads nominal apparent power from the powsybl network when
        available. If the value is missing or invalid, the converter falls back to the
        configured default base apparent power.

        """
        base_sn_mva = self._to_float(
            getattr(self.pyp_net, "nominal_apparent_power", np.nan), np.nan
        )
        base_sn_mva = (
            base_sn_mva / 1000.0
            if np.isfinite(base_sn_mva) and base_sn_mva > 0
            else self.default_base_sn_mva
        )

        self.pandap_net = create_empty_network(
            f_hz=self.default_frequency_hz, sn_mva=base_sn_mva
        )

    # ======================================================
    # Collect base data
    # ======================================================

    def _collect_voltage_levels(self) -> dict[str, float]:
        """
        Collect nominal voltages for all powsybl voltage levels.

        Returns:
            Dictionary mapping voltage-level ids to nominal voltages in kilovolts.

        Raises:
            ValueError: If a voltage level has no valid positive nominal voltage.

        """
        voltage_level_map = {}
        voltage_levels = self.pyp_net.get_voltage_levels()

        for vl_id, vl in voltage_levels.iterrows():
            nominal_v = self._to_float(vl.get("nominal_v"), np.nan)
            if not np.isfinite(nominal_v) or nominal_v <= 0:
                raise ValueError(f"VoltageLevel {vl_id} has no valid nominal_v.")

            voltage_level_map[vl_id] = nominal_v

        return voltage_level_map

    def _collect_buses(
        self,
    ) -> tuple[dict[Any, Any], dict[Any, Any], dict[str, list[str]]]:
        """
        Collect powsybl bus-breaker buses and create lookup mappings.

        Returns:
            A tuple ``(elements_map, bus_id_to_bus_names, bus_map)`` used by later
            element creation steps to resolve powsybl references to pandapower buses.

        """
        elements_map = {}
        bus_id_to_bus_names: dict[Any, Any] = {}
        bus_map: dict[Any, Any] = {}

        buses = self.pyp_net.get_bus_breaker_view_buses()

        for bus_name, row in buses.iterrows():
            vl_id = row["voltage_level_id"]
            bus_id = row["bus_id"]

            if vl_id not in bus_map:
                bus_map[vl_id] = []
            bus_map[vl_id].append(bus_name)

            bus_id_to_bus_names.setdefault(bus_id, []).append(bus_name)

            elements_map[bus_name] = {
                "type": "bus",
                "bus_id": bus_id,
                "voltage_level_id": vl_id,
            }

        return elements_map, bus_id_to_bus_names, bus_map

    # ======================================================
    # Create pandapower elements
    # ======================================================

    def _create_buses(self, bus_map, voltage_level_map, elements_map) -> None:
        """
        Create pandapower buses from collected powsybl bus data.

        This method creates one pandapower bus for every bus-breaker bus and stores the
        created pandapower index in ``elements_map`` for later element creation.

        """
        bus_names = [
            bus_name
            for vl_id, names in bus_map.items()
            for bus_name in names
        ]

        vn_kv_values = [
            voltage_level_map.get(vl_id)
            for vl_id, names in bus_map.items()
            for _ in names
        ]

        if not bus_names:
            return

        pp_bus_indices = create_buses(
            self.pandap_net,
            nr_buses=len(bus_names),
            vn_kv=vn_kv_values,
            name=bus_names,
        )

        for bus_name, pp_bus_idx in zip(bus_names, pp_bus_indices):
            elements_map[bus_name]["pandap_bus_idx"] = pp_bus_idx

    def _create_generators(self, elements_map, bus_id_to_bus_names) -> None:
        """
        Create pandapower generators from powsybl generators.

        The voltage magnitude is derived by dividing the powsybl target voltage by the
        corresponding pandapower bus nominal voltage. Exactly one converted generator is
        marked as slack so that pandapower can run a load flow.

        """
        generators = self.pyp_net.get_generators(all_attributes=True)
        slack_generator_id = self._select_slack_generator_id(generators)

        for gen_id, row in generators.iterrows():
            bus_ref = self._get_first_non_missing_value(
                row, "bus_breaker_bus_id", "connectable_bus_id", "bus_id", default=None
            )

            bus_idx = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, bus_ref)

            if bus_idx is None:
                continue

            connected = self._to_bool(row.get("connected"), True)

            target_p_mw = self._to_float(
                row.get("target_p"), self.default_active_power_mw
            )

            target_v_kv = self._to_float(row.get("target_v"), np.nan)

            bus_vn_kv = self._get_bus_vn_kv(bus_idx)

            if np.isfinite(target_v_kv) and bus_vn_kv > 0:
                vm_pu = target_v_kv / bus_vn_kv
            else:
                vm_pu = self.default_vm_pu

            is_slack = str(gen_id) == str(slack_generator_id)

            gen_idx = create_gen(
                self.pandap_net,
                bus=bus_idx,
                p_mw=target_p_mw,
                vm_pu=vm_pu,
                name=gen_id,
                slack=is_slack,
                in_service=connected,
                controllable=True,
            )

            self._set_generator_optional_parameters(
                gen_idx=gen_idx,
                gen_id=gen_id,
                row=row,
                is_slack=is_slack,
            )

    def _set_generator_optional_parameters(
        self,
        gen_idx: int | np.integer,
        gen_id: Any,
        row: pd.Series,
        is_slack: bool,
    ) -> None:
        """Set optional generator limits and metadata after generator creation."""
        self.pandap_net.gen.at[gen_idx, "slack_weight"] = 1.0 if is_slack else 0.0

        rated_s = self._to_float(row.get("rated_s"), np.nan)
        min_p = self._to_float(row.get("min_p"), np.nan)
        max_p = self._to_float(row.get("max_p"), np.nan)
        target_q = self._to_float(row.get("target_q"), np.nan)

        min_q = self._to_float(
            self._get_first_non_missing_value(
                row, "min_q", "min_q_mvar", "minimum_q", default=np.nan
            ),
            np.nan,
        )

        max_q = self._to_float(
            self._get_first_non_missing_value(
                row, "max_q", "max_q_mvar", "maximum_q", default=np.nan
            ),
            np.nan,
        )

        if not np.isfinite(min_q):
            if np.isfinite(rated_s) and rated_s > 0:
                min_q = -rated_s
            else:
                min_q = -9999.0

        if not np.isfinite(max_q):
            if np.isfinite(rated_s) and rated_s > 0:
                max_q = rated_s
            else:
                max_q = 9999.0

        if min_q > max_q:
            min_q, max_q = max_q, min_q

        if np.isfinite(rated_s) and rated_s > 0:
            self.pandap_net.gen.at[gen_idx, "sn_mva"] = rated_s

        if np.isfinite(min_p):
            self.pandap_net.gen.at[gen_idx, "min_p_mw"] = min_p

        if np.isfinite(max_p):
            self.pandap_net.gen.at[gen_idx, "max_p_mw"] = max_p

        if np.isfinite(target_q):
            self.pandap_net.gen.at[gen_idx, "target_q_mvar"] = target_q

        self.pandap_net.gen.at[gen_idx, "min_q_mvar"] = min_q
        self.pandap_net.gen.at[gen_idx, "max_q_mvar"] = max_q
        self.pandap_net.gen.at[gen_idx, "powsybl_id"] = gen_id

    def _get_bus_vn_kv(self, pp_bus_idx) -> float:
        """Return the nominal voltage of a pandapower bus in kV."""
        return self._to_float(self.pandap_net.bus.at[pp_bus_idx, "vn_kv"], np.nan)

    def _buses_have_different_nominal_voltage(
        self, pp_from_bus, pp_to_bus, atol=1e-6
    ) -> bool:
        """Return whether two pandapower buses have different voltages."""
        from_vn_kv = self._get_bus_vn_kv(pp_from_bus)
        to_vn_kv = self._get_bus_vn_kv(pp_to_bus)

        if not np.isfinite(from_vn_kv) or not np.isfinite(to_vn_kv):
            return False

        return not np.isclose(from_vn_kv, to_vn_kv, rtol=0.0, atol=atol)

    def _create_cross_voltage_line_as_impedance(
        self,
        line_id,
        pp_from_bus,
        pp_to_bus,
        r_ohm,
        x_ohm,
        g_total_s,
        b_total_s,
        in_service,
    ) -> None:
        """
        Create a powsybl line between different voltage levels as impedance.

        Normal pandapower lines are not suitable here because they are parameterized
        as line elements with per-kilometre values. Cross-voltage powsybl line
        branches are safer as impedance elements.

        """
        from_vn_kv = self._get_bus_vn_kv(pp_from_bus)
        to_vn_kv = self._get_bus_vn_kv(pp_to_bus)

        sn_mva = self._to_float(
            getattr(self.pandap_net, "sn_mva", np.nan), self.default_base_sn_mva
        )

        if not np.isfinite(sn_mva) or sn_mva <= 0:
            sn_mva = self.default_base_sn_mva

        if not np.isfinite(from_vn_kv) or from_vn_kv <= 0:
            raise ValueError(
                f"Cannot create impedance for line {line_id!r}: "
                f"invalid from-bus nominal voltage {from_vn_kv} kV."
            )

        if not np.isfinite(to_vn_kv) or to_vn_kv <= 0:
            raise ValueError(
                f"Cannot create impedance for line {line_id!r}: "
                f"invalid to-bus nominal voltage {to_vn_kv} kV."
            )

        if not np.isfinite(r_ohm):
            r_ohm = self.default_impedance_ohm

        if not np.isfinite(x_ohm):
            x_ohm = self.default_impedance_ohm

        if abs(r_ohm) < 1e-12 and abs(x_ohm) < 1e-12:
            x_ohm = 1e-6

        rft_pu = self._ohm_to_pu(r_ohm, from_vn_kv, sn_mva)
        xft_pu = self._ohm_to_pu(x_ohm, from_vn_kv, sn_mva)

        rtf_pu = self._ohm_to_pu(r_ohm, to_vn_kv, sn_mva)
        xtf_pu = self._ohm_to_pu(x_ohm, to_vn_kv, sn_mva)

        impedance_idx = create_impedance(
            self.pandap_net,
            from_bus=pp_from_bus,
            to_bus=pp_to_bus,
            rft_pu=rft_pu,
            xft_pu=xft_pu,
            rtf_pu=rtf_pu,
            xtf_pu=xtf_pu,
            sn_mva=sn_mva,
            name=line_id,
            in_service=in_service,
        )

        self.pandap_net.impedance.at[impedance_idx, "powsybl_id"] = str(line_id)
        self.pandap_net.impedance.at[impedance_idx, "powsybl_element_type"] = (
            "cross_voltage_line"
        )
        self.pandap_net.impedance.at[impedance_idx, "from_vn_kv"] = from_vn_kv
        self.pandap_net.impedance.at[impedance_idx, "to_vn_kv"] = to_vn_kv
        self.pandap_net.impedance.at[impedance_idx, "powsybl_g_total_s"] = g_total_s
        self.pandap_net.impedance.at[impedance_idx, "powsybl_b_total_s"] = b_total_s

    def _create_lines(self, elements_map, bus_id_to_bus_names) -> None:
        """
        Create pandapower lines from powsybl line data.

        This method resolves both terminal buses, converts total line impedance and
        shunt data into pandapower per-kilometre parameters, applies line length and
        parallel-circuit defaults, and preserves the connection status.

        """
        lines = self.pyp_net.get_lines(all_attributes=True)

        for line_id, row in lines.iterrows():
            pp_from_bus = self._get_pp_bus_idx(
                elements_map, bus_id_to_bus_names, row.get("bus1_id")
            )
            pp_to_bus = self._get_pp_bus_idx(
                elements_map, bus_id_to_bus_names, row.get("bus2_id")
            )

            if pp_from_bus is None or pp_to_bus is None:
                continue

            r_ohm = self._to_float(row.get("r"), self.default_impedance_ohm)
            x_ohm = self._to_float(row.get("x"), self.default_impedance_ohm)
            g1_s = self._to_float(row.get("g1"), self.default_admittance_s)
            g2_s = self._to_float(row.get("g2"), g1_s)

            b1_s = self._to_float(row.get("b1"), self.default_admittance_s)
            b2_s = self._to_float(row.get("b2"), b1_s)

            g_total_s = g1_s + g2_s
            b_total_s = b1_s + b2_s

            in_service = self._to_bool(row.get("connected1"), True) and self._to_bool(
                row.get("connected2"), True
            )

            if self._buses_have_different_nominal_voltage(pp_from_bus, pp_to_bus):
                self._create_cross_voltage_line_as_impedance(
                    line_id=line_id,
                    pp_from_bus=pp_from_bus,
                    pp_to_bus=pp_to_bus,
                    r_ohm=r_ohm,
                    x_ohm=x_ohm,
                    g_total_s=g_total_s,
                    b_total_s=b_total_s,
                    in_service=in_service,
                )

                continue

            parallel = self._to_float(row.get("parallel"), np.nan)
            if not np.isfinite(parallel) or parallel <= 0:
                parallel = self.default_parallel

            length_km = self._to_float(row.get("length_km"), np.nan)
            if not np.isfinite(length_km):
                length_km = self._to_float(row.get("length"), np.nan)

            if not np.isfinite(length_km) or length_km <= 0:
                length_km = self.default_length_km

            r_ohm_per_km = r_ohm * parallel / length_km
            x_ohm_per_km = x_ohm * parallel / length_km

            c_nf_per_km = (b_total_s * 1e9) / (
                2 * np.pi * self.pandap_net.f_hz * length_km * parallel
            )
            g_us_per_km = (g_total_s * 1e6) / (length_km * parallel)

            create_line_from_parameters(
                self.pandap_net,
                from_bus=pp_from_bus,
                to_bus=pp_to_bus,
                length_km=length_km,
                r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=x_ohm_per_km,
                c_nf_per_km=c_nf_per_km,
                g_us_per_km=g_us_per_km,
                max_i_ka=self.default_line_max_i_ka,
                parallel=int(parallel),
                name=line_id,
                in_service=in_service,
            )

    def _create_2_windings_transformers(
        self, elements_map, bus_id_to_bus_names
    ) -> None:
        """
        Create pandapower two-windings transformers from powsybl data.

        The method maps the higher rated voltage side to pandapower's high-voltage side,
        converts powsybl impedance and admittance values to pandapower transformer
        parameters, and transfers the connection state and phase shift.

        """
        trafos = self.pyp_net.get_2_windings_transformers(all_attributes=True)

        for trafo_id, row in trafos.iterrows():
            pp_bus1 = self._get_pp_bus_idx(
                elements_map, bus_id_to_bus_names, row.get("bus1_id")
            )
            pp_bus2 = self._get_pp_bus_idx(
                elements_map, bus_id_to_bus_names, row.get("bus2_id")
            )

            if pp_bus1 is None or pp_bus2 is None:
                continue

            r_ohm = self._to_float(row.get("r"), self.default_impedance_ohm)
            x_ohm = self._to_float(row.get("x"), self.default_impedance_ohm)
            g_s = self._to_float(row.get("g"), self.default_admittance_s)
            b_s = self._to_float(row.get("b"), self.default_admittance_s)

            rated_u1 = self._to_float(row.get("rated_u1"), np.nan)
            rated_u2 = self._to_float(row.get("rated_u2"), np.nan)
            rated_s = self._to_float(row.get("rated_s"), np.nan)

            if not np.isfinite(rated_u1) or rated_u1 <= 0:
                continue

            if not np.isfinite(rated_u2) or rated_u2 <= 0:
                continue

            if not np.isfinite(rated_s) or rated_s <= 0:
                rated_s = self.default_trafo_sn_mva

            in_service = self._to_bool(row.get("connected1"), True) and self._to_bool(
                row.get("connected2"), True
            )

            if rated_u1 >= rated_u2:
                hv_bus = pp_bus1
                lv_bus = pp_bus2
                vn_hv_kv = rated_u1
                vn_lv_kv = rated_u2
            else:
                hv_bus = pp_bus2
                lv_bus = pp_bus1
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

            shift_degree = self._to_float(row.get("alpha"), self.default_shift_degree)

            create_transformer_from_parameters(
                self.pandap_net,
                hv_bus=hv_bus,
                lv_bus=lv_bus,
                sn_mva=rated_s,
                vn_hv_kv=vn_hv_kv,
                vn_lv_kv=vn_lv_kv,
                vk_percent=vk_percent,
                vkr_percent=vkr_percent,
                pfe_kw=pfe_kw,
                i0_percent=i0_percent,
                shift_degree=shift_degree,
                name=trafo_id,
                in_service=in_service,
            )

    def _create_3_windings_transformers(
        self, elements_map, bus_id_to_bus_names
    ) -> None:
        """
        Create pandapower three-windings transformer from powsybl data.

        Each transformer leg is collected, validated, sorted by rated voltage, and then
        mapped to the pandapower high-, medium-, low-voltage sides. The method also
        derives pairwise short-circuit values, no-load losses, no-load current, phase
        shifts, and service state.

        """
        trafos = self.pyp_net.get_3_windings_transformers(all_attributes=True)

        for trafo_id, row in trafos.iterrows():
            legs: list[dict[str, Any]] = []

            for leg_no in (1, 2, 3):
                bus_ref = self._get_first_non_missing_value(
                    row,
                    f"bus{leg_no}_id",
                    f"bus_breaker_bus{leg_no}_id",
                    f"bus{leg_no}",
                )

                connectable_bus_ref = self._get_first_non_missing_value(
                    row,
                    f"bus_breaker_bus{leg_no}_id",
                    f"connectable_bus{leg_no}_id",
                    f"connectableBus{leg_no}",
                )

                connected = self._to_bool(
                    self._get_first_non_missing_value(
                        row, f"connected{leg_no}", default=not self._is_missing(bus_ref)
                    ),
                    default=not self._is_missing(bus_ref),
                )

                if self._is_missing(bus_ref):
                    bus_ref = connectable_bus_ref
                    connected = False

                pp_bus = self._get_pp_bus_idx(
                    elements_map, bus_id_to_bus_names, bus_ref
                )

                if pp_bus is None:
                    legs = []
                    break

                rated_u = self._to_float(
                    self._get_first_non_missing_value(
                        row, f"rated_u{leg_no}", f"ratedU{leg_no}", default=np.nan
                    )
                )

                rated_s = self._to_float(
                    self._get_first_non_missing_value(
                        row, f"rated_s{leg_no}", f"ratedS{leg_no}", default=np.nan
                    )
                )

                if not np.isfinite(rated_u) or rated_u <= 0:
                    legs = []
                    break

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
                        "bus": pp_bus,
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
                continue

            legs.sort(key=lambda leg: leg["rated_u"], reverse=True)

            hv_leg = legs[0]
            mv_leg = legs[1]
            lv_leg = legs[2]

            in_service = (
                hv_leg["connected"] and mv_leg["connected"] and lv_leg["connected"]
            )

            reference_u_kv = hv_leg["rated_u"]

            vk_hv_percent, vkr_hv_percent = (
                self._calculate_3w_pair_short_circuit_values(
                    hv_leg, mv_leg, reference_u_kv
                )
            )
            vk_mv_percent, vkr_mv_percent = (
                self._calculate_3w_pair_short_circuit_values(
                    mv_leg, lv_leg, reference_u_kv
                )
            )
            vk_lv_percent, vkr_lv_percent = (
                self._calculate_3w_pair_short_circuit_values(
                    hv_leg, lv_leg, reference_u_kv
                )
            )

            pfe_kw = hv_leg["pfe_kw"] + mv_leg["pfe_kw"] + lv_leg["pfe_kw"]
            i0_percent = (
                hv_leg["i0_percent"] + mv_leg["i0_percent"] + lv_leg["i0_percent"]
            )

            shift_mv_degree = mv_leg["alpha_degree"] - hv_leg["alpha_degree"]
            shift_lv_degree = lv_leg["alpha_degree"] - hv_leg["alpha_degree"]

            create_transformer3w_from_parameters(
                self.pandap_net,
                hv_bus=hv_leg["bus"],
                mv_bus=mv_leg["bus"],
                lv_bus=lv_leg["bus"],
                vn_hv_kv=hv_leg["rated_u"],
                vn_mv_kv=mv_leg["rated_u"],
                vn_lv_kv=lv_leg["rated_u"],
                sn_hv_mva=hv_leg["rated_s"],
                sn_mv_mva=mv_leg["rated_s"],
                sn_lv_mva=lv_leg["rated_s"],
                vk_hv_percent=vk_hv_percent,
                vk_mv_percent=vk_mv_percent,
                vk_lv_percent=vk_lv_percent,
                vkr_hv_percent=vkr_hv_percent,
                vkr_mv_percent=vkr_mv_percent,
                vkr_lv_percent=vkr_lv_percent,
                pfe_kw=pfe_kw,
                i0_percent=i0_percent,
                shift_mv_degree=shift_mv_degree,
                shift_lv_degree=shift_lv_degree,
                name=trafo_id,
                in_service=in_service,
            )

    def _create_loads(self, elements_map, bus_id_to_bus_names) -> None:
        """
        Create pandapower loads from powsybl load data.

        The active and reactive load powers are copied from powsybl ``p0`` and ``q0``.
        Missing values are replaced by converter defaults, and disconnected loads are
        created out of service.

        """
        loads = self.pyp_net.get_loads(all_attributes=True)

        for load_id, row in loads.iterrows():
            pp_bus_idx = self._get_pp_bus_idx(
                elements_map, bus_id_to_bus_names, row.get("bus_id")
            )
            if pp_bus_idx is None:
                continue

            create_load(
                self.pandap_net,
                bus=pp_bus_idx,
                p_mw=self._to_float(row.get("p0"), self.default_active_power_mw),
                q_mvar=self._to_float(row.get("q0"), self.default_reactive_power_mvar),
                name=load_id,
                in_service=self._to_bool(row.get("connected"), True),
            )

    def _create_switches(self, elements_map, bus_id_to_bus_names) -> None:
        """
        Create pandapower bus-bus switches from powsybl switches.

        Only switches with two resolvable and distinct pandapower buses are created.
        The powsybl open/closed state and switch kind are mapped to pandapower fields.

        """
        switches = self.pyp_net.get_switches(all_attributes=True)

        for switch_id, row in switches.iterrows():
            bus1_id = row.get("bus_breaker_bus1_id")
            bus2_id = row.get("bus_breaker_bus2_id")

            pp_bus1 = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, bus1_id)
            pp_bus2 = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, bus2_id)

            if pp_bus1 is None or pp_bus2 is None:
                continue

            if pp_bus1 == pp_bus2:
                continue

            is_open = self._to_bool(row.get("open"), False)
            closed = not is_open

            pp_type = self._get_pp_switch_type(row)

            create_switch(
                self.pandap_net,
                bus=pp_bus1,
                element=pp_bus2,
                et="b",
                closed=closed,
                type=pp_type,
                name=switch_id,
                z_ohm=self.default_switch_z_ohm,
            )

    # ======================================================
    # EXPORT
    # ======================================================

    def _export_to_json(self, filename: str) -> str:
        """
        Export the generated pandapower network as JSON.

        Args:
            filename: Source file name used to derive the output file path.

        Returns:
            Path of the written JSON file.

        """
        base_path, _ = os.path.splitext(filename)
        json_filename = base_path + ".json"
        to_json(self.pandap_net, json_filename)
        return json_filename

    def _calculate_3w_pair_short_circuit_values(
        self, leg_a, leg_b, reference_u_kv
    ) -> tuple[float, float]:
        """
        Calculate pairwise short-circuit values for the three-winding transformer.

        Args:
            leg_a: First transformer leg dictionary.
            leg_b: Second transformer leg dictionary.
            reference_u_kv: Voltage base used for the pairwise impedance conversion.

        Returns:
            Tuple ``(vk_percent, vkr_percent)`` for the selected leg pair.

        """
        s_ref_mva = min(leg_a["rated_s"], leg_b["rated_s"])
        z_base_ohm = (reference_u_kv**2) / s_ref_mva

        z_a_relative = leg_a["z_ohm"] / z_base_ohm
        z_b_relative = leg_b["z_ohm"] / z_base_ohm

        z_pair_relative = z_a_relative + z_b_relative

        vk_percent = 100.0 * abs(z_pair_relative)
        vkr_percent = 100.0 * z_pair_relative.real

        if not np.isfinite(vk_percent) or vk_percent <= 0:
            vk_percent = self.default_trafo_vk_percent

        if not np.isfinite(vkr_percent) or vkr_percent <= 0:
            vkr_percent = 0.0

        return vk_percent, vkr_percent

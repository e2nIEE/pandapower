"""Power grid conversion utilities for powsybl and pandapower.

This module contains the ``PyPowSyBlConverter`` class, a conversion helper that
loads a powsybl ``.xiidm`` network, creates an equivalent pandapower network,
exports the result as JSON and optionally prints detailed transfer and load-flow debug tables.

The debug methods are designed to make numerical deviations visible without
changing the converted network itself.
"""
from __future__ import annotations
from typing import Any

import pandapower as pandap
import pypowsybl as pyp
import pypowsybl.loadflow as pyp_lf
import numpy as np
import pandas as pd

import os

class PyPowSyBlConverter:
    """Convert powsybl network models into pandapower networks.

    The class stores both the source powsybl network and the generated pandapower
    network on the instance. During conversion it creates buses, generators,
    lines, transformers, loads and switches, then writes the pandapower network to
    a JSON file.

    Several debug helpers build comparison tables between the expected values from
    powsybl and the values actually created in pandapower. A second debug mode can
    run AC load-flow calculations in both toolchains and compare the resulting
    voltages, powers, currents and element statuses.

    Attributes:
        pandap_net: Generated pandapower network.
        pyp_net: Loaded powsybl network.
        transfer_table: DataFrame containing static conversion checks.
        loadflow_table: DataFrame containing load-flow result checks.
    """
    def __init__(self) -> None:
        """Initialize converter state and default conversion parameters.

        The constructor creates empty placeholders for the powsybl source network, the
        pandapower target network, debug tables and load-flow results. It also defines
        fallback values that are used whenever the source data is missing, invalid, or
        not directly available in powsybl.

        No file is loaded and no network element is created at construction time.
        """
        self.pandap_net = None
        self.pyp_net = None

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
        self.default_switch_type = "CB"
        self.switch_type_mapping = {
            "BREAKER": "CB",
            "DISCONNECTOR": "DS",
            "LOAD_BREAK_SWITCH": "LBS"
        }
        self.default_switch_z_ohm = 0.0
        self.default_rho = 1.0
        self.default_ratio_tap_position = np.nan
        self.default_phase_tap_position = np.nan

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
            debug: bool = False,
            debug_run: bool = False,
            default_shift_degree: float = 0.0,
            default_length_km: float = 1.0
    ) -> tuple[Any, Any, str]:
        """Convert a powsybl ``.xiidm`` file into a pandapower network.

        The method is the main conversion workflow. It loads the powsybl network,
        initializes an empty pandapower network, collects voltage levels and bus
        references, creates all supported pandapower elements, exports the result as a
        JSON file, and optionally prints debug comparison tables.

        Args:
            filename: Path to powsybl ``.xiidm`` file.
            debug: If ``True``, build and print a static transfer comparison table.
            debug_run: If ``True``, run AC load-flow calculations in both frameworks
                and print a result comparison table.
            default_shift_degree: Fallback phase-shift angle in degrees.
            default_length_km: Fallback line length in kilometres.

        Returns:
            A tuple containing the generated pandapower network, the loaded powsybl
            network, and the path of the exported pandapower JSON file.
        """
        self.default_shift_degree = self._to_float(default_shift_degree, self.default_shift_degree)

        self.default_length_km = self._to_float(default_length_km, self.default_length_km)
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

        if debug:
            self.transfer_table = self._build_transfer_table()
            self._print_transfer_table()

        if debug_run:
            self._print_loadflow_debug_table()

        return self.pandap_net, self.pyp_net, json_filename

    # ======================================================
    # Helper
    # ======================================================

    def _to_float(self, value, default = np.nan) -> float:
        """Convert a value to ``float`` while applying a safe fallback.

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
        
    def _get_pp_bus_names(self, elements_map, bus_id_to_bus_names, pyp_bus_ref) -> list[str]:
        """Resolve a powsybl bus reference to pandapower bus names.

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
    
    def _get_pp_bus_indices(self, elements_map, bus_id_to_bus_names, pyp_bus_ref) -> list[int]:
        """Resolve a powsybl bus reference to pandapower bus indices.

        Args:
            elements_map: Mapping that stores created pandapower indices per element.
            bus_id_to_bus_names: Mapping from powsybl bus ids to bus-breaker names.
            pyp_bus_ref: Bus reference from a powsybl element row.

        Returns:
            List of matching pandapower bus indices.
        """
        bus_indices = []

        for bus_name in self._get_pp_bus_names(elements_map, bus_id_to_bus_names, pyp_bus_ref):
            pp_bus_idx = elements_map.get(bus_name, {}).get("pandap_bus_idx")
            if pp_bus_idx is not None:
                bus_indices.append(pp_bus_idx)

        return bus_indices
        
    def _get_pp_bus_idx(self, elements_map, bus_id_to_bus_names, pyp_bus_ref) -> int | None:
        """Return the first pandapower bus index for a powsybl bus reference.

        This helper is used by element creation methods that require exactly one bus
        index. If the reference cannot be resolved, ``None`` is returned and the caller
        usually skips the affected element.
        """
        bus_indices = self._get_pp_bus_indices(elements_map, bus_id_to_bus_names, pyp_bus_ref)
        if not bus_indices:
            return None
        
        return bus_indices[0]
    
    def _get_first_existing_value(self, row, *keys, default = None) -> Any:
        """Return the first existing and non-missing row value."""

        for key in keys:
            if key in row.index:
                value = row.get(key)

                if not self._debug_is_missing(value):
                    return value
                
        return default
    
    def _select_slack_generator_id(self, generators) -> Any | None:
        """Select exactly one powsybl generator as pandapower slack generator.
        
        powsybl generators should remain pandapower generators.
        Only one of them receives slack = True so pandapower can run a load flow.
        """

        if generators is None or generators.empty:
            return None
        
        try:
            pyp_buses = self.pyp_net.get_buses(all_attributes = True)
        except Exception:
            pyp_buses = pd.DataFrame()

        candidates = []

        for gen_id, row in generators.iterrows():
            connected = self._debug_to_bool(row.get("connected"), True)

            if not connected:
                continue

            bus_id = row.get("bus_id")

            if (
                bus_id in pyp_buses.index
                and "v_angle" in pyp_buses.columns
            ):
                angle_score = abs(
                    self._to_float(
                        pyp_buses.at[bus_id, "v_angle"],
                        np.inf
                    )
                )
            else:
                angle_score = np.inf

            target_p_score = abs(
                self._to_float(
                    row.get("target_p"),
                    self.default_active_power_mw
                )
            )

            candidates.append(
                (
                    angle_score,
                    -target_p_score,
                    str(gen_id),
                    gen_id
                )
            )

        if not candidates:
            return generators.index[0]
        
        candidates.sort(key = lambda item: (item[0], item[1], item[2]))

        return candidates[0][3]
    
    def _normalize_switch_kind(self, value) -> str | None:
        """Normalize a powsybl switch kind to a comparable uppercase token.

        Empty strings, missing values, textual ``nan`` values, and ``None``-like values
        are treated as missing and return ``None``.
        """
        if value is None:
            return None
        
        if hasattr(value, "item"):
            value = value.item()

        value = str(value).strip().upper()

        if value == "" or value in ("NAN", "NONE", "<NA>"):
            return None
        
        return value
    
    def _get_pp_switch_type(self, switch_row, default = None) -> str:
        """Map a powsybl switch kind to pandapower switch type.

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
        """Convert an impedance from ohms to per-unit.

        Args:
            z_ohm: Physical impedance in ohms.
            vn_kv: Nominal voltage in kilovolts.
            sn_mva: Base apparent power in megavolt-amperes

        Returns:
            Impedance in per-unit on the specified voltage and power base.
        """
        z_base = (vn_kv ** 2) / sn_mva
        return z_ohm / z_base
    
    def _siemens_to_pu(self, y_s, vn_kv, sn_mva) -> float:
        """Convert an admittance from siemens to per-unit.

        Args:
            y_s: Physical admittance in siemens.
            vn_kv: Nominal voltage in kilovolts.
            sn_mva: Base apparent power in megavolt-amperes

        Returns:
            Admittance in per-unit on the specified voltage and power base.
        """
        z_base = (vn_kv ** 2) / sn_mva
        return y_s * z_base
    
    def _leg_impedance_to_percent(self, r_ohm, x_ohm, rated_u_kv, rated_s_mva) -> tuple[float, float]:
        """Convert transformer leg impedance values to pandapower percentages.

        Args:
            r_ohm: Resistive part of the impedance in ohms.
            x_ohm: Reactive part of the impedance in ohms.
            rated_u_kv: Rated voltage of the transformer leg in kilovolts.
            rated_s_mva: Rated apparent power of the transformer leg in MVA.

        Returns:
            Tuple ``(vk-percent, vkr_percent)`` for pandapower transformer parameters.
        """
        z_base = (rated_u_kv ** 2) / rated_s_mva
        r_pu = r_ohm / z_base
        x_pu = x_ohm / z_base

        vk_percent = 100.0 * np.hypot(r_pu, x_pu)
        vkr_percent = 100.0 * r_pu

        return vk_percent, vkr_percent
    
    def _star_to_pairwise_percent(self, hv_value, mv_value, lv_value, sn_hv_mva, sn_mv_mva, sn_lv_mva) -> tuple[float, float, float]:
        """Convert star-equivalent transformer values to pairwise percentages.

        The method scales medium- and low-voltage side values to the high-voltage power
        base and combines the star legs into the pairwise values required by
        pandapower's three-windings transformer model.

        Returns:
            Tuple containing the high-medium, medium-low, and high-low pair values.
        """
        hv_star = hv_value
        mv_star = mv_value * sn_hv_mva / sn_mv_mva
        lv_star = lv_value * sn_hv_mva / sn_lv_mva

        hv_mv_prime = hv_star + mv_star
        mv_lv_prime = mv_star + lv_star
        hv_lv_prime = hv_value + lv_star

        hv_mv_value = hv_mv_prime * min(sn_hv_mva, sn_mv_mva) / sn_hv_mva
        mv_lv_value = mv_lv_prime * min(sn_mv_mva, sn_lv_mva) / sn_hv_mva
        hv_lv_value = hv_lv_prime * min(sn_hv_mva, sn_lv_mva) / sn_hv_mva

        return hv_mv_value, mv_lv_value, hv_lv_value
    
    # ======================================================
    # Initialization
    # ======================================================

    def _load_powsybl_network(self, filename: str) -> None:
        """Load and validate the powsybl source network.

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
        
        try:
            self.pyp_net = pyp.network.load(filename)
        except Exception as e:
            raise RuntimeError(f"Failed to load powsybl file: {e}")
        
        if getattr(self.pyp_net, "per_unit", False):
            raise ValueError(
                "This converter mapping expects physical powsybl data "
                "(per_unit == False)."
            )
        
    def _init_pandapower_network(self) -> None:
        """Create an empty pandapower network using the source base power.

        The method reads nominal apparent power from the powsybl network when
        available. If the value is missing or invalid, the converter falls back to the
        configured default base apparent power.
        """
        base_sn_mva = self._to_float(getattr(self.pyp_net, "nominal_apparent_power", np.nan), np.nan)
        base_sn_mva = base_sn_mva / 1000.0 if np.isfinite(base_sn_mva) and base_sn_mva > 0 else self.default_base_sn_mva
        
        self.pandap_net = pandap.create_empty_network(
            f_hz = self.default_frequency_hz,
            sn_mva = base_sn_mva)
        
    # ======================================================
    # Collect base data
    # ======================================================

    def _collect_voltage_levels(self) -> dict[str, float]:
        """Collect nominal voltages for all powsybl voltage levels.

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

    def _collect_buses(self) -> tuple[dict[Any, Any], dict[Any, Any], dict[str, list[str]]]:
        """Collect powsybl bus-breaker buses and create lookup mappings.

        Returns:
            A tuple ``(elements_map, bus_id_to_bus_names, bus_map)`` used by later
            element creation steps to resolve powsybl references to pandapower buses.
        """
        elements_map = {}
        bus_id_to_bus_names = {}
        bus_map = {}

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
                "voltage_level_id": vl_id
            }
        
        return elements_map, bus_id_to_bus_names, bus_map
    
    # ======================================================
    # Create pandapower elements
    # ======================================================

    def _create_buses(self, bus_map, voltage_level_map, elements_map) -> None:
        """Create pandapower buses from collected powsybl bus data.
        
        This method creates one pandapower bus for every bus-breaker bus and stores the
        created pandapower index in ``elements_map`` for later element creation.
        """
        for vl_id, bus_names in bus_map.items():
            vn_kv = voltage_level_map.get(vl_id)

            for bus_name in bus_names:
                pp_bus_idx = pandap.create_bus(self.pandap_net, name = bus_name, vn_kv = vn_kv)

                elements_map[bus_name]["pandap_bus_idx"] = pp_bus_idx

    def _create_generators(self, elements_map, bus_id_to_bus_names) -> None:
        """Create pandapower generators from powsybl generators.

        The voltage magnitude is derived by dividing the powsybl target voltage by the
        corresponding pandapower bus nominal voltage. Exactly one converted generator is
        marked as slack so that pandapower can run a load flow.
        """

        generators = self.pyp_net.get_generators(all_attributes = True)
        slack_generator_id = self._select_slack_generator_id(generators)

        for gen_id, row in generators.iterrows():
            bus_ref = self._get_first_existing_value(
                row,
                "bus_breaker_bus_id",
                "connectable_bus_id",
                "bus_id",
                default = None
            )

            bus_idx = self._get_pp_bus_idx(
                elements_map,
                bus_id_to_bus_names,
                bus_ref
            )

            if bus_idx is None:
                continue

            connected = self._debug_to_bool(row.get("connected"), True)

            target_p_mw = self._to_float(
                row.get("target_p"),
                self.default_active_power_mw
            )

            target_v_kv = self._to_float(row.get("target_v"), np.nan)

            bus_vn_kv = float(self.pandap_net.bus.at[bus_idx, "vn_kv"])

            if np.isfinite(target_v_kv) and bus_vn_kv > 0:
                vm_pu = target_v_kv / bus_vn_kv
            else:
                vm_pu = self.default_vm_pu

            is_slack = str(gen_id) == str(slack_generator_id)

            gen_idx = pandap.create_gen(
                self.pandap_net,
                bus = bus_idx,
                p_mw = target_p_mw,
                vm_pu = vm_pu,
                name = gen_id,
                slack = is_slack,
                in_service = connected,
                controllable = True
            )

            self.pandap_net.gen.at[gen_idx, "slack_weight"] = 1.0 if is_slack else 0.0

            rated_s = self._to_float(row.get("rated_s"), np.nan)
            min_p = self._to_float(row.get("min_p"), np.nan)
            max_p = self._to_float(row.get("max_p"), np.nan)
            target_q = self._to_float(row.get("target_q"), np.nan)

            min_q = self._to_float(
                self._get_first_existing_value(
                    row,
                    "min_q",
                    "min_q_mvar",
                    "minimum_q",
                    default = np.nan
                ),
                np.nan
            )

            max_q = self._to_float(
                self._get_first_existing_value(
                    row,
                    "max_q",
                    "max_q_mvar",
                    "maximum_q",
                    default = np.nan
                ),
                np.nan
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
        return self._to_float(
            self.pandap_net.bus.at[pp_bus_idx, "vn_kv"],
            np.nan
        )
    
    def _buses_have_different_nominal_voltage(
            self,
            pp_from_bus,
            pp_to_bus,
            atol = 1e-6
    ) -> bool:
        """Return whether two pandapower buses have different voltages."""
        from_vn_kv = self._get_bus_vn_kv(pp_from_bus)
        to_vn_kv = self._get_bus_vn_kv(pp_to_bus)

        if not np.isfinite(from_vn_kv) or not np.isfinite(to_vn_kv):
            return False
        
        return not np.isclose(
            from_vn_kv,
            to_vn_kv,
            rtol = 0.0,
            atol = atol
        )
    
    def _create_cross_voltage_line_as_impedance(
            self,
            line_id,
            pp_from_bus,
            pp_to_bus,
            r_ohm,
            x_ohm,
            g_total_s,
            b_total_s,
            in_service
    ) -> None:
        """Create a powsybl line between different voltage levels as impedance.
        
        Normal pandapower lines are not suitable here because they are parameterized
        as line elements with per-kilometre values. Cross-voltage powsybl line
        branches are safer as impedance elements.
        """

        from_vn_kv = self._get_bus_vn_kv(pp_from_bus)
        to_vn_kv = self._get_bus_vn_kv(pp_to_bus)

        sn_mva = self._to_float(
            getattr(self.pandap_net, "sn_mva", np.nan),
            self.default_base_sn_mva
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

        impedance_idx = pandap.create_impedance(
            self.pandap_net,
            from_bus = pp_from_bus,
            to_bus = pp_to_bus,
            rft_pu = rft_pu,
            xft_pu = xft_pu,
            rtf_pu = rtf_pu,
            xtf_pu = xtf_pu,
            sn_mva = sn_mva,
            name = line_id,
            in_service = in_service
        )

        self.pandap_net.impedance.at[impedance_idx, "powsybl_id"] = str(line_id)
        self.pandap_net.impedance.at[impedance_idx, "powsybl_element_type"] = "cross_voltage_line"
        self.pandap_net.impedance.at[impedance_idx, "from_vn_kv"] = from_vn_kv
        self.pandap_net.impedance.at[impedance_idx, "to_vn_kv"] = to_vn_kv
        self.pandap_net.impedance.at[impedance_idx, "powsybl_g_total_s"] = g_total_s
        self.pandap_net.impedance.at[impedance_idx, "powsybl_b_total_s"] = b_total_s

    def _create_lines(self, elements_map, bus_id_to_bus_names) -> None:
        """Create pandapower lines from powsybl line data.
        
        This method resolves both terminal buses, converts total line impedance and 
        shunt data into pandapower per-kilometre parameters, applies line length and 
        parallel-circuit defaults, and preserves the connection status.
        """
        lines = self.pyp_net.get_lines(all_attributes = True)

        for line_id, row in lines.iterrows():
            pp_from_bus = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, row.get("bus1_id"))
            pp_to_bus = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, row.get("bus2_id"))

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

            in_service = (
                self._debug_to_bool(row.get("connected1"), True)
                and self._debug_to_bool(row.get("connected2"), True)
            )

            if self._buses_have_different_nominal_voltage(pp_from_bus, pp_to_bus):
                self._create_cross_voltage_line_as_impedance(
                    line_id = line_id,
                    pp_from_bus = pp_from_bus,
                    pp_to_bus = pp_to_bus,
                    r_ohm = r_ohm,
                    x_ohm = x_ohm,
                    g_total_s = g_total_s,
                    b_total_s = b_total_s,
                    in_service = in_service
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

            c_nf_per_km = (b_total_s * 1e9) / (2 * np.pi * self.pandap_net.f_hz * length_km * parallel)
            g_us_per_km = (g_total_s * 1e6) / (length_km * parallel)

            pandap.create_line_from_parameters(
                self.pandap_net,
                from_bus = pp_from_bus,
                to_bus = pp_to_bus,
                length_km = length_km,
                r_ohm_per_km = r_ohm_per_km,
                x_ohm_per_km = x_ohm_per_km,
                c_nf_per_km = c_nf_per_km,
                g_us_per_km = g_us_per_km,
                max_i_ka = self.default_line_max_i_ka,
                parallel = parallel,
                name = line_id,
                in_service = in_service
            )

    def _create_2_windings_transformers(self, elements_map, bus_id_to_bus_names) -> None:
        """Create pandapower two-windings transformers from powsybl data.
        
        The method maps the higher rated voltage side to pandapower's high-voltage side,
        converts powsybl impedance and admittance values to pandapower transformer
        parameters, and transfers the connection state and phase shift.
        """
        trafos = self.pyp_net.get_2_windings_transformers(all_attributes = True)

        for trafo_id, row in trafos.iterrows():
            pp_bus1 = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, row.get("bus1_id"))
            pp_bus2 = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, row.get("bus2_id"))

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

            in_service = (
                self._debug_to_bool(row.get("connected1"), True)
                and self._debug_to_bool(row.get("connected2"), True)
            )

            if  rated_u1 >= rated_u2:
                hv_bus = pp_bus1
                lv_bus = pp_bus2
                vn_hv_kv = rated_u1
                vn_lv_kv = rated_u2
            else:
                hv_bus = pp_bus2
                lv_bus = pp_bus1
                vn_hv_kv = rated_u2
                vn_lv_kv = rated_u1

            z_base_side2 = (rated_u2 ** 2) / rated_s

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

            pfe_kw = (rated_u2 ** 2) * g_s * 1000.0

            shift_degree = self._to_float(
                row.get("alpha"),
                self.default_shift_degree
            )

            pandap.create_transformer_from_parameters(
                self.pandap_net,
                hv_bus = hv_bus,
                lv_bus = lv_bus,
                sn_mva = rated_s,
                vn_hv_kv = vn_hv_kv,
                vn_lv_kv = vn_lv_kv,
                vk_percent = vk_percent,
                vkr_percent = vkr_percent,
                pfe_kw = pfe_kw,
                i0_percent = i0_percent,
                shift_degree = shift_degree,
                name = trafo_id,
                in_service = in_service
            )

    def _create_3_windings_transformers(self, elements_map, bus_id_to_bus_names) -> None:
        """Create pandapower three-windings transformer from powsybl data.
        
        Each transformer leg is collected, validated, sorted by rated voltage, and then
        mapped to the pandapower high-, medium-, low-voltage sides. The method also
        derives pairwise short-circuit values, no-load losses, no-load current, phase
        shifts, and service state.
        """
        trafos = self.pyp_net.get_3_windings_transformers(all_attributes = True)

        def is_missing(value) -> bool:
            """Return whether a local powsybl row value should be treated as missing."""
            if value is None:
                return True
            
            try:
                if isinstance(value, (float, np.floating)) and np.isnan(value):
                    return True
            except TypeError:
                pass

            value_str = str(value).strip()
            return value_str == "" or value_str.lower() in ("nan", "none", "<na>")
        
        def get_first(row, *keys, default = None) -> Any:
            """Return the first non-missing value for the requested row keys."""
            for key in keys:
                if key in row.index:
                    value = row.get(key)
                    if not is_missing(value):
                        return value
                    
            return default
        
        def to_bool(value, default = True) -> bool:
            """Convert a local powsybl row value to ``bool`` with a fallback."""
            if is_missing(value):
                return default
            
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            
            value_str = str(value).strip().lower()

            if value_str == "true":
                return True
            
            if value_str == "false":
                return False
            
            return bool(value)
        
        def sanitize_vk_percent(value) -> float:
            """Replace invalid short-circuit voltage values with the configured fallback."""
            if not np.isfinite(value) or value <= 0:
                return self.default_trafo_vk_percent
            
            return value
        
        def sanitize_vkr_percent(value) -> float:
            """Replace invalid resistive short-circuit voltage values with zero."""
            if not np.isfinite(value) or value <= 0:
                return 0.0
            
            return value
        
        def pair_short_circuit_values(leg_a, leg_b, reference_u_kv) -> tuple[float, float]:
            """Calculate pairwise short-circuit values for two transformer legs."""
            s_ref_mva = min(leg_a["rated_s"], leg_b["rated_s"])

            z_base_ohm = (reference_u_kv ** 2) / s_ref_mva

            z_a_relative = leg_a["z_ohm"] / z_base_ohm
            z_b_relative = leg_b["z_ohm"] / z_base_ohm

            z_pair_relative = z_a_relative + z_b_relative

            vk_percent = 100.0 * abs(z_pair_relative)
            vkr_percent = 100.0 * z_pair_relative.real

            return (
                sanitize_vk_percent(vk_percent),
                sanitize_vkr_percent(vkr_percent)
            )

        for trafo_id, row in trafos.iterrows():
            legs = []

            for leg_no in (1, 2, 3):
                bus_ref = get_first(
                    row,
                    f"bus{leg_no}_id",
                    f"bus_breaker_bus{leg_no}_id",
                    f"bus{leg_no}"
                )

                connectable_bus_ref = get_first(
                    row,
                    f"bus_breaker_bus{leg_no}_id",
                    f"connectable_bus{leg_no}_id",
                    f"connectableBus{leg_no}"
                )

                connected = to_bool(
                    get_first(row, f"connected{leg_no}", default = not is_missing(bus_ref)),
                    default = not is_missing(bus_ref)
                )

                if is_missing(bus_ref):
                    bus_ref = connectable_bus_ref
                    connected = False

                pp_bus = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, bus_ref)

                if pp_bus is None:
                    legs = []
                    break

                rated_u = self._to_float(
                    get_first(row, f"rated_u{leg_no}", f"ratedU{leg_no}", default = np.nan)
                )

                rated_s = self._to_float(
                    get_first(row, f"rated_s{leg_no}", f"ratedS{leg_no}", default = np.nan)
                )

                rho = self._to_float(
                    get_first(row, f"rho{leg_no}", default = self.default_rho),
                    self.default_rho
                )

                if not np.isfinite(rho) or rho <= 0:
                    rho = self.default_rho

                if not np.isfinite(rated_u) or rated_u <= 0:
                    legs = []
                    break

                if not np.isfinite(rated_s) or rated_s <= 0:
                    rated_s = self.default_trafo_sn_mva

                r_ohm = self._to_float(
                    get_first(row, f"r{leg_no}_at_current_tap", f"r{leg_no}", default = self.default_impedance_ohm),
                    self.default_impedance_ohm
                )

                x_ohm = self._to_float(
                    get_first(row, f"x{leg_no}_at_current_tap", f"x{leg_no}", default = self.default_impedance_ohm),
                    self.default_impedance_ohm
                )

                g_s = self._to_float(
                    get_first(row, f"g{leg_no}_at_current_tap", f"g{leg_no}", default = self.default_admittance_s),
                    self.default_admittance_s
                )

                b_s = self._to_float(
                    get_first(row, f"b{leg_no}_at_current_tap", f"b{leg_no}", default = self.default_admittance_s),
                    self.default_admittance_s
                )

                alpha_degree = self._to_float(
                    get_first(row, f"alpha{leg_no}", default = self.default_shift_degree),
                    self.default_shift_degree
                )

                z_ohm = complex(r_ohm, x_ohm)

                z_base_ohm = (rated_u ** 2) / rated_s
                y_relative = complex(g_s, b_s) * z_base_ohm

                i0_percent = 100.0 * abs(y_relative)
                pfe_kw = (rated_u ** 2) * g_s * 1000.0

                legs.append({
                    "leg_no": leg_no,
                    "bus": pp_bus,
                    "rated_u": rated_u,
                    "rho": rho,
                    "rated_s": rated_s,
                    "z_ohm": z_ohm,
                    "i0_percent": i0_percent,
                    "pfe_kw": pfe_kw,
                    "connected": connected,
                    "alpha_degree": alpha_degree
                })

            if len(legs) != 3:
                continue

            legs.sort(key = lambda leg: leg["rated_u"], reverse = True)

            hv_leg = legs[0]
            mv_leg = legs[1]
            lv_leg = legs[2]

            in_service = (
                hv_leg["connected"]
                and mv_leg["connected"]
                and lv_leg["connected"]
            )

            reference_u_kv = hv_leg["rated_u"]

            vk_hv_percent, vkr_hv_percent = pair_short_circuit_values(hv_leg, mv_leg, reference_u_kv)
            vk_mv_percent, vkr_mv_percent = pair_short_circuit_values(mv_leg, lv_leg, reference_u_kv)
            vk_lv_percent, vkr_lv_percent = pair_short_circuit_values(hv_leg, lv_leg, reference_u_kv)

            pfe_kw = (
                hv_leg["pfe_kw"]
                + mv_leg["pfe_kw"]
                + lv_leg["pfe_kw"]
            )
            i0_percent = (
                hv_leg["i0_percent"]
                + mv_leg["i0_percent"]
                + lv_leg["i0_percent"]
            )

            shift_mv_degree = mv_leg["alpha_degree"] - hv_leg["alpha_degree"]
            shift_lv_degree = lv_leg["alpha_degree"] - hv_leg["alpha_degree"]

            pandap.create_transformer3w_from_parameters(
                self.pandap_net,
                hv_bus = hv_leg["bus"],
                mv_bus = mv_leg["bus"],
                lv_bus = lv_leg["bus"],
                vn_hv_kv = hv_leg["rated_u"],
                vn_mv_kv = mv_leg["rated_u"],
                vn_lv_kv = lv_leg["rated_u"],
                sn_hv_mva = hv_leg["rated_s"],
                sn_mv_mva = mv_leg["rated_s"],
                sn_lv_mva = lv_leg["rated_s"],
                vk_hv_percent = vk_hv_percent,
                vk_mv_percent = vk_mv_percent,
                vk_lv_percent = vk_lv_percent,
                vkr_hv_percent = vkr_hv_percent,
                vkr_mv_percent = vkr_mv_percent,
                vkr_lv_percent = vkr_lv_percent,
                pfe_kw = pfe_kw,
                i0_percent = i0_percent,
                shift_mv_degree = shift_mv_degree,
                shift_lv_degree = shift_lv_degree,
                name = trafo_id,
                in_service = in_service
            )

    def _create_loads(self, elements_map, bus_id_to_bus_names) -> None:
        """Create pandapower loads from powsybl load data.
        
        The active and reactive load powers are copied from powsybl ``p0`` and ``q0``.
        Missing values are replaced by converter defaults, and disconnected loads are
        created out of service.
        """
        loads = self.pyp_net.get_loads(all_attributes = True)

        for load_id, row in loads.iterrows():
            pp_bus_idx = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, row.get("bus_id"))
            if pp_bus_idx is None:
                continue

            pandap.create_load(
                self.pandap_net,
                bus = pp_bus_idx,
                p_mw = self._to_float(row.get("p0"), self.default_active_power_mw),
                q_mvar = self._to_float(row.get("q0"), self.default_reactive_power_mvar),
                name = load_id,
                in_service = self._debug_to_bool(row.get("connected"), True)
            )

    def _create_switches(self, elements_map, bus_id_to_bus_names) -> None:
        """Create pandapower bus-bus switches from powsybl switches.
        
        Only switches with two resolvable and distinct pandapower buses are created.
        The powsybl open/closed state and switch kind are mapped to pandapower fields.        
        """
        switches = self.pyp_net.get_switches(all_attributes = True)

        for switch_id, row in switches.iterrows():
            bus1_id = row.get("bus_breaker_bus1_id")
            bus2_id = row.get("bus_breaker_bus2_id")

            pp_bus1 = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, bus1_id)
            pp_bus2 = self._get_pp_bus_idx(elements_map, bus_id_to_bus_names, bus2_id)

            if pp_bus1 is None or pp_bus2 is None:
                continue

            if pp_bus1 == pp_bus2:
                continue

            is_open = self._debug_to_bool(row.get("open"), False)
            closed = not is_open

            pp_type = self._get_pp_switch_type(row)

            pandap.create_switch(
                self.pandap_net,
                bus = pp_bus1,
                element = pp_bus2,
                et = "b",
                closed = closed,
                type = pp_type,
                name = switch_id,
                z_ohm = self.default_switch_z_ohm
            )

    # ======================================================
    # EXPORT
    # ======================================================

    def _export_to_json(self, filename: str) -> str:
        """Export the generated pandapower network as JSON.
        
        Args:
            filename: Source file name used to derive the output file path.

        Returns:
            Path of the written JSON file.
        """
        base_path, _ = os.path.splitext(filename)
        json_filename = base_path + ".json"
        pandap.to_json(self.pandap_net, json_filename)
        return json_filename
    
    # ======================================================
    # DEBUG-OUTPUT
    # ======================================================

    def _print_debug_outputs(self) -> None:
        """Build and print the static transfer debug output.

        This is a convenience wrapper around the transfer table builder and printer.        
        """
        self.transfer_table = self._build_transfer_table()
        self._print_transfer_table()

    def _build_transfer_table(self) -> pd.DataFrame:
        """Build the complete static powsybl-to-pandapower transfer table.

        Returns:
            DataFrame with one row per checked element parameter.
        """
        rows = []

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
            "status"
        ]

        return pd.DataFrame(rows, columns = columns)
    
    def _build_transfer_dataframes(self, only_errors = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Split the transfer table into grouped DataFrames.
        
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
            for element_type, dataframe in table.groupby("element_type", sort = False):
                self.transfer_dataframes[element_type] = dataframe.reset_index(drop = True).copy()

        self.transfer_summary = self._build_transfer_summary_dataframe()

        return self.transfer_dataframes, self.transfer_summary
    
    def _build_transfer_summary_dataframe(self) -> pd.DataFrame:
        """Build a status-count summary for the static transfer table.
        
        Returns:
            DataFrame containing counts per status and a total error count.
        """
        if self.transfer_table is None or self.transfer_table.empty:
            return pd.DataFrame(
                [{"status": "TOTAL_ERRORS", "count": 0}],
                columns = ["status", "count"]
            )
        
        summary = (
            self.transfer_table["status"]
            .value_counts(dropna = False)
            .rename_axis("status")
            .reset_index(name = "count")
        )

        error_count = int((self.transfer_table["status"] != "OK").sum())
        error_summary = pd.DataFrame([{"status": "TOTAL_ERRORS", "count": error_count}])

        return pd.concat([summary, error_summary], ignore_index = True)
    
    def _print_transfer_table(self, only_errors = False) -> None:
        """Print grouped static transfer comparison tables to stdout.
        
        Args:
            only_errors: If ``True``, print only failing comparison rows.
        """
        self.transfer_dataframes, self.transfer_summary = self._build_transfer_dataframes(only_errors = only_errors)

        print("\nDEBUG COMPARISON POWSYBL -> PANDAPOWER")
        print("Note: IDs and names are used only for mapping, not as value comparisons.")
        print("Expected pandapower values derived from powsybl are compared against the actually created values.")

        if not self.transfer_dataframes:
            print("\nNo comparison values available.")
            print("\nDEBUG_SUMMARY")
            print(self.transfer_summary)
            return
        
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 80,
            "display.float_format", "{:.12g}".format
        ):
            for element_type, dataframe in self.transfer_dataframes.items():
                print(f"\nDEBUG_DATAFRAME_{str(element_type).upper()}")
                print(dataframe)

            print("\nDEBUG_SUMMARY")
            print(self.transfer_summary)

    # ======================================================
    # DEBUG-RUN / LOAD-FLOW COMPARISON
    # ======================================================

    def _print_loadflow_debug_table(self, only_errors = False) -> None:
        """Print grouped AC load-flow comparison tables to stdout:
        
        Args:
            only_errors: If ``True``, print only failing load-flow comparison rows.
        """
        self.loadflow_table = self._build_loadflow_table()
        self.loadflow_dataframes, self.loadflow_summary = self._build_loadflow_dataframes(only_errors = only_errors)

        print("\nDEBUG RUN LOAD FLOW POWSYBL -> PANDAPOWER")
        print("Note: After conversion, an AC load flow is calculated in both networks.")
        print("Result values from powsybl are compared against the pandapower res_* tables.")

        if not self.loadflow_dataframes:
            print("\nNo load flow comparison values available.")
            print("\nDEBUG_RUN_SUMMARY")
            print(self.loadflow_summary)
            return
        
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 80,
            "display.float_format", "{:.12g}".format
        ):
            for element_type, dataframe in self.loadflow_dataframes.items():
                print(f"\nDEBUG_RUN_DATAFRAME_{str(element_type).upper()}")
                print(dataframe)

            print("\nDEBUG_RUN_SUMMARY")
            print(self.loadflow_summary)

    def _build_loadflow_table(self) -> pd.DataFrame:
        """Run debug load flows and build the result comparison table.
        
        The method first executes powsybl and pandapower load flows. If both converge,
        it compares result values for supported element types. Otherwise, it records a
        status row explaining why the numerical comparison was skipped.

        Returns:
            DataFrame with load-flow static rows and result comparison rows.
        """
        rows = []

        pyp_status = self._run_powsybl_loadflow_for_debug()
        pp_status = self._run_pandapower_loadflow_for_debug()

        self._add_loadflow_execution_rows(rows, pyp_status, pp_status)

        if pyp_status.get("success") and pp_status.get("success"):
            self._build_loadflow_bus_rows(rows)
            self._build_loadflow_generator_rows(rows)
            self._build_loadflow_line_rows(rows)
            self._build_loadflow_2w_trafo_rows(rows)
            self._build_loadflow_3w_trafo_rows(rows)
            self._build_loadflow_load_rows(rows)
        else:
            rows.append({
                "element_type": "loadflow_status",
                "element": "comparison",
                "parameter": "comparison_executed",
                "unit": "",
                "powsybl_result": pyp_status.get("message", ""),
                "pandapower_result": pp_status.get("message", ""),
                "delta": "",
                "status": "ERROR"
            })

        columns = [
            "element_type",
            "element",
            "parameter",
            "unit",
            "powsybl_result",
            "pandapower_result",
            "delta",
            "status"
        ]

        return pd.DataFrame(rows, columns = columns)
    
    def _build_loadflow_dataframes(self, only_errors = False) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Split the load-flow into grouped DataFrames.
        
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
            for element_type, dataframe in table.groupby("element_type", sort = False):
                self.loadflow_dataframes[element_type] = dataframe.reset_index(drop = True).copy()

        self.loadflow_summary = self._build_loadflow_summary_dataframe()

        return self.loadflow_dataframes, self.loadflow_summary
    
    def _build_loadflow_summary_dataframe(self) -> pd.DataFrame:
        """Build a status-count summary for the load-flow comparison table.
        
        Returns:
            DataFrame containing counts per status and a total error count.
        """
        if self.loadflow_table is None or self.loadflow_table.empty:
            return pd.DataFrame(
                [{"status": "TOTAL_ERRORS", "count": 0}],
                columns = ["status", "count"]
            )
        
        summary = (
            self.loadflow_table["status"]
            .value_counts(dropna = False)
            .rename_axis("status")
            .reset_index(name = "count")
        )

        error_count = int((self.loadflow_table["status"] != "OK").sum())
        error_summary = pd.DataFrame([{"status": "TOTAL_ERRORS", "count": error_count}])

        return pd.concat([summary, error_summary], ignore_index = True)
    
    def _run_powsybl_loadflow_for_debug(self) -> dict[str, Any]:
        """Run a powsybl AC load-flow for debug comparison.
        
        Returns:
            Dictionary containing success flag, a readable status message, and the
            raw powsybl component results.
        """
        try:
            self.powsybl_loadflow_result = pyp_lf.run_ac(self.pyp_net)
            results = self.powsybl_loadflow_result or []

            statuses = []
            for index, result in enumerate(results):
                status = getattr(result, "status", None)
                statuses.append(f"component_{index}={status}")

            success = bool(results) and all(
                self._debug_loadflow_status_is_ok(getattr(result, "status", "OK"))
                for result in results
            )

            if not results:
                success = False
                message = "No powsybl ComponentResults received."
            else:
                message = "; ".join(statuses)

            return {
                "success": success,
                "message": message,
                "results": results
            }
        
        except Exception as exc:
            self.powsybl_loadflow_result = None
            return {
                "success": False,
                "message": f"powsybl load flow failed: {exc}",
                "results": []
            }
        
    def _run_pandapower_loadflow_for_debug(self) -> dict[str, Any]:
        """Run a pandapower AC load-flow for debug comparison.
        
        Returns:
            Dictionary containing a success flag and a readable convergence message.
        """
        try:
            pandap.runpp(
                self.pandap_net,
                algorithm = "nr",
                calculate_voltage_angles = True,
                init = "auto"
            )

            converged = bool(getattr(self.pandap_net, "converged", False))
            self.pandapower_loadflow_result = {
                "converged": converged
            }

            return {
                "success": converged,
                "message": "converged" if converged else "pandapower load flow did not converge"
            }
        
        except Exception as exc:
            self.pandapower_loadflow_result = None
            return {
                "success": False,
                "message": f"pandapower load flow failed: {exc}"
            }
        
    def _debug_loadflow_status_is_ok(self, status) -> bool:
        """Return whether a powsybl load-flow status indicates convergence."""
        status_text = str(status).upper()
        return "CONVERGED" in status_text
    
    def _add_loadflow_execution_rows(self, rows, pyp_status, pp_status) -> None:
        """Append load-flow execution status rows to the debug table.
        
        Args:
            rows: Mutable list that receives result dictionaries.
            pyp_status: Status dictionary returned by the powsybl load-flow runner.
            pp_status: Status dictionary returned by the pandapower load-flow runner.
        """
        pyp_results = pyp_status.get("results", []) or []

        if pyp_results:
            for index, result in enumerate(pyp_results):
                status = getattr(result, "status", None)
                rows.append({
                    "element_type": "loadflow_status",
                    "element": f"powsybl_component_{index}",
                    "parameter": "status",
                    "unit": "",
                    "powsybl_result": status,
                    "pandapower_result": "",
                    "delta": "",
                    "status": "OK" if self._debug_loadflow_status_is_ok(status) else "ERROR"
                })

        else:
            rows.append({
                "element_type": "loadflow_status",
                "element": "powsybl",
                "parameter": "status",
                "unit": "",
                "powsybl_result": pyp_status.get("message", ""),
                "pandapower_result": "",
                "delta": "",
                "status": "OK" if pyp_status.get("success") else "ERROR"
            })

        rows.append({
            "element_type": "loadflow_status",
            "element": "pandapower",
            "parameter": "converged",
            "unit": "",
            "powsybl_result": "",
            "pandapower_result": pp_status.get("message", ""),
            "delta": "",
            "status": "OK" if pp_status.get("success") else "ERROR"
        })

    def _add_loadflow_check(
            self,
            rows,
            element_type,
            element,
            parameter,
            powsybl_result,
            pandapower_result,
            unit = "",
            abs_tol = 1e-5,
            rel_tol = 1e-5,
    ) -> None:
        """Append one numerical load-flow comparison row.
        
        The method compares the powsybl result value against the pandapower result
        value, calculates the delta where possible, and stores an ``OK`` or ``ERROR``
        status according to the given tolerances.
        """
        is_equal = self._debug_values_equal(
            powsybl_result,
            pandapower_result,
            abs_tol = abs_tol,
            rel_tol = rel_tol
        )

        rows.append({
            "element_type": element_type,
            "element": element,
            "parameter": parameter,
            "unit": unit,
            "powsybl_result": powsybl_result,
            "pandapower_result": pandapower_result,
            "delta": self._debug_delta(powsybl_result, pandapower_result),
            "status": "OK" if is_equal else "ERROR"
        })

    def _debug_get_result_values(self, row, *keys, default = np.nan) -> Any:
        """Return the first available result value from a powsybl result row.
        
        Args:
            row: Source row that may contain different naming variants.
            *keys: Candidate column names to check in order.
            default: Value returned if no usable result is found.
        """
        if row is None:
            return default
        
        for key in keys:
            if key in row.index:
                value = row.get(key)
                if not self._debug_is_missing(value):
                    return value
        
        return default
    
    def _debug_ampere_to_kiloampere(self, value) -> float:
        """Convert a current value from ampere to kiloampere for comparison."""
        value = self._debug_to_float(value, np.nan)

        if np.isfinite(value):
            return value / 1000.0
        
        return np.nan
    
    def _debug_get_pp_result_row(self, result_table, element_index) -> pd.Series | None:
        """Return a pandapower result row by element index.
        
        Returns:
            The matching pandas row or ``None`` if the result table is missing, empty,
            or does not contain the requested index.
        """
        if result_table is None or result_table.empty:
            return None
        
        if element_index in result_table.index:
            return result_table.loc[element_index]
        
        return None
    
    def _build_loadflow_bus_rows(self, rows) -> None:
        """Append bus voltage magnitude and angle comparison rows.
        
        The powsybl voltage magnitude is compared in kilovolts. The pandapower result
        is converted from per-unit to kilovolts using the nominal bus voltage.
        """
        pyp_bb_buses = self.pyp_net.get_bus_breaker_view_buses()
        pyp_buses = self.pyp_net.get_buses(all_attributes = True)
        pp_buses = self.pandap_net.bus
        pp_res_buses = self.pandap_net.res_bus

        for bus_name, row in pyp_bb_buses.iterrows():
            pp_row = self._debug_get_pp_match(pp_buses, bus_name)

            if pp_row is None:
                continue

            pp_result_row = self._debug_get_pp_result_row(pp_res_buses, pp_row.name)

            if pp_result_row is None:
                continue

            pyp_bus_id = row.get("bus_id")
            pyp_row = None

            if pyp_bus_id in pyp_buses.index:
                pyp_row = pyp_buses.loc[pyp_bus_id]
            else:
                pyp_row = row

            pp_vn_kv = self._debug_to_float(pp_row.get("vn_kv"), np.nan)
            pp_vm_pu = self._debug_to_float(pp_result_row.get("vm_pu"), np.nan)
            pp_vm_kv = pp_vm_pu * pp_vn_kv if np.isfinite(pp_vm_pu) and np.isfinite(pp_vn_kv) else np.nan

            self._add_loadflow_check(
                rows,
                "bus",
                bus_name,
                "v_mag",
                self._debug_get_result_values(pyp_row, "v_mag", "voltage", "voltage_kv"),
                pp_vm_kv,
                "kV",
                abs_tol = 1e-3,
                rel_tol = 1e-5
            )

            self._add_loadflow_check(
                rows,
                "bus",
                bus_name,
                "v_angle",
                self._debug_get_result_values(pyp_row, "v_angle", "angle", "va_degree"),
                pp_result_row.get("va_degree"),
                "deg",
                abs_tol = 1e-2,
                rel_tol = 1e-5
            )

    def _build_loadflow_generator_rows(self, rows) -> None:
        """Append generator active and reactive power comparison rows."""
        pyp_gens = self.pyp_net.get_generators(all_attributes = True)
        pp_gens = self.pandap_net.gen
        pp_res_gens = self.pandap_net.res_gen

        for gen_id, row in pyp_gens.iterrows():
            pp_row = self._debug_get_pp_match(pp_gens, gen_id)

            if pp_row is None:
                continue

            pp_result_row = self._debug_get_pp_result_row(pp_res_gens, pp_row.name)

            if pp_result_row is None:
                continue

            pyp_p_terminal = self._debug_get_result_values(row, "p")
            pyp_q_terminal = self._debug_get_result_values(row, "q")

            if not self._debug_is_missing(pyp_p_terminal):
                pyp_p = -self._debug_to_float(pyp_p_terminal, np.nan)
            else:
                pyp_p = self._debug_to_float(
                    self._debug_get_result_values(row, "p_mw", "target_p"),
                    np.nan
                )

            if not self._debug_is_missing(pyp_q_terminal):
                pyp_q = -self._debug_to_float(pyp_q_terminal, np.nan)
            else:
                pyp_q = self._debug_to_float(
                    self._debug_get_result_values(row, "q_mvar", "target_q"),
                    np.nan
                )

            self._add_loadflow_check(
                rows,
                "gen",
                gen_id,
                "p_mw",
                pyp_p,
                pp_result_row.get("p_mw"),
                "MW",
                abs_tol = 5e-2,
                rel_tol = 5e-3
            )

            self._add_loadflow_check(
                rows,
                "gen",
                gen_id,
                "q_mvar",
                pyp_q,
                pp_result_row.get("q_mvar"),
                "MVar",
                abs_tol = 5e-2,
                rel_tol = 5e-3
            )

    def _build_loadflow_line_rows(self, rows) -> None:
        """Append line or impedance power and current comparison rows for both terminals."""
        pyp_lines = self.pyp_net.get_lines(all_attributes = True)
        pp_lines = self.pandap_net.line
        pp_res_line = self.pandap_net.res_line

        pp_res_impedance = getattr(
            self.pandap_net,
            "res_impedance",
            pd.DataFrame()
        )

        for line_id, row in pyp_lines.iterrows():
            pp_row = self._debug_get_pp_match(pp_lines, line_id)

            if pp_row is not None:
                pp_result_row = self._debug_get_pp_result_row(
                    pp_res_line,
                    pp_row.name
                )

                element_type = "line"
            else:
                pp_row = self._debug_get_pp_impedance_match(line_id)

                if pp_row is None:
                    continue

                pp_result_row = self._debug_get_pp_result_row(
                    pp_res_impedance,
                    pp_row.name
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
                ("i2", "i_to_ka", "kA")
            ):
                
                pyp_value = self._debug_get_result_values(row, pyp_parameter)

                if pyp_parameter.startswith("i"):
                    pyp_value = self._debug_ampere_to_kiloampere(pyp_value)

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
                    abs_tol = abs_tol,
                    rel_tol = rel_tol
                )

    def _build_loadflow_2w_trafo_rows(self, rows) -> None:
        """Append two-windings transformer load-flow comparison rows.
        
        The method maps powsybl side 1 and side 2 to pandapower high- and low-voltage
        result columns according to the rated voltages.
        """
        pyp_trafos = self.pyp_net.get_2_windings_transformers(all_attributes = True)
        pp_trafos = self.pandap_net.trafo
        pp_res_trafo = self.pandap_net.res_trafo

        for trafo_id, row in pyp_trafos.iterrows():
            pp_row = self._debug_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                continue

            pp_result_row = self._debug_get_pp_result_row(pp_res_trafo, pp_row.name)

            if pp_result_row is None:
                continue

            rated_u1 = self._debug_to_float(row.get("rated_u1"), np.nan)
            rated_u2 = self._debug_to_float(row.get("rated_u2"), np.nan)

            if np.isfinite(rated_u1) and np.isfinite(rated_u2) and rated_u1 >= rated_u2:
                side_mapping = {
                    "1": ("p_hv_mw", "q_hv_mvar", "i_hv_ka"),
                    "2": ("p_lv_mw", "q_lv_mvar", "i_lv_ka")
                }
            else:
                side_mapping = {
                    "1": ("p_lv_mw", "q_lv_mvar", "i_lv_ka"),
                    "2": ("p_hv_mw", "q_hv_mvar", "i_hv_ka")
                }

            for pyp_side, pp_parameters in side_mapping.items():
                for pyp_prefix, pp_parameter, unit in (
                    ("p", pp_parameters[0], "MW"),
                    ("q", pp_parameters[1], "MVar"),
                    ("i", pp_parameters[2], "kA")
                ):
                    pyp_parameter = f"{pyp_prefix}{pyp_side}"
                    pyp_value = self._debug_get_result_values(row, pyp_parameter)

                    if pyp_prefix == "i":
                        pyp_value = self._debug_ampere_to_kiloampere(pyp_value)

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
                        abs_tol = abs_tol,
                        rel_tol = rel_tol
                    )

    def _build_loadflow_3w_trafo_rows(self, rows) -> None:
        """Append three-windings transformer load-flow comparison rows.
        
        Transformer legs are sorted by rated voltage before they are compared to
        pandapower high-, medium-, and low-voltage result columns.
        """
        pyp_trafos = self.pyp_net.get_3_windings_transformers(all_attributes = True)
        pp_trafos = self.pandap_net.trafo3w
        pp_res_trafo = self.pandap_net.res_trafo3w

        for trafo_id, row in pyp_trafos.iterrows():
            pp_row = self._debug_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                continue

            pp_result_row = self._debug_get_pp_result_row(pp_res_trafo, pp_row.name)

            if pp_result_row is None:
                continue

            legs = []
            for leg_no in (1, 2, 3):
                rated_u = self._debug_to_float(
                    self._debug_get_first(row, f"rated_u{leg_no}", f"ratedU{leg_no}", default = np.nan),
                    np.nan
                )

                if np.isfinite(rated_u):
                    legs.append({"leg_no": leg_no, "rated_u": rated_u})

            if len(legs) != 3:
                continue

            legs.sort(key = lambda leg: leg["rated_u"], reverse = True)
            side_names = ["hv", "mv", "lv"]

            for leg, side_name in zip(legs, side_names):
                leg_no = leg["leg_no"]

                for pyp_prefix, pp_parameter, unit in (
                    ("p", f"p_{side_name}_mw", "MW"),
                    ("q", f"q_{side_name}_mvar", "MVar"),
                    ("i", f"i_{side_name}_ka", "kA")
                ):
                    pyp_parameter = f"{pyp_prefix}{leg_no}"
                    pyp_value = self._debug_get_result_values(row, pyp_parameter)

                    if pyp_prefix == "i":
                        pyp_value = self._debug_ampere_to_kiloampere(pyp_value)

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
                        abs_tol = abs_tol,
                        rel_tol = rel_tol
                    )

    def _build_loadflow_load_rows(self, rows) -> None:
        """Append load active and reactive power comparison rows."""
        pyp_loads = self.pyp_net.get_loads(all_attributes = True)
        pp_loads = self.pandap_net.load
        pp_res_loads = self.pandap_net.res_load

        for load_id, row in pyp_loads.iterrows():
            pp_row = self._debug_get_pp_match(pp_loads, load_id)

            if pp_row is None:
                continue

            pp_result_row = self._debug_get_pp_result_row(pp_res_loads, pp_row.name)

            if pp_result_row is None:
                continue

            self._add_loadflow_check(
                rows,
                "load",
                load_id,
                "p_mw",
                self._debug_get_result_values(row, "p", "p0", "p_mw"),
                pp_result_row.get("p_mw"),
                "MW",
                abs_tol = 1e-4,
                rel_tol = 1e-5
            )

            self._add_loadflow_check(
                rows,
                "load",
                load_id,
                "q_mvar",
                self._debug_get_result_values(row, "q", "q0", "q_mvar"),
                pp_result_row.get("q_mvar"),
                "MVar",
                abs_tol = 1e-4,
                rel_tol = 1e-5
            )

    # ======================================================
    # DEBUG-HELPER
    # ======================================================

    def _debug_is_missing(self, value) -> bool:
        """Return whether a value should be treated as missing in debug logic.
        
        The check handles Python ``None``, NumPy/Pandas null values, and column textual
        representations such as ``nan``, ``none``, and ``<na>``.
        """
        if value is None:
            return True
        
        try:
            if hasattr(value, "item"):
                value = value.item()
        except Exception:
            pass

        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)):
                return bool(missing)
        except Exception:
            pass

        value_str = str(value).strip().lower()
        return value_str in ("", "nan", "none", "<na>")
    
    def _debug_to_float(self, value, default = np.nan) -> float:
        """Convert a debug value to ``float`` while rejecting invalid values.
        
        Boolean values are intentionally not converted to floats because they represent
        states rather than numeric quantities in the comparison tables.
        """
        try:
            if self._debug_is_missing(value):
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
        
    def _debug_to_bool(self, value, default = False) -> bool:
        """Convert a debug value to ``bool`` with support for textual states."""
        if self._debug_is_missing(value):
            return default
        
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        
        value_str = str(value).strip().lower()

        if value_str in ("true", "1", "yes", "y"):
            return True
        
        if value_str in ("false", "0", "no", "n"):
            return False
        
        return bool(value)
    
    def _debug_get_first(self, row, *keys, default = None) -> Any:
        """Return the first non-missing value for requested row keys."""
        for key in keys:
            if key in row.index:
                value = row.get(key)
                if not self._debug_is_missing(value):
                    return value
                
        return default
    
    def _debug_get_pp_match(self, pp_table, element_id) -> pd.Series | None:
        """Find a pandapower element row by its ``name`` column.
        
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
    
    def _debug_get_pp_impedance_match(self, element_id) -> pd.Series | None:
        """Find a pandapower impedance row by name or stored powsybl id."""

        pp_impedances = getattr(self.pandap_net, "impedance", pd.DataFrame())

        pp_row = self._debug_get_pp_match(pp_impedances, element_id)

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
    
    def _debug_values_equal(self, expected, actual, abs_tol = 1e-6, rel_tol = 1e-6) -> bool:
        """Compare two debug values with tolerant numerical handling.
        
        Numeric values are compared with ``numpy.isclose``. Missing values compare equal
        only when both sides are missing. Non-numeric values are compared as stripped
        strings, while booleans are compared as booleans.
        """
        expected_missing = self._debug_is_missing(expected)
        actual_missing = self._debug_is_missing(actual)

        if expected_missing and actual_missing:
            return True
        
        if expected_missing != actual_missing:
            return False
        
        if isinstance(expected, (bool, np.bool_)) or isinstance(actual, (bool, np.bool_)):
            return bool(expected) == bool(actual)
        
        expected_float = self._debug_to_float(expected, np.nan)
        actual_float = self._debug_to_float(actual, np.nan)

        if np.isfinite(expected_float) and np.isfinite(actual_float):
            return bool(np.isclose(
                actual_float,
                expected_float,
                rtol = rel_tol,
                atol = abs_tol
            ))
        
        return str(expected).strip() == str(actual).strip()
    
    def _debug_delta(self, expected, actual) -> float | str:
        """Return the numerical delta ``actual - expected`` when possible."""
        expected_float = self._debug_to_float(expected, np.nan)
        actual_float = self._debug_to_float(actual, np.nan)

        if np.isfinite(expected_float) and np.isfinite(actual_float):
            return actual_float - expected_float
        
        return ""
    
    def _add_transfer_check(
            self,
            rows,
            element_type,
            element,
            parameter,
            powsybl_expected,
            pandapower_created,
            unit = "",
            abs_tol = 1e-6,
            rel_tol = 1e-6
    ) -> None:
        """Append one static conversion row.
        
        The method compares the expected value derived from powsybl with the value that
        was created in pandapower and stores the result status in the transfer table.
        """
        
        is_equal = self._debug_values_equal(
            powsybl_expected,
            pandapower_created,
            abs_tol = abs_tol,
            rel_tol = rel_tol
        )

        rows.append({
            "element_type": element_type,
            "element": element,
            "parameter": parameter,
            "unit": unit,
            "powsybl_expected": powsybl_expected,
            "pandapower_created": pandapower_created,
            "delta": self._debug_delta(powsybl_expected, pandapower_created),
            "status": "OK" if is_equal else "ERROR"
        })

    def _add_missing_pp_element(self, rows, element_type, element) -> None:
        """Append a transfer-table row for a missing pandapower element."""
        rows.append({
            "element_type": element_type,
            "element": element,
            "parameter": "element_present",
            "unit": "",
            "powsybl_expected": True,
            "pandapower_created": False,
            "delta": "",
            "status": "ERROR"
        })

    def _build_bus_transfer_rows(self, rows) -> None:
        """Append static bus nominal-voltage comparison rows."""
        pyp_buses = self.pyp_net.get_bus_breaker_view_buses()
        pyp_voltage_levels = self.pyp_net.get_voltage_levels()
        pp_buses = self.pandap_net.bus

        for bus_name, row in pyp_buses.iterrows():
            pp_row = self._debug_get_pp_match(pp_buses, bus_name)

            if pp_row is None:
                self._add_missing_pp_element(rows, "bus", bus_name)
                continue

            voltage_level_id = row.get("voltage_level_id")

            if voltage_level_id in pyp_voltage_levels.index:
                expected_vn_kv = self._debug_to_float(
                    pyp_voltage_levels.at[voltage_level_id, "nominal_v"],
                    np.nan
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
                "kV"
            )

    def _build_generator_transfer_rows(self, rows) -> None:
        """Append static generator parameter comparison rows.
        
        The method checks active power, voltage magnitude, limits, slack state,
        and service state.
        """
        pyp_gens = self.pyp_net.get_generators(all_attributes = True)
        pp_gens = self.pandap_net.gen
        slack_generator_id = self._select_slack_generator_id(pyp_gens)

        for gen_id, row in pyp_gens.iterrows():
            pp_row = self._debug_get_pp_match(pp_gens, gen_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "gen", gen_id)
                continue

            pp_bus_idx = pp_row.get("bus")
            bus_vn_kv = np.nan
            
            if pp_bus_idx in self.pandap_net.bus.index:
                bus_vn_kv = self._debug_to_float(
                    self.pandap_net.bus.at[pp_bus_idx, "vn_kv"],
                    np.nan
                )

            target_v_kv = self._debug_to_float(row.get("target_v"), np.nan)

            if np.isfinite(target_v_kv) and np.isfinite(bus_vn_kv) and bus_vn_kv > 0:
                expected_vm_pu = target_v_kv / bus_vn_kv
            else:
                expected_vm_pu = self.default_vm_pu

            checks = [
                ("p_mw", self._debug_to_float(row.get("target_p"), self.default_active_power_mw), pp_row.get("p_mw"), "MW"),
                ("vm_pu", expected_vm_pu, pp_row.get("vm_pu"), "pu"),
                ("sn_mva", self._debug_to_float(row.get("rated_s"), np.nan), pp_row.get("sn_mva"), "MVA"),
                ("min_p_mw", self._debug_to_float(row.get("min_p"), np.nan), pp_row.get("min_p_mw"), "MW"),
                ("max_p_mw", self._debug_to_float(row.get("max_p"), np.nan), pp_row.get("max_p_mw"), "MW"),
                ("slack", str(gen_id) == str(slack_generator_id), pp_row.get("slack"), ""),
                ("in_service", self._debug_to_bool(row.get("connected"), True), pp_row.get("in_service"), "")
            ]

            for parameter, expected, actual, unit in checks:
                self._add_transfer_check(
                    rows,
                    "gen",
                    gen_id,
                    parameter,
                    expected,
                    actual,
                    unit
                )

    def _build_line_transfer_rows(self, rows) -> None:
        """Append static line parameter comparison rows."""
        pyp_lines = self.pyp_net.get_lines(all_attributes = True)
        pp_lines = self.pandap_net.line
        pp_impedances = getattr(self.pandap_net, "impedance", pd.DataFrame())

        for line_id, row in pyp_lines.iterrows():
            pp_row = self._debug_get_pp_match(pp_lines, line_id)

            if pp_row is None:
                pp_impedance_row = self._debug_get_pp_match(pp_impedances, line_id)

                if pp_impedance_row is None and not pp_impedances.empty and "powsybl_id" in pp_impedances.columns:
                    match = pp_impedances[
                        pp_impedances["powsybl_id"].astype(str) == str(line_id)
                    ]

                    if not match.empty:
                        pp_impedance_row = match.iloc[0]

                if pp_impedance_row is None:
                    self._add_missing_pp_element(rows, "line", line_id)
                    continue

                expected_impedance = self._debug_expected_cross_voltage_impedance_values(
                    row,
                    pp_impedance_row
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
                    ("in_service", "")
                ):
                    self._add_transfer_check(
                        rows,
                        "impedance",
                        line_id,
                        parameter,
                        expected_impedance.get(parameter),
                        pp_impedance_row.get(parameter),
                        unit
                    )
                
                continue

            expected = self._debug_expected_line_values(row)

            for parameter, unit in (
                ("length_km", "km"),
                ("r_ohm_per_km", "Ohm/km"),
                ("x_ohm_per_km", "Ohm/km"),
                ("c_nf_per_km", "nF/km"),
                ("g_us_per_km", "uS/km"),
                ("parallel", ""),
                ("in_service", "")
            ):
                self._add_transfer_check(
                    rows,
                    "line",
                    line_id,
                    parameter,
                    expected.get(parameter),
                    pp_row.get(parameter),
                    unit
                )

    def _debug_expected_line_values(self, row) -> dict[str, Any]:
        """Calculate expected pandapower line parameters from a powsybl line row.
        
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
            
        parallel = self._debug_to_float(row.get("parallel"), np.nan)
        if not np.isfinite(parallel) or parallel <= 0:
            parallel = self.default_parallel

        length_km = self._debug_to_float(row.get("length_km"), np.nan)

        if not np.isfinite(length_km):
            length_km = self._debug_to_float(row.get("length"), np.nan)

        if not np.isfinite(length_km) or length_km <= 0:
            length_km = self.default_length_km

        r_ohm_per_km = r_ohm * parallel / length_km
        x_ohm_per_km = x_ohm * parallel / length_km

        c_nf_per_km = (
            b_total_s * 1e9
        ) / (
            2.0 * np.pi * self.pandap_net.f_hz * length_km * parallel
        )

        g_us_per_km = (
            g_total_s * 1e6
        ) / (
            length_km * parallel
        )

        in_service = (
            self._debug_to_bool(row.get("connected1"), True)
            and self._debug_to_bool(row.get("connected2"), True)
        )

        return {
            "length_km": length_km,
            "r_ohm_per_km": r_ohm_per_km,
            "x_ohm_per_km": x_ohm_per_km,
            "c_nf_per_km": c_nf_per_km,
            "g_us_per_km": g_us_per_km,
            "parallel": parallel,
            "in_service": in_service
        }
    
    def _debug_expected_cross_voltage_impedance_values(self, row, pp_impedance_row) -> dict[str, Any]:
        """Calculate expected pandapower impedance values for cross-voltage powsybl lines."""

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
            pp_impedance_row.get("sn_mva"),
            self.default_base_sn_mva
        )

        if not np.isfinite(sn_mva) or sn_mva <= 0:
            sn_mva = self.default_base_sn_mva

        in_service = (
            self._debug_to_bool(row.get("connected1"), True)
            and self._debug_to_bool(row.get("connected2"), True)
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
            "in_service": in_service
        }

    def _build_2w_trafo_transfer_rows(self, rows) -> None:
        """Append static two-windings transformer comparison rows."""
        pyp_2w_trafos = self.pyp_net.get_2_windings_transformers(all_attributes = True)
        pp_trafos = self.pandap_net.trafo

        for trafo_id, row in pyp_2w_trafos.iterrows():
            pp_row = self._debug_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "trafo_2w", trafo_id)
                continue
            
            expected = self._debug_expected_2w_trafo_values(row)

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
                ("in_service", "")
            ):
                self._add_transfer_check(
                    rows,
                    "trafo_2w",
                    trafo_id,
                    parameter,
                    expected.get(parameter),
                    pp_row.get(parameter),
                    unit
                )

    def _debug_expected_2w_trafo_values(self, row) -> dict[str, Any] | None:
        """Calculate expected pandapower values for a two-winding transformer.
        
        Returns:
            Dictionary with pandapower transformer parameters or ``None`` when required
            rated-voltage data is missing or invalid.
        """
        r_ohm = self._to_float(row.get("r"), self.default_impedance_ohm)
        x_ohm = self._to_float(row.get("x"), self.default_impedance_ohm)
        g_s = self._to_float(row.get("g"), self.default_admittance_s)
        b_s = self._to_float(row.get("b"), self.default_admittance_s)

        rated_u1 = self._debug_to_float(row.get("rated_u1"), np.nan)
        rated_u2 = self._debug_to_float(row.get("rated_u2"), np.nan)
        rated_s = self._debug_to_float(row.get("rated_s"), np.nan)

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

        z_base_side2 = (rated_u2 ** 2) / rated_s

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

        pfe_kw = (rated_u2 ** 2) * g_s * 1000.0

        expected_alpha = self._debug_to_float(
            row.get("alpha"),
            self.default_shift_degree
        )

        in_service = (
            self._debug_to_bool(row.get("connected1"), True)
            and self._debug_to_bool(row.get("connected2"), True)
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
            "in_service": in_service
        }

    def _build_3w_trafo_transfer_rows(self, rows) -> None:
        """Append static three-winding transfer comparison rows."""
        pyp_3w_trafos = self.pyp_net.get_3_windings_transformers(all_attributes = True)
        pp_trafos = self.pandap_net.trafo3w

        for trafo_id, row in pyp_3w_trafos.iterrows():
            pp_row = self._debug_get_pp_match(pp_trafos, trafo_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "trafo_3w", trafo_id)
                continue

            expected = self._debug_expected_3w_trafo_values(row)

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
                ("in_service", "")
            ):
                self._add_transfer_check(
                    rows,
                    "trafo_3w",
                    trafo_id,
                    parameter,
                    expected.get(parameter),
                    pp_row.get(parameter),
                    unit
                )

    def _debug_expected_3w_trafo_values(self, row) -> dict[str, Any] | None:
        """Calculate expected pandapower values for a three-winding transformer.
        
        The method mirrors the creation logic used during conversion so that the debug
        table can verify voltage-side ordering, pairwise impedance values, losses,
        phase shifts, and service state.
        
        Returns:
            Dictionary with expected pandapower parameters or ``None`` when required
            data is missing.
        """
        legs = []

        for leg_no in (1, 2, 3):
            bus_ref = self._debug_get_first(
                row,
                f"bus{leg_no}_id",
                f"bus_breaker_bus{leg_no}_id",
                f"bus{leg_no}"
            )

            connected = self._debug_to_bool(
                self._debug_get_first(
                    row,
                    f"connected{leg_no}",
                    default = not self._debug_is_missing(bus_ref)
                    ),
                default = not self._debug_is_missing(bus_ref)
            )

            if self._debug_is_missing(bus_ref):
                connected = False

            rated_u = self._to_float(
                self._debug_get_first(
                    row,
                    f"rated_u{leg_no}",
                    f"ratedU{leg_no}",
                    default = np.nan
                ),
                np.nan
            )

            rated_s = self._to_float(
                self._debug_get_first(
                    row,
                    f"rated_s{leg_no}",
                    f"ratedS{leg_no}",
                    default = np.nan
                ),
                np.nan
            )

            if not np.isfinite(rated_u) or rated_u <= 0:
                return None

            if not np.isfinite(rated_s) or rated_s <= 0:
                rated_s = self.default_trafo_sn_mva

            r_ohm = self._to_float(
                self._debug_get_first(
                    row,
                    f"r{leg_no}_at_current_tap",
                    f"r{leg_no}",
                    default = self.default_impedance_ohm
                ),
                self.default_impedance_ohm
            )

            x_ohm = self._to_float(
                self._debug_get_first(
                    row,
                    f"x{leg_no}_at_current_tap",
                    f"x{leg_no}",
                    default = self.default_impedance_ohm
                ),
                self.default_impedance_ohm
            )

            g_s = self._to_float(
                self._debug_get_first(
                    row,
                    f"g{leg_no}_at_current_tap",
                    f"g{leg_no}",
                    default = self.default_admittance_s
                ),
                self.default_admittance_s
            )

            b_s = self._to_float(
                self._debug_get_first(
                    row,
                    f"b{leg_no}_at_current_tap",
                    f"b{leg_no}",
                    default = self.default_admittance_s
                ),
                self.default_admittance_s
            )

            alpha_degree = self._to_float(
                self._debug_get_first(
                    row,
                    f"alpha{leg_no}",
                    default = self.default_shift_degree
                ),
                self.default_shift_degree
            )

            z_ohm = complex(r_ohm, x_ohm)

            z_base_ohm = (rated_u ** 2) / rated_s
            y_relative = complex(g_s, b_s) * z_base_ohm

            i0_percent = 100.0 * abs(y_relative)
            pfe_kw = (rated_u ** 2) * g_s * 1000.0

            legs.append({
                "leg_no": leg_no,
                "rated_u": rated_u,
                "rated_s": rated_s,
                "z_ohm": z_ohm,
                "i0_percent": i0_percent,
                "pfe_kw": pfe_kw,
                "connected": connected,
                "alpha_degree": alpha_degree
            })

        if len(legs) != 3:
            return None

        legs.sort(key = lambda leg: leg["rated_u"], reverse = True)

        hv_leg = legs[0]
        mv_leg = legs[1]
        lv_leg = legs[2]
        reference_u_kv = hv_leg["rated_u"]

        vk_hv_percent, vkr_hv_percent = self._debug_3w_pair_short_circuit_values(
            hv_leg,
            mv_leg,
            reference_u_kv
        )
    
        vk_mv_percent, vkr_mv_percent = self._debug_3w_pair_short_circuit_values(
            mv_leg,
            lv_leg,
            reference_u_kv
        )
    
        vk_lv_percent, vkr_lv_percent = self._debug_3w_pair_short_circuit_values(
            hv_leg,
            lv_leg,
            reference_u_kv
        )

        pfe_kw = (
            hv_leg["pfe_kw"]
            + mv_leg["pfe_kw"]
            + lv_leg["pfe_kw"]
        )
    
        i0_percent = (
            hv_leg["i0_percent"]
            + mv_leg["i0_percent"]
            + lv_leg["i0_percent"]
        )

        shift_mv_degree = mv_leg["alpha_degree"] - hv_leg["alpha_degree"]
        shift_lv_degree = lv_leg["alpha_degree"] - hv_leg["alpha_degree"]
    
        in_service = (
            hv_leg["connected"]
            and mv_leg["connected"]
            and lv_leg["connected"]
        )

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
            "in_service": in_service
        }
    
    def _debug_3w_pair_short_circuit_values(self, leg_a, leg_b, reference_u_kv) -> tuple[float, float]:
        """Calculate pairwise short-circuit values for debug verification.
        
        Args:
            leg_a: First transformer leg dictionary
            leg_b: Second transformer leg dictionary.
            reference_u_kv: Voltage base used for the pairwise impedance conversion.
            
        Returns:
            Tuple ``(vk_percent, vkr_percent)`` for the selected leg pair.
        """
        s_ref_mva = min(leg_a["rated_s"], leg_b["rated_s"])
        z_base_ohm = (reference_u_kv ** 2) / s_ref_mva

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

    def _build_load_transfer_rows(self, rows) -> None:
        """Append static load parameter comparison rows."""
        pyp_loads = self.pyp_net.get_loads(all_attributes = True)
        pp_loads = self.pandap_net.load

        for load_id, row in pyp_loads.iterrows():
            pp_row = self._debug_get_pp_match(pp_loads, load_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "load", load_id)
                continue

            self._add_transfer_check(
                rows,
                "load",
                load_id,
                "p_mw",
                self._debug_to_float(row.get("p0"), self.default_active_power_mw),
                pp_row.get("p_mw"),
                "MW"
            )

            self._add_transfer_check(
                rows,
                "load",
                load_id,
                "q_mvar",
                self._debug_to_float(row.get("q0"), self.default_reactive_power_mvar),
                pp_row.get("q_mvar"),
                "MVar"
            )

            self._add_transfer_check(
                rows,
                "load",
                load_id,
                "in_service",
                self._debug_to_bool(row.get("connected"), True),
                pp_row.get("in_service")
            )

    def _build_switch_transfer_rows(self, rows) -> None:
        """Append static switch parameter comparison rows.
        
        The method checks the resolved buses, switch element type, closed state,
        switch type, and switch impedance.
        """
        pyp_switches = self.pyp_net.get_switches(all_attributes = True)
        pp_switches = self.pandap_net.switch

        for switch_id, row in pyp_switches.iterrows():
            pp_row = self._debug_get_pp_match(pp_switches, switch_id)

            if pp_row is None:
                self._add_missing_pp_element(rows, "switch", switch_id)
                continue

            expected_closed = not self._debug_to_bool(row.get("open"), False)
            expected_type = self._get_pp_switch_type(row)
            expected_z_ohm = self.default_switch_z_ohm

            self._add_transfer_check(
                rows,
                "switch",
                switch_id,
                "closed",
                expected_closed,
                pp_row.get("closed")
            )

            self._add_transfer_check(
                rows,
                "switch",
                switch_id,
                "type",
                expected_type,
                pp_row.get("type")
            )

            self._add_transfer_check(
                rows,
                "switch",
                switch_id,
                "z_ohm",
                expected_z_ohm,
                pp_row.get("z_ohm"),
                "Ohm"
            )
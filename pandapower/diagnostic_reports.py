# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# initialize custom log levels for compact and detailed output
def add_log_level(levelno: int, levelname: str) -> None:
    """
    add a custom log level to the logger

    :param levelno: int number for the level
    :param levelname: name for the level
    """

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(levelno):
            self._log(levelno, message, args, **kwargs)

    logging.addLevelName(level=levelno, levelName=levelname.upper())
    setattr(logging, levelname.upper(), levelno)
    setattr(logging.getLoggerClass(), levelname.lower(), log_for_level)
    setattr(logging, levelname.lower(), log_for_level)


add_log_level(logging.WARNING + 4, "compact")
add_log_level(logging.WARNING + 2, "detailed")


class NotCompactFilter(logging.Filter):
    """
    Filter compact messages if detailed view is required

    Returns True if the message is not on log level 'COMPACT'
    """
    def filter(self, record):
        return record.levelno != logging.COMPACT

class NotDetailedFilter(logging.Filter):
    """
    Filter compact messages if detailed view is required

    Returns True if the message is not on log level 'DETAILED'
    """
    def filter(self, record):
        return record.levelno != logging.DETAILED

class LogCount(logging.Filter):
    """
    logging Message Filter that counts Messages but does not Filter any out.

    This is used to detect if a separator should be printed or not
    """
    def __init__(self):
        super().__init__()
        self.count = defaultdict(int)

    def filter(self, record):
        self.count[record.levelno] += 1
        return True

    def cd_in_count(self):
        return logging.COMPACT in self.count or logging.DETAILED in self.count

    def reset(self):
        self.count = defaultdict(int)


# separator between log messages
log_format_len = 60
log_message_sep = f"\n{'':-<{log_format_len}}\n"


def diagnostic_report(net, diag_results, diag_errors, diag_params, compact_report, warnings_only):
    diag_report = DiagnosticReports(net, diag_results, diag_errors, diag_params, compact_report)
    if warnings_only:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logger.setLevel(log_level)

    report_methods = {
        "missing_bus_indices": diag_report.report_missing_bus_indices,
        "disconnected_elements": diag_report.report_disconnected_elements,
        "different_voltage_levels_connected": diag_report.report_different_voltage_levels_connected,
        "impedance_values_close_to_zero": diag_report.report_impedance_values_close_to_zero,
        "nominal_voltages_dont_match": diag_report.report_nominal_voltages_dont_match,
        "invalid_values": diag_report.report_invalid_values,
        "overload": diag_report.report_overload,
        "multiple_voltage_controlling_elements_per_bus": diag_report.report_multiple_voltage_controlling_elements_per_bus,
        "wrong_switch_configuration": diag_report.report_wrong_switch_configuration,
        "no_ext_grid": diag_report.report_no_ext_grid,
        "wrong_reference_system": diag_report.report_wrong_reference_system,
        "deviation_from_std_type": diag_report.report_deviation_from_std_type,
        "numba_comparison": diag_report.report_numba_comparison,
        "parallel_switches": diag_report.report_parallel_switches,
    }

    log_counter = LogCount()

    logger.warning(f"\n\n{' PANDAPOWER DIAGNOSTIC TOOL ':-^{log_format_len}}\n")
    logger.addFilter(log_counter)

    for key in report_methods:
        if (key in diag_results) or not warnings_only:
            if log_counter.cd_in_count():
                logger.warning(log_message_sep)
            log_counter.reset()
            report_methods[key]()

    logger.removeFilter(log_counter)
    logger.warning(f"\n\n{' END OF PANDAPOWER DIAGNOSTIC ':-^{log_format_len}}\n")


class DiagnosticReports:
    def __init__(self, net, diag_results, diag_errors, diag_params, compact_report):
        self.net = net
        self.diag_results = diag_results
        self.diag_errors = diag_errors
        self.diag_params = diag_params
        # if compact report filter out detailed messages and vice versa
        if compact_report:
            logger.addFilter(NotDetailedFilter())
        else:
            logger.addFilter(NotCompactFilter())

    def report_disconnected_elements(self):
        # error and success checks
        if "disconnected_elements" in self.diag_errors:
            logger.warning("Check for disconnected elements failed due to the following error:")
            logger.warning(self.diag_errors["disconnected_elements"])
            return
        if "disconnected_elements" not in self.diag_results:
            logger.info("PASSED: No problematic switches found")
            return
        # message header
        logger.compact("disconnected_elements:\n")
        logger.detailed("Checking for elements without a connection to an external grid...\n")

        # message body
        diag_result = self.diag_results["disconnected_elements"]
        element_counter = 0
        for disc_section in diag_result:
            logger.compact(f"disconnected_section: {disc_section}")
            logger.detailed("Disconnected section found,"
                           " consisting of the following elements:")
            for key in disc_section:
                element_counter += len(disc_section[key])
                logger.detailed(f"{key}: {disc_section[key]}")

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} disconnected element(s) found.")

    def report_different_voltage_levels_connected(self):
        from pandapower.toolbox import get_connected_buses_at_element
        # error and success checks
        if "different_voltage_levels_connected" in self.diag_errors:
            logger.warning("Check for connection of different voltage levels failed due to the following error:")
            logger.warning(self.diag_errors["different_voltage_levels_connected"])
            return
        if "different_voltage_levels_connected" not in self.diag_results:
            logger.info("PASSED: No connection of different voltage levels found")
            return
        # message header
        logger.compact("different_voltage_levels_connected:\n")
        logger.detailed("Checking for connections of different voltage levels...\n")

        # message body
        diag_result = self.diag_results["different_voltage_levels_connected"]
        element_counter = 0
        for key in diag_result:
            element_counter += len(diag_result[key])
            element_type = ""
            if key == "lines":
                element_type = "line"
            elif key == "switches":
                element_type = "switch"
            logger.compact(f"{key}:")
            for element in diag_result[key]:
                buses = list(get_connected_buses_at_element(self.net, element, key[0]))
                logger.compact(f"{element_type} {element}: buses {buses}")
                logger.detailed(
                    f"{element_type} {element} connects bus {buses[0]}: {self.net.bus.name.at[buses[0]]} "
                    f"(vn_kv = {self.net.bus.vn_kv.at[buses[0]]}) and "
                    f"bus {buses[1]}: {self.net.bus.name.at[buses[1]]} "
                    f"(vn_kv = {self.net.bus.vn_kv.at[buses[1]]})"
                )
        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} element(s) that connect different voltage levels found.")


    def report_impedance_values_close_to_zero(self):
        # error and success checks
        if "impedance_values_close_to_zero" in self.diag_errors:
            logger.warning("Check for elements with impedance values close to zero failed due "
                           "to the following error:")
            logger.warning(self.diag_errors["impedance_values_close_to_zero"])
            return
        if "impedance_values_close_to_zero" not in self.diag_results:
            logger.info("PASSED: No elements with impedance values close to zero found...")
            return

        # message header
        logger.compact("impedance_values_close_to_zero:\n")
        logger.detailed("Checking for impedance values close to zero...\n")

        # message body
        diag_result = self.diag_results["impedance_values_close_to_zero"][0]
        element_counter = 0
        for key in diag_result:
            element_counter += len(diag_result[key])
            for element in diag_result[key]:
                min_r_type = ""
                min_x_type = ""
                if key in ("line", "line_dc", "xward"):
                    min_r_type = "r_ohm"
                    min_x_type = "x_ohm"
                elif key == "impedance":
                    min_r_type = "r_pu"
                    min_x_type = "x_pu"
                elif key == "vsc":
                    min_r_type = "r_dc_ohm"
                    min_x_type = "x_ohm"
                logger.warning(
                    f"{key} {element}: {min_r_type} <= {self.diag_params['min_'+min_r_type]} or "
                    f"{min_x_type} <= {self.diag_params['min_'+min_x_type]}"
                )

        if len(self.diag_results["impedance_values_close_to_zero"]) > 1:
            switch_replacement = self.diag_results["impedance_values_close_to_zero"][1]
            if switch_replacement["loadflow_converges_with_switch_replacement"]:
                logger.warning("Switch replacement successful: Power flow converges after "
                               "replacing implausible elements with switches.")
            else:
                logger.warning("Power flow still does not converge after replacing implausible "
                               "elements with switches.")

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} element(s) with impedance values close to zero found.")


    def report_nominal_voltages_dont_match(self):
        # error and success checks
        if "nominal_voltages_dont_match" in self.diag_errors:
            logger.warning("Check for components with deviating nominal voltages failed due "
                           "to the following error:")
            logger.warning(self.diag_errors["nominal_voltages_dont_match"])
            return
        if "nominal_voltages_dont_match" not in self.diag_results:
            logger.info("PASSED: No components with deviating nominal voltages found")
            return
        # message header
        logger.compact("nominal_voltages_dont_match:\n")
        logger.detailed("Checking for components with deviating nominal voltages...\n")

        # message body
        diag_result = self.diag_results["nominal_voltages_dont_match"]
        nom_voltage_tolerance = self.diag_params["nom_voltage_tolerance"]
        element_counter = 0
        for element in diag_result:
            logger.compact(f"{element}:")
            for key in diag_result[element]:
                element_counter += len(diag_result[element][key])
                if element == "trafo":
                    logger.compact(f"{key}: {diag_result[element][key]}")
                    if key == "hv_lv_swapped":
                        logger.detailed(
                            f"Trafo(s) {diag_result[element][key]}: hv and lv connectors seem to be swapped"
                        )
                    elif key == "hv_bus":
                        for trafo in diag_result[element][key]:
                            logger.detailed(
                                f"Trafo {trafo}: Nominal voltage on hv_side"
                                f"({self.net.trafo.vn_hv_kv.at[trafo]} kV) and voltage_level of hv_bus "
                                f"(bus {self.net.trafo.hv_bus.at[trafo]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo.hv_bus.at[trafo]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                    elif key == "lv_bus":
                        for trafo in diag_result[element][key]:
                            logger.detailed(
                                f"Trafo {trafo}: Nominal voltage on lv_side "
                                f"({self.net.trafo.vn_lv_kv.at[trafo]} kV) and voltage_level of lv_bus "
                                f"(bus {self.net.trafo.lv_bus.at[trafo]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo.lv_bus.at[trafo]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                if element == "trafo3w":
                    logger.compact(f"{key}: {diag_result[element][key]}")
                    if key == "connectors_swapped_3w":
                        logger.detailed(f"Trafo3w {diag_result[element][key]}: connectors seem to be swapped")
                    elif key == "hv_bus":
                        for trafo3w in diag_result[element][key]:
                            logger.detailed(
                                f"Trafo3w {trafo3w}: Nominal voltage on hv_side "
                                f"({self.net.trafo3w.vn_hv_kv.at[trafo3w]} kV) and voltage_level of hv_bus "
                                f"(bus {self.net.trafo3w.hv_bus.at[trafo3w]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo3w.hv_bus.at[trafo3w]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                    elif key == "mv_bus":
                        for trafo3w in diag_result[element][key]:
                            logger.detailed(
                                f"Trafo3w {trafo3w}: Nominal voltage on mv_side "
                                f"({self.net.trafo3w.vn_mv_kv.at[trafo3w]} kV) and voltage_level of mv_bus "
                                f"(bus {self.net.trafo3w.mv_bus.at[trafo3w]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo3w.mv_bus.at[trafo3w]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                    elif key == "lv_bus":
                        for trafo3w in diag_result[element][key]:
                            logger.detailed(
                                f"Trafo3w {trafo3w}: Nominal voltage on lv_side "
                                f"({self.net.trafo3w.vn_lv_kv.at[trafo3w]} kV) and voltage_level of lv_bus "
                                f"(bus {self.net.trafo3w.lv_bus.at[trafo3w]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo3w.lv_bus.at[trafo3w]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} component(s) with deviating nominal voltages found")

    def report_invalid_values(self):
        # error and success checks
        if "invalid_values" in self.diag_errors:
            logger.warning("Check for invalid values failed due to the following error:")
            logger.warning(self.diag_errors["invalid_values"])
            return
        if "invalid_values" not in self.diag_results:
            logger.info("PASSED: No invalid values found")
            return
        # message header
        logger.compact("invalid_values:\n")
        logger.detailed("Checking for invalid_values...\n")

        # message body
        diag_result = self.diag_results["invalid_values"]
        element_counter = 0
        for element_type in diag_result:
            element_counter += len(diag_result[element_type])
            logger.warning(f"{element_type}:")
            for inv_value in diag_result[element_type]:
                logger.compact(
                    f"{element_type} {inv_value[0]}: '{inv_value[1]}' = {inv_value[2]} "
                    f"(restriction: {inv_value[3]})"
                )
                logger.detailed(
                    f"Invalid value found: '{element_type} {inv_value[0]}' with attribute "
                    f"'{inv_value[1]}' = {inv_value[2]} "
                    f"(data type: {type(inv_value[2])}). Valid input needs to be {inv_value[3]}."
                )

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} invalid values found.")

    def report_overload(self):
        # error and success checks
        if "overload" in self.diag_errors:
            logger.warning("Check for overload failed due to the following error:")
            logger.warning(self.diag_errors["overload"])
            return
        if "overload" not in self.diag_results:
            logger.info("PASSED: Power flow converges. No overload found.")
            return
        # message header
        logger.compact("overload:\n")
        logger.detailed("Checking for overload...\n")

        # message body
        diag_result = self.diag_results["overload"]
        overload_scaling_factor = self.diag_params["overload_scaling_factor"]
        osf_percent = f"{overload_scaling_factor * 100} percent."
        if not diag_result["load"] and not diag_result["generation"]:
            logger.warning(
                "Overload check failed: Power flow still does not converge with load and generation scaled down to "
                f"{osf_percent}"
            )
        elif diag_result["load"] and diag_result["generation"]:
            logger.warning(
                f"overload found: Power flow converges with load and generation scaled down to {osf_percent}"
            )
        else:
            if diag_result["load"]:
                logger.warning(f"overload found: Power flow converges with load scaled down to {osf_percent}")
            else:
                assert diag_result["generation"]
                logger.warning(f"overload found: Power flow converges with generation scaled down to {osf_percent}")

    def report_wrong_switch_configuration(self):
        # error and success checks
        if "wrong_switch_configuration" in self.diag_errors:
            logger.warning("Check for wrong switch configuration failed due to the following error:")
            logger.warning(self.diag_errors["wrong_switch_configuration"])
            return
        if "wrong_switch_configuration" not in self.diag_results:
            logger.info("PASSED: Power flow converges. Switch configuration seems ok.")
            return
        # message header
        logger.compact("wrong_switch_configuration:\n")
        logger.detailed("Checking switch configuration...\n")

        # message body
        diag_result = self.diag_results["wrong_switch_configuration"]
        if diag_result:
            logger.warning("Possibly wrong switch configuration found: power flow "
                           "converges with all switches closed.")
        else:
            logger.warning("Power flow still does not converge with all switches closed.")


    def report_no_ext_grid(self):
        # error and success checks
        if "no_ext_grid" in self.diag_errors:
            logger.warning("Check for external grid failed due to the following error:")
            logger.warning(self.diag_errors["no_ext_grid"])
            return
        if "no_ext_grid" not in self.diag_results:
            logger.info("PASSED: External grid found.")
            return
        # message header
        logger.compact("no_external_grid:\n")
        logger.detailed("Checking if there is at least one external grid...\n")

        # message body
        diag_result = self.diag_results["no_ext_grid"]
        if diag_result is True:
            logger.warning("No ext_grid found. There has to be at least one ext_grid!")

    def report_multiple_voltage_controlling_elements_per_bus(self):
        # error and success checks
        if "multiple_voltage_controlling_elements_per_bus" in self.diag_errors:
            logger.warning("Check for multiple voltage controlling elements per bus failed due to the following error:")
            logger.warning(self.diag_errors["multiple_voltage_controlling_elements_per_bus"])
            return
        if "multiple_voltage_controlling_elements_per_bus" not in self.diag_results:
            logger.info("PASSED: No buses with multiple gens and/or ext_grids found.")
            return
        # message header
        logger.compact("multiple_voltage_controlling_elements_per_bus:\n")
        logger.detailed("Checking for multiple gens and/or external grids per bus...\n")

        # message body
        diag_result = self.diag_results["multiple_voltage_controlling_elements_per_bus"]
        element_counter = 0
        for feeder_type in diag_result:
            element_counter += len(diag_result[feeder_type])
            logger.compact(f"{feeder_type}: {diag_result[feeder_type]}")
            for bus in diag_result[feeder_type]:
                if feeder_type == "buses_with_mult_ext_grids":
                    logger.detailed(
                        f"External grids {list(self.net.ext_grid[self.net.ext_grid.bus == bus].index)} "
                        f"are connected to bus {bus}. Only one external grid per bus is allowed."
                    )
                elif feeder_type == "buses_with_gens_and_ext_grids":
                    logger.detailed(
                        f"Generator(s) {list(self.net.gen[self.net.gen.bus == bus].index)} and "
                        f"external grid(s) {list(self.net.ext_grid[self.net.ext_grid.bus == bus].index)} "
                        f"are connected to bus {bus}. "
                        "Only one generator OR one external grid per bus is allowed."
                    )

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} bus(ses) with multiple gens and/or ext_grids found.")


    def report_wrong_reference_system(self):
        # error and success checks
        if "wrong_reference_system" in self.diag_errors:
            logger.warning("Check for wrong reference system failed due to the following error:")
            logger.warning(self.diag_errors["wrong_reference_system"])
            return
        if "wrong_reference_system" not in self.diag_results:
            logger.info("PASSED: correct reference system")
            return
        # message header
        logger.compact("wrong_reference_system:\n")
        logger.detailed("Checking for usage of wrong reference system...\n")

        # message body
        diag_result = self.diag_results["wrong_reference_system"]
        for element_type in diag_result:
            logger.compact(f"{element_type} {diag_result[element_type]}: wrong reference system.")
            for element in diag_result[element_type]:
                _element_type = element_type[:-1]  # remove s at end (element_type can be 'loads', 'gens' or 'sgens'
                element_name = self.net[_element_type].name.at[element]
                element_p_mw = self.net[_element_type].p_mw.at[element]
                logger.detailed(
                    f"Found {_element_type} {element}: '{element_name}' with p_mw = "
                    f"{element_p_mw}. In load reference system p_mw should be positive."
                )

        # message summary
        if 'loads' in diag_result:
            logger.detailed(
                f"\nSUMMARY: Found {len(diag_result['loads'])} load(s) with negative p_mw. "
                "In load reference system, p_mw should be positive. "
                "If the intention was to model a constant generation, please use an sgen instead."
            )
        if 'gens' in diag_result:
            logger.detailed(
                f"\nSUMMARY: Found {len(diag_result['gens'])} gen(s) with positive p_mw. "
                "In load reference system, p_mw should be negative. "
                "If the intention was to model a load, please use a load instead."
            )
        if 'sgens' in diag_result:
            logger.detailed(
                f"\nSUMMARY: Found {len(diag_result['sgens'])} sgen(s) with positive p_mw. "
                "In load reference system, p_mw should be negative. "
                "If the intention was to model a load, please use a load instead."
            )

    def report_deviation_from_std_type(self):
        # error and success checks
        if "deviation_from_std_type" in self.diag_errors:
            logger.warning("Check for deviation from std_type failed due to the following error:")
            logger.warning(self.diag_errors["deviation_from_std_type"])
            return
        if "deviation_from_std_type" not in self.diag_results:
            logger.info("PASSED: No elements with deviations from std_type found.")
            return
        # message header
        logger.compact("deviation_from_std_type:\n")
        logger.detailed("Checking for deviation from std type...\n")

        # message body
        diag_result = self.diag_results["deviation_from_std_type"]
        element_counter = 0
        for et in diag_result:
            for eid in diag_result[et]:
                element_counter += 1
                values = diag_result[et][eid]
                if values['std_type_in_lib']:
                    logger.warning(
                        f"{et} {eid}: {values['param']} = {values['e_value']}, "
                        f"std_type_value = {values['std_type_value']}"
                    )
                else:
                    logger.warning(f"{et} {eid}: No valid std_type or std_type not in net.std_types")

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} elements with deviations from std_type found.")


    def report_numba_comparison(self):
        # error and success checks
        if "numba_comparison" in self.diag_errors:
            logger.warning("numba_comparison failed due to the following error:")
            logger.warning(self.diag_errors["numba_comparison"])
            return
        if "numba_comparison" not in self.diag_results:
            logger.info("PASSED: No results with deviations between numba = True vs. False found.")
            return
        # message header
        logger.compact("numba_comparison:\n")
        logger.detailed("Checking for deviations between numba = True vs. False...\n")

        # message body
        diag_result = self.diag_results["numba_comparison"]
        element_counter = 0
        for element_type in diag_result:
            for res_type in diag_result[element_type]:
                logger.compact(
                    f"{element_type}.{res_type} absolute deviations:\n{diag_result[element_type][res_type]}"
                )
                for idx in diag_result[element_type][res_type].index:
                    element_counter += 1
                    dev = diag_result[element_type][res_type].loc[idx]
                    logger.detailed(f"{element_type}.{res_type} at index {idx}: absolute deviation = {dev}")

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} results with deviations between numba = True vs. False found.")

    def report_parallel_switches(self):
        # error and success checks
        if "parallel_switches" in self.diag_errors:
            logger.warning("Check for parallel_switches failed due to the following error:")
            logger.warning(self.diag_errors["parallel_switches"])
            return
        if "parallel_switches" not in self.diag_results:
            logger.info("PASSED: No parallel switches found.")
            return
        # message header
        logger.compact("parallel_switches:\n")
        logger.detailed("Checking for parallel switches...\n")

        # message body
        diag_result = self.diag_results["parallel_switches"]
        for switch_tuple in diag_result:
            logger.warning(f"switches {switch_tuple} are parallel.")

        # message summary
        logger.detailed(f"\nSUMMARY: {len(diag_result)} occurrences of parallel switches found.")

    def report_missing_bus_indices(self):
        # error and success checks
        if "missing_bus_indices" in self.diag_errors:
            logger.warning("Check for missing bus indices failed due to the following error:")
            logger.warning(self.diag_errors["missing_bus_indices"])
            return
        if "missing_bus_indices" not in self.diag_results:
            logger.info("PASSED: No missing bus indices found.")
            return
        # message header
        logger.compact("missing_bus_indices:\n")
        logger.detailed("Checking for missing bus indices...\n")

        # message body
        diag_result = self.diag_results["missing_bus_indices"]
        element_counter = 0
        for element_type in diag_result:
            for element in diag_result[element_type]:
                element_counter += 1
                logger.warning(f"{element_type} {element[0]}: {element[1]} ({element[2]}) not in net.bus.index")

        # message summary
        logger.detailed(f"\nSUMMARY: {element_counter} missing bus indices found.")

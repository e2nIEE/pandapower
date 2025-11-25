import copy
import sys
from collections import defaultdict
from functools import partial
import logging
from typing import Any

import numpy as np
import pandas as pd

from pandapower import ADict, select_subnet
from pandapower.toolbox import replace_xward_by_ward, get_connected_elements
from pandapower.create import create_impedance, create_switch
from pandapower.run import runpp
from pandapower.auxiliary import (
    LoadflowNotConverged,
    OPFNotConverged,
    ControllerNotConverged,
    NetCalculationNotConverged,
    pandapowerNet,
)
from pandapower.diagnostic.diagnostic_helpers import (
    DiagnosticFunction,
    T, # TODO: remove T import
    check_boolean,
    check_less_zero,
    check_number,
    check_less_15,
    check_less_20,
    check_pos_int,
    check_greater_zero,
    check_greater_equal_zero,
    check_switch_type,
    check_less_equal_zero,
    check_greater_zero_less_equal_one
)

logger = logging.getLogger(__name__)

expected_exceptions = (LoadflowNotConverged, OPFNotConverged, ControllerNotConverged, NetCalculationNotConverged)

default_argument_values = {
    "overload_scaling_factor": 0.001,
    "capacitance_scaling_factor": 0.01,
    "min_r_ohm": 0.001,
    "min_x_ohm": 0.001,
    "max_r_ohm": 100.,
    "max_x_ohm": 100.,
    "nominal_voltage_tolerance": 0.3,
    "numba_tolerance": 1e-05
}


class InvalidValues(DiagnosticFunction):
    """
    Applies type check functions to find violations of input type restrictions.
    """
    def diagnostic(self, net, **kwargs):
        """
        :param pandapowerNet net: pandapower network
        :param kwargs:
        :return: dict that contains all input type restriction violations
                 grouped by element (keys)
                 Format: {'element': [element_index, 'element_attribute', attribute_value]}
        :rtype: dict[str, Any]
        """
        check_results = {}

        # Contains all element attributes that are necessary to initiate a power flow calculation.
        # There's a tuple with the structure (attribute_name, input type restriction)
        # for each attribute according to pandapower data structure documantation
        # (see also type_checks function)

        important_values = {
            "bus": [("vn_kv", ">0"), ("in_service", "boolean")],
            "line": [
                ("from_bus", "positive_integer"),
                ("to_bus", "positive_integer"),
                ("length_km", ">0"),
                ("r_ohm_per_km", ">=0"),
                ("x_ohm_per_km", ">=0"),
                ("c_nf_per_km", ">=0"),
                ("max_i_ka", ">0"),
                ("df", "0<x<=1"),
                ("in_service", "boolean"),
            ],
            "trafo": [
                ("hv_bus", "positive_integer"),
                ("lv_bus", "positive_integer"),
                ("sn_mva", ">0"),
                ("vn_hv_kv", ">0"),
                ("vn_lv_kv", ">0"),
                ("vkr_percent", ">=0"),
                ("vk_percent", ">0"),
                ("vkr_percent", "<15"),
                ("vk_percent", "<20"),
                ("pfe_kw", ">=0"),
                ("i0_percent", ">=0"),
                ("in_service", "boolean"),
            ],
            "trafo3w": [
                ("hv_bus", "positive_integer"),
                ("mv_bus", "positive_integer"),
                ("lv_bus", "positive_integer"),
                ("sn_hv_mva", ">0"),
                ("sn_mv_mva", ">0"),
                ("sn_lv_mva", ">0"),
                ("vn_hv_kv", ">0"),
                ("vn_mv_kv", ">0"),
                ("vn_lv_kv", ">0"),
                ("vkr_hv_percent", ">=0"),
                ("vkr_mv_percent", ">=0"),
                ("vkr_lv_percent", ">=0"),
                ("vk_hv_percent", ">0"),
                ("vk_mv_percent", ">0"),
                ("vk_lv_percent", ">0"),
                ("vkr_hv_percent", "<15"),
                ("vkr_mv_percent", "<15"),
                ("vkr_lv_percent", "<15"),
                ("vk_hv_percent", "<20"),
                ("vk_mv_percent", "<20"),
                ("vk_lv_percent", "<20"),
                ("pfe_kw", ">=0"),
                ("i0_percent", ">=0"),
                ("in_service", "boolean"),
            ],
            "load": [
                ("bus", "positive_integer"),
                ("p_mw", "number"),
                ("q_mvar", "number"),
                ("scaling", ">=0"),
                ("in_service", "boolean"),
            ],
            "sgen": [
                ("bus", "positive_integer"),
                ("p_mw", "number"),
                ("q_mvar", "number"),
                ("scaling", ">=0"),
                ("in_service", "boolean"),
            ],
            "gen": [
                ("bus", "positive_integer"),
                ("p_mw", "number"),
                ("scaling", ">=0"),
                ("in_service", "boolean")
            ],
            "ext_grid": [
                ("bus", "positive_integer"),
                ("vm_pu", ">0"),
                ("va_degree", "number")
            ],
            "switch": [
                ("bus", "positive_integer"),
                ("element", "positive_integer"),
                ("et", "switch_type"),
                ("closed", "boolean"),
            ],
        }

        # matches a check function to each single input type restriction
        type_checks = {
            ">0": check_greater_zero,
            ">=0": check_greater_equal_zero,
            "<0": check_less_zero,
            "<15": check_less_15,
            "<20": check_less_20,
            "<=0": check_less_equal_zero,
            "boolean": check_boolean,
            "positive_integer": check_pos_int,
            "number": check_number,
            "0<x<=1": check_greater_zero_less_equal_one,
            "switch_type": check_switch_type,
        }

        for key in important_values:
            if len(net[key]) > 0:
                for value in important_values[key]:
                    for i, element in net[key].iterrows():
                        check_result = type_checks[value[1]](element, i, value[0])
                        if check_result is not None:
                            if key not in check_results:
                                check_results[key] = []
                            # converts np.nan to str for easier usage of assert in pytest
                            nan_check = pd.isnull(net[key][value[0]].at[i])
                            if nan_check:
                                check_results[key].append((i, value[0], str(net[key][value[0]].at[i]), value[1]))
                            else:
                                check_results[key].append((i, value[0], net[key][value[0]].at[i], value[1]))

        return check_results if check_results else None

    def report(self, error: Exception | None, results: Any | None):
        # error and success checks
        if error is not None:
            self.out.warning("Check for invalid values failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No invalid values found")
            return
        # message header
        self.out.compact("invalid_values:\n")
        self.out.detailed("Checking for invalid_values...\n")

        # message body
        element_counter = 0
        for element_type in results:
            element_counter += len(results[element_type])
            self.out.warning(f"{element_type}:")
            for inv_value in results[element_type]:
                self.out.compact(
                    f"{element_type} {inv_value[0]}: '{inv_value[1]}' = {inv_value[2]} (restriction: {inv_value[3]})"
                )
                self.out.detailed(
                    f"Invalid value found: '{element_type} {inv_value[0]}' with attribute "
                    f"'{inv_value[1]}' = {inv_value[2]} "
                    f"(data type: {type(inv_value[2])}). Valid input needs to be {inv_value[3]}."
                )

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} invalid values found.")


class NoExtGrid(DiagnosticFunction):
    """
    Checks, if at least one external grid exists.
    """
    def diagnostic(self, net, **kwargs):
        if net.ext_grid.in_service.sum() + (net.gen.slack & net.gen.in_service).sum() == 0:
            return True

    def report(self, error: Exception, results: bool | None):
        # error and success checks
        if error is not None:
            self.out.warning("Check for external grid failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: External grid found.")
            return
        # message header
        self.out.compact("no_external_grid:\n")
        self.out.detailed("Checking if there is at least one external grid...\n")

        # message body
        if results:
            self.out.warning("No ext_grid found. There has to be at least one ext_grid!")


class MultipleVoltageControllingElementsPerBus(DiagnosticFunction):
    """
    Checks, if there are buses with more than one generator and/or more than one external grid.
    """
    def __init__(self):
        super().__init__()
        self.net = None

    def diagnostic(self, net, **kwargs) -> dict[str, list[Any]] | None:
        """
        :returns: dict that contains all buses with multiple generator and
                  all buses with multiple external grids
                  Format: {'mult_ext_grids': [buses], 'buses_with_mult_gens', [buses]}
        :rtype: dict

        """
        self.net = net
        check_results = {}
        buses_with_mult_ext_grids = list(net.ext_grid.groupby("bus").count().query("vm_pu > 1").index)
        if buses_with_mult_ext_grids:
            check_results["buses_with_mult_ext_grids"] = buses_with_mult_ext_grids
        buses_with_gens_and_ext_grids = set(net.ext_grid.bus).intersection(set(net.gen.bus))
        if buses_with_gens_and_ext_grids:
            check_results["buses_with_gens_and_ext_grids"] = list(buses_with_gens_and_ext_grids)

        return check_results if check_results else None

    def report(self, error: Exception | None, results: dict[str, list[Any]] | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning(
                "Check for multiple voltage controlling elements per bus failed due to the following error:"
            )
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No buses with multiple gens and/or ext_grids found.")
            return
        # message header
        self.out.compact("multiple_voltage_controlling_elements_per_bus:\n")
        self.out.detailed("Checking for multiple gens and/or external grids per bus...\n")

        # message body
        element_counter = 0
        for feeder_type in results:
            element_counter += len(results[feeder_type])
            self.out.compact(f"{feeder_type}: {results[feeder_type]}")
            for bus in results[feeder_type]:
                if feeder_type == "buses_with_mult_ext_grids":
                    self.out.detailed(
                        f"External grids {list(self.net.ext_grid[self.net.ext_grid.bus == bus].index)} "
                        f"are connected to bus {bus}. Only one external grid per bus is allowed."
                    )
                elif feeder_type == "buses_with_gens_and_ext_grids":
                    self.out.detailed(
                        f"Generator(s) {list(self.net.gen[self.net.gen.bus == bus].index)} and "
                        f"external grid(s) {list(self.net.ext_grid[self.net.ext_grid.bus == bus].index)} "
                        f"are connected to bus {bus}. "
                        "Only one generator OR one external grid per bus is allowed."
                    )

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} bus(ses) with multiple gens and/or ext_grids found.")


class Overload(DiagnosticFunction):
    """
    Checks, if a loadflow calculation converges. If not, checks, if an overload is the reason for
    that by scaling down the loads, gens and sgens to 0.1%.
    """
    def __init__(self) -> None:
        super().__init__()
        self.overload_scaling_factor: float | None = None

    def diagnostic(self, net: pandapowerNet, **kwargs) -> dict[str, bool] | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
            is replaced by the function kwargs["run"]

        :returns: dict with the results of the overload check
                  Format: {'load_overload': True/False, 'generation_overload', True/False}
        """
        # get function to run power flow
        run = partial(kwargs.pop("run", runpp), **kwargs)
        check_result = {}
        load_scaling = copy.deepcopy(net.load.scaling)
        gen_scaling = copy.deepcopy(net.gen.scaling)
        sgen_scaling = copy.deepcopy(net.sgen.scaling)

        overload_scaling_factor = kwargs.pop(
            "overload_scaling_factor", default_argument_values["overload_scaling_factor"]
        )
        self.overload_scaling_factor = overload_scaling_factor
        try:
            run(net)
        except expected_exceptions:
            check_result["load"] = False
            check_result["generation"] = False
            try:
                net.load.scaling = overload_scaling_factor
                run(net)
                check_result["load"] = True
            except expected_exceptions:
                net.load.scaling = load_scaling
                try:
                    net.gen.scaling = overload_scaling_factor
                    net.sgen.scaling = overload_scaling_factor
                    run(net)
                    check_result["generation"] = True
                except expected_exceptions:
                    net.sgen.scaling = sgen_scaling
                    net.gen.scaling = gen_scaling
                    try:
                        net.load.scaling = overload_scaling_factor
                        net.gen.scaling = overload_scaling_factor
                        net.sgen.scaling = overload_scaling_factor
                        run(net)
                        check_result["generation"] = True
                        check_result["load"] = True
                    except expected_exceptions:
                        self.out.debug("Overload check did not help")
            net.sgen.scaling = sgen_scaling
            net.gen.scaling = gen_scaling
            net.load.scaling = load_scaling
        except Exception as e:
            self.out.error(f"Overload check failed: {str(e)}")
            raise e

        return check_result if check_result else None

    def report(self, error: Exception | None, results: dict[str, bool] | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for overload failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: Power flow converges. No overload found.")
            return
        # message header
        self.out.compact("overload:\n")
        self.out.detailed("Checking for overload...\n")

        # message body
        if self.overload_scaling_factor is not None:
            overload_scaling_factor = self.overload_scaling_factor
        else:
            raise RuntimeError('diagnostic was not executed before calling results?')
        osf_percent = f"{overload_scaling_factor * 100} percent."
        if not results["load"] and not results["generation"]:
            self.out.warning(
                "Overload check failed: Power flow still does not converge with load and generation scaled down to "
                f"{osf_percent}"
            )
        elif results["load"] and results["generation"]:
            self.out.warning(
                f"overload found: Power flow converges with load and generation scaled down to {osf_percent}"
            )
        else:
            if results["load"]:
                self.out.warning(f"overload found: Power flow converges with load scaled down to {osf_percent}")
            else:
                self.out.warning(f"overload found: Power flow converges with generation scaled down to {osf_percent}")


class WrongLineCapacitance(DiagnosticFunction):
    """
    Checks, if a loadflow calculation converges. If not, checks, if line capacitance is too high, by scaling it to 1%.
    """
    def __init__(self) -> None:
        super().__init__()
        self.capacitance_scaling_factor: float | None = None

    def diagnostic(self, net: pandapowerNet, **kwargs) -> bool | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
            is replaced by the function kwargs["run"]

        :returns: dict with the results of the overload check
                  Format: {'load_overload': True/False, 'generation_overload', True/False}
        """
        # get function to run power flow
        run = partial(kwargs.pop("run", runpp), **kwargs)
        check_result = None
        line_capacitance = copy.copy(net.line.c_nf_per_km)

        capacitance_scaling_factor = kwargs.pop(
            "capacitance_scaling_factor", default_argument_values["capacitance_scaling_factor"]
        )

        self.capacitance_scaling_factor = capacitance_scaling_factor
        try:
            run(net)
        except expected_exceptions:
            check_result = False
            try:
                net.line.c_nf_per_km *= capacitance_scaling_factor
                run(net)
                check_result = True
            except expected_exceptions:
                self.out.debug("Line capacitance check failed.")

        except Exception as e:
            self.out.error(f"Line capacitance check failed: {str(e)}")
            raise e

        # teardown
        net.line.c_nf_per_km = line_capacitance

        return check_result

    def report(self, error: Exception | None, results: bool | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for convergence error failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: Power flow converges. No line capacitance problems found.")
            return

        # message header
        self.out.compact("line problems:\n")
        self.out.detailed("Checking for too high line capacitance...\n")

        # message body
        if self.capacitance_scaling_factor is not None:
            capacitance_scaling_factor = self.capacitance_scaling_factor
        else:
            raise RuntimeError('diagnostic was not executed before calling results?')

        osf_percent = f"{capacitance_scaling_factor * 100} percent."

        if results:
            self.out.warning(
                f"Too high capacitance found: Power flow converges with line.c_nf_per_km scaled down to {osf_percent}")
        else:
            self.out.warning(
                f"Too high capacitance tested: Power flow did not converge with line.c_nf_per_km scaled down to {osf_percent}")


class SubNetProblemTest(DiagnosticFunction):
    """
    Checks, if subnets are converging. This is done using the zone attribute.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net: pandapowerNet | None = None
        self.zones: list[str] = []

    def diagnostic(self, net: pandapowerNet, **kwargs) -> dict[str, bool] | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
            is replaced by the function kwargs["run"]

        :returns: dict with the results of the overload check
                  Format: {'load_overload': True/False, 'generation_overload', True/False}
        """
        # get function to run power flow
        run = partial(kwargs.pop("run", runpp), **kwargs)
        self.net = copy.deepcopy(net)

        self.zones = self.net.bus.zone.unique()
        check_result = {}

        if len(self.zones) < 1:
            return None

        for zone, buses in self.net.bus.groupby(net.bus.zone):
            subnet = select_subnet(self.net, buses=buses, include_switch_buses=True, keep_everything_else=True)
            try:
                run(subnet)
                check_result[zone] = True
            except expected_exceptions:
                check_result[zone] = False
                self.out.debug(f"Calculation on zone: {zone} failed.")
            except Exception as e:
                self.out.error(f"Calculation on subnets failed: {str(e)}")
                raise e

        return check_result if not all(check_result.values()) else None

    def report(self, error: Exception | None, results: dict[str, bool] | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for subnets error failed due to the following error:")
            self.out.warning(error)
            return

        if results is None:
            if len(self.zones) > 1:
                self.out.info("PASSED: Power flow converges. No problems in zones found.")
            else:
                self.out.info("SKIPPED: Did not check zones, no zones are defined.")
            return

        # message header
        self.out.compact("Zone problems:\n")
        self.out.detailed("Checking problems in subnets derived from net.bus.zone...\n")

        # message body
        self.out.warning(f"Found loadflow problems in the following zones: {results.keys()}")


class OptimisticPowerflow(DiagnosticFunction):
    """
    Checks, if powerflow converges, if a set of 'optimistic' tests is performed.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net: pandapowerNet | None = None
        self.zones: list[str] = []

    def diagnostic(self, net: pandapowerNet, **kwargs) -> dict[str, bool] | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
            is replaced by the function kwargs["run"]

        :returns: dict with the results of the overload check
                  Format: {'load_overload': True/False, 'generation_overload', True/False}
        """
        # get function to run power flow
        run = partial(kwargs.pop("run", runpp), **kwargs)
        self.net = copy.deepcopy(net)

        check_result = {}

        try:
            self.net.line.c_nf_per_km = 0

            run(self.net)
            return None
        except expected_exceptions:
            check_result["c_nf_per_km_zero"] = False
            self.out.debug(f"Line susceptance = 0, did not solve the problem.")
            try:
                if 'trafo' in self.net:
                    self.net.trafo.pfe_kw = 0
                if 'trafo3w' in self.net:
                    self.net.trafo3w.pfe_kw = 0

                run(self.net)
                check_result["trafo_pfe_kw_zero"] = True
            except expected_exceptions:
                check_result["trafo_pfe_kw_zero"] = False
                self.out.debug(f"Line susceptance = 0 and iron losses = 0, did not solve the problem.")
                try:
                    if 'load' in self.net:
                        self.net.load.p_mw = 0
                        self.net.load.q_mvar = 0
                    if 'sgen' in self.net:
                        self.net.sgen.p_mw = 0
                        self.net.sgen.q_mvar = 0

                    run(self.net)
                    check_result["load_sgen_zero"] = True
                except expected_exceptions:
                    check_result["load_sgen_zero"] = False
                    self.out.debug(f"Line susceptance = 0, iron losses = 0 and no injection / consumption, did not solve the problem.")

        except Exception as e:
            self.out.error(f"Optimistic calculation on failed: {str(e)}")
            raise e

        return check_result if check_result else None

    def report(self, error: Exception | None, results: dict[str, bool] | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for subnets error failed due to the following error:")
            self.out.warning(error)
            return

        if results is None:
            self.out.info("PASSED: Power flow converges. No problems in zones found.")
            return

        # message header
        self.out.compact("Optimistic_powerflow:\n")
        self.out.detailed(
            "Checking if optimistic powerflow, line.c_nf_per_km = 0, trafo(3w).pfe_kw = 0 and load, sgen.p_mw/q_mvar = 0\n")

        # message body
        if "load_sgen_zero" in results:
            if results['load_sgen_zero']:
                self.out.warning(
                    f"Powerflow problems solved, by setting line.c_nf_per_km = 0, trafo(3w).pfe_kw = 0 and load, sgen.p_mw/q_mvar = 0.")
            else:
                self.out.warning("Optimistic powerflow failed...")
        elif 'trafo_pfe_kw_zero' in results:
            self.out.warning(
                f"Powerflow problems solved, by setting line.c_nf_per_km = 0 and trafo(3w).pfe_kw = 0.")
        elif 'c_nf_per_km_zero' in results:
            self.out.warning(
                f"Powerflow problems solved, by setting line.c_nf_per_km = 0.")


class SlackGenPlacement(DiagnosticFunction):
    """
    Checks, if powerflow converges/losses are minimized, if a different gen is used as slack
    """
    def __init__(self) -> None:
        super().__init__()
        self.net: pandapowerNet | None = None

    def diagnostic(self, net: pandapowerNet, **kwargs) -> dict[str, float] | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
            is replaced by the function kwargs["run"]

        :returns: dict with the results of the overload check
                  Format: {'load_overload': True/False, 'generation_overload', True/False}
        """
        # get function to run power flow
        run = partial(kwargs.pop("run", runpp), **kwargs)

        check_result = {}

        if 'gen' not in net:
            return None

        orig_gen = copy.copy(net.gen)

        def _calculate_losses(net):
            idx = net.gen.loc[net.gen.slack==True].index
            res = np.linalg.norm(net.res_gen.loc[idx].p_mw + net.res_gen.loc[idx].q_mvar * 1j)
            res += np.linalg.norm(net.res_ext_grid.p_mw + net.res_ext_grid.q_mvar * 1j)
            return res

        try:
            run(net)

            res = _calculate_losses(net)
            check_result['base_case'] = res
        except expected_exceptions:
            self.out.debug(f"Base case did not converge, trying different combinations")
            check_result['base_case'] = sys.float_info.max
        except Exception as e:
            self.out.error(f"Slack gen placement calculation failed: {str(e)}")
            raise e

        # disable all slack gen
        net.gen.slack = False

        # try all gen combinations
        for idx, gen in net.gen.iterrows():
            net.gen.loc[idx, 'slack'] = True

            try:
                run(net)

                res = _calculate_losses(net)
                check_result[idx] = res
            except expected_exceptions:
                self.out.debug(f"Gen[{idx}]=Slack did not converge, trying different one")
                check_result[idx] = sys.float_info.max
            except Exception as e:
                self.out.error(f"Slack gen placement calculation failed: {str(e)}")
                raise e

            net.gen.loc[idx, 'slack'] = False

        # teardown
        net.gen = orig_gen

        bc = check_result.pop('base_case')
        if all(map(lambda x: bc < x, check_result.values())):
            return None

        return check_result

    def report(self, error: Exception | None, results: dict[str, float] | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for slack gen placement failed due to the following error:")
            self.out.warning(error)
            return

        if results is None:
            self.out.info("PASSED: Power flow converges, no modifications needed.")
            return

        # message header
        self.out.compact("slack_gen_placement:\n")
        self.out.detailed("Checking if another generator as slack reduces losses / enables powerflow\n")

        # message body
        t = min(results.items())
        self.out.warning(f'Gen idx: {t[0]} as slack, reduces apparent power to: {t[1]}')


class WrongSwitchConfiguration(DiagnosticFunction):
    """
    Checks, if a loadflow calculation converges. If not, checks, if the switch configuration is
    the reason for that by closing all switches
    """
    def diagnostic(self, net: pandapowerNet, **kwargs) -> bool | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
                       is replaced by the function kwargs["run"]
        """
        run = partial(kwargs.pop("run", runpp), **kwargs)
        switch_configuration = copy.deepcopy(net.switch.closed)
        try:
            run(net)
        except expected_exceptions:
            try:
                net.switch.closed = True
                run(net)
                net.switch.closed = switch_configuration
                return True
            except expected_exceptions:
                net.switch.closed = switch_configuration
                return False
        except Exception as e:
            self.out.error(f"Switch check failed: {str(e)}")
        return None

    def report(self, error: Exception | None, results: bool | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for wrong switch configuration failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: Power flow converges. Switch configuration seems ok.")
            return
        # message header
        self.out.compact("wrong_switch_configuration:\n")
        self.out.detailed("Checking switch configuration...\n")

        # message body
        if results:
            self.out.warning("Possibly wrong switch configuration found: power flow converges with all switches closed.")
        else:
            self.out.warning("Power flow still does not converge with all switches closed.")


class MissingBusIndices(DiagnosticFunction):
    """
    Checks for missing bus indices.
    """
    def diagnostic(self, net: pandapowerNet, **kwargs) -> dict | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: unused

        :returns: List of tuples each containing missing bus indices.
                  Format: [(element_index, bus_name (e.g. "from_bus",  bus_index), ...]
        """
        check_results = {}
        bus_indices = set(net.bus.index)
        element_bus_names = {
            "ext_grid": ["bus"],
            "load": ["bus"],
            "gen": ["bus"],
            "sgen": ["bus"],
            "trafo": ["lv_bus", "hv_bus"],
            "trafo3w": ["lv_bus", "mv_bus", "hv_bus"],
            "switch": ["bus", "element"],
            "line": ["from_bus", "to_bus"],
        }
        for element in element_bus_names:
            element_check = []
            for i, row in net[element].iterrows():
                for bus_name in element_bus_names[element]:
                    if row[bus_name] not in bus_indices:
                        if not ((element == "switch") and (bus_name == "element") and (row.et in ["l", "t", "t3"])):
                            element_check.append((i, bus_name, row[bus_name]))
            if element_check:
                check_results[element] = element_check

        return check_results if check_results else None

    def report(self, error: Exception | None, results: dict | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for missing bus indices failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No missing bus indices found.")
            return
        # message header
        self.out.compact("missing_bus_indices:\n")
        self.out.detailed("Checking for missing bus indices...\n")

        # message body
        element_counter = 0
        for element_type in results:
            for element in results[element_type]:
                element_counter += 1
                self.out.warning(f"{element_type} {element[0]}: {element[1]} ({element[2]}) not in net.bus.index")

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} missing bus indices found.")


class DifferentVoltageLevelsConnected(DiagnosticFunction):
    """
    Checks if there are lines or switches that connect different voltage levels.
    """
    def __init__(self):
        super().__init__()
        self.net = None

    def diagnostic(self, net: pandapowerNet, **kwargs) -> dict | None:
        """
        :param pandapowerNet net: pandapower network
        :param kwargs:

        :returns: dict that contains all lines and switches that connect different voltage levels.
                  Format: {'lines': lines, 'switches': switches}
        """
        self.net = net
        check_results = {}
        inconsistent_lines = []
        for i, line in net.line.iterrows():
            buses = net.bus.loc[[line.from_bus, line.to_bus]]
            if buses.vn_kv.iloc[0] != buses.vn_kv.iloc[1]:
                inconsistent_lines.append(i)

        inconsistent_switches = []
        for i, switch in net.switch[net.switch.et == "b"].iterrows():
            buses = net.bus.loc[[switch.bus, switch.element]]
            if buses.vn_kv.iloc[0] != buses.vn_kv.iloc[1]:
                inconsistent_switches.append(i)

        if inconsistent_lines:
            check_results["lines"] = inconsistent_lines
        if inconsistent_switches:
            check_results["switches"] = inconsistent_switches

        return check_results if check_results else None

    def report(self, error: Exception | None, results: dict | None) -> None:
        from pandapower.toolbox import get_connected_buses_at_element

        # error and success checks
        if error is not None:
            self.out.warning("Check for connection of different voltage levels failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No connection of different voltage levels found")
            return
        # message header
        self.out.compact("different_voltage_levels_connected:\n")
        self.out.detailed("Checking for connections of different voltage levels...\n")

        # message body
        element_counter = 0
        for key in results:
            element_counter += len(results[key])
            element_type = ""
            if key == "lines":
                element_type = "line"
            elif key == "switches":
                element_type = "switch"
            self.out.compact(f"{key}:")
            for element in results[key]:
                buses = list(get_connected_buses_at_element(self.net, element, key[0]))
                self.out.compact(f"{element_type} {element}: buses {buses}")
                self.out.detailed(
                    f"{element_type} {element} connects bus {buses[0]}: {self.net.bus.name.at[buses[0]]} "
                    f"(vn_kv = {self.net.bus.vn_kv.at[buses[0]]}) and "
                    f"bus {buses[1]}: {self.net.bus.name.at[buses[1]]} "
                    f"(vn_kv = {self.net.bus.vn_kv.at[buses[1]]})"
                )
        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} element(s) that connect different voltage levels found.")


class ImplausibleImpedanceValues(DiagnosticFunction):
    """
    Checks, if there are lines, xwards or impedances with an impedance value close to zero.
    """
    def __init__(self):
        super().__init__()
        self.params = {}

    def diagnostic(self, net: pandapowerNet, **kwargs) -> T | None: # TODO: T typing
        """
        :param pandapowerNet net: pandapower network
        :param kwargs: Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
                       is replaced by the function kwargs["run"]

        :returns: list that contains the indices of all lines with an impedance value of zero.
        """
        # get function to run power flow
        run = partial(kwargs.pop("run", runpp), **kwargs)
        check_results: list[dict] = []
        implausible_elements = {}

        max_r_ohm = kwargs.pop("max_r_ohm", default_argument_values.get("max_r_ohm", None))
        min_r_ohm = kwargs.pop("min_r_ohm", default_argument_values.get("min_r_ohm", None))
        max_x_ohm = kwargs.pop("max_x_ohm", default_argument_values.get("max_x_ohm", None))
        min_x_ohm = kwargs.pop("min_x_ohm", default_argument_values.get("min_x_ohm", None))
        zb_f_ohm = np.square(net.bus.loc[net.impedance.from_bus.values, "vn_kv"].values) / net.impedance.sn_mva
        zb_t_ohm = np.square(net.bus.loc[net.impedance.to_bus.values, "vn_kv"].values) / net.impedance.sn_mva

        self.params['max_r_ohm'] = max_r_ohm
        self.params['min_r_ohm'] = min_r_ohm
        self.params['max_x_ohm'] = max_x_ohm
        self.params['min_x_ohm'] = min_x_ohm
        self.params['zb_f_ohm'] = zb_f_ohm
        self.params['zb_t_ohm'] = zb_t_ohm
        for n, v in self.params.items():
            if v is None:
                raise RuntimeError(f"missing default argument value: '{n}'")

        line = net.line.loc[
            (
                (net.line.r_ohm_per_km * net.line.length_km >= max_r_ohm)
                | (net.line.r_ohm_per_km * net.line.length_km <= min_r_ohm)
                | (net.line.x_ohm_per_km * net.line.length_km >= max_x_ohm)
                | (net.line.x_ohm_per_km * net.line.length_km <= min_x_ohm)
            )
            & net.line.in_service
        ].index

        xward = net.xward.loc[(
                (net.xward.r_ohm.abs() >= max_r_ohm)
                | (net.xward.r_ohm.abs() <= min_r_ohm)
                | (net.xward.x_ohm.abs() >= max_x_ohm)
                | (net.xward.x_ohm.abs() <= min_x_ohm)
            ) & net.xward.in_service
        ].index

        impedance = net.impedance.loc[(
            (np.abs(net.impedance.rft_pu) >= max_r_ohm / zb_f_ohm)
            | (np.abs(net.impedance.rft_pu) <= min_r_ohm / zb_f_ohm)
            | (np.abs(net.impedance.xft_pu) >= max_x_ohm / zb_f_ohm)
            | (np.abs(net.impedance.xft_pu) <= min_x_ohm / zb_f_ohm)
            | (np.abs(net.impedance.rtf_pu) >= max_r_ohm / zb_t_ohm)
            | (np.abs(net.impedance.rtf_pu) <= min_r_ohm / zb_t_ohm)
            | (np.abs(net.impedance.xtf_pu) >= max_x_ohm / zb_t_ohm)
            | (np.abs(net.impedance.xtf_pu) <= min_x_ohm / zb_t_ohm)
        ) & net.impedance.in_service].index

        trafo = net.trafo.loc[((
            (net.trafo.vk_percent / 100 * np.square(net.trafo.vn_hv_kv) / net.trafo.sn_mva >= max_x_ohm)
            | (net.trafo.vk_percent / 100 * np.square(net.trafo.vn_lv_kv) / net.trafo.sn_mva <= min_x_ohm)
        ) & net.trafo.in_service)].index

        trafo3w = net.trafo3w.loc[((
            (net.trafo3w.vk_hv_percent / 100 * np.square(net.trafo3w.vn_hv_kv) / net.trafo3w.sn_hv_mva >= max_x_ohm)
            | (net.trafo3w.vk_hv_percent / 100 * np.square(net.trafo3w.vn_mv_kv) / net.trafo3w.sn_hv_mva <= min_x_ohm)
            | (net.trafo3w.vk_mv_percent / 100 * np.square(net.trafo3w.vn_mv_kv) / net.trafo3w.sn_mv_mva >= max_x_ohm)
            | (net.trafo3w.vk_mv_percent / 100 * np.square(net.trafo3w.vn_lv_kv) / net.trafo3w.sn_mv_mva <= min_x_ohm)
            | (net.trafo3w.vk_lv_percent / 100 * np.square(net.trafo3w.vn_hv_kv) / net.trafo3w.sn_lv_mva >= max_x_ohm)
            | (net.trafo3w.vk_lv_percent / 100 * np.square(net.trafo3w.vn_lv_kv) / net.trafo3w.sn_lv_mva <= min_x_ohm)
        ) & net.trafo3w.in_service)].index

        vsc = net.vsc.loc[
            ((net.vsc.r_ohm <= min_r_ohm) | (net.vsc.x_ohm <= min_x_ohm) | (net.vsc.r_dc_ohm <= min_r_ohm))
            & net.vsc.in_service
        ].index

        line_dc = net.line_dc.loc[(
            (net.line_dc.r_ohm_per_km * net.line_dc.length_km) <= min_r_ohm
        ) & net.line_dc.in_service].index

        if len(line) > 0:
            implausible_elements["line"] = list(line)
        if len(xward) > 0:
            implausible_elements["xward"] = list(xward)
        if len(impedance) > 0:
            implausible_elements["impedance"] = list(impedance)
        if len(trafo) > 0:
            implausible_elements["trafo"] = list(trafo)
        if len(trafo3w) > 0:
            implausible_elements["trafo3w"] = list(trafo3w)
        if len(vsc) > 0:
            implausible_elements["vsc"] = list(vsc)
        if len(line_dc) > 0:
            implausible_elements["line_dc"] = list(line_dc)

        check_results.append(implausible_elements)
        # checks if loadflow converges when implausible lines or impedances are replaced by switches
        if ("line" in implausible_elements) or ("impedance" in implausible_elements) or ("xward" in implausible_elements):
            switch_copy = copy.deepcopy(net.switch)
            line_copy = copy.deepcopy(net.line)
            impedance_copy = copy.deepcopy(net.impedance)
            vsc_copy = copy.deepcopy(net.vsc)
            line_dc_copy = copy.deepcopy(net.line_dc)
            ward_copy = copy.deepcopy(net.ward)
            xward_copy = copy.deepcopy(net.xward)
            trafo_copy = copy.deepcopy(net.trafo)
            trafo3w_copy = copy.deepcopy(net.trafo3w)
            try:
                run(net)
            except (*expected_exceptions, FloatingPointError):
                try:
                    for key in implausible_elements:
                        implausible_idx = implausible_elements[key]
                        if key == "vsc":
                            net.vsc.x_ohm = np.fmax(net.vsc.x_ohm, 0.5)
                            net.vsc.r_dc_ohm = np.fmax(net.vsc.r_dc_ohm, 0.5)
                        if key == "line_dc":
                            net.line_dc.length_km = np.fmax(net.line_dc.length_km, 0.5)
                            net.line_dc.r_dc_ohm = np.fmax(net.line_dc.r_dc_ohm, 0.5)
                        net[key].loc[implausible_idx, "in_service"] = False
                        if key == "xward":
                            replace_xward_by_ward(net, implausible_idx)
                        elif key == "trafo":
                            for idx in implausible_idx:
                                create_impedance(
                                    net, net.trafo.at[idx, "hv_bus"], net.trafo.at[idx, "lv_bus"], 0, 0.01, 100
                                )
                        elif key == "trafo3w":
                            for idx in implausible_idx:
                                create_impedance(
                                    net, net.trafo3w.at[idx, "hv_bus"], net.trafo3w.at[idx, "mv_bus"], 0, 0.01, 100
                                )
                                create_impedance(
                                    net, net.trafo3w.at[idx, "mv_bus"], net.trafo3w.at[idx, "lv_bus"], 0, 0.01, 100
                                )
                                create_impedance(
                                    net, net.trafo3w.at[idx, "hv_bus"], net.trafo3w.at[idx, "lv_bus"], 0, 0.01, 100
                                )
                        else:
                            for idx in implausible_idx:
                                create_switch(net, net[key].from_bus.at[idx], net[key].to_bus.at[idx], et="b")
                    run(net)
                    switch_replacement = True
                except expected_exceptions:
                    switch_replacement = False
                check_results.append({"loadflow_converges_with_switch_replacement": switch_replacement})
            except Exception as e:
                self.out.error(f"Impedance values check failed: {str(e)}")
            net.switch = switch_copy
            net.line = line_copy
            net.impedance = impedance_copy
            net.vsc = vsc_copy
            net.line_dc = line_dc_copy
            net.ward = ward_copy
            net.xward = xward_copy
            net.trafo = trafo_copy
            net.trafo3w = trafo3w_copy

        return check_results if implausible_elements else None

    def report(self, error: Exception | None, results: T | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for elements with impedance values close to zero failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No elements with impedance values close to zero found...")
            return

        # message header
        self.out.compact("impedance_values_close_to_zero:\n")
        self.out.detailed("Checking for impedance values close to zero...\n")

        # message body
        element_counter = 0
        for key, value in results[0].items():
            element_counter += len(value)
            min_r_type = ""
            min_x_type = ""
            if key in ("line", "line_dc", "xward", "vsc"):
                min_r_type = self.params["min_r_ohm"]
                min_x_type = self.params["min_x_ohm"]
            elif key == "impedance":
                min_r_type = self.params["min_r_ohm"] / self.params["zb_f_ohm"]
                min_x_type = self.params["min_x_ohm"] / self.params["zb_f_ohm"]
            for element in value:
                self.out.warning(f"{key} {element}: r_ <= {min_r_type} or x_ <= {min_x_type}") # TODO r_ and x_ should output violating column

        if len(results) > 1:
            switch_replacement = results[1]
            if switch_replacement["loadflow_converges_with_switch_replacement"]:
                self.out.warning(
                    "Switch replacement successful: Power flow converges after "
                    "replacing implausible elements with switches."
                )
            else:
                self.out.warning(
                    "Power flow still does not converge after replacing implausible elements with switches."
                )

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} element(s) with impedance values close to zero found.")


class NominalVoltagesMismatch(DiagnosticFunction):
    """
    Checks, if there are components whose nominal voltages differ from the nominal voltages of the
    buses they're connected to. At the moment, only trafos and trafo3w are checked.
    Also checks for trafos with swapped hv and lv connectors.
    """
    def __init__(self) -> None:
        super().__init__()
        self.nom_voltage_tolerance: float | None = None
        self.net: pandapowerNet | None = None

    def diagnostic(self, net: pandapowerNet, **kwargs) -> T | None:
        """
        :param pandapowerNet net: pandapower network

        :returns: dict that contains all components whose nominal voltages
                  differ from the nominal voltages of the buses they're
                  connected to.
                  Format:
                  {trafo': {'hv_bus' : trafos_indices,
                            'lv_bus' : trafo_indices,
                            'hv_lv_swapped' : trafo_indices},
                  trafo3w': {'hv_bus' : trafos3w_indices,
                             'mv_bus' : trafos3w_indices
                             'lv_bus' : trafo3w_indices,
                             'connectors_swapped_3w' : trafo3w_indices}}
        """
        self.net = net
        results = {}
        trafo_results = {}
        trafo3w_results = {}

        hv_bus = []
        lv_bus = []
        hv_lv_swapped = []

        hv_bus_3w = []
        mv_bus_3w = []
        lv_bus_3w = []
        connectors_swapped_3w = []

        nom_voltage_tolerance = kwargs.pop("nom_voltage_tolerance", default_argument_values.get("nom_voltage_tolerance", None))
        if nom_voltage_tolerance is None:
            raise RuntimeError("Missing default argument value for 'nom_voltage_tolerance'")
        self.nom_voltage_tolerance = nom_voltage_tolerance

        for i, trafo in net.trafo.iterrows():
            hv_bus_violation = False
            lv_bus_violation = False
            connectors_swapped = False
            hv_bus_vn_kv = net.bus.vn_kv.at[trafo.hv_bus]
            lv_bus_vn_kv = net.bus.vn_kv.at[trafo.lv_bus]

            if abs(1 - (trafo.vn_hv_kv / hv_bus_vn_kv)) > nom_voltage_tolerance:
                hv_bus_violation = True
            if abs(1 - (trafo.vn_lv_kv / lv_bus_vn_kv)) > nom_voltage_tolerance:
                lv_bus_violation = True
            if hv_bus_violation and lv_bus_violation:
                trafo_voltages = np.array(([trafo.vn_hv_kv, trafo.vn_lv_kv]))
                bus_voltages = np.array([hv_bus_vn_kv, lv_bus_vn_kv])
                trafo_voltages.sort()
                bus_voltages.sort()
                if all((abs(trafo_voltages - bus_voltages) / bus_voltages) < (nom_voltage_tolerance)):
                    connectors_swapped = True

            if connectors_swapped:
                hv_lv_swapped.append(i)
            else:
                if hv_bus_violation:
                    hv_bus.append(i)
                if lv_bus_violation:
                    lv_bus.append(i)

        if hv_bus:
            trafo_results["hv_bus"] = hv_bus
        if lv_bus:
            trafo_results["lv_bus"] = lv_bus
        if hv_lv_swapped:
            trafo_results["hv_lv_swapped"] = hv_lv_swapped
        if trafo_results:
            results["trafo"] = trafo_results

        for i, trafo3w in net.trafo3w.iterrows():
            hv_bus_violation = False
            mv_bus_violation = False
            lv_bus_violation = False
            connectors_swapped = False
            hv_bus_vn_kv = net.bus.vn_kv.at[trafo3w.hv_bus]
            mv_bus_vn_kv = net.bus.vn_kv.at[trafo3w.mv_bus]
            lv_bus_vn_kv = net.bus.vn_kv.at[trafo3w.lv_bus]

            if abs(1 - (trafo3w.vn_hv_kv / hv_bus_vn_kv)) > nom_voltage_tolerance:
                hv_bus_violation = True
            if abs(1 - (trafo3w.vn_mv_kv / mv_bus_vn_kv)) > nom_voltage_tolerance:
                mv_bus_violation = True
            if abs(1 - (trafo3w.vn_lv_kv / lv_bus_vn_kv)) > nom_voltage_tolerance:
                lv_bus_violation = True
            if hv_bus_violation and mv_bus_violation and lv_bus_violation:
                trafo_voltages = np.array(([trafo3w.vn_hv_kv, trafo3w.vn_mv_kv, trafo3w.vn_lv_kv]))
                bus_voltages = np.array([hv_bus_vn_kv, mv_bus_vn_kv, lv_bus_vn_kv])
                trafo_voltages.sort()
                bus_voltages.sort()
                if all((abs(trafo_voltages - bus_voltages) / bus_voltages) < (nom_voltage_tolerance)):
                    connectors_swapped = True

            if connectors_swapped:
                connectors_swapped_3w.append(i)
            else:
                if hv_bus_violation:
                    hv_bus_3w.append(i)
                if mv_bus_violation:
                    mv_bus_3w.append(i)
                if lv_bus_violation:
                    lv_bus_3w.append(i)

        if hv_bus_3w:
            trafo3w_results["hv_bus"] = hv_bus_3w
        if mv_bus_3w:
            trafo3w_results["mv_bus"] = mv_bus_3w
        if lv_bus_3w:
            trafo3w_results["lv_bus"] = lv_bus_3w
        if connectors_swapped_3w:
            trafo3w_results["connectors_swapped_3w"] = connectors_swapped_3w
        if trafo3w_results:
            results["trafo3w"] = trafo3w_results

        return results if len(results) > 0 else None

    def report(self, error: Exception | None, results: T | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for components with deviating nominal voltages failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No components with deviating nominal voltages found")
            return
        # message header
        self.out.compact("nominal_voltages_dont_match:\n")
        self.out.detailed("Checking for components with deviating nominal voltages...\n")

        # message body
        nom_voltage_tolerance = self.nom_voltage_tolerance
        if nom_voltage_tolerance is None or self.net is None:
            raise RuntimeError('NominalVoltagesMismatch report called before diagnostic')
        element_counter = 0
        for element in results:
            self.out.compact(f"{element}:")
            for key in results[element]:
                element_counter += len(results[element][key])
                if element == "trafo":
                    self.out.compact(f"{key}: {results[element][key]}")
                    if key == "hv_lv_swapped":
                        self.out.detailed(
                            f"Trafo(s) {results[element][key]}: hv and lv connectors seem to be swapped"
                        )
                    elif key == "hv_bus":
                        for trafo in results[element][key]:
                            self.out.detailed(
                                f"Trafo {trafo}: Nominal voltage on hv_side"
                                f"({self.net.trafo.vn_hv_kv.at[trafo]} kV) and voltage_level of hv_bus "
                                f"(bus {self.net.trafo.hv_bus.at[trafo]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo.hv_bus.at[trafo]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                    elif key == "lv_bus":
                        for trafo in results[element][key]:
                            self.out.detailed(
                                f"Trafo {trafo}: Nominal voltage on lv_side "
                                f"({self.net.trafo.vn_lv_kv.at[trafo]} kV) and voltage_level of lv_bus "
                                f"(bus {self.net.trafo.lv_bus.at[trafo]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo.lv_bus.at[trafo]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                if element == "trafo3w":
                    self.out.compact(f"{key}: {results[element][key]}")
                    if key == "connectors_swapped_3w":
                        self.out.detailed(f"Trafo3w {results[element][key]}: connectors seem to be swapped")
                    elif key == "hv_bus":
                        for trafo3w in results[element][key]:
                            self.out.detailed(
                                f"Trafo3w {trafo3w}: Nominal voltage on hv_side "
                                f"({self.net.trafo3w.vn_hv_kv.at[trafo3w]} kV) and voltage_level of hv_bus "
                                f"(bus {self.net.trafo3w.hv_bus.at[trafo3w]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo3w.hv_bus.at[trafo3w]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                    elif key == "mv_bus":
                        for trafo3w in results[element][key]:
                            self.out.detailed(
                                f"Trafo3w {trafo3w}: Nominal voltage on mv_side "
                                f"({self.net.trafo3w.vn_mv_kv.at[trafo3w]} kV) and voltage_level of mv_bus "
                                f"(bus {self.net.trafo3w.mv_bus.at[trafo3w]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo3w.mv_bus.at[trafo3w]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )
                    elif key == "lv_bus":
                        for trafo3w in results[element][key]:
                            self.out.detailed(
                                f"Trafo3w {trafo3w}: Nominal voltage on lv_side "
                                f"({self.net.trafo3w.vn_lv_kv.at[trafo3w]} kV) and voltage_level of lv_bus "
                                f"(bus {self.net.trafo3w.lv_bus.at[trafo3w]} with voltage_level "
                                f"{self.net.bus.vn_kv.at[self.net.trafo3w.lv_bus.at[trafo3w]]} kV) "
                                f"deviate more than +/- {nom_voltage_tolerance * 100} percent."
                            )

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} component(s) with deviating nominal voltages found")


class DisconnectedElements(DiagnosticFunction):
    """
    Checks, if there are network sections without a connection to an ext_grid. Returns all network
    elements in these sections, that are in service. Elements belonging to the same disconnected
    networks section are grouped in lists (e.g. disconnected lines: [[1, 2, 3], [4, 5]]
    means, that lines 1, 2 and 3 are in one disconncted section but are connected to each other.
    The same stands for lines 4, 5.)
    """
    def diagnostic(self, net: ADict, **kwargs) -> T | None: # TODO: T typing
        """

         INPUT:
            **net** (pandapowerNet)         - pandapower network

         OUTPUT:
            **disc_elements** (dict)        - list that contains all network elements, without a
                                              connection to an ext_grid.

                                              format: {'disconnected buses'   : bus_indices,
                                                       'disconnected switches' : switch_indices,
                                                       'disconnected lines'    : line_indices,
                                                       'disconnected trafos'   : trafo_indices
                                                       'disconnected loads'    : load_indices,
                                                       'disconnected gens'     : gen_indices,
                                                       'disconnected sgens'    : sgen_indices}
        """
        from pandapower.topology import create_nxgraph, connected_components

        mg = create_nxgraph(net)
        sections = connected_components(mg)
        disc_elements = []

        for section in sections:
            section_dict = {}

            if not section & set(net.ext_grid.bus[net.ext_grid.in_service]).union(
                net.gen.bus[net.gen.slack & net.gen.in_service]
            ) and any(net.bus.in_service.loc[list(section)]):
                section_buses = list(net.bus[net.bus.index.isin(section) & net.bus.in_service].index)
                section_switches = list(net.switch[net.switch.bus.isin(section_buses)].index)
                section_lines = list(
                    get_connected_elements(net, "line", section_buses, respect_switches=True, respect_in_service=True)
                )
                section_trafos = list(
                    get_connected_elements(net, "trafo", section_buses, respect_switches=True, respect_in_service=True)
                )

                section_trafos3w = list(
                    get_connected_elements(net, "trafo3w", section_buses, respect_switches=True, respect_in_service=True)
                )
                section_gens = list(net.gen[net.gen.bus.isin(section) & net.gen.in_service].index)
                section_sgens = list(net.sgen[net.sgen.bus.isin(section) & net.sgen.in_service].index)
                section_loads = list(net.load[net.load.bus.isin(section) & net.load.in_service].index)

                if section_buses:
                    section_dict["buses"] = section_buses
                if section_switches:
                    section_dict["switches"] = section_switches
                if section_lines:
                    section_dict["lines"] = section_lines
                if section_trafos:
                    section_dict["trafos"] = section_trafos
                if section_trafos3w:
                    section_dict["trafos3w"] = section_trafos3w
                if section_loads:
                    section_dict["loads"] = section_loads
                if section_gens:
                    section_dict["gens"] = section_gens
                if section_sgens:
                    section_dict["sgens"] = section_sgens

                if any(section_dict.values()):
                    disc_elements.append(section_dict)

        open_trafo_switches = net.switch[(net.switch.et == "t") & (net.switch.closed == 0)]
        isolated_trafos = set(open_trafo_switches.groupby("element").count().query("bus > 1").index)
        isolated_trafos_is = isolated_trafos.intersection((set(net.trafo[net.trafo.in_service].index)))
        if isolated_trafos_is:
            disc_elements.append({"isolated_trafos": list(isolated_trafos_is)})

        isolated_trafos3w = set(open_trafo_switches.groupby("element").count().query("bus > 2").index)
        isolated_trafos3w_is = isolated_trafos3w.intersection(set(net.trafo[net.trafo.in_service].index))
        if isolated_trafos3w_is:
            disc_elements.append({"isolated_trafos3w": list(isolated_trafos3w_is)})

        return disc_elements if disc_elements else None

    def report(self, error: Exception | None, results: T | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for disconnected elements failed due to the following error:")
            self.out.warning(self.diag_errors["disconnected_elements"])
            return
        if results is None:
            self.out.info("PASSED: No problematic switches found")
            return
        # message header
        self.out.compact("disconnected_elements:\n")
        self.out.detailed("Checking for elements without a connection to an external grid...\n")

        # message body
        element_counter = 0
        for disc_section in results:
            self.out.compact(f"disconnected_section: {disc_section}")
            self.out.detailed("Disconnected section found, consisting of the following elements:")
            for key in disc_section:
                element_counter += len(disc_section[key])
                self.out.detailed(f"{key}: {disc_section[key]}")

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} disconnected element(s) found.")


class WrongReferenceSystem(DiagnosticFunction):
    """
    Checks usage of wrong reference system for loads, sgens and gens.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net: pandapowerNet | None = None

    def diagnostic(self, net: pandapowerNet, **kwargs) -> T | None: # TODO: T typing
        """

         INPUT:
            **net** (pandapowerNet)    - pandapower network

         OUTPUT:
            **check_results** (dict)        - dict that contains the indices of all components where the
                                              usage of the wrong reference system was found.

                                              Format: {'element_type': element_indices}

        """
        self.net = net
        check_results = {}
        neg_loads = list(net.load[net.load.p_mw < 0].index)
        neg_gens = list(net.gen[net.gen.p_mw < 0].index)
        neg_sgens = list(net.sgen[net.sgen.p_mw < 0].index)

        if neg_loads:
            check_results["loads"] = neg_loads
        if neg_gens:
            check_results["gens"] = neg_gens
        if neg_sgens:
            check_results["sgens"] = neg_sgens

        return check_results if check_results else None

    def report(self, error: Exception | None, results: T | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for wrong reference system failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: correct reference system")
            return
        # message header
        self.out.compact("wrong_reference_system:\n")
        self.out.detailed("Checking for usage of wrong reference system...\n")

        # message body
        if self.net is None:
            raise RuntimeError('report called before diagnostic')
        for element_type in results:
            self.out.compact(f"{element_type} {results[element_type]}: wrong reference system.")
            for element in results[element_type]:
                _element_type: str = element_type[:-1]  # remove s at end (element_type can be 'loads', 'gens' or 'sgens'
                element_name = self.net[_element_type].name.at[element]
                element_p_mw = self.net[_element_type].p_mw.at[element]
                self.out.detailed(
                    f"Found {_element_type} {element}: '{element_name}' with p_mw = "
                    f"{element_p_mw}. In load reference system p_mw should be positive."
                )

        # message summary
        if "loads" in results:
            self.out.detailed(
                f"\nSUMMARY: Found {len(results['loads'])} load(s) with negative p_mw. "
                "In load reference system, p_mw should be positive. "
                "If the intention was to model a constant generation, please use an sgen instead."
            )
        if "gens" in results:
            self.out.detailed(
                f"\nSUMMARY: Found {len(results['gens'])} gen(s) with positive p_mw. "
                "In load reference system, p_mw should be negative. "
                "If the intention was to model a load, please use a load instead."
            )
        if "sgens" in results:
            self.out.detailed(
                f"\nSUMMARY: Found {len(results['sgens'])} sgen(s) with positive p_mw. "
                "In load reference system, p_mw should be negative. "
                "If the intention was to model a load, please use a load instead."
            )


class NumbaComparison(DiagnosticFunction):
    """
    Compares the results of loadflows with numba=True vs. numba=False.
    """
    def diagnostic(self, net: ADict, **kwargs) -> T | None: # TODO: T typing
        """
         INPUT:
            **net** (pandapowerNet)    - pandapower network
            **numba_tolerance** (float) - Maximum absolute deviation allowed between
                                          numba=True/False results.
         OPTIONAL:
            **kwargs** - Keyword arguments for power flow function. If "run" is in kwargs the default call to runpp()
                         is replaced by the function kwargs["run"]
         OUTPUT:
            **check_result** (dict)    - Absolute deviations between numba=True/False results.
        """
        numba_tolerance = kwargs.pop("numba_tolerance", default_argument_values.get("numba_tolerance", None))
        if numba_tolerance is None:
            raise RuntimeError("missing default argument value for 'numba_tolerance'")
        run = partial(kwargs.pop("run", runpp), **kwargs)
        check_results: dict = {}
        run(net, numba=True)
        result_numba_true = copy.deepcopy(net)
        run(net, numba=False)
        result_numba_false = copy.deepcopy(net)
        res_keys = [key for key in result_numba_true if (key in [
            "res_bus", "res_ext_grid", "res_gen", "res_impedance", "res_line", "res_load", "res_sgen",
            "res_shunt", "res_trafo", "res_trafo3w", "res_ward", "res_xward"
        ])]
        for key in res_keys:
            diffs = abs(result_numba_true[key] - result_numba_false[key]) > numba_tolerance
            if any(diffs.any()):
                if key not in check_results:
                    check_results[key] = {}
                for col in diffs.columns:
                    if (col not in check_results[key]) and (diffs.any()[col]):
                        check_results[key][col] = {}
                        numba_true = result_numba_true[key][col][diffs[col]]
                        numba_false = result_numba_false[key][col][diffs[col]]
                        check_results[key][col] = abs(numba_true - numba_false)

        return check_results if check_results else None

    def report(self, error: Exception | None, results: T | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("numba_comparison failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No results with deviations between numba = True vs. False found.")
            return
        # message header
        self.out.compact("numba_comparison:\n")
        self.out.detailed("Checking for deviations between numba = True vs. False...\n")

        # message body
        element_counter = 0
        for element_type in results:
            for res_type in results[element_type]:
                self.out.compact(f"{element_type}.{res_type} absolute deviations:\n{results[element_type][res_type]}")
                for idx in results[element_type][res_type].index:
                    element_counter += 1
                    dev = results[element_type][res_type].loc[idx]
                    self.out.detailed(f"{element_type}.{res_type} at index {idx}: absolute deviation = {dev}")

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} results with deviations between numba = True vs. False found.")


class DeviationFromStdType(DiagnosticFunction):
    """
    Checks, if element parameters match the values in the standard type library.
    """
    def diagnostic(self, net: ADict, **kwargs) -> dict[str, dict] | None:
        """
         INPUT:
            **net** (pandapowerNet)    - pandapower network


         OUTPUT:
            **check_results** (dict)   - All elements, that don't match the values in the
                                         standard type library

                                         Format: (element_type, element_index, parameter)
        """
        check_results: dict[str, dict] = defaultdict(dict)
        for key, std_types in net.std_types.items():
            if key not in net:
                continue
            for i, element in net[key].iterrows():
                std_type = element.std_type
                if std_type not in std_types:
                    if std_type is not None:
                        check_results[key][i] = {"std_type_in_lib": False}
                    continue
                std_type_values = std_types[std_type]
                for param in std_type_values:
                    if param == "tap_pos":
                        continue
                    if param in net[key].columns:
                        try:
                            isclose = np.isclose(element[param], std_type_values[param], equal_nan=True)
                        except TypeError:
                            isclose = element[param] == std_type_values[param]
                        if not isclose:
                            check_results[key][i] = {
                                "param": param,
                                "e_value": element[param],
                                "std_type_value": std_type_values[param],
                                "std_type_in_lib": True,
                            }

        return check_results if check_results else None

    def report(self, error: Exception | None, results: dict[str, dict] | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for deviation from std_type failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No elements with deviations from std_type found.")
            return
        # message header
        self.out.compact("deviation_from_std_type:\n")
        self.out.detailed("Checking for deviation from std type...\n")

        # message body
        element_counter = 0
        for et in results:
            for eid in results[et]:
                element_counter += 1
                values = results[et][eid]
                if values["std_type_in_lib"]:
                    self.out.warning(
                        f"{et} {eid}: {values['param']} = {values['e_value']}, "
                        f"std_type_value = {values['std_type_value']}"
                    )
                else:
                    self.out.warning(f"{et} {eid}: No valid std_type or std_type not in net.std_types")

        # message summary
        self.out.detailed(f"\nSUMMARY: {element_counter} elements with deviations from std_type found.")


class ParallelSwitches(DiagnosticFunction):
    """
    Checks for parallel switches.
    """
    def diagnostic(self, net: ADict, **kwargs) -> T | None: # TODO: T typing
        """

         INPUT:
            **net** (PandapowerNet)    - pandapower network

         OUTPUT:
            **found_parallel_switches** (list)   - List of tuples each containing parallel switches.
        """
        found_parallel_switches = []
        compare_parameters = ["bus", "element", "et"]
        parallels_bus_and_element = list(net.switch.groupby(compare_parameters).count().query("closed > 1").index)
        for bus, element, et in parallels_bus_and_element:
            found_parallel_switches.append(list(net.switch.query("bus==@bus & element==@element & et==@et").index))

        return found_parallel_switches if found_parallel_switches else None

    def report(self, error: Exception | None, results: T | None) -> None:
        # error and success checks
        if error is not None:
            self.out.warning("Check for parallel_switches failed due to the following error:")
            self.out.warning(error)
            return
        if results is None:
            self.out.info("PASSED: No parallel switches found.")
            return
        # message header
        self.out.compact("parallel_switches:\n")
        self.out.detailed("Checking for parallel switches...\n")

        # message body
        for switch_tuple in results:
            self.out.warning(f"switches {switch_tuple} are parallel.")

        # message summary
        self.out.detailed(f"\nSUMMARY: {len(results)} occurrences of parallel switches found.")

# (name in result_dict, instance of class, list of arguments: None = all kwargs / [] = no arguments)
default_diagnostic_functions: list[tuple[str, DiagnosticFunction, list[str] | None]] = [
    ("missing_bus_indices", MissingBusIndices(), []),
    ("disconnected_elements", DisconnectedElements(), []),
    ("different_voltage_levels_connected", DifferentVoltageLevelsConnected(), []),
    ("implausible_impedance_values", ImplausibleImpedanceValues(), None),
    ("nominal_voltages_dont_match", NominalVoltagesMismatch(), ["nominal_voltage_tolerance"]),
    ("invalid_values", InvalidValues(), []),
    ("overload", Overload(), None),
    ("wrong_line_capacitance", WrongLineCapacitance(), None),
    ("wrong_switch_configuration", WrongSwitchConfiguration(), None),
    ("test_subnet_from_zone", SubNetProblemTest(), None),
    ("multiple_voltage_controlling_elements_per_bus", MultipleVoltageControllingElementsPerBus(), []),
    ("no_ext_grid", NoExtGrid(), []),
    ("wrong_reference_system", WrongReferenceSystem(), []),
    ("deviation_from_std_type", DeviationFromStdType(), []),
    ("numba_comparison", NumbaComparison(), None),
    ("parallel_switches", ParallelSwitches(), []),
    ("optimistic_powerflow", OptimisticPowerflow(), None)
]
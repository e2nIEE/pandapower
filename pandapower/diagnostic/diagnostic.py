# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from typing import Literal, Any
from collections.abc import Callable

from pandapower.network import ADict
from pandapower.diagnostic.diagnostic_functions import default_diagnostic_functions, default_argument_values
from pandapower.diagnostic.diagnostic_helpers import logger, DiagnosticFunction, NotCompactFilter, NotDetailedFilter, LogCount

# LOGGER:
# there is no local logger here because it requires the custom log levels from diagnostic_helpers

# separator between log messages
log_format_len = 60
log_message_sep = f"\n{'':-<{log_format_len}}\n"


class Diagnostic:
    """
    A Diagnostic Tool Class for diagnosing and reporting on issues in a ADict subclassed network

    default Diagnostic Functions are for networks of type pandapowerNet

    Example:
        >>> from pandapower.diagnostic import Diagnostic
        >>> from pandapower.networks.mv_oberrhein import mv_oberrhein
        >>>
        >>> net = mv_oberrhein()
        >>> diag = Diagnostic()
        >>> result = diag.diagnose_network(net, report_style="detailed")

    """
    def __init__(self, add_default_functions: bool = True):
        """

        Parameters:
            add_default_functions: Should the DiagnosticFunctions for pandapower networks be added?
        """
        self._functions: list[tuple[str, DiagnosticFunction, list[str] | None]] = []
        self._report_functions: list[Callable] = []
        self.kwargs = {}
        if add_default_functions:
            self.kwargs = default_argument_values
            self._functions = default_diagnostic_functions

        self.net: ADict | None = None
        self.diag_results: dict[str, Any] = {}
        self.diag_errors: dict[str, Any] = {}

    def register_function(
            self, diagnostic_function: DiagnosticFunction, argument_names: list[str] | None, name: str | None
    ):
        """
        register a diagnostic function to run when running network diagnostics and the associated report function.

        Parameters:
            diagnostic_function: instance of class implementing the DiagnosticFunction Base Class
            argument_names: the kwargs that should be passed to the diagnostic function. If None is provided all kwargs
                will be passed.
            name: name to use for results dict and reports, if None will use name of the class
            
        Example:
            >>> from pandapower.diagnostic import Diagnostic
            >>> from pandapower.diagnostic.diagnostic_functions import DeviationFromStdType
            >>>
            >>> diag = Diagnostic(add_default_functions=False)
            >>> diag.register_function(DeviationFromStdType(), None)
            >>> diag.register_function(DeviationFromStdType(), None, "dev_from_std_twice")
        
        """
        if name is None:
            name = diagnostic_function.__class__.__name__
        self._functions.append((name, diagnostic_function, argument_names))

    def diagnose_network(
        self,
        net: ADict,
        report_style: Literal["detailed", "compact"] | None = "detailed",
        warnings_only: bool = False,
        return_result_dict: bool = True,
        **kwargs,
    ) -> dict[str, Any] | None:
        """
        Tool for diagnosis of pandapower networks. Identifies possible reasons for non converging loadflows.

        Parameters:
             net: the network to run the registerd diagnostic functions on
             report_style: style of the report, that gets ouput in the console
                  - 'detailled': full report with high level of additional descriptions
                  - 'compact'  : more compact report, containing essential information only
                  - 'None'     : no report
             warnings_only: Filters logging output for warnings
                 True: logging output for errors only
                 False: logging output for all checks, regardless if errors were found or not
             return_result_dict: returns a dictionary containing all check results
                 True: returns dict with all check results
                 False: no result dict
             overload_scaling_factor: downscaling factor for loads and generation for overload check
             lines_min_length_km: minimum length_km allowed for lines
             lines_min_z_ohm: minimum z_ohm allowed for lines
             nom_voltage_tolerance: highest allowed relative deviation between nominal voltages and bus voltages

        Keyword arguments:
            any: for the power flow function to use during tests. If "run" is in kwargs the default call to runpp() is
                replaced by the function kwargs["run"]

        Returns:
            A dict that contains the the result of each diagnostic function, can be passed to its report function for
            interpretation.

        Example:
            >>> from pandapower.diagnostic import Diagnostic
            >>> d = Diagnostic()
            >>> results = d.diagnose_network(net, report_style='compact', warnings_only=True)
            >>> d.compact_report()
            >>> d.detailed_report()

        """
        # clear old diagnostic output
        self.diag_results = {}
        self.diag_errors = {}
        self.net = net
        # update with new kwargs
        self.kwargs.update(kwargs)
        for name, diag_class, arg_names in self._functions:
            args: dict[str, Any] = self.kwargs if arg_names is None else {}
            if arg_names is not None:
                for arg_name in arg_names:
                    args[arg_name] = self.kwargs.get(arg_name, ValueError(
                        f"Diagnostic function '{name}' expects argument '{arg_name}', which was not provided."
                    ))
                    if isinstance(args[arg_name], ValueError):
                        raise args[arg_name]
            try:
                diag_result = diag_class.diagnostic(net, **args)
                if diag_result is not None:
                    self.diag_results[name] = diag_result
            except Exception as e:
                self.diag_errors[name] = e

        if report_style is not None:
            self.report(compact_report=report_style == "compact", warnings_only=warnings_only)

        return self.diag_results if return_result_dict else None

    def compact_report(self, warnings_only: bool = False):
        """
        Generate the compact diagnostic report.
        
        Parameters:
            warnings_only: If True only warnings are printed
        
        Raises:
            RuntimeError: When called and no diagnostic results are available.
        """
        self.report(warnings_only=warnings_only)

    def detailed_report(self, warnings_only: bool = False):
        """
        Generate the detailed diagnostic report.
        
        Parameters:
            warnings_only: If True only warnings are printed
        
        Raises:
            RuntimeError: When called and no diagnostic results are available.
        """
        self.report(compact_report=False, warnings_only=warnings_only)

    def report(self, compact_report: bool = True, warnings_only: bool = False) -> None:
        """
        Generate a diagnostic report.
        
        Parameters:
            compact_report: diagnostic report should be compact or detailed
            warnings_only: diagnostic report should be only warnings or all info
        
        Raises:
            RuntimeError: When called and no diagnostic results are available.
        """
        if self.net is None:
            raise RuntimeError(
                "Network has not been diagnosed yet. Call 'diagnose_network' before retrieving the report."
            )
        # setup logger
        if warnings_only:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        original_log_level = logger.getEffectiveLevel()
        logger.setLevel(log_level)
        log_counter = LogCount()
        log_detail_filter = NotDetailedFilter() if compact_report else NotCompactFilter()
        logger.addFilter(log_counter)
        logger.addFilter(log_detail_filter)

        # generate diagnostic output
        logger.warning(f"\n\n{' PANDAPOWER DIAGNOSTIC TOOL ':-^{log_format_len}}\n")

        for name, diag_class, _ in self._functions:
            if log_counter.cd_in_count():
                logger.warning(log_message_sep)
            log_counter.reset()
            diag_class.report(self.diag_errors.get(name, None), self.diag_results.get(name, None))

        logger.warning(f"\n\n{' END OF PANDAPOWER DIAGNOSTIC ':-^{log_format_len}}\n")

        # teardown logger (if this is omitted it might affect next run)
        logger.removeFilter(log_counter)
        logger.removeFilter(log_detail_filter)
        logger.setLevel(original_log_level)

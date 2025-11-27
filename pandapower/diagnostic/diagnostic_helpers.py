# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from typing import Generic, TypeVar, Literal
from abc import abstractmethod, ABC
from collections import defaultdict

import numpy as np

from pandapower import pandapowerNet
from pandapower.auxiliary import ADict

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


add_log_level(logging.WARNING + 2, "compact")
add_log_level(logging.WARNING + 4, "detailed")

T = TypeVar('T')


class DiagnosticFunction(ABC, Generic[T]):
    """
    A meta class for creating custom Diagnostic Functions that can be executed with the pandapower Diagnostic API.
    """
    def __init__(self):
        self.out = logger

    @abstractmethod
    def diagnostic(self, net: ADict, **kwargs) -> T | None:
        """
        A diagnostic method that should be run on the network.
        
        Parameters:
            net: the network to run it on
            kwargs: any kwargs passed to diagnose_network
        
        Returns:
            the diagnostic result or None if successful
        """
        pass

    @abstractmethod
    def report(self, error: Exception | None, results: T | None) -> None:
        """
        A method to generate a report for the result of diagnostic.
        
        Parameters:
            error: when the diagnostic encountered an error it will be passed here
            results: when the diagnostic produced an output it will be passed here

        The first lines should always check if an error occurred or there are no results
        
        >>> # error and success checks
        >>> if error is not None:
        >>>     self.out.warning("Check for < > failed due to the following error:")
        >>>     self.out.warning(error)
        >>>     return
        >>> if results is None:
        >>>     self.out.info("PASSED: < > successful.")
        >>>     return
        >>> # message header
        >>> ...
        >>> # message body
        >>> ...
        >>> # message summary

        There are additional levels defined for the `self.out` logger.
        
        >>> self.out.detailed
        
        and
        
        >>> self.out.compact
        
        These will only print in the resepctive reports.
        Use info if the message should only show when warnings_only is False.
        Use warning if the message should always be shown.
        """
        pass


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


def check_boolean(element, element_index, column):
    if element[column] not in [True, False, 0, 1, 0.0, 1.0]:
        return element_index


def check_greater_equal_zero(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] < 0:
            return element_index
    else:
        return element_index


def check_greater_zero(element, element_index, column):
    """
    functions that check, if a certain input type restriction for attribute values of a pandapower
    elements are fulfilled. Exemplary description for all type check functions.

    INPUT:
       **element (pandas.Series)** - pandapower element instance (e.g. net.bus.loc[1])

       **element_index (int)**     - index of the element instance

       **column (string)**         - element attribute (e.g. 'vn_kv')


    OUTPUT:
       **element_index (index)**   - index of element instance, if input type restriction is not
                                     fulfilled
    """
    if check_number(element, element_index, column) is None:
        if element[column] <= 0:
            return element_index
    else:
        return element_index


def check_greater_zero_less_equal_one(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if not (0 < element[column] <= 1):
            return element_index


def check_less_15(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] >= 15:
            return element_index
    else:
        return element_index


def check_less_20(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] >= 20:
            return element_index
    else:
        return element_index


def check_less_equal_zero(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] > 0:
            return element_index
    else:
        return element_index


def check_less_zero(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] >= 0:
            return element_index
    else:
        return element_index


def check_number(element, element_index, column):
    try:
        nan_check = np.isnan(element[column])
        if nan_check or isinstance(element[column], bool):
            return element_index
    except TypeError:
        return element_index


def check_pos_int(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if not ((element[column] % 1 == 0) and element[column] >= 0):
            return element_index
    else:
        return element_index


def check_switch_type(element, element_index, column):
    if element[column] not in ["b", "l", "t", "t3"]:
        return element_index
    return None


def diagnostic(
        net: pandapowerNet,
        report_style: Literal['compact', 'detailed'] | None,
        warnings_only: bool,
        return_result_dict: bool,
        overload_scaling_factor: float = 0.001,
        lines_min_length_km: float = 0.,
        lines_min_z_ohm: float = 0.,
        nom_voltage_tolerance: float = 0.3,
        **kwargs
):
    """
    Tool for diagnosis of pandapower networks. Identifies possible reasons for non converging loadflows.

    Parameters:
        net: A pandapower network
        report_style: style of the report, that gets ouput in the console
            'detailed': full report with high level of additional descriptions
            'compact'  : more compact report, containing essential information only
            'None'     : no report
        warnings_only: Filters logging output for warnings
            True: logging output for errors only
            False: logging output for all checks, regardless if errors were found or not
        return_result_dict: returns a dictionary containing all check results
            True: returns dict with all check results
            False: no result dict
        overload_scaling_factor: downscaling factor for loads and generation for overload check
        lines_min_length_km: minimum length_km allowed for lines
        lines_min_z_ohm: minimum z_ohm allowed for lines
        nom_voltage_tolerance** (float, 0.3): highest allowed relative deviation between nominal voltages and bus
            voltages
            
    Keyword Arguments:
        Any: Keyword arguments for the power flow function to use during tests. If "run" is in kwargs the default call to
            runpp() is replaced by the function kwargs["run"]

    Returns:
        dict that contains the indices of all elements where errors were found
            Format: {'check_name': check_results}

    Example:
        >>> from pandapower.diagnostic.diagnostic_helpers import diagnostic
        >>> diagnostic(net, report_style='compact', warnings_only=True)

    """
    from pandapower.diagnostic.diagnostic import Diagnostic
    d = Diagnostic()
    kwargs['overload_scaling_factor'] = overload_scaling_factor
    kwargs['lines_min_length_km'] = lines_min_length_km
    kwargs['nom_voltage_tolerance'] = nom_voltage_tolerance
    kwargs['lines_min_z_ohm'] = lines_min_z_ohm
    return d.diagnose_network(net, report_style, warnings_only, return_result_dict, **kwargs)

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import time
from typing import Union, List, Type, Dict
import pandapower.auxiliary
from . import build_pp_net
from .. import cim_classes
from .. import interfaces
from . import converter_classes as std_converter_classes

logger = logging.getLogger('cim.cim2pp.from_cim')


def from_cim_dict(cim_parser: cim_classes.CimParser, log_debug=False, convert_line_to_switch: bool = False,
                  line_r_limit: float = 0.1, line_x_limit: float = 0.1,
                  repair_cim: Union[str, interfaces.CIMRepair] = None,
                  repair_cim_class: Type[interfaces.CIMRepair] = None,
                  repair_pp: Union[str, interfaces.PandapowerRepair] = None,
                  repair_pp_class: Type[interfaces.PandapowerRepair] = None,
                  custom_converter_classes: Dict = None,
                  **kwargs) -> pandapower.auxiliary.pandapowerNet:
    """
    Create a pandapower net from a CIM data structure.

    :param cim_parser: The CimParser with parsed cim data.
    :param log_debug: Set this parameter to True to enable logging at debug level. Optional, default: False
    :param convert_line_to_switch: Set this parameter to True to enable line -> switch conversion. All lines with a
        resistance lower or equal than line_r_limit or a reactance lower or equal than line_x_limit will become a
        switch. Optional, default: False
    :param line_r_limit: The limit from resistance. Optional, default: 0.1
    :param line_x_limit: The limit from reactance. Optional, default: 0.1
    :param repair_cim: The CIMRepair object or a path to its serialized object. Optional, default: None
    :param repair_cim_class: The CIMRepair class. Optional, default: None
    :param repair_pp: The PandapowerRepair object or a path to its serialized object. Optional, default: None
    :param repair_pp_class: The PandapowerRepair class. Optional, default: None
    :param custom_converter_classes: Dict to inject classes for different functionality. Optional, default: None
    :return: The pandapower net.
    """
    converter_classes = get_converter_classes()
    if custom_converter_classes is not None:
        for key in custom_converter_classes:
            converter_classes[key] = custom_converter_classes.get(key)

    # repair the input CIM data
    if repair_cim is not None and repair_cim_class is not None:
        repair_cim = repair_cim_class().deserialize(repair_cim, report_container=cim_parser.get_report_container())
        repair_cim.repair(cim_parser.get_cim_dict(), report_container=cim_parser.get_report_container())

    cim_converter = build_pp_net.CimConverter(cim_parser=cim_parser, converter_classes=converter_classes, **kwargs)
    pp_net = cim_converter.convert_to_pp(convert_line_to_switch=convert_line_to_switch, line_r_limit=line_r_limit,
                                         line_x_limit=line_x_limit, log_debug=log_debug, **kwargs)

    # repair the output pandapower network
    if repair_pp is not None and repair_pp_class is not None:
        repair_pp = repair_pp_class().deserialize(repair_pp, report_container=cim_parser.get_report_container())
        repair_pp.repair(pp_net, report_container=cim_parser.get_report_container())

    return pp_net


def get_converter_classes():
    converter_classes: Dict[str,classmethod] = {
        'ConnectivityNodesCim16': std_converter_classes.connectivitynodes.connectivityNodesCim16.ConnectivityNodesCim16,
        'externalNetworkInjectionsCim16':
            std_converter_classes.externalnetworks.externalNetworkInjectionsCim16.ExternalNetworkInjectionsCim16,
        'acLineSegmentsCim16': std_converter_classes.lines.acLineSegmentsCim16.AcLineSegmentsCim16,
        'dcLineSegmentsCim16': std_converter_classes.lines.dcLineSegmentsCim16.DcLineSegmentsCim16,
        'switchesCim16': std_converter_classes.switches.switchesCim16.SwitchesCim16,
        'energyConcumersCim16': std_converter_classes.loads.energyConcumersCim16.EnergyConsumersCim16,
        'conformLoadsCim16': std_converter_classes.loads.conformLoadsCim16.ConformLoadsCim16,
        'nonConformLoadsCim16': std_converter_classes.loads.nonConformLoadsCim16.NonConformLoadsCim16,
        'stationSuppliesCim16': std_converter_classes.loads.stationSuppliesCim16.StationSuppliesCim16,
        'synchronousMachinesCim16': std_converter_classes.generators.synchronousMachinesCim16.SynchronousMachinesCim16,
        'asynchronousMachinesCim16':
            std_converter_classes.generators.asynchronousMachinesCim16.AsynchronousMachinesCim16,
        'energySourcesCim16': std_converter_classes.generators.energySourcesCim16.EnergySourceCim16,
        'linearShuntCompensatorCim16':
            std_converter_classes.shunts.linearShuntCompensatorCim16.LinearShuntCompensatorCim16,
        'nonLinearShuntCompensatorCim16':
            std_converter_classes.shunts.nonLinearShuntCompensatorCim16.NonLinearShuntCompensatorCim16,
        'staticVarCompensatorCim16': std_converter_classes.shunts.staticVarCompensatorCim16.StaticVarCompensatorCim16,
        'equivalentBranchesCim16': std_converter_classes.impedance.equivalentBranchesCim16.EquivalentBranchesCim16,
        'seriesCompensatorsCim16': std_converter_classes.impedance.seriesCompensatorsCim16.SeriesCompensatorsCim16,
        'equivalentInjectionsCim16': std_converter_classes.wards.equivalentInjectionsCim16.EquivalentInjectionsCim16,
        'powerTransformersCim16': std_converter_classes.transformers.powerTransformersCim16.PowerTransformersCim16,
        'tapController': std_converter_classes.transformers.tapController.TapController,
        'geoCoordinatesFromGLCim16':
            std_converter_classes.coordinates.geoCoordinatesFromGLCim16.GeoCoordinatesFromGLCim16,
        'coordinatesFromDLCim16': std_converter_classes.coordinates.coordinatesFromDLCim16.CoordinatesFromDLCim16,
    }
    return converter_classes


def from_cim(file_list: List[str] = None, encoding: str = 'utf-8', convert_line_to_switch: bool = False,
             line_r_limit: float = 0.1, line_x_limit: float = 0.1,
             repair_cim: Union[str, interfaces.CIMRepair] = None,
             repair_cim_class: Type[interfaces.CIMRepair] = None,
             repair_pp: Union[str, interfaces.PandapowerRepair] = None,
             repair_pp_class: Type[interfaces.PandapowerRepair] = None,
             custom_converter_classes: Dict = None, **kwargs) -> \
        pandapower.auxiliary.pandapowerNet:
    """
    Convert a CIM net to a pandapower net from XML files.
    Additional parameters for kwargs:
    - create_measurements (str): Set this parameter to 'SV' to create measurements for the pandapower net from the SV
    profile. Set it to 'Analog' to create measurements from Analogs. If the parameter is not set or is set to None, no
    measurements will be created.
    - use_GL_or_DL_profile (str): Choose the profile to use for converting coordinates. Set it to 'GL' to use the GL
    profile (Usually lat and long coordinates). Set it to 'DL' to use the DL profile (Usually x, y coordinates for
    displaying control room schema). Set it to 'both' to let the converter choose the profile. The converter will
    choose the GL profile first if available, otherwise the DL profile. Optional, default: both.
    - diagram_name (str): The name from the Diagram from the diagram layout profile for the geo coordinates. Default:
    The first diagram sorted ascending by name. Set the parameter to "all" to use available diagrams for creating the
    coordinates.
    - create_tap_controller (bool): If True, create pandapower controllers for transformer tap changers. If False, skip
    creating them. Default: True
    - sn_mva (float): Set the sn_mva from the pandapower net to a specific value. This value is not given in CGMES.
    Default: None (pandapower default will be chosen)
    - run_powerflow (bool): Option to run to powerflow inside the converter to create res tables directly.
    Default: False.
    - ignore_errors (bool): Option to disable raising of internal errors. Useful if you need to get a network not matter
    if there are errors in the conversion. Default: True.

    :param file_list: The path to the CGMES files as a list.
    :param encoding: The encoding from the files. Optional, default: utf-8
    :param convert_line_to_switch: Set this parameter to True to enable line -> switch conversion. All lines with a
        resistance lower or equal than line_r_limit or a reactance lower or equal than line_x_limit will become a
        switch. Optional, default: False
    :param line_r_limit: The limit from resistance. Optional, default: 0.1
    :param line_x_limit: The limit from reactance. Optional, default: 0.1
    :param repair_cim: The CIMRepair object or a path to its serialized object. Optional, default: None
    :param repair_cim_class: The CIMRepair class. Optional, default: None
    :param repair_pp: The PandapowerRepair object or a path to its serialized object. Optional, default: None
    :param repair_pp_class: The PandapowerRepair class. Optional, default: None
    :param custom_converter_classes: Dict to inject classes for different functionality. Optional, default: None
    :return: The pandapower net.
    """
    time_start_parsing = time.time()

    cim_parser = cim_classes.CimParser()
    cim_parser.parse_files(file_list=file_list, encoding=encoding, prepare_cim_net=True, set_data_types=True)

    time_start_converting = time.time()
    pp_net = from_cim_dict(cim_parser, convert_line_to_switch=convert_line_to_switch,
                           line_r_limit=line_r_limit, line_x_limit=line_x_limit, repair_cim=repair_cim,
                           repair_cim_class=repair_cim_class, repair_pp=repair_pp, repair_pp_class=repair_pp_class,
                           custom_converter_classes=custom_converter_classes, **kwargs)
    time_end_converting = time.time()
    logger.info("The pandapower net: \n%s" % pp_net)

    logger.info("Needed time for parsing: %s" % (time_start_converting - time_start_parsing))
    logger.info("Needed time for converting: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time: %s" % (time_end_converting - time_start_parsing))

    return pp_net

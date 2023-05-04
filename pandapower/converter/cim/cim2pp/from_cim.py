# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import time
from typing import Union, List, Type
import pandapower.auxiliary
from . import build_pp_net
from .. import cim_classes
from .. import interfaces

logger = logging.getLogger('cim.cim2pp.from_cim')


def from_cim_dict(cim_parser: cim_classes.CimParser, log_debug=False, convert_line_to_switch: bool = False,
                  line_r_limit: float = 0.1, line_x_limit: float = 0.1,
                  repair_cim: Union[str, interfaces.CIMRepair] = None,
                  repair_cim_class: Type[interfaces.CIMRepair] = None,
                  repair_pp: Union[str, interfaces.PandapowerRepair] = None,
                  repair_pp_class: Type[interfaces.PandapowerRepair] = None,
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
    :return: The pandapower net.
    """

    # repair the input CIM data
    if repair_cim is not None and repair_cim_class is not None:
        repair_cim = repair_cim_class().deserialize(repair_cim, report_container=cim_parser.get_report_container())
        repair_cim.repair(cim_parser.get_cim_dict(), report_container=cim_parser.get_report_container())

    cim_converter = build_pp_net.CimConverter(cim_parser=cim_parser, **kwargs)
    pp_net = cim_converter.convert_to_pp(convert_line_to_switch=convert_line_to_switch, line_r_limit=line_r_limit,
                                         line_x_limit=line_x_limit, log_debug=log_debug, **kwargs)

    # repair the output pandapower network
    if repair_pp is not None and repair_pp_class is not None:
        repair_pp = repair_pp_class().deserialize(repair_pp, report_container=cim_parser.get_report_container())
        repair_pp.repair(pp_net, report_container=cim_parser.get_report_container())

    return pp_net


def from_cim(file_list: List[str] = None, encoding: str = 'utf-8', convert_line_to_switch: bool = False,
             line_r_limit: float = 0.1, line_x_limit: float = 0.1,
             repair_cim: Union[str, interfaces.CIMRepair] = None,
             repair_cim_class: Type[interfaces.CIMRepair] = None,
             repair_pp: Union[str, interfaces.PandapowerRepair] = None,
             repair_pp_class: Type[interfaces.PandapowerRepair] = None, **kwargs) -> \
        pandapower.auxiliary.pandapowerNet:
    """
    Convert a CIM net to a pandapower net from XML files.
    Additional parameters for kwargs:
    create_measurements (str): Set this parameter to 'SV' to create measurements for the pandapower net from the SV
    profile. Set it to 'Analog' to create measurements from Analogs. If the parameter is not set or is set to None, no
    measurements will be created.
    update_assets_from_sv (bool): Set this parameter to True to update the assets (sgens, loads, wards, ...) with values
    from the SV profile. Default: False.
    use_GL_or_DL_profile (str): Choose the profile to use for converting coordinates. Set it to 'GL' to use the GL
    profile (Usually lat and long coordinates). Set it to 'DL' to use the DL profile (Usually x, y coordinates for
    displaying control room schema). Set it to 'both' to let the converter choose the profile. The converter will
    choose the GL profile first if available, otherwise the DL profile. Optional, default: both.
    diagram_name (str): The name from the Diagram from the diagram layout profile for the geo coordinates. Default: The
    first diagram sorted ascending by name.
    create_tap_controller (bool): If True, create pandapower controllers for transformer tap changers. If False, skip
    creating them. Default: True
    sn_mva (float): Set the sn_mva from the pandapower net to a specific value. This value is not given in CGMES.
    Default: None (pandapower default will be chosen)

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
    :return: The pandapower net.
    """
    time_start_parsing = time.time()

    cim_parser = cim_classes.CimParser()
    cim_parser.parse_files(file_list=file_list, encoding=encoding, prepare_cim_net=True, set_data_types=True)

    time_start_converting = time.time()
    pp_net = from_cim_dict(cim_parser, convert_line_to_switch=convert_line_to_switch,
                           line_r_limit=line_r_limit, line_x_limit=line_x_limit, repair_cim=repair_cim,
                           repair_cim_class=repair_cim_class, repair_pp=repair_pp, repair_pp_class=repair_pp_class,
                           **kwargs)
    time_end_converting = time.time()
    logger.info("The pandapower net: \n%s" % pp_net)

    logger.info("Needed time for parsing: %s" % (time_start_converting - time_start_parsing))
    logger.info("Needed time for converting: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time: %s" % (time_end_converting - time_start_parsing))

    return pp_net

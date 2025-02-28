# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import time
from typing import Union
import pandapower.auxiliary
from . import build_pp_net
from .. import cim_classes

logger = logging.getLogger('cim.cim2pp.from_cim')


def from_cim_dict(cim_parser: cim_classes.CimParser, log_debug=False, convert_line_to_switch: bool = False,
                  line_r_limit: float = 0.1, line_x_limit: float = 0.1, **kwargs) -> pandapower.auxiliary.pandapowerNet:
    """
    Create a pandapower net from a CIM dictionary.
    :param cim_parser: The CimParser with parsed cim data.
    :param log_debug: Set this parameter to True to enable logging at debug level. Optional, default: False
    :param convert_line_to_switch: Set this parameter to True to enable line -> switch conversion. All lines with a
    resistance lower or equal than line_r_limit or a reactance lower or equal than line_x_limit will become a switch.
    :param line_r_limit: The limit from resistance. Optional, default: 0.1
    :param line_x_limit: The limit from reactance. Optional, default: 0.1
    :return: The pandapower net.
    """
    cim_converter = build_pp_net.CimConverter(cim_parser=cim_parser, **kwargs)
    return cim_converter.convert_to_pp(convert_line_to_switch=convert_line_to_switch, line_r_limit=line_r_limit,
                                       line_x_limit=line_x_limit, log_debug=log_debug, **kwargs)


def from_cim(eq_file: str = None, ssh_file: str = None, tp_file: str = None, sv_file: str = None,
             encoding: str = 'utf-8', convert_line_to_switch: bool = False, line_r_limit: float = 0.1,
             line_x_limit: float = 0.1, repair_cim: Union[str, str] = None,
             repair_pp: Union[str, str] = None, **kwargs) -> pandapower.auxiliary.pandapowerNet:
    """
    Convert a CIM net to a pandapower net from XML files.
    Additional parameters for kwargs:
    eq_bd_file (str): The path to the EQ Boundary file
    tp_bd_file (str): The path to the TP Boundary file
    file_list (str): The path to the CGMES files as a list. Note: The files need a FullModel to parse the CGMES profile.
    create_measurements (str): Set this parameter to 'SV' to create measurements for the pandapower net from the SV
    profile. Set it to 'Analog' to create measurements from Analogs. If the parameter is not set or is set to None, no
    measurements will be created.
    update_assets_from_sv (bool): Set this parameter to True to update the assets (sgens, loads, wards, ...) with values
    from the SV profile. Default: False.
    diagram_name (str): The name from the Diagram from the diagram layout profile for the geo coordinates. Default: The
    first diagram sorted ascending by name.
    create_tap_controller (bool): If True, create pandapower controllers for transformer tap changers. If False, skip
    creating them. Default: True
    sn_mva (float): Set the sn_mva from the pandapower net to a specific value. This value is not given in CGMES.
    Default: None (pandapower default will be chosen)
    :param eq_file: The path to the EQ file.
    :param ssh_file: The path to zhe SSH file.
    :param tp_file: The path to the TP file. Optional, default: None. Note: If the source data is in Bus-Branch format, the TP profile is needed to get the topological structure from the net!
    :param sv_file: The path to the SV file. Optional, default: None
    :param encoding: The encoding from the files. Optional, default: utf-8
    :param convert_line_to_switch: Set this parameter to True to enable line -> switch conversion. All lines with a resistance lower or equal than line_r_limit or a reactance lower or equal than line_x_limit will become a switch. Optional, default: False
    :param line_r_limit: The limit from resistance. Optional, default: 0.1
    :param line_x_limit: The limit from reactance. Optional, default: 0.1
    :param repair_cim: The CIMRepair object or a path to its serialized object. Optional, default: None
    :param repair_pp: The PandapowerRepair object or a path to its serialized object. Optional, default: None
    :return: The pandapower net.
    """
    time_start_parsing = time.time()

    cim_parser = cim_classes.CimParser()
    cim_parser.parse_files(eq_file=eq_file, ssh_file=ssh_file, sv_file=sv_file, tp_file=tp_file,
                           eq_bd_file=kwargs.get('eq_bd_file', None), tp_bd_file=kwargs.get('tp_bd_file', None),
                           file_list=kwargs.get('file_list', None), encoding=encoding, prepare_cim_net=True,
                           set_data_types=True)

    time_start_converting = time.time()
    pp_net = from_cim_dict(cim_parser, convert_line_to_switch=convert_line_to_switch,
                           line_r_limit=line_r_limit, line_x_limit=line_x_limit, **kwargs)

    time_end_converting = time.time()
    logger.info("The pandapower net: \n%s" % pp_net)

    logger.info("Needed time for parsing: %s" % (time_start_converting - time_start_parsing))
    logger.info("Needed time for converting: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time: %s" % (time_end_converting - time_start_parsing))

    return pp_net
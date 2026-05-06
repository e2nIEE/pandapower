# -*- coding: utf-8 -*-
# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import os
import json
from typing import Dict, List
import numpy as np

from pandapower.network_structure import get_structure_dict
from pandapower.auxiliary import pandapowerNet
import pandas as pd

logger = logging.getLogger(__name__)


def get_pp_net_special_columns_dict() -> Dict[str, str]:
    """
    Get a dictionary with the special CIM fields, used as columns in a pandapower network.
    :return Dict[str, str]: The dictionary with the special CIM fields.
    """
    return {'o_id': 'origin_id', 'sub': 'substation', 't': 'terminal', 't_from': 'terminal_from',
            'from_bus': 'terminal_from', 't_to': 'terminal_to', 'to_bus': 'terminal_to', 't_bus': 'terminal_bus',
            't_ele': 'terminal_element', 't_hv': 'terminal_hv', 'hv_bus': 'terminal_hv', 't_mv': 'terminal_mv',
            'mv_bus': 'terminal_mv', 't_lv': 'terminal_lv', 'lv_bus': 'terminal_lv', 'o_cl': 'origin_class',
            'o_prf': 'origin_profile', 'ct': 'cim_topnode', 'tc': 'tapchanger_class', 'tc2': 'tapchanger2_class',
            'tc_id': 'tapchanger_id','tc2_id': 'tapchanger2_id', 'pte_id': 'PowerTransformerEnd_id',
            'pte_id_hv': 'PowerTransformerEnd_id_hv', 'pte_id_mv': 'PowerTransformerEnd_id_mv',
            'pte_id_lv': 'PowerTransformerEnd_id_lv', 'cnc_id': 'ConnectivityNodeContainer_id',
            'sub_id': 'Substation_id', 'src': 'source', 'name': 'name', 'desc': 'description',
            'a_id': 'analog_id', 'bus': 'terminal', 'bb_id': 'Busbar_id', 'bb_name': 'Busbar_name'}


def extend_pp_net_cim(net: pandapowerNet, override: bool = True) -> pandapowerNet:
    """
    Extend pandapower element DataFrames with special columns for the CIM converter, e.g. a column for the RDF ID.
    :param net: The pandapower net to extend.
    :param override: If True, all existing special CIM columns will be overwritten (content will be erased). If False,
    only missing columns will be created. Optional, default: True
    :return: The extended pandapower network.
    """
    # some special items
    if override:
        net['CGMES'] = {}
        net['CGMES']['BaseVoltage'] = pd.DataFrame(None, columns=['rdfId', 'nominalVoltage'])
    else:
        if 'CGMES' not in net:
            net['CGMES'] = {}
        if 'BaseVoltage' not in net['CGMES']:
            net['CGMES']['BaseVoltage'] = pd.DataFrame(None, columns=['rdfId', 'nominalVoltage'])

    return net


def get_cim_schema(cgmes_version: str = '2.4.15') -> Dict[str, Dict[str, Dict[str, str or Dict[str, Dict[str, str]]]]]:
    """
    Parses the CIM schema from the serialized CIM schema json files which have been created from the RDF schema files.
    The schema is parsed for the serializer from the CIM data structure used by the cim2pp and pp2cim converters.

    :param cgmes_version: CIM version to use, '2.4.15', '3.0' or LTDS, default '2.4.15'
    :return: The CIM schema as dictionary.
    """
    path_with_serialized_schemas = os.path.dirname(__file__) + os.sep + 'serialized_schemas'
    if not os.path.isdir(path_with_serialized_schemas):
        os.mkdir(path_with_serialized_schemas)
    for one_file in os.listdir(path_with_serialized_schemas):
        path_to_schema = path_with_serialized_schemas + os.sep + one_file
        if one_file.lower().startswith('cim16_') and cgmes_version == '2.4.15':
            logger.info("Parsing the schema from CIM 16 from disk: %s" % path_to_schema)
            with open(path_to_schema, encoding='UTF-8', mode='r') as f:
                cim_schema = json.load(f)
            return cim_schema
        elif one_file.lower().startswith('cim100_') and (cgmes_version == '3.0' or cgmes_version.lower() == 'ltds'):
            logger.info("Parsing the schema from CIM 100 from disk: %s" % path_to_schema)
            with open(path_to_schema, encoding='UTF-8', mode='r') as f:
                cim_schema = json.load(f)
            return cim_schema

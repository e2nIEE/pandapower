# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import os
import json
from typing import Dict, List
import numpy as np
from pandapower.auxiliary import pandapowerNet
import pandas as pd

logger = logging.getLogger(__name__)


def get_pp_net_special_columns_dict() -> Dict[str, str]:
    """
    Get a dictionary with the special CIM fields, used as columns in a pandapower network.
    :return Dict[str, str]: The dictionary with the special CIM fields.
    """
    return dict({'o_id': 'origin_id', 'sub': 'substation', 't': 'terminal', 't_from': 'terminal_from',
                 'from_bus': 'terminal_from', 't_to': 'terminal_to', 'to_bus': 'terminal_to', 't_bus': 'terminal_bus',
                 't_ele': 'terminal_element', 't_hv': 'terminal_hv', 'hv_bus': 'terminal_hv', 't_mv': 'terminal_mv',
                 'mv_bus': 'terminal_mv', 't_lv': 'terminal_lv', 'lv_bus': 'terminal_lv', 'o_cl': 'origin_class',
                 'o_prf': 'origin_profile', 'ct': 'cim_topnode', 'tc': 'tapchanger_class', 'tc_id': 'tapchanger_id',
                 'pte_id': 'PowerTransformerEnd_id', 'pte_id_hv': 'PowerTransformerEnd_id_hv',
                 'pte_id_mv': 'PowerTransformerEnd_id_mv', 'pte_id_lv': 'PowerTransformerEnd_id_lv',
                 'cnc_id': 'ConnectivityNodeContainer_id', 'sub_id': 'substation_id'})


def extend_pp_net_cim(net: pandapowerNet, override: bool = True) -> pandapowerNet:
    """
    Extend pandapower element DataFrames with special columns for the CIM converter, e.g. a column for the RDF ID.
    :param net: The pandapower net to extend.
    :param override: If True, all existing special CIM columns will be overwritten (content will be erased). If False,
    only missing columns will be created. Optional, default: True
    :return: The extended pandapower network.
    """
    np_str_type = 'str'
    np_float_type = 'float'
    np_bool_type = 'bool'

    sc = get_pp_net_special_columns_dict()

    # all pandapower element types like bus, line, trafo will get the following special columns
    fill_dict_all: Dict[str, List[str]] = dict({})
    fill_dict_all[np_str_type] = [sc['o_id'], sc['o_cl']]

    # special elements
    fill_dict: Dict[str, Dict[str, List[str]]] = dict()

    fill_dict['bus'] = dict()
    fill_dict['bus'][np_str_type] = [sc['o_prf'], sc['ct'], sc['cnc_id'], sc['sub_id'], 'description', 'busbar_id',
                                     'busbar_name']

    fill_dict['ext_grid'] = dict()
    fill_dict['ext_grid'][np_str_type] = [sc['t'], sc['sub'], 'description']
    fill_dict['ext_grid'][np_float_type] = ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar', 'p_mw', 'q_mvar',
                                            's_sc_max_mva', 's_sc_min_mva', 'rx_max', 'rx_min', 'r0x0_max', 'x0x_max']

    fill_dict['load'] = dict()
    fill_dict['load'][np_str_type] = [sc['t'], 'description']
    fill_dict['gen'] = dict()
    fill_dict['gen'][np_str_type] = [sc['t'], 'description']
    fill_dict['gen'][np_float_type] = \
        ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar', 'vn_kv', 'rdss_ohm', 'xdss_pu', 'cos_phi', 'pg_percent']
    fill_dict['sgen'] = dict()
    fill_dict['sgen'][np_str_type] = [sc['t'], 'description']
    fill_dict['sgen'][np_float_type] = ['k', 'rx', 'vn_kv', 'rdss_ohm', 'xdss_pu', 'lrc_pu', 'generator_type']
    fill_dict['motor'] = dict()
    fill_dict['motor'][np_str_type] = [sc['t'], 'description']
    fill_dict['storage'] = dict()
    fill_dict['storage'][np_str_type] = [sc['t'], 'description']
    fill_dict['shunt'] = dict()
    fill_dict['shunt'][np_str_type] = [sc['t'], 'description']
    fill_dict['ward'] = dict()
    fill_dict['ward'][np_str_type] = [sc['t'], 'description']
    fill_dict['xward'] = dict()
    fill_dict['xward'][np_str_type] = [sc['t'], 'description']

    fill_dict['line'] = dict()
    fill_dict['line'][np_str_type] = [sc['t_from'], sc['t_to'], 'description']
    fill_dict['line'][np_float_type] = ['r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km', 'g0_us_per_km',
                                        'endtemp_degree']

    fill_dict['dcline'] = dict()
    fill_dict['dcline'][np_str_type] = [sc['t_from'], sc['t_to'], 'description']

    fill_dict['switch'] = dict()
    fill_dict['switch'][np_str_type] = [sc['t_bus'], sc['t_ele'], 'description']

    fill_dict['impedance'] = dict()
    fill_dict['impedance'][np_str_type] = [sc['t_from'], sc['t_to'], 'description']
    fill_dict['impedance'][np_float_type] = ['rft0_pu', 'xft0_pu', 'rtf0_pu', 'xtf0_pu']

    fill_dict['trafo'] = dict()
    fill_dict['trafo'][np_str_type] = [sc['t_hv'], sc['t_lv'], sc['pte_id_hv'], sc['pte_id_lv'], sc['tc'], sc['tc_id'],
                                       'description', 'vector_group', 'id_characteristic']
    fill_dict['trafo'][np_float_type] = ['vk0_percent', 'vkr0_percent', 'xn_ohm']
    fill_dict['trafo'][np_bool_type] = ['power_station_unit', 'oltc']

    fill_dict['trafo3w'] = dict()
    fill_dict['trafo3w'][np_str_type] = [sc['t_hv'], sc['t_mv'], sc['t_lv'], sc['pte_id_hv'], sc['pte_id_mv'],
                                         sc['pte_id_lv'], sc['tc'], sc['tc_id'], 'description', 'vector_group',
                                         'id_characteristic']
    fill_dict['trafo3w'][np_float_type] = ['vk0_hv_percent', 'vk0_mv_percent', 'vk0_lv_percent', 'vkr0_hv_percent',
                                           'vkr0_mv_percent', 'vkr0_lv_percent']
    fill_dict['trafo3w'][np_bool_type] = ['power_station_unit']

    for pp_type, one_fd in fill_dict.items():
        for np_type, fields in fill_dict_all.items():
            np_type = np.sctypeDict.get(np_type)
            for field in fields:
                if override or field not in net[pp_type].columns:
                    net[pp_type][field] = pd.Series([], dtype=np_type)
        for np_type, fields in one_fd.items():
            np_type = np.sctypeDict.get(np_type)
            for field in fields:
                if override or field not in net[pp_type].columns:
                    net[pp_type][field] = pd.Series([], dtype=np_type)

    # some special items
    if override:
        net['CGMES'] = dict()
        net['CGMES']['BaseVoltage'] = pd.DataFrame(None, columns=['rdfId', 'nominalVoltage'])
    else:
        if 'CGMES' not in net.keys():
            net['CGMES'] = dict()
        if 'BaseVoltage' not in net['CGMES'].keys():
            net['CGMES']['BaseVoltage'] = pd.DataFrame(None, columns=['rdfId', 'nominalVoltage'])

    return net


def get_cim16_schema() -> Dict[str, Dict[str, Dict[str, str or Dict[str, Dict[str, str]]]]]:
    """
    Parses the CIM 16 schema from the CIM 16 RDF schema files for the serializer from the CIM data structure used by
    the cim2pp and pp2cim converters.
    :return: The CIM 16 schema as dictionary.
    """
    path_with_serialized_schemas = os.path.dirname(__file__) + os.sep + 'serialized_schemas'
    if not os.path.isdir(path_with_serialized_schemas):
        os.mkdir(path_with_serialized_schemas)
    for one_file in os.listdir(path_with_serialized_schemas):
        if one_file.lower().endswith('_schema.json'):
            path_to_schema = path_with_serialized_schemas + os.sep + one_file
            logger.info("Parsing the schema from CIM 16 from disk: %s" % path_to_schema)
            with open(path_to_schema, encoding='UTF-8', mode='r') as f:
                cim16_schema = json.load(f)
            return cim16_schema

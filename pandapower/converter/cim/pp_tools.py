# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import time
from typing import Union, Dict, List

import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.create import create_bus, create_ext_grid, create_lines
from pandapower.network_structure import get_structure_dict
from pandapower.std_types import create_std_type
from . import cim_tools

logger = logging.getLogger(__name__)


def _set_column_to_type(input_df: pd.DataFrame, column: str, data_type: str) -> None:
    try:
        input_df[column] = input_df[column].astype(data_type)
    except Exception as e:
        logger.error("Couldn't set data type %s for column %s!" % (data_type, column))
        logger.exception(e)


def set_pp_col_types(net: Union[pandapowerNet, Dict], ignore_errors: bool = False) -> pandapowerNet:
    """
    Set the data types for some columns from pandapower assets. This mainly effects bus columns (to int, e.g.
    sgen.bus or line.from_bus) and in_service and other boolean columns (to bool, e.g. line.in_service or gen.slack).
    :param net: The pandapower network to update the data types.
    :param ignore_errors: Ignore problems if set to True (no warnings displayed). Optional, default: False.
    :return: The pandapower network with updated data types.
    """
    time_start = time.time()
    structure_dict = get_structure_dict()
    pp_elements = ['bus', 'dcline', 'ext_grid', 'gen', 'impedance', 'line', 'load', 'motor', 'sgen', 'shunt', 'storage',
                   'switch', 'trafo', 'trafo3w', 'ward', 'xward']
    columns = ['bus', 'element', 'to_bus', 'from_bus', 'hv_bus', 'mv_bus', 'lv_bus', 'in_service', 'controllable',
               'parallel']
    logger.info("Setting the columns data types for buses to int and in_service to bool for the following elements: "
                "%s" % pp_elements)
    for ele in pp_elements:
        logger.info("Accessing pandapower element %s." % ele)
        if not hasattr(net, ele):
            if not ignore_errors:
                logger.warning("Missing the pandapower element %s in the input pandapower network!" % ele)
            continue
        for column in columns:
            if column in net[ele].columns:
                _set_column_to_type(net[ele], column, structure_dict[ele][column])
    # some individual things
    if hasattr(net, 'switch'):
        _set_column_to_type(net['switch'], 'closed', structure_dict['switch']['closed'])
    if hasattr(net, 'trafo'):
        _set_column_to_type(net['trafo'], 'id_characteristic_table',
                            structure_dict['trafo']['id_characteristic_table'])
    if hasattr(net, 'trafo3w'):
        _set_column_to_type(net['trafo3w'], 'id_characteristic_table',
                            structure_dict['trafo3w']['id_characteristic_table'])
    if hasattr(net, 'sgen'):
        _set_column_to_type(net['sgen'], 'current_source', structure_dict['sgen']['current_source'])
        _set_column_to_type(net['sgen'], 'id_q_capability_characteristic',
                            structure_dict['sgen']['id_q_capability_characteristic'])
    if hasattr(net, 'gen'):
        _set_column_to_type(net['gen'], 'slack', structure_dict['gen']['slack'])
        _set_column_to_type(net['gen'], 'id_q_capability_characteristic',
                            structure_dict['gen']['id_q_capability_characteristic'])
    if hasattr(net, 'shunt'):
        _set_column_to_type(net['shunt'], 'step', structure_dict['shunt']['step'])
        _set_column_to_type(net['shunt'], 'max_step', structure_dict['shunt']['max_step'])
        _set_column_to_type(net['shunt'], 'id_characteristic_table',
                            structure_dict['shunt']['id_characteristic_table'])
    logger.info("Finished setting the data types for the pandapower network in %ss." % (time.time() - time_start))
    return net


def add_slack_and_lines_to_boundary_nodes(net: pandapowerNet, voltage_levels: List[int] = None):
    """
    Add lines with low impedance and a slack to the boundary nodes with the highest voltage.
    :param net: The pandapower network
    :param voltage_levels: The voltage levels to add lines and slacks. For each given voltage level, lines and one
    slack will be connected to the corresponding boundary nodes. Optional, default: Highest voltage level from
    boundary nodes.
    :return:
    """
    sc = cim_tools.get_pp_net_special_columns_dict()
    busses = net.bus.loc[(net.bus[sc['o_prf']] == 'eq_bd') | (net.bus[sc['o_prf']] == 'tp_bd')]
    if voltage_levels is None:
        max_voltage = busses['vn_kv'].max()
        logger.info("Highest voltage level: %skV" % max_voltage)
        voltage_levels = [max_voltage]
    create_std_type(net, data={'r_ohm_per_km': 0, 'x_ohm_per_km': .05, 'c_nf_per_km': 0, 'max_i_ka': 9999},
                    name='low_impedance_line', element='line')
    for one_voltage_level in voltage_levels:
        logger.info("Processing voltage level %skV" % one_voltage_level)
        busses_t = busses.loc[busses['vn_kv'] == one_voltage_level]
        new_bus_id = create_bus(net, vn_kv=one_voltage_level,
                                name='virtual slack bus at voltage level ' + str(one_voltage_level))
        logger.info("Added virtual slack bus with ID: %s" % new_bus_id)
        create_ext_grid(net, bus=new_bus_id, vm_pu=1.0,
                        name='virtual slack at voltage level ' + str(one_voltage_level))
        logger.info("Added slack at bus ID: %s" % new_bus_id)
        new_bus_id_array = [new_bus_id for _ in busses_t.index.values]
        create_lines(net, from_buses=busses_t.index.values, to_buses=new_bus_id_array, std_type='low_impedance_line',
                     name='virtual line to slack node with voltage level ' + str(one_voltage_level), length_km=1)
        logger.info("Created %s low impedance lines." % len(busses_t.index.values))
        del busses_t


def get_not_existing_column(df: pd.DataFrame) -> str:
    """
    Get a not existing column name in a DataFrame
    :param df: The DataFrame where to get a not existing column
    :return: The name of a not existing column, e.g. temp_col_0 or temp_col_1
    """
    i = 0
    col = 'temp_col_'
    while col + str(i) in df.columns:
        i += 1
    return col + str(i)

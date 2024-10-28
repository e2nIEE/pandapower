# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import time
from typing import Dict, Union
import numpy as np
from pandapower.create import create_empty_network
from pandapower.auxiliary import pandapowerNet
import pandas as pd


class UCTE2pandapower:

    def __init__(self):
        """
        Convert UCTE data to pandapower.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.u_d = dict()
        self.net: pandapowerNet = create_empty_network()

    def convert(self, ucte_dict: Dict) -> pandapowerNet:
        self.logger.info("Converting UCTE data to a pandapower network.")
        time_start = time.time()
        self.net: pandapowerNet = create_empty_network()
        # create a temporary copy from the origin input data
        self.u_d = dict()
        for ucte_element, items in ucte_dict.items():
            self.u_d[ucte_element] = items.copy()
        # first reset the index to get indices for pandapower
        for ucte_element, df in self.u_d.items():
            if ucte_element == 'R':
                continue
            df.reset_index(level=0, inplace=True)
            df.rename(columns={'index': 'id'}, inplace=True)
        # now replace the node1 and node2 columns with the node index at lines, transformers, ...
        merge_nodes = self.u_d['N'][['id', 'node']]
        self.u_d['L'] = pd.merge(
            self.u_d['L'], merge_nodes.rename(columns={'node': 'node1', 'id': 'from_bus'}),
            how='left', on='node1')
        self.u_d['L'] = pd.merge(
            self.u_d['L'], merge_nodes.rename(columns={'node': 'node2', 'id': 'to_bus'}),
            how='left', on='node2')
        self.u_d['L'].drop(columns=['node1', 'node2'], inplace=True)
        for one_asset in ['T', 'R', 'TT']:
            self.u_d[one_asset] = pd.merge(
                self.u_d[one_asset], merge_nodes.rename(columns={'node': 'node1', 'id': 'hv_bus'}),
                how='left', on='node1')
            self.u_d[one_asset] = pd.merge(
                self.u_d[one_asset], merge_nodes.rename(columns={'node': 'node2', 'id': 'lv_bus'}),
                how='left', on='node2')
            self.u_d[one_asset].drop(columns=['node1', 'node2'], inplace=True)
        # prepare the nodes
        self._convert_nodes()
        # prepare the loads
        self._convert_loads()
        # prepare the generators
        self._convert_gens()
        # prepare the lines
        self._convert_lines()
        # prepare the impedances
        self._convert_impedances()
        # prepare the transformers
        self._convert_trafos()
        self.net = self.set_pp_col_types(self.net)
        self.logger.info("Finished converting the input data to pandapower in %ss." % (
            time.time() - time_start))
        return self.net

    def _copy_to_pp(self, pp_type: str, input_df: pd.DataFrame):
        self.logger.debug("Copy %s datasets to pandapower network with type %s" % (
            input_df.index.size, pp_type))
        if pp_type not in self.net.keys():
            self.logger.warning("Missing pandapower type %s in the pandapower network!" % pp_type)
            return
        start_index_pp_net = self.net[pp_type].index.size
        self.net[pp_type] = pd.concat([
                self.net[pp_type],
                pd.DataFrame(None, index=[list(range(input_df.index.size))])],
            ignore_index=True, sort=False)
        for one_attr in self.net[pp_type].columns:
            if one_attr in input_df.columns:
                self.net[pp_type][one_attr][start_index_pp_net:] = input_df[one_attr][:]

    def _convert_nodes(self):
        self.logger.info("Converting the nodes.")
        nodes = self.u_d['N']  # Note: Do not use a copy, the columns 'ur_kv' and 'node_geo' are needed later
        nodes['volt_str'] = nodes['node'].str[6:7]
        volt_map = {'0': 750, '1': 380, '2': 220, '3': 150, '4': 120, '5': 110, '6': 70, '7': 27, '8': 330, '9': 500,
                    'A': 26, 'B': 25, 'C': 24, 'D': 23, 'E': 22, 'F': 21, 'G': 20, 'H': 19, 'I': 18, 'J': 17, 'K': 15.7,
                    'L': 15, 'M': 13.7, 'N': 13, 'O': 12, 'P': 11, 'Q': 9.8, 'R': 9, 'S': 8, 'T': 7, 'U': 6, 'V': 5,
                    'W': 4, 'X': 3, 'Y': 2, 'Z': 1}
        nodes['vn_kv'] = nodes['volt_str'].map(volt_map)
        # make sure that 'node_geo' has a valid value
        nodes.loc[nodes['node_geo'] == '', 'node_geo'] = nodes.loc[nodes['node_geo'] == '',
                                                                   'node'].str[:6]
        nodes['node2'] = nodes['node'].str[:6]
        nodes['grid_area_id'] = nodes['node'].str[:2]
        # drop all voltages at non pu nodes
        nodes['voltage'].loc[(nodes['node_type'] != 2) & (nodes['node_type'] != 3)] = np.nan
        nodes.rename(columns={'node': 'name'}, inplace=True)
        nodes['in_service'] = True
        self._copy_to_pp('bus', nodes)
        self.logger.info("Finished converting the nodes.")

    def _convert_loads(self):
        self.logger.info("Converting the loads.")
        # select the loads from the nodes and drop not given values
        loads = self.u_d['N'].dropna(subset=['p_load', 'q_load'])
        # select all with p != 0 or q != 0
        loads = loads.loc[(loads['p_load'] != 0) | (loads['q_load'] != 0)]
        loads.rename(columns={'id': 'bus', 'node': 'name', 'p_load': 'p_mw', 'q_load': 'q_mvar'},
                     inplace=True)
        # get a new index
        loads.reset_index(level=0, inplace=True, drop=True)
        loads['scaling'] = 1
        loads['in_service'] = True
        self._copy_to_pp('load', loads)
        self.logger.info("Finished converting the loads.")

    def _convert_gens(self):
        self.logger.info("Converting the generators.")
        # select the gens from the nodes and drop not given values
        gens = self.u_d['N'].dropna(subset=['p_gen', 'q_gen'])
        # select all with p != 0 or q != 0 or voltage != 0
        gens = gens.loc[(gens['p_gen'] != 0) | (gens['q_gen'] != 0) | (gens['voltage'] > 0)]
        # change the signing
        gens['p_gen'] = gens['p_gen'] * -1
        gens['q_gen'] = gens['q_gen'] * -1
        gens['min_p_gen'] = gens['min_p_gen'] * -1
        gens['max_p_gen'] = gens['max_p_gen'] * -1
        gens['min_q_gen'] = gens['min_q_gen'] * -1
        gens['max_q_gen'] = gens['max_q_gen'] * -1
        # drop all voltages at non pu nodes
        gens['voltage'].loc[(gens['node_type'] != 2) & (gens['node_type'] != 3)] = np.nan
        gens['vm_pu'] = gens['voltage'] / gens['vn_kv']
        gens.rename(columns={
            'id': 'bus', 'node': 'name', 'p_gen': 'p_mw', 'q_gen': 'q_mvar',
            'min_p_gen': 'min_p_mw', 'max_p_gen': 'max_p_mw', 'min_q_gen': 'min_q_mvar',
            'max_q_gen': 'max_q_mvar'}, inplace=True)
        # get a new index
        gens.reset_index(level=0, inplace=True, drop=True)
        gens['scaling'] = 1
        gens['va_degree'] = 0
        gens['slack_weight'] = 1
        gens['slack'] = False
        gens['current_source'] = True
        gens['in_service'] = True
        self._copy_to_pp('ext_grid', gens.loc[gens['node_type'] == 3])
        self._copy_to_pp('gen', gens.loc[gens['node_type'] == 2])
        self._copy_to_pp('sgen', gens.loc[(gens['node_type'] == 0) | (gens['node_type'] == 1)])
        self.logger.info("Finished converting the generators.")

    def _convert_lines(self):
        self.logger.info("Converting the lines.")
        # get the lines
        # status 9 & 1 stands for equivalent line that can be interpreted as impedance
        lines = self.u_d['L'].loc[self.u_d['L'].status.isin([9, 1]) == False, :]
        # lines = self.u_d['L']
        # create the in_service column from the UCTE status
        in_service_map = dict({0: True, 1: True, 2: True, 7: False, 8: False, 9: False})
        lines['in_service'] = lines['status'].map(in_service_map)
        status_map = dict({0: 0, 1: 1, 2: 2, 7: 2, 8: 0, 9: 1})
        lines['status'] = lines['status'].map(status_map)
        # i in A to i in kA
        lines['max_i_ka'] = lines['i'] / 1e3
        lines['c_nf_per_km'] = 1e3 * lines['b'] / (2 * np.pi * 50)
        lines['g_us_per_km'] = 0
        lines['df'] = 1
        lines['parallel'] = 1
        lines['length_km'] = 1
        # rename the columns to the pandapower schema
        lines.rename(columns={'r': 'r_ohm_per_km', 'x': 'x_ohm_per_km', 'name': 'name'},
                     inplace=True)
        self._copy_to_pp('line', lines)
        self.logger.info("Finished converting the lines.")

    def _convert_impedances(self):
        self.logger.info("Converting the impedances.")
        # get the impedances
        # status 9 & 1 stands for equivalent line that can be interpreted as impedance
        impedances = self.u_d['L'].loc[self.u_d['L'].status.isin([9, 1]), :]
        # create the in_service column from the UCTE status
        in_service_map = dict({0: True, 1: True, 2: True, 7: False, 8: False, 9: False})
        impedances['in_service'] = impedances['status'].map(in_service_map)
        impedances['z_ohm'] = (impedances['r'] ** 2 + impedances['x'] ** 2) ** .5
        self._set_column_to_type(impedances, 'from_bus', int)
        impedances = pd.merge(impedances, self.u_d['N'][['vn_kv']],
                              how='left', left_on='from_bus', right_index=True)
        # rename the columns to the pandapower schema
        impedances.rename(columns={'name': 'name'}, inplace=True)
        impedances['sn_mva'] = impedances['vn_kv'] ** 2 / impedances['z_ohm']
        # relative values
        impedances['rft_pu'] = impedances['r'] / impedances['z_ohm']
        impedances['rtf_pu'] = impedances['r'] / impedances['z_ohm']
        impedances['xft_pu'] = impedances['x'] / impedances['z_ohm']
        impedances['xtf_pu'] = impedances['x'] / impedances['z_ohm']
        self._copy_to_pp('impedance', impedances)
        self.logger.info("Finished converting the impedances.")

    def _convert_trafos(self):
        self.logger.info("Converting the transformers.")
        trafos = pd.merge(self.u_d['T'], self.u_d['R'], how='left', on=[
            'hv_bus', 'lv_bus', 'order_code'])
        # create the in_service column from the UCTE status
        status_map = dict({0: True, 1: True, 8: False, 9: False})
        trafos['in_service'] = trafos['status'].map(status_map)
        # calculate the derating factor
        trafos['df'] = trafos['voltage1'] * (trafos['i'] / 1e3) * 3**.5 / trafos['p']
        # calculate the relative short-circuit voltage
        trafos['vk_percent'] = (abs(trafos.r) ** 2 + abs(trafos.x) ** 2) ** 0.5 * \
                               (trafos.p * 1e3) / (10. * trafos.voltage1 ** 2)
        # calculate vkr_percent
        trafos['vkr_percent'] = abs(trafos.r) * trafos.p * 100 / trafos.voltage1 ** 2
        # calculate iron losses in kW
        trafos['pfe_kw'] = trafos.g * trafos.voltage1 ** 2 / 1e3
        # calculate open loop losses in percent of rated current
        trafos['i0_percent'] = \
            (((trafos.b * 1e-6 * trafos.voltage1 ** 2) ** 2 +
              (trafos.g * 1e-6 * trafos.voltage1 ** 2) ** 2) ** .5) * 100 / trafos.p
        # fill the phase regulated tap changer with the angle regulated ones
        trafos.phase_reg_delta_u.fillna(trafos.angle_reg_delta_u, inplace=True)
        trafos.phase_reg_n.fillna(trafos.angle_reg_n, inplace=True)
        trafos.phase_reg_n2.fillna(trafos.angle_reg_n2, inplace=True)
        trafos['tap_min'] = -trafos.phase_reg_n
        # set the hv and lv voltage sides to voltage1 and voltage2 (The non-regulated transformer side is currently
        # voltage1, not the hv side!)
        trafos['vn_hv_kv'] = trafos[['voltage1', 'voltage2']].max(axis=1)
        trafos['vn_lv_kv'] = trafos[['voltage1', 'voltage2']].min(axis=1)
        # swap the 'fid_node_start' and 'fid_node_end' if need
        trafos['swap'] = trafos['vn_hv_kv'] != trafos['voltage1']
        # copy the 'fid_node_start' and 'fid_node_end'
        trafos['hv_bus2'] = trafos['hv_bus'].copy()
        trafos['lv_bus2'] = trafos['lv_bus'].copy()
        trafos['hv_bus'].loc[trafos.swap] = trafos['lv_bus2'].loc[trafos.swap]
        trafos['lv_bus'].loc[trafos.swap] = trafos['hv_bus2'].loc[trafos.swap]
        # set the tap side, default is lv Correct it for other windings
        trafos['tap_side'] = 'lv'
        trafos['tap_side'].loc[trafos.swap] = 'hv'
        # now set it to nan for not existing tap changers
        trafos['tap_side'].loc[trafos.phase_reg_n.isnull()] = None
        trafos['tap_neutral'] = 0
        trafos['tap_neutral'].loc[trafos.phase_reg_n.isnull()] = np.nan
        trafos['shift_degree'] = 0
        trafos['parallel'] = 1
        trafos['tap_phase_shifter'] = False
        # rename the columns to the pandapower schema
        trafos.rename(columns={
            'p': 'sn_mva', 'phase_reg_n2': 'tap_pos', 'phase_reg_delta_u': 'tap_step_percent',
            'angle_reg_theta': 'tap_step_degree', 'phase_reg_n': 'tap_max'},
            inplace=True)
        self._copy_to_pp('trafo', trafos)
        self.logger.info("Finished converting the transformers.")

    def _set_column_to_type(self, input_df: pd.DataFrame, column: str, data_type):
        try:
            input_df[column] = input_df[column].astype(data_type)
        except Exception as e:
            self.logger.error("Couldn't set data type %s for column %s!" % (data_type, column))
            self.logger.exception(e)

    def set_pp_col_types(self, net: Union[pandapowerNet, Dict], ignore_errors: bool = False) -> \
            pandapowerNet:
        """
        Set the data types for some columns from pandapower assets. This mainly effects bus columns (to int, e.g.
        sgen.bus or line.from_bus) and in_service and other boolean columns (to bool, e.g. line.in_service or gen.slack).
        :param net: The pandapower network to update the data types.
        :param ignore_errors: Ignore problems if set to True (no warnings displayed). Optional, default: False.
        :return: The pandapower network with updated data types.
        """
        time_start = time.time()
        pp_elements = [
            'bus', 'dcline', 'ext_grid', 'gen', 'impedance', 'line', 'load', 'sgen', 'shunt',
            'storage', 'switch', 'trafo', 'trafo3w', 'ward', 'xward']
        to_int = ['bus', 'element', 'to_bus', 'from_bus', 'hv_bus', 'mv_bus', 'lv_bus']
        to_bool = ['in_service', 'closed', 'tap_phase_shifter']
        self.logger.info(
            "Setting the columns data types for buses to int and in_service to bool for the "
            "following elements: %s" % pp_elements)
        int_type = int
        bool_type = bool
        for ele in pp_elements:
            self.logger.info("Accessing pandapower element %s." % ele)
            if not hasattr(net, ele):
                if not ignore_errors:
                    self.logger.warning(
                        "Missing the pandapower element %s in the input pandapower network!" % ele)
                continue
            for one_int in to_int:
                if one_int in net[ele].columns:
                    self._set_column_to_type(net[ele], one_int, int_type)
            for one_bool in to_bool:
                if one_bool in net[ele].columns:
                    self._set_column_to_type(net[ele], one_bool, bool_type)
        # some individual things
        if hasattr(net, 'sgen'):
            self._set_column_to_type(net['sgen'], 'current_source', bool_type)
        if hasattr(net, 'gen'):
            self._set_column_to_type(net['gen'], 'slack', bool_type)
        if hasattr(net, 'shunt'):
            self._set_column_to_type(net['shunt'], 'step', int_type)
            self._set_column_to_type(net['shunt'], 'max_step', int_type)
        self.logger.info("Finished setting the data types for the pandapower network in %ss." %
                         (time.time() - time_start))
        return net

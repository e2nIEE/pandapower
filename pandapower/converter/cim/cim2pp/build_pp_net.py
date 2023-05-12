# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import math
from typing import Dict, Tuple, List
import traceback
import pandapower as pp
import pandapower.auxiliary
from pandapower.control.controller.trafo.ContinuousTapControl import ContinuousTapControl
from pandapower.control.controller.trafo.DiscreteTapControl import DiscreteTapControl
import pandas as pd
import numpy as np
import time
from .convert_measurements import CreateMeasurements
from .. import cim_tools
from .. import pp_tools
from .. import cim_classes
from ..other_classes import ReportContainer, Report, LogLevel, ReportCode
logger = logging.getLogger('cim.cim2pp.build_pp_net')

pd.set_option('display.max_columns', 900)
pd.set_option('display.max_rows', 90000)
sc = cim_tools.get_pp_net_special_columns_dict()


class CimConverter:

    def __init__(self, cim_parser: cim_classes.CimParser, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cim_parser: cim_classes.CimParser = cim_parser
        self.kwargs = kwargs
        self.cim: Dict[str, Dict[str, pd.DataFrame]] = self.cim_parser.get_cim_dict()
        self.net: pandapower.auxiliary.pandapowerNet = pp.create_empty_network()
        self.bus_merge: pd.DataFrame = pd.DataFrame()
        self.power_trafo2w: pd.DataFrame = pd.DataFrame()
        self.power_trafo3w: pd.DataFrame = pd.DataFrame()
        self.report_container: ReportContainer = cim_parser.get_report_container()

    def _merge_eq_ssh_profile(self, cim_type: str, add_cim_type_column: bool = False) -> pd.DataFrame:
        df = pd.merge(self.cim['eq'][cim_type], self.cim['ssh'][cim_type], how='left', on='rdfId')
        if add_cim_type_column:
            df[sc['o_cl']] = cim_type
        return df

    def _convert_connectivity_nodes_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting ConnectivityNodes / TopologicalNodes.")
        connectivity_nodes, eqssh_terminals = self._prepare_connectivity_nodes_cim16()

        # self._create_busses(connectivity_nodes)
        self._copy_to_pp('bus', connectivity_nodes)

        # a prepared and modified copy of eqssh_terminals to use for lines, switches, loads, sgens and so on
        eqssh_terminals = eqssh_terminals[
            ['rdfId', 'ConductingEquipment', 'ConnectivityNode', 'sequenceNumber', 'connected']].copy()
        eqssh_terminals.rename(columns={'rdfId': 'rdfId_Terminal'}, inplace=True)
        eqssh_terminals.rename(columns={'ConductingEquipment': 'rdfId'}, inplace=True)
        # buses for merging with assets:
        bus_merge = pd.DataFrame(data=self.net['bus'].loc[:, [sc['o_id'], 'vn_kv']])
        bus_merge.rename(columns={'vn_kv': 'base_voltage_bus'}, inplace=True)
        bus_merge.reset_index(level=0, inplace=True)
        bus_merge.rename(columns={'index': 'index_bus', sc['o_id']: 'ConnectivityNode'}, inplace=True)
        bus_merge = pd.merge(eqssh_terminals, bus_merge, how='left', on='ConnectivityNode')
        self.bus_merge = bus_merge

        self.logger.info("Created %s busses in %ss" % (connectivity_nodes.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s busses from ConnectivityNodes / TopologicalNodes in %ss" %
                    (connectivity_nodes.index.size, time.time() - time_start)))

    def _prepare_connectivity_nodes_cim16(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # check the model: Bus-Branch or Node-Breaker: In the Bus-Branch model are no ConnectivityNodes
        node_breaker = True if self.cim['eq']['ConnectivityNode'].index.size > 0 else False
        # use this dictionary to store the source profile from the element (normal or boundary profile)
        cn_dict = dict({'eq': {sc['o_prf']: 'eq'}, 'eq_bd': {sc['o_prf']: 'eq_bd'},
                        'tp': {sc['o_prf']: 'tp'}, 'tp_bd': {sc['o_prf']: 'tp_bd'}})
        if node_breaker:
            # Node-Breaker model
            connectivity_nodes = pd.concat([self.cim['eq']['ConnectivityNode'].assign(**cn_dict['eq']),
                                            self.cim['eq_bd']['ConnectivityNode'].assign(**cn_dict['eq_bd'])],
                                           ignore_index=True, sort=False)
            connectivity_nodes[sc['o_cl']] = 'ConnectivityNode'
            connectivity_nodes[sc['cnc_id']] = connectivity_nodes['ConnectivityNodeContainer'][:]
            # the buses are modeled as ConnectivityNode(s), but the voltage is stored as a BaseVoltage
            # to get the BaseVoltage:
            # 1: ConnectivityNode -> [ConnectivityNodeContainer] VoltageLevel -> [BaseVoltage] BaseVoltage
            # 2: ConnectivityNode -> [ConnectivityNodeContainer] Bay -> [VoltageLevel] VoltageLevel ->
            # [BaseVoltage] BaseVoltage
            # 3: ConnectivityNode -> [ConnectivityNodeContainer] Substation ||| VoltageLevel -> [Substation] Substation
            # 4: ConnectivityNodes from the boundary profile have the BaseVoltage in their TopologicalNode
            # The idea is to add the Bays and Substations to the VoltageLevels to handle them similar to (1)
            # prepare the bays (2)
            eq_bay = self.cim['eq']['Bay'].copy()
            eq_bay.rename(columns={'rdfId': 'ConnectivityNodeContainer'}, inplace=True)
            eq_bay = pd.merge(self.cim['eq']['ConnectivityNode'][['ConnectivityNodeContainer']], eq_bay,
                              how='inner', on='ConnectivityNodeContainer')
            eq_bay.dropna(subset=['VoltageLevel'], inplace=True)
            eq_bay = pd.merge(eq_bay, self.cim['eq']['VoltageLevel'][['rdfId', 'BaseVoltage', 'Substation']],
                              how='left', left_on='VoltageLevel', right_on='rdfId')
            eq_bay.drop(columns=['VoltageLevel', 'rdfId'], inplace=True)
            eq_bay.rename(columns={'ConnectivityNodeContainer': 'rdfId'}, inplace=True)
            # now prepare the substations (3)
            # first get only the needed substation used as ConnectivityNodeContainer in ConnectivityNode
            eq_subs = pd.merge(self.cim['eq']['ConnectivityNode'][['ConnectivityNodeContainer']].rename(
                columns={'ConnectivityNodeContainer': 'Substation'}),
                               self.cim['eq']['Substation'][['rdfId']].rename(columns={'rdfId': 'Substation'}),
                               how='inner', on='Substation')
            # now merge them with the VoltageLevel
            eq_subs = pd.merge(self.cim['eq']['VoltageLevel'][['rdfId', 'BaseVoltage', 'Substation']], eq_subs,
                               how='inner', on='Substation')
            eq_subs_duplicates = eq_subs[eq_subs.duplicated(['Substation'], keep='first')]
            eq_subs['rdfId'] = eq_subs['Substation']
            if eq_subs_duplicates.index.size > 0:
                self.logger.warning(
                    "More than one VoltageLevel refers to one Substation, maybe the voltages from some buses "
                    "are incorrect, the problematic VoltageLevels and Substations:\n%s" % eq_subs_duplicates)
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="More than one VoltageLevel refers to one Substation, maybe the voltages from some buses "
                            "are incorrect, the problematic VoltageLevels and Substations:\n%s" % eq_subs_duplicates))
            eq_subs.drop_duplicates(['rdfId'], keep='first', inplace=True)
            # now merge the VoltageLevel with the ConnectivityNode
            eq_voltage_levels = self.cim['eq']['VoltageLevel'][['rdfId', 'BaseVoltage', 'Substation']]
            eq_voltage_levels = pd.concat([eq_voltage_levels, eq_bay], ignore_index=True, sort=False)
            eq_voltage_levels = pd.concat([eq_voltage_levels, eq_subs], ignore_index=True, sort=False)
            eq_voltage_levels.drop_duplicates(['rdfId'], keep='first', inplace=True)
            del eq_bay, eq_subs, eq_subs_duplicates
            eq_substations = self.cim['eq']['Substation'][['rdfId', 'name']]
            eq_substations.rename(columns={'rdfId': 'Substation', 'name': 'name_substation'}, inplace=True)
            eq_voltage_levels = pd.merge(eq_voltage_levels, eq_substations, how='left', on='Substation')
            eq_voltage_levels.drop_duplicates(subset=['rdfId'], inplace=True)
            eq_voltage_levels.rename(columns={'rdfId': 'ConnectivityNodeContainer'}, inplace=True)

            connectivity_nodes = pd.merge(connectivity_nodes, eq_voltage_levels, how='left',
                                          on='ConnectivityNodeContainer')
            connectivity_nodes[sc['sub_id']] = connectivity_nodes['Substation'][:]
            # now prepare the BaseVoltage from the boundary profile at the ConnectivityNode (4)
            eq_bd_cns = pd.merge(self.cim['eq_bd']['ConnectivityNode'][['rdfId']],
                                 self.cim['tp_bd']['ConnectivityNode'][['rdfId', 'TopologicalNode']],
                                 how='inner', on='rdfId')
            # eq_bd_cns.drop(columns=['rdfId'], inplace=True)
            # eq_bd_cns.rename(columns={'TopologicalNode': 'rdfId'}, inplace=True)
            eq_bd_cns = pd.merge(eq_bd_cns, self.cim['tp_bd']['TopologicalNode'][['rdfId', 'BaseVoltage']].rename(
                columns={'rdfId': 'TopologicalNode'}), how='inner', on='TopologicalNode')
            # eq_bd_cns.drop(columns=['TopologicalNode'], inplace=True)
            eq_bd_cns.rename(columns={'BaseVoltage': 'BaseVoltage_2', 'TopologicalNode': 'TopologicalNode_2'},
                             inplace=True)
            connectivity_nodes = pd.merge(connectivity_nodes, eq_bd_cns, how='left', on='rdfId')
            connectivity_nodes['BaseVoltage'].fillna(connectivity_nodes['BaseVoltage_2'], inplace=True)
            connectivity_nodes.drop(columns=['BaseVoltage_2'], inplace=True)
            # check if there is a mix between BB and NB models
            terminals_temp = \
                self.cim['eq']['Terminal'].loc[self.cim['eq']['Terminal']['ConnectivityNode'].isna(), 'rdfId']
            if terminals_temp.index.size > 0:
                terminals_temp = pd.merge(terminals_temp, self.cim['tp']['Terminal'][['rdfId', 'TopologicalNode']],
                                          how='left', on='rdfId')
                terminals_temp.drop(columns=['rdfId'], inplace=True)
                terminals_temp.rename(columns={'TopologicalNode': 'rdfId'}, inplace=True)
                terminals_temp.drop_duplicates(subset=['rdfId'], inplace=True)
                tp_temp = self.cim['tp']['TopologicalNode'][['rdfId', 'name', 'description', 'BaseVoltage']]
                tp_temp[sc['o_prf']] = 'tp'
                tp_temp = pd.concat([tp_temp, self.cim['tp_bd']['TopologicalNode'][['rdfId', 'name', 'BaseVoltage']]],
                                    sort=False)
                tp_temp[sc['o_prf']].fillna('tp_bd', inplace=True)
                tp_temp[sc['o_cl']] = 'TopologicalNode'
                tp_temp = pd.merge(terminals_temp, tp_temp, how='inner', on='rdfId')
                connectivity_nodes = pd.concat([connectivity_nodes, tp_temp], ignore_index=True, sort=False)
        else:
            # Bus-Branch model
            # concat the TopologicalNodes from the tp and boundary profile and keep the source profile for each element
            # as column using the pandas assign method
            connectivity_nodes = pd.concat([self.cim['tp']['TopologicalNode'].assign(**cn_dict['tp']),
                                            self.cim['tp_bd']['TopologicalNode'].assign(**cn_dict['tp_bd'])],
                                           ignore_index=True, sort=False)
            connectivity_nodes[sc['o_cl']] = 'TopologicalNode'
            connectivity_nodes['name_substation'] = ''
        # prepare the voltages from the buses
        eq_base_voltages = pd.concat([self.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']],
                                      self.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']]],
                                     ignore_index=True, sort=False)
        eq_base_voltages.drop_duplicates(subset=['rdfId'], inplace=True)
        eq_base_voltages.rename(columns={'rdfId': 'BaseVoltage'}, inplace=True)
        # make sure that the BaseVoltage has string datatype
        connectivity_nodes['BaseVoltage'] = connectivity_nodes['BaseVoltage'].astype(str)
        connectivity_nodes = pd.merge(connectivity_nodes, eq_base_voltages, how='left', on='BaseVoltage')
        connectivity_nodes.drop(columns=['BaseVoltage'], inplace=True)
        eqssh_terminals = self.cim['eq']['Terminal'][['rdfId', 'ConnectivityNode', 'ConductingEquipment',
                                                      'sequenceNumber']]
        eqssh_terminals = \
            pd.concat([eqssh_terminals, self.cim['eq_bd']['Terminal'][['rdfId', 'ConductingEquipment',
                                                                       'ConnectivityNode', 'sequenceNumber']]],
                      ignore_index=True, sort=False)
        eqssh_terminals = pd.merge(eqssh_terminals, self.cim['ssh']['Terminal'], how='left', on='rdfId')
        eqssh_terminals = pd.merge(eqssh_terminals, self.cim['tp']['Terminal'], how='left', on='rdfId')
        eqssh_terminals['ConnectivityNode'].fillna(eqssh_terminals['TopologicalNode'], inplace=True)
        # concat the DC terminals
        dc_terminals = pd.merge(pd.concat([self.cim['eq']['DCTerminal'], self.cim['eq']['ACDCConverterDCTerminal']],
                                          ignore_index=True, sort=False),
                                pd.concat([self.cim['ssh']['DCTerminal'], self.cim['ssh']['ACDCConverterDCTerminal']],
                                          ignore_index=True, sort=False), how='left', on='rdfId')
        dc_terminals = pd.merge(dc_terminals,
                                pd.concat([self.cim['tp']['DCTerminal'], self.cim['tp']['ACDCConverterDCTerminal']],
                                          ignore_index=True, sort=False), how='left', on='rdfId')
        dc_terminals.rename(columns={'DCNode': 'ConnectivityNode', 'DCConductingEquipment': 'ConductingEquipment',
                                     'DCTopologicalNode': 'TopologicalNode'}, inplace=True)
        eqssh_terminals = pd.concat([eqssh_terminals, dc_terminals], ignore_index=True, sort=False)
        # special fix for concat tp profiles
        eqssh_terminals.drop_duplicates(subset=['rdfId', 'TopologicalNode'], inplace=True)
        eqssh_terminals_temp = eqssh_terminals[['ConnectivityNode', 'TopologicalNode']]
        eqssh_terminals_temp.dropna(subset=['TopologicalNode'], inplace=True)
        eqssh_terminals_temp.drop_duplicates(inplace=True)
        connectivity_nodes_size = connectivity_nodes.index.size
        if node_breaker:
            connectivity_nodes = pd.merge(connectivity_nodes, eqssh_terminals_temp, how='left', left_on='rdfId',
                                          right_on='ConnectivityNode')
        else:
            connectivity_nodes = pd.merge(connectivity_nodes, eqssh_terminals_temp, how='left', left_on='rdfId',
                                          right_on='TopologicalNode')
            eqssh_terminals['ConnectivityNode'] = eqssh_terminals['TopologicalNode'].copy()
        # fill the column TopologicalNode for the ConnectivityNodes from the eq_bd profile if exists
        if 'TopologicalNode_2' in connectivity_nodes.columns:
            connectivity_nodes['TopologicalNode'].fillna(connectivity_nodes['TopologicalNode_2'], inplace=True)
            connectivity_nodes.drop(columns=['TopologicalNode_2'], inplace=True)
        if connectivity_nodes.index.size != connectivity_nodes_size:
            self.logger.warning("There is a problem at the busses!")
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="There is a problem at the busses!"))
            dups = connectivity_nodes.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 1]
            for rdfId, count in dups.items():
                self.logger.warning("The ConnectivityNode with RDF ID %s has %s TopologicalNodes!" % (rdfId, count))
                self.logger.warning("The ConnectivityNode data: \n%s" %
                                    connectivity_nodes[connectivity_nodes['rdfId'] == rdfId])
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The ConnectivityNode with RDF ID %s has %s TopologicalNodes!" % (rdfId, count)))
            # raise ValueError("The number of ConnectivityNodes increased after merging with Terminals, number of "
            #                  "ConnectivityNodes before merge: %s, number of ConnectivityNodes after merge: %s" %
            #                  (connectivity_nodes_size, connectivity_nodes.index.size))
            connectivity_nodes.drop_duplicates(subset=['rdfId'], keep='first', inplace=True)
        connectivity_nodes.rename(columns={'rdfId': sc['o_id'], 'TopologicalNode': sc['ct'], 'nominalVoltage': 'vn_kv',
                                           'name_substation': 'zone'}, inplace=True)
        connectivity_nodes['in_service'] = True
        connectivity_nodes['type'] = 'b'
        return connectivity_nodes, eqssh_terminals

    def _convert_external_network_injections_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting ExternalNetworkInjections.")

        eqssh_eni = self._prepare_external_network_injections_cim16()

        # choose the slack
        eni_ref_prio_min = eqssh_eni.loc[eqssh_eni['enabled'], 'slack_weight'].min()
        # check if the slack is a SynchronousMachine
        sync_machines = self._merge_eq_ssh_profile('SynchronousMachine')
        regulation_controllers = self._merge_eq_ssh_profile('RegulatingControl')
        regulation_controllers = regulation_controllers.loc[regulation_controllers['mode'] == 'voltage']
        regulation_controllers = regulation_controllers[['rdfId', 'targetValue', 'enabled']]
        regulation_controllers.rename(columns={'rdfId': 'RegulatingControl'}, inplace=True)
        sync_machines = pd.merge(sync_machines, regulation_controllers, how='left', on='RegulatingControl')

        sync_ref_prio_min = sync_machines.loc[(sync_machines['referencePriority'] > 0) & (sync_machines['enabled']),
                                              'referencePriority'].min()
        if pd.isna(eni_ref_prio_min):
            ref_prio_min = sync_ref_prio_min
        elif pd.isna(sync_ref_prio_min):
            ref_prio_min = eni_ref_prio_min
        else:
            ref_prio_min = min(eni_ref_prio_min, sync_ref_prio_min)

        eni_slacks = eqssh_eni.loc[(eqssh_eni['slack_weight'] == ref_prio_min) & (eqssh_eni['controllable'])]
        eni_gens = eqssh_eni.loc[(eqssh_eni['slack_weight'] != ref_prio_min) & (eqssh_eni['controllable'])]
        eni_sgens = eqssh_eni.loc[~eqssh_eni['controllable']]

        self._copy_to_pp('ext_grid', eni_slacks)
        self._copy_to_pp('gen', eni_gens)
        self._copy_to_pp('sgen', eni_sgens)

        self.logger.info("Created %s external networks, %s generators and %s static generators in %ss" %
                         (eni_slacks.index.size, eni_gens.index.size, eni_sgens.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s external networks, %s generators and %s static generators from "
                    "ExternalNetworkInjections in %ss" %
                    (eni_slacks.index.size, eni_gens.index.size, eni_sgens.index.size, time.time() - time_start)))

    def _prepare_external_network_injections_cim16(self) -> pd.DataFrame:

        eqssh_eni = self._merge_eq_ssh_profile('ExternalNetworkInjection', add_cim_type_column=True)

        # merge with busses
        eqssh_eni = pd.merge(eqssh_eni, self.bus_merge, how='left', on='rdfId')

        # get the voltage from controllers
        regulation_controllers = self._merge_eq_ssh_profile('RegulatingControl')
        regulation_controllers = regulation_controllers.loc[regulation_controllers['mode'] == 'voltage']
        regulation_controllers = regulation_controllers[['rdfId', 'targetValue', 'enabled']]
        regulation_controllers.rename(columns={'rdfId': 'RegulatingControl'}, inplace=True)

        eqssh_eni = pd.merge(eqssh_eni, regulation_controllers, how='left', on='RegulatingControl')

        # get slack voltage and angle from SV profile
        eqssh_eni = pd.merge(eqssh_eni, self.net.bus[['vn_kv', sc['ct']]],
                             how='left', left_on='index_bus', right_index=True)
        eqssh_eni = pd.merge(eqssh_eni, self.cim['sv']['SvVoltage'][['TopologicalNode', 'v', 'angle']],
                             how='left', left_on=sc['ct'], right_on='TopologicalNode')
        eqssh_eni['controlEnabled'] = eqssh_eni['controlEnabled'] & eqssh_eni['enabled']
        eqssh_eni['vm_pu'] = eqssh_eni['targetValue'] / eqssh_eni['vn_kv']  # voltage from regulation
        eqssh_eni['vm_pu'].fillna(eqssh_eni['v'] / eqssh_eni['vn_kv'], inplace=True)  # voltage from measurement
        eqssh_eni['vm_pu'].fillna(1., inplace=True)  # default voltage
        eqssh_eni['angle'].fillna(0., inplace=True)  # default angle
        eqssh_eni['ratedU'] = eqssh_eni['targetValue'][:]  # targetValue in kV
        eqssh_eni['ratedU'].fillna(eqssh_eni['v'], inplace=True)  # v in kV
        eqssh_eni['ratedU'].fillna(eqssh_eni['vn_kv'], inplace=True)
        eqssh_eni['s_sc_max_mva'] = 3 ** .5 * eqssh_eni['ratedU'] * (eqssh_eni['maxInitialSymShCCurrent'] / 1e3)
        eqssh_eni['s_sc_min_mva'] = 3 ** .5 * eqssh_eni['ratedU'] * (eqssh_eni['minInitialSymShCCurrent'] / 1e3)
        # get the substations
        eqssh_eni = pd.merge(eqssh_eni, self.net.bus[[sc['o_id'], 'zone']].rename({sc['o_id']: 'b_id'}, axis=1),
                             how='left', left_on='ConnectivityNode', right_on='b_id')

        eqssh_eni['referencePriority'].loc[eqssh_eni['referencePriority'] == 0] = np.NaN
        eqssh_eni['p'] = -eqssh_eni['p']
        eqssh_eni['q'] = -eqssh_eni['q']
        eqssh_eni['x0x_max'] = ((eqssh_eni['maxR1ToX1Ratio'] + 1j) /
                                (eqssh_eni['maxR0ToX0Ratio'] + 1j)).abs() * eqssh_eni['maxZ0ToZ1Ratio']

        eqssh_eni.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'zone': sc['sub'],
                                  'angle': 'va_degree', 'index_bus': 'bus', 'connected': 'in_service',
                                  'minP': 'min_p_mw', 'maxP': 'max_p_mw', 'minQ': 'min_q_mvar', 'maxQ': 'max_q_mvar',
                                  'p': 'p_mw', 'q': 'q_mvar', 'controlEnabled': 'controllable',
                                  'maxR1ToX1Ratio': 'rx_max', 'minR1ToX1Ratio': 'rx_min', 'maxR0ToX0Ratio': 'r0x0_max',
                                  'referencePriority': 'slack_weight'},
                         inplace=True)
        eqssh_eni['scaling'] = 1.
        eqssh_eni['type'] = None
        eqssh_eni['slack'] = False

        return eqssh_eni

    def _convert_ac_line_segments_cim16(self, convert_line_to_switch, line_r_limit, line_x_limit):
        time_start = time.time()
        self.logger.info("Start converting ACLineSegments.")
        eq_ac_line_segments = self._prepare_ac_line_segments_cim16(convert_line_to_switch, line_r_limit, line_x_limit)

        # now create the lines and the switches
        # -------- lines --------
        if 'line' in eq_ac_line_segments['kindOfType'].values:
            line_df = eq_ac_line_segments.loc[eq_ac_line_segments['kindOfType'] == 'line']
            line_df.rename(columns={
                'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'],
                'index_bus': 'from_bus', 'index_bus2': 'to_bus', 'length': 'length_km',
                'shortCircuitEndTemperature': 'endtemp_degree'}, inplace=True)
            line_df[sc['o_cl']] = 'ACLineSegment'
            line_df['in_service'] = line_df.connected & line_df.connected2
            line_df['r_ohm_per_km'] = abs(line_df.r) / line_df.length_km
            line_df['x_ohm_per_km'] = abs(line_df.x) / line_df.length_km
            line_df['c_nf_per_km'] = abs(line_df.bch) / (2 * 50 * np.pi * line_df.length_km) * 1e9
            line_df['g_us_per_km'] = abs(line_df.gch) * 1e6 / line_df.length_km
            line_df['r0_ohm_per_km'] = abs(line_df.r0) / line_df.length_km
            line_df['x0_ohm_per_km'] = abs(line_df.x0) / line_df.length_km
            line_df['c0_nf_per_km'] = abs(line_df.b0ch) / (2 * 50 * np.pi * line_df.length_km) * 1e9
            line_df['g0_us_per_km'] = abs(line_df.g0ch) * 1e6 / line_df.length_km
            line_df['parallel'] = 1
            line_df['df'] = 1.
            line_df['type'] = None
            line_df['std_type'] = None
            self._copy_to_pp('line', line_df)
        else:
            line_df = pd.DataFrame(None)
        # -------- switches --------
        if 'switch' in eq_ac_line_segments['kindOfType'].values:
            switch_df = eq_ac_line_segments.loc[eq_ac_line_segments['kindOfType'] == 'switch']

            switch_df.rename(columns={
                'rdfId': sc['o_id'], 'index_bus': 'bus', 'index_bus2': 'element', 'rdfId_Terminal': sc['t_bus'],
                'rdfId_Terminal2': sc['t_ele']}, inplace=True)
            switch_df['et'] = 'b'
            switch_df['type'] = None
            switch_df['z_ohm'] = 0
            if switch_df.index.size > 0:
                switch_df['closed'] = switch_df.connected & switch_df.connected2
            self._copy_to_pp('switch', switch_df)
        else:
            switch_df = pd.DataFrame(None)

        self.logger.info("Created %s lines and %s switches in %ss" %
                         (line_df.index.size, switch_df.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s lines and %s switches from ACLineSegments in %ss" %
                    (line_df.index.size, switch_df.index.size, time.time() - time_start)))

    def _prepare_ac_line_segments_cim16(self, convert_line_to_switch, line_r_limit, line_x_limit) -> pd.DataFrame:
        line_length_before_merge = self.cim['eq']['ACLineSegment'].index.size
        # until now self.cim['eq']['ACLineSegment'] looks like:
        #   rdfId   name    r       ...
        #   _x01    line1   0.056   ...
        #   _x02    line2   0.471   ...
        # now join with the terminals
        eq_ac_line_segments = pd.merge(self.cim['eq']['ACLineSegment'], self.bus_merge, how='left', on='rdfId')
        eq_ac_line_segments[sc['o_cl']] = 'ACLineSegment'
        # now eq_ac_line_segments looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    line1   0.056   termi025        True        ...
        #   _x01    line1   0.056   termi223        True        ...
        #   _x02    line2   0.471   termi154        True        ...
        #   _x02    line2   0.471   termi199        True        ...
        # if each switch got two terminals, reduce back to one line to use fast slicing
        if eq_ac_line_segments.index.size != line_length_before_merge * 2:
            self.logger.error("Error processing the ACLineSegments, there is a problem with Terminals in the source "
                              "data!")
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Error processing the ACLineSegments, there is a problem with Terminals in the source data!"))
            dups = eq_ac_line_segments.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The ACLineSegment with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The ACLineSegment data: \n%s" %
                                    eq_ac_line_segments[eq_ac_line_segments['rdfId'] == rdfId])
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The ACLineSegment with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eq_ac_line_segments = eq_ac_line_segments[0:0]
        eq_ac_line_segments.reset_index(inplace=True)
        # now merge with OperationalLimitSets and CurrentLimits
        eq_operational_limit_sets = self.cim['eq']['OperationalLimitSet'][['rdfId', 'Terminal']]
        eq_operational_limit_sets.rename(columns={'rdfId': 'rdfId_OperationalLimitSet',
                                                  'Terminal': 'rdfId_Terminal'}, inplace=True)
        eq_ac_line_segments = pd.merge(eq_ac_line_segments, eq_operational_limit_sets, how='left',
                                       on='rdfId_Terminal')
        eq_current_limits = self.cim['eq']['CurrentLimit'][['rdfId', 'OperationalLimitSet', 'value']]
        eq_current_limits.rename(columns={'rdfId': 'rdfId_CurrentLimit',
                                          'OperationalLimitSet': 'rdfId_OperationalLimitSet'}, inplace=True)
        eq_ac_line_segments = pd.merge(eq_ac_line_segments, eq_current_limits, how='left',
                                       on='rdfId_OperationalLimitSet')
        eq_ac_line_segments.value = eq_ac_line_segments.value.astype(float)
        # sort by rdfId, sequenceNumber and value. value is max_i_ka, choose the lowest one if more than one is
        # given (A line may have more than one max_i_ka in CIM, different modes e.g. normal)
        eq_ac_line_segments.sort_values(by=['rdfId', 'sequenceNumber', 'value'], inplace=True)
        eq_ac_line_segments.drop_duplicates(['rdfId', 'rdfId_Terminal'], keep='first', inplace=True)

        # copy the columns which are needed to reduce the eq_ac_line_segments to one row per line
        eq_ac_line_segments['rdfId_Terminal2'] = eq_ac_line_segments['rdfId_Terminal'].copy()
        eq_ac_line_segments['connected2'] = eq_ac_line_segments['connected'].copy()
        eq_ac_line_segments['index_bus2'] = eq_ac_line_segments['index_bus'].copy()
        eq_ac_line_segments['value2'] = eq_ac_line_segments['value'].copy()
        eq_ac_line_segments = eq_ac_line_segments.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eq_ac_line_segments.rdfId_Terminal2 = eq_ac_line_segments.rdfId_Terminal2.iloc[
                                              1:].reset_index().rdfId_Terminal2
        eq_ac_line_segments.connected2 = eq_ac_line_segments.connected2.iloc[1:].reset_index().connected2
        eq_ac_line_segments.index_bus2 = eq_ac_line_segments.index_bus2.iloc[1:].reset_index().index_bus2
        eq_ac_line_segments.value2 = eq_ac_line_segments.value2.iloc[1:].reset_index().value2
        eq_ac_line_segments.drop_duplicates(['rdfId'], keep='first', inplace=True)
        # get the max_i_ka
        eq_ac_line_segments['max_i_ka'] = eq_ac_line_segments['value'].fillna(eq_ac_line_segments['value2']) * 1e-3

        # filter if line or switches will be added
        eq_ac_line_segments['kindOfType'] = 'line'
        if convert_line_to_switch:
            eq_ac_line_segments.loc[(abs(eq_ac_line_segments['r']) <= line_r_limit) |
                                    (abs(eq_ac_line_segments['x']) <= line_x_limit), 'kindOfType'] = 'switch'
        return eq_ac_line_segments

    def _convert_dc_line_segments_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting DCLineSegments.")
        eq_dc_line_segments = self._prepare_dc_line_segments_cim16()

        self._copy_to_pp('dcline', eq_dc_line_segments)

        self.logger.info("Created %s DC lines in %ss" % (eq_dc_line_segments.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s DC lines from DCLineSegments in %ss" %
                    (eq_dc_line_segments.index.size, time.time() - time_start)))

    def _prepare_dc_line_segments_cim16(self) -> pd.DataFrame:
        line_length_before_merge = self.cim['eq']['DCLineSegment'].index.size
        # until now self.cim['eq']['DCLineSegment'] looks like:
        #   rdfId   name    ...
        #   _x01    line1   ...
        #   _x02    line2   ...
        # now join with the terminals
        dc_line_segments = pd.merge(self.cim['eq']['DCLineSegment'], self.bus_merge, how='left', on='rdfId')
        dc_line_segments = dc_line_segments[['rdfId', 'name', 'ConnectivityNode', 'sequenceNumber']]
        dc_line_segments[sc['o_cl']] = 'DCLineSegment'
        # now dc_line_segments looks like:
        #   rdfId   name    rdfId_Terminal  connected   ...
        #   _x01    line1   termi025        True        ...
        #   _x01    line1   termi223        True        ...
        #   _x02    line2   termi154        True        ...
        #   _x02    line2   termi199        True        ...
        # if each switch got two terminals, reduce back to one line to use fast slicing
        if dc_line_segments.index.size != line_length_before_merge * 2:
            self.logger.error("Error processing the DCLineSegments, there is a problem with Terminals in the source "
                              "data!")
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Error processing the DCLineSegments, there is a problem with Terminals in the source data!"))
            return pd.DataFrame(None)
        dc_line_segments.reset_index(inplace=True)

        # now merge with the Converters
        converters = pd.merge(pd.concat([self.cim['eq']['CsConverter'], self.cim['eq']['VsConverter']],
                                        ignore_index=True, sort=False),
                              pd.concat([self.cim['ssh']['CsConverter'], self.cim['ssh']['VsConverter']],
                                        ignore_index=True, sort=False),
                              how='left', on='rdfId')
        if 'name' in converters.columns:
            converters.drop(columns=['name'], inplace=True)
        # merge with the terminals
        converters = pd.merge(converters, self.bus_merge, how='left', on='rdfId')
        converters.drop(columns=['sequenceNumber'], inplace=True)
        converters.rename(columns={'rdfId': 'converters'}, inplace=True)
        converter_terminals = pd.concat([self.cim['eq']['Terminal'], self.cim['eq_bd']['Terminal']],
                                        ignore_index=True, sort=False)
        converter_terminals = converter_terminals[['rdfId']].rename(columns={'rdfId': 'rdfId_Terminal'})
        converters_t = pd.merge(converters, converter_terminals, how='inner', on='rdfId_Terminal')

        dc_line_segments = pd.merge(dc_line_segments, converters[['converters', 'ConnectivityNode']],
                                    how='left', on='ConnectivityNode')
        # get the missing converters (maybe there is a switch or something else between the line and the converter)
        t = self.cim['eq']['DCTerminal'][['DCNode', 'DCConductingEquipment', 'sequenceNumber']]
        t.rename(columns={'DCNode': 'ConnectivityNode', 'DCConductingEquipment': 'ConductingEquipment'}, inplace=True)

        def search_converter(cn_ids: Dict[str, str], visited_cns: List[str]) -> str:
            new_cn_dict = dict()
            for one_cn, from_dev in cn_ids.items():
                # get the Terminals
                t_temp = t.loc[t['ConnectivityNode'] == one_cn, :]
                for _, one_t in t_temp.iterrows():
                    # prevent running backwards
                    if one_t['ConductingEquipment'] == from_dev:
                        continue
                    ids_temp = t.loc[(t['ConductingEquipment'] == one_t['ConductingEquipment']) & (
                            t['sequenceNumber'] != one_t['sequenceNumber'])]['ConnectivityNode'].values
                    for id_temp in ids_temp:
                        # check if the ConnectivityNode has a converter
                        if id_temp in converters['ConnectivityNode'].values:
                            # found the converter
                            return converters.loc[converters['ConnectivityNode'] == id_temp, 'converters'].values[0]
                        if id_temp not in visited_cns:
                            new_cn_dict[id_temp] = one_t['ConductingEquipment']
            if len(list(new_cn_dict.keys())) > 0:
                visited_cns.extend(list(cn_ids.keys()))
                return search_converter(cn_ids=new_cn_dict, visited_cns=visited_cns)
        for row_index, row in dc_line_segments[dc_line_segments['converters'].isna()].iterrows():
            conv = search_converter(cn_ids=dict({row['ConnectivityNode']: row['rdfId']}),
                                    visited_cns=[row['ConnectivityNode']])
            dc_line_segments.loc[row_index, 'converters'] = conv
            if conv is None:
                self.logger.warning("Problem with converting tht DC line %s: No ACDC converter found, maybe the DC "
                                    "part is too complex to reduce it to pandapower requirements!")
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="Error processing the DCLineSegments, there is a problem with Terminals in the source "
                            "data!"))
        dc_line_segments.drop(columns=['ConnectivityNode'], inplace=True)
        dc_line_segments = pd.merge(dc_line_segments, converters_t, how='left', on='converters')
        dc_line_segments['targetUpcc'].fillna(dc_line_segments['base_voltage_bus'], inplace=True)

        # copy the columns which are needed to reduce the dc_line_segments to one row per line
        dc_line_segments.sort_values(by=['rdfId', 'sequenceNumber'], inplace=True)
        # a list of DC line parameters which are used for each DC line end
        dc_line_segments.reset_index(inplace=True)
        copy_list = ['index_bus', 'rdfId_Terminal', 'connected', 'p', 'ratedUdc', 'targetUpcc', 'base_voltage_bus']
        for one_item in copy_list:
            # copy the columns which are required for each line end
            dc_line_segments[one_item + '2'] = dc_line_segments[one_item].copy()
            # cut the first element from the copied columns
            dc_line_segments[one_item + '2'] = dc_line_segments[one_item + '2'].iloc[1:].reset_index()[one_item + '2']
        del copy_list, one_item
        dc_line_segments.drop_duplicates(['rdfId'], keep='first', inplace=True)
        dc_line_segments = pd.merge(dc_line_segments,
                                    pd.DataFrame(dc_line_segments.pivot_table(index=['converters'], aggfunc='size'),
                                                 columns=['converter_dups']), how='left', on='converters')
        dc_line_segments['loss_mw'] = \
            abs(abs(dc_line_segments['p']) - abs(dc_line_segments['p2'])) / dc_line_segments['converter_dups']
        dc_line_segments['p_mw'] = dc_line_segments['p'] / dc_line_segments['converter_dups']
        dc_line_segments['loss_percent'] = 0
        dc_line_segments['vm_from_pu'] = dc_line_segments['targetUpcc'] / dc_line_segments['base_voltage_bus']
        dc_line_segments['vm_to_pu'] = dc_line_segments['targetUpcc2'] / dc_line_segments['base_voltage_bus2']
        dc_line_segments['in_service'] = dc_line_segments.connected & dc_line_segments.connected2
        dc_line_segments.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'index_bus': 'from_bus',
            'index_bus2': 'to_bus'}, inplace=True)

        return dc_line_segments

    def _convert_switches_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting Breakers, Disconnectors, LoadBreakSwitches and Switches.")
        eqssh_switches = self._prepare_switches_cim16()
        self._copy_to_pp('switch', eqssh_switches)
        self.logger.info("Created %s switches in %ss." % (eqssh_switches.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s switches from Breakers, Disconnectors, LoadBreakSwitches and Switches in %ss." %
                    (eqssh_switches.index.size, time.time() - time_start)))

    def _prepare_switches_cim16(self) -> pd.DataFrame:
        eqssh_switches = self._merge_eq_ssh_profile('Breaker', add_cim_type_column=True)
        eqssh_switches['type'] = 'CB'
        start_index_cim_net = eqssh_switches.index.size
        eqssh_switches = \
            pd.concat([eqssh_switches, self._merge_eq_ssh_profile('Disconnector', add_cim_type_column=True)],
                      ignore_index=True, sort=False)
        eqssh_switches.type[start_index_cim_net:] = 'DS'
        start_index_cim_net = eqssh_switches.index.size
        eqssh_switches = \
            pd.concat([eqssh_switches, self._merge_eq_ssh_profile('LoadBreakSwitch', add_cim_type_column=True)],
                      ignore_index=True, sort=False)
        eqssh_switches.type[start_index_cim_net:] = 'LBS'
        start_index_cim_net = eqssh_switches.index.size
        # switches needs to be the last which getting appended because of class inherit problem in jpa
        eqssh_switches = pd.concat([eqssh_switches, self._merge_eq_ssh_profile('Switch', add_cim_type_column=True)],
                                   ignore_index=True, sort=False)
        eqssh_switches.type[start_index_cim_net:] = 'LS'
        # drop all duplicates to fix class inherit problem in jpa
        eqssh_switches.drop_duplicates(subset=['rdfId'], keep='first', inplace=True)
        switch_length_before_merge = eqssh_switches.index.size
        # until now eqssh_switches looks like:
        #   rdfId   name    open    ...
        #   _x01    switch1 True    ...
        #   _x02    switch2 False   ...
        # now join with the terminals
        eqssh_switches = pd.merge(eqssh_switches, self.bus_merge, how='left', on='rdfId')
        eqssh_switches.sort_values(by=['rdfId', 'sequenceNumber'], inplace=True)
        eqssh_switches.reset_index(inplace=True)
        # copy the columns which are needed to reduce the eqssh_switches to one line per switch
        eqssh_switches['rdfId_Terminal2'] = eqssh_switches['rdfId_Terminal'].copy()
        eqssh_switches['connected2'] = eqssh_switches['connected'].copy()
        eqssh_switches['index_bus2'] = eqssh_switches['index_bus'].copy()
        if eqssh_switches.index.size == switch_length_before_merge * 2:
            # here is where the magic happens: just remove the first value from the copied columns, reset the index
            # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
            eqssh_switches.rdfId_Terminal2 = eqssh_switches.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
            eqssh_switches.connected2 = eqssh_switches.connected2.iloc[1:].reset_index().connected2
            eqssh_switches.index_bus2 = eqssh_switches.index_bus2.iloc[1:].reset_index().index_bus2
            eqssh_switches.drop_duplicates(subset=['rdfId'], keep='first', inplace=True)
        else:
            self.logger.error("Something went wrong at switches, seems like that terminals for connection with "
                              "connectivity nodes are missing!")
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Something went wrong at switches, seems like that terminals for connection with "
                        "connectivity nodes are missing!"))
            dups = eqssh_switches.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The switch with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The switch data: \n%s" % eqssh_switches[eqssh_switches['rdfId'] == rdfId])
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The switch with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eqssh_switches = eqssh_switches[0:0]
        eqssh_switches.rename(columns={'rdfId': sc['o_id'], 'index_bus': 'bus', 'index_bus2': 'element',
                                       'rdfId_Terminal': sc['t_bus'], 'rdfId_Terminal2': sc['t_ele']}, inplace=True)
        eqssh_switches['et'] = 'b'
        eqssh_switches['z_ohm'] = 0
        if eqssh_switches.index.size > 0:
            eqssh_switches['closed'] = ~eqssh_switches.open & eqssh_switches.connected & eqssh_switches.connected2
        return eqssh_switches

    def _convert_energy_consumers_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EnergyConsumers.")
        eqssh_energy_consumers = self._prepare_energy_consumers_cim16()
        self._copy_to_pp('load', eqssh_energy_consumers)
        self.logger.info("Created %s loads in %ss." % (eqssh_energy_consumers.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from EnergyConsumers in %ss." %
                    (eqssh_energy_consumers.index.size, time.time() - time_start)))

    def _prepare_energy_consumers_cim16(self) -> pd.DataFrame:
        eqssh_energy_consumers = self._merge_eq_ssh_profile('EnergyConsumer', add_cim_type_column=True)
        eqssh_energy_consumers = pd.merge(eqssh_energy_consumers, self.bus_merge, how='left', on='rdfId')
        eqssh_energy_consumers.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                               'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'}, inplace=True)
        eqssh_energy_consumers['const_i_percent'] = 0.
        eqssh_energy_consumers['const_z_percent'] = 0.
        eqssh_energy_consumers['scaling'] = 1.
        eqssh_energy_consumers['type'] = None
        return eqssh_energy_consumers

    def _convert_conform_loads_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting ConformLoads.")
        eqssh_conform_loads = self._prepare_conform_loads_cim16()
        self._copy_to_pp('load', eqssh_conform_loads)
        self.logger.info("Created %s loads in %ss." % (eqssh_conform_loads.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from ConformLoads in %ss." %
                    (eqssh_conform_loads.index.size, time.time() - time_start)))

    def _prepare_conform_loads_cim16(self) -> pd.DataFrame:
        eqssh_conform_loads = self._merge_eq_ssh_profile('ConformLoad', add_cim_type_column=True)
        eqssh_conform_loads = pd.merge(eqssh_conform_loads, self.bus_merge, how='left', on='rdfId')
        eqssh_conform_loads.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                            'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'}, inplace=True)
        eqssh_conform_loads['const_i_percent'] = 0.
        eqssh_conform_loads['const_z_percent'] = 0.
        eqssh_conform_loads['scaling'] = 1.
        eqssh_conform_loads['type'] = None
        return eqssh_conform_loads

    def _convert_non_conform_loads_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting NonConformLoads.")
        eqssh_non_conform_loads = self._prepare_non_conform_loads_cim16()
        self._copy_to_pp('load', eqssh_non_conform_loads)
        self.logger.info("Created %s loads in %ss." % (eqssh_non_conform_loads.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from NonConformLoads in %ss." %
                    (eqssh_non_conform_loads.index.size, time.time() - time_start)))

    def _prepare_non_conform_loads_cim16(self) -> pd.DataFrame:
        eqssh_non_conform_loads = self._merge_eq_ssh_profile('NonConformLoad', add_cim_type_column=True)
        eqssh_non_conform_loads = pd.merge(eqssh_non_conform_loads, self.bus_merge, how='left', on='rdfId')
        eqssh_non_conform_loads.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                                'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'}, inplace=True)
        eqssh_non_conform_loads['const_i_percent'] = 0.
        eqssh_non_conform_loads['const_z_percent'] = 0.
        eqssh_non_conform_loads['scaling'] = 1.
        eqssh_non_conform_loads['type'] = None
        return eqssh_non_conform_loads

    def _convert_station_supplies_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting StationSupplies.")
        eqssh_station_supplies = self._prepare_station_supplies_cim16()
        self._copy_to_pp('load', eqssh_station_supplies)
        self.logger.info("Created %s loads in %ss." % (eqssh_station_supplies.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from StationSupplies in %ss." %
                    (eqssh_station_supplies.index.size, time.time() - time_start)))

    def _prepare_station_supplies_cim16(self) -> pd.DataFrame:
        eqssh_station_supplies = self._merge_eq_ssh_profile('StationSupply', add_cim_type_column=True)
        eqssh_station_supplies = pd.merge(eqssh_station_supplies, self.bus_merge, how='left', on='rdfId')
        eqssh_station_supplies.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                               'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'}, inplace=True)
        eqssh_station_supplies['const_i_percent'] = 0.
        eqssh_station_supplies['const_z_percent'] = 0.
        eqssh_station_supplies['scaling'] = 1.
        eqssh_station_supplies['type'] = None
        return eqssh_station_supplies

    def _convert_synchronous_machines_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting SynchronousMachines.")
        eqssh_synchronous_machines = self._prepare_synchronous_machines_cim16()

        # convert the SynchronousMachines with voltage control to gens
        eqssh_sm_gens = eqssh_synchronous_machines.loc[(eqssh_synchronous_machines['mode'] == 'voltage') &
                                                       (eqssh_synchronous_machines['enabled'])]
        self._copy_to_pp('gen', eqssh_sm_gens)
        # now deal with the pq generators
        eqssh_synchronous_machines = eqssh_synchronous_machines.loc[(eqssh_synchronous_machines['mode'] != 'voltage') |
                                                                    (~eqssh_synchronous_machines['enabled'])]
        self._copy_to_pp('sgen', eqssh_synchronous_machines)
        self.logger.info("Created %s gens and %s sgens in %ss." %
                         (eqssh_sm_gens.index.size, eqssh_synchronous_machines.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s gens and %s sgens from SynchronousMachines in %ss." %
                         (eqssh_sm_gens.index.size, eqssh_synchronous_machines.index.size, time.time() - time_start)))

    def _prepare_synchronous_machines_cim16(self) -> pd.DataFrame:
        eq_generating_units = self.cim['eq']['GeneratingUnit'][['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']]
        # a column for the type of the static generator in pandapower
        eq_generating_units['type'] = 'GeneratingUnit'
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['WindGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('WP', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['HydroGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('Hydro', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['SolarGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('PV', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['ThermalGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('Thermal', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['NuclearGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('Nuclear', inplace=True)
        eq_generating_units.rename(columns={'rdfId': 'GeneratingUnit'}, inplace=True)
        eqssh_synchronous_machines = self._merge_eq_ssh_profile('SynchronousMachine', add_cim_type_column=True)
        if 'type' in eqssh_synchronous_machines.columns:
            eqssh_synchronous_machines.drop(columns=['type'], inplace=True)
        if 'EquipmentContainer' in eqssh_synchronous_machines.columns:
            eqssh_synchronous_machines.drop(columns=['EquipmentContainer'], inplace=True)
        eqssh_synchronous_machines = pd.merge(eqssh_synchronous_machines, eq_generating_units,
                                              how='left', on='GeneratingUnit')
        # merge with RegulatingControl to check if it is a voltage controlled generator
        eqssh_reg_control = self._merge_eq_ssh_profile('RegulatingControl')[['rdfId', 'mode', 'enabled', 'targetValue']]
        eqssh_reg_control = eqssh_reg_control.loc[eqssh_reg_control['mode'] == 'voltage']
        eqssh_synchronous_machines = pd.merge(
            eqssh_synchronous_machines, eqssh_reg_control.rename(columns={'rdfId': 'RegulatingControl'}),
            how='left', on='RegulatingControl')
        eqssh_synchronous_machines = pd.merge(eqssh_synchronous_machines, self.bus_merge, how='left', on='rdfId')
        eqssh_synchronous_machines.drop_duplicates(['rdfId'], keep='first', inplace=True)
        # add the voltage from the bus
        eqssh_synchronous_machines = pd.merge(eqssh_synchronous_machines, self.net.bus[['vn_kv']],
                                              how='left', left_on='index_bus', right_index=True)
        eqssh_synchronous_machines['vm_pu'] = eqssh_synchronous_machines.targetValue / eqssh_synchronous_machines.vn_kv
        eqssh_synchronous_machines['vm_pu'].fillna(1., inplace=True)
        eqssh_synchronous_machines.rename(columns={'vn_kv': 'bus_voltage'}, inplace=True)
        eqssh_synchronous_machines['slack'] = False
        # set the slack = True for gens with highest prio
        # get the highest prio from SynchronousMachines
        sync_ref_prio_min = eqssh_synchronous_machines.loc[
            (eqssh_synchronous_machines['referencePriority'] > 0) & (eqssh_synchronous_machines['enabled']),
            'referencePriority'].min()
        # get the highest prio from ExternalNetworkInjection and check if the slack is an ExternalNetworkInjection
        enis = self._merge_eq_ssh_profile('ExternalNetworkInjection')
        regulation_controllers = self._merge_eq_ssh_profile('RegulatingControl')
        regulation_controllers = regulation_controllers.loc[regulation_controllers['mode'] == 'voltage']
        regulation_controllers = regulation_controllers[['rdfId', 'targetValue', 'enabled']]
        regulation_controllers.rename(columns={'rdfId': 'RegulatingControl'}, inplace=True)
        enis = pd.merge(enis, regulation_controllers, how='left', on='RegulatingControl')

        eni_ref_prio_min = enis.loc[(enis['referencePriority'] > 0) & (enis['enabled']), 'referencePriority'].min()
        if pd.isna(sync_ref_prio_min):
            ref_prio_min = eni_ref_prio_min
        elif pd.isna(eni_ref_prio_min):
            ref_prio_min = sync_ref_prio_min
        else:
            ref_prio_min = min(eni_ref_prio_min, sync_ref_prio_min)

        eqssh_synchronous_machines.loc[eqssh_synchronous_machines['referencePriority'] == ref_prio_min, 'slack'] = True
        eqssh_synchronous_machines['p_mw'] = -eqssh_synchronous_machines['p']
        eqssh_synchronous_machines['q_mvar'] = -eqssh_synchronous_machines['q']
        eqssh_synchronous_machines['current_source'] = True
        eqssh_synchronous_machines['sn_mva'] = \
            eqssh_synchronous_machines['ratedS'].fillna(eqssh_synchronous_machines['nominalP'])
        # SC data
        eqssh_synchronous_machines['vn_kv'] = eqssh_synchronous_machines['ratedU'][:]
        eqssh_synchronous_machines['rdss_ohm'] = \
            eqssh_synchronous_machines['r2'] * \
            (eqssh_synchronous_machines['ratedU']**2 / eqssh_synchronous_machines['ratedS'])
        eqssh_synchronous_machines['xdss_pu'] = eqssh_synchronous_machines['x2'][:]
        eqssh_synchronous_machines['voltageRegulationRange'].fillna(0., inplace=True)
        eqssh_synchronous_machines['pg_percent'] = eqssh_synchronous_machines['voltageRegulationRange']
        eqssh_synchronous_machines['k'] = (eqssh_synchronous_machines['ratedS'] * 1e3 / eqssh_synchronous_machines[
            'ratedU']) / (eqssh_synchronous_machines['ratedU'] / (
                    3 ** .5 * (eqssh_synchronous_machines['r2'] ** 2 + eqssh_synchronous_machines['x2'] ** 2)))
        eqssh_synchronous_machines['rx'] = eqssh_synchronous_machines['r2'] / eqssh_synchronous_machines['x2']
        eqssh_synchronous_machines['scaling'] = 1.
        eqssh_synchronous_machines['generator_type'] = 'current_source'
        eqssh_synchronous_machines.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'],
                                                   'connected': 'in_service', 'index_bus': 'bus',
                                                   'minOperatingP': 'min_p_mw', 'maxOperatingP': 'max_p_mw',
                                                   'minQ': 'min_q_mvar', 'maxQ': 'max_q_mvar',
                                                   'ratedPowerFactor': 'cos_phi'}, inplace=True)
        return eqssh_synchronous_machines

    def _convert_asynchronous_machines_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting AsynchronousMachines.")
        eqssh_asynchronous_machines = self._prepare_asynchronous_machines_cim16()
        self._copy_to_pp('motor', eqssh_asynchronous_machines)
        self.logger.info("Created %s motors in %ss." %
                         (eqssh_asynchronous_machines.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s motors from AsynchronousMachines in %ss." %
                         (eqssh_asynchronous_machines.index.size, time.time() - time_start)))

    def _prepare_asynchronous_machines_cim16(self) -> pd.DataFrame:
        eq_generating_units = self.cim['eq']['WindGeneratingUnit'].copy()
        # a column for the type of the static generator in pandapower
        eq_generating_units['type'] = 'WP'
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['GeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('GeneratingUnit', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['HydroGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('Hydro', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['SolarGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('PV', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['ThermalGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('Thermal', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cim['eq']['NuclearGeneratingUnit']], sort=False)
        eq_generating_units['type'].fillna('Nuclear', inplace=True)
        eq_generating_units.rename(columns={'rdfId': 'GeneratingUnit'}, inplace=True)
        eqssh_asynchronous_machines = self._merge_eq_ssh_profile('AsynchronousMachine', add_cim_type_column=True)
        # prevent conflict of merging two dataframes each containing column 'name'
        eq_generating_units.drop('name', axis=1, inplace=True)
        eqssh_asynchronous_machines = pd.merge(eqssh_asynchronous_machines, eq_generating_units,
                                               how='left', on='GeneratingUnit')
        eqssh_asynchronous_machines = pd.merge(eqssh_asynchronous_machines, self.bus_merge, how='left', on='rdfId')
        eqssh_asynchronous_machines['p_mw'] = -eqssh_asynchronous_machines['p']
        eqssh_asynchronous_machines['q_mvar'] = -eqssh_asynchronous_machines['q']
        eqssh_asynchronous_machines['current_source'] = True
        eqssh_asynchronous_machines['cos_phi_n'] = eqssh_asynchronous_machines['ratedPowerFactor'][:]
        eqssh_asynchronous_machines['sn_mva'] = \
            eqssh_asynchronous_machines['ratedS'].fillna(eqssh_asynchronous_machines['nominalP'])
        eqssh_asynchronous_machines['generator_type'] = 'async'
        eqssh_asynchronous_machines['loading_percent'] = \
            100 * eqssh_asynchronous_machines['p_mw'] / eqssh_asynchronous_machines['ratedMechanicalPower']
        eqssh_asynchronous_machines.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'],
                                                    'connected': 'in_service', 'index_bus': 'bus',
                                                    'rxLockedRotorRatio': 'rx', 'iaIrRatio': 'lrc_pu',
                                                    'ratedPowerFactor': 'cos_phi', 'ratedU': 'vn_kv',
                                                    'efficiency': 'efficiency_n_percent',
                                                    'ratedMechanicalPower': 'pn_mech_mw'}, inplace=True)
        eqssh_asynchronous_machines['scaling'] = 1
        eqssh_asynchronous_machines['efficiency_percent'] = 100
        return eqssh_asynchronous_machines

    def _convert_energy_sources_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EnergySources.")
        eqssh_energy_sources = self._prepare_energy_sources_cim16()
        es_slack = eqssh_energy_sources.loc[eqssh_energy_sources.vm_pu.notna()]
        es_sgen = eqssh_energy_sources.loc[eqssh_energy_sources.vm_pu.isna()]
        self._copy_to_pp('ext_grid', es_slack)
        self._copy_to_pp('sgen', es_sgen)
        # self._copy_to_pp('sgen', eqssh_energy_sources)
        self.logger.info("Created %s sgens in %ss." % (eqssh_energy_sources.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s sgens from EnergySources in %ss." %
                    (eqssh_energy_sources.index.size, time.time() - time_start)))

    def _prepare_energy_sources_cim16(self) -> pd.DataFrame:
        eq_energy_scheduling_type = \
            pd.concat([self.cim['eq']['EnergySchedulingType'], self.cim['eq_bd']['EnergySchedulingType']], sort=False)
        eq_energy_scheduling_type.rename(columns={'rdfId': 'EnergySchedulingType', 'name': 'type'}, inplace=True)
        eqssh_energy_sources = self._merge_eq_ssh_profile('EnergySource', add_cim_type_column=True)
        eqssh_energy_sources = pd.merge(eqssh_energy_sources, eq_energy_scheduling_type, how='left',
                                        on='EnergySchedulingType')
        eqssh_energy_sources = pd.merge(eqssh_energy_sources, self.bus_merge, how='left', on='rdfId')
        eqssh_energy_sources.drop_duplicates(['rdfId'], keep='first', inplace=True)
        sgen_type = dict({'WP': 'WP', 'Wind': 'WP', 'PV': 'PV', 'SolarPV': 'PV', 'BioGas': 'BioGas',
                          'OtherRES': 'OtherRES', 'CHP': 'CHP'})  # todo move?
        eqssh_energy_sources['type'] = eqssh_energy_sources['type'].map(sgen_type)
        eqssh_energy_sources['p_mw'] = -eqssh_energy_sources['activePower']
        eqssh_energy_sources['q_mvar'] = -eqssh_energy_sources['reactivePower']
        eqssh_energy_sources['va_degree'] = eqssh_energy_sources['voltageAngle'] * 180 / math.pi
        eqssh_energy_sources['vm_pu'] = \
            eqssh_energy_sources['voltageMagnitude'] / eqssh_energy_sources['base_voltage_bus']
        eqssh_energy_sources['scaling'] = 1.
        eqssh_energy_sources['current_source'] = True
        eqssh_energy_sources['generator_type'] = 'current_source'
        eqssh_energy_sources.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'connected': 'in_service',
                                             'index_bus': 'bus'}, inplace=True)
        return eqssh_energy_sources

    def _convert_static_var_compensator_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting StaticVarCompensator.")
        eq_stat_coms = self._prepare_static_var_compensator_cim16()
        self._copy_to_pp('shunt', eq_stat_coms)
        self.logger.info("Created %s generators in %ss." % (eq_stat_coms.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s generators from StaticVarCompensator in %ss." %
                    (eq_stat_coms.index.size, time.time() - time_start)))

    def _prepare_static_var_compensator_cim16(self) -> pd.DataFrame:
        eq_stat_coms = self._merge_eq_ssh_profile('StaticVarCompensator', True)
        eq_stat_coms = pd.merge(eq_stat_coms, self.bus_merge, how='left', on='rdfId')
        eq_stat_coms.rename(columns={'q': 'q_mvar'}, inplace=True)
        # get the active power and reactive power from SV profile
        eq_stat_coms = pd.merge(eq_stat_coms, self.cim['sv']['SvPowerFlow'][['p', 'q', 'Terminal']],
                                how='left', left_on='rdfId_Terminal', right_on='Terminal')
        eq_stat_coms['q_mvar'].fillna(eq_stat_coms['q'], inplace=True)
        eq_stat_coms.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'p': 'p_mw',
                                     'voltageSetPoint': 'vn_kv', 'index_bus': 'bus', 'connected': 'in_service'},
                            inplace=True)
        eq_stat_coms['step'] = 1
        eq_stat_coms['max_step'] = 1
        return eq_stat_coms

    def _convert_linear_shunt_compensator_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting LinearShuntCompensator.")
        eqssh_shunts = self._prepare_linear_shunt_compensator_cim16()
        self._copy_to_pp('shunt', eqssh_shunts)
        self.logger.info("Created %s shunts in %ss." % (eqssh_shunts.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s shunts from LinearShuntCompensator in %ss." %
                    (eqssh_shunts.index.size, time.time() - time_start)))

    def _prepare_linear_shunt_compensator_cim16(self) -> pd.DataFrame:
        eqssh_shunts = self._merge_eq_ssh_profile('LinearShuntCompensator', add_cim_type_column=True)
        eqssh_shunts = pd.merge(eqssh_shunts, self.bus_merge, how='left', on='rdfId')
        eqssh_shunts.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'connected': 'in_service', 'index_bus': 'bus',
            'nomU': 'vn_kv', 'sections': 'step', 'maximumSections': 'max_step'}, inplace=True)
        y = eqssh_shunts['gPerSection'] + eqssh_shunts['bPerSection'] * 1j
        s = eqssh_shunts['vn_kv']**2 * np.conj(y)
        eqssh_shunts['p_mw'] = s.values.real
        eqssh_shunts['q_mvar'] = s.values.imag
        return eqssh_shunts

    def _convert_nonlinear_shunt_compensator_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting NonlinearShuntCompensator.")
        if self.cim['eq']['NonlinearShuntCompensator'].index.size > 0:
            eqssh_shunts = self._prepare_nonlinear_shunt_compensator_cim16()
            self._copy_to_pp('shunt', eqssh_shunts)
        else:
            eqssh_shunts = pd.DataFrame(None)
        self.logger.info("Created %s shunts in %ss." % (eqssh_shunts.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s shunts from NonlinearShuntCompensator in %ss." %
                    (eqssh_shunts.index.size, time.time() - time_start)))

    def _prepare_nonlinear_shunt_compensator_cim16(self) -> pd.DataFrame:
        eqssh_shunts = self._merge_eq_ssh_profile('NonlinearShuntCompensator', add_cim_type_column=True)
        eqssh_shunts = pd.merge(eqssh_shunts, self.bus_merge, how='left', on='rdfId')

        eqssh_shunts['p'] = float('NaN')
        eqssh_shunts['q'] = float('NaN')
        eqssh_shunts_cols = list(eqssh_shunts.columns.values)
        nscp = self.cim['eq']['NonlinearShuntCompensatorPoint'][
            ['NonlinearShuntCompensator', 'sectionNumber', 'b', 'g']].rename(
            columns={'NonlinearShuntCompensator': 'rdfId'})
        # calculate p and q from b, g, and all the sections
        for i in range(1, int(nscp['sectionNumber'].max()) + 1):
            nscp_t = nscp.loc[nscp['sectionNumber'] == i]
            eqssh_shunts = pd.merge(eqssh_shunts, nscp_t, how='left', on='rdfId')
            y = eqssh_shunts['g'] + eqssh_shunts['b'] * 1j
            s = eqssh_shunts['nomU'] ** 2 * np.conj(y)
            eqssh_shunts['p_temp'] = s.values.real
            eqssh_shunts['q_temp'] = s.values.imag
            if i == 1:
                eqssh_shunts['p'] = eqssh_shunts['p_temp'][:]
                eqssh_shunts['q'] = eqssh_shunts['q_temp'][:]
            else:
                eqssh_shunts.loc[eqssh_shunts['sections'] >= eqssh_shunts['sectionNumber'], 'p'] = \
                    eqssh_shunts['p'] + eqssh_shunts['p_temp']
                eqssh_shunts.loc[eqssh_shunts['sections'] >= eqssh_shunts['sectionNumber'], 'q'] = \
                    eqssh_shunts['q'] + eqssh_shunts['q_temp']
            eqssh_shunts = eqssh_shunts[eqssh_shunts_cols]
        eqssh_shunts.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'connected': 'in_service', 'index_bus': 'bus',
            'nomU': 'vn_kv', 'p': 'p_mw', 'q': 'q_mvar'}, inplace=True)
        eqssh_shunts['step'] = 1
        eqssh_shunts['max_step'] = 1
        return eqssh_shunts

    def _convert_equivalent_branches_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EquivalentBranches.")
        eq_eb = self._prepare_equivalent_branches_cim16()
        self._copy_to_pp('impedance', eq_eb)
        self.logger.info("Created %s impedance elements in %ss." % (eq_eb.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s impedance elements from EquivalentBranches in %ss." %
                    (eq_eb.index.size, time.time() - time_start)))

    def _prepare_equivalent_branches_cim16(self) -> pd.DataFrame:
        eq_eb = pd.merge(self.cim['eq']['EquivalentBranch'],
                         pd.concat([self.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']].rename(
                             columns={'rdfId': 'BaseVoltage'}),
                             self.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']].rename(
                                 columns={'rdfId': 'BaseVoltage'})], ignore_index=True, sort=False),
                         how='left', on='BaseVoltage')
        # the r21 and x21 are optional if they are equal to r and x, so fill up missing values
        eq_eb['r21'].fillna(eq_eb['r'], inplace=True)
        eq_eb['x21'].fillna(eq_eb['x'], inplace=True)
        eq_eb['zeroR21'].fillna(eq_eb['zeroR12'], inplace=True)
        eq_eb['zeroX21'].fillna(eq_eb['zeroX12'], inplace=True)
        # set cim type
        eq_eb[sc['o_cl']] = 'EquivalentBranch'

        # add the buses
        eqb_length_before_merge = self.cim['eq']['EquivalentBranch'].index.size
        # until now self.cim['eq']['EquivalentBranch'] looks like:
        #   rdfId   name    r       ...
        #   _x01    bran1   0.056   ...
        #   _x02    bran2   0.471   ...
        # now join with the terminals and bus indexes
        eq_eb = pd.merge(eq_eb, self.bus_merge, how='left', on='rdfId')
        # now eq_eb looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    bran1   0.056   termi025        True        ...
        #   _x01    bran1   0.056   termi223        True        ...
        #   _x02    bran2   0.471   termi154        True        ...
        #   _x02    bran2   0.471   termi199        True        ...
        # if each equivalent branch got two terminals, reduce back to one row to use fast slicing
        if eq_eb.index.size != eqb_length_before_merge * 2:
            self.logger.error("There is a problem at the EquivalentBranches source data: Not each EquivalentBranch "
                              "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                              (eqb_length_before_merge * 2, eq_eb.index.size))
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="There is a problem at the EquivalentBranches source data: Not each EquivalentBranch "
                        "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                        (eqb_length_before_merge * 2, eq_eb.index.size)))
            dups = eq_eb.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The EquivalentBranch with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The EquivalentBranch data: \n%s" % eq_eb[eq_eb['rdfId'] == rdfId])
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The EquivalentBranch with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eq_eb = eq_eb[0:0]
        # sort by RDF ID and the sequenceNumber to make sure r12 and r21 are in the correct order
        eq_eb.sort_values(by=['rdfId', 'sequenceNumber'], inplace=True)
        # copy the columns which are needed to reduce the eq_eb to one row per equivalent branch
        eq_eb['rdfId_Terminal2'] = eq_eb['rdfId_Terminal'].copy()
        eq_eb['connected2'] = eq_eb['connected'].copy()
        eq_eb['index_bus2'] = eq_eb['index_bus'].copy()
        eq_eb = eq_eb.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eq_eb.rdfId_Terminal2 = eq_eb.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
        eq_eb.connected2 = eq_eb.connected2.iloc[1:].reset_index().connected2
        eq_eb.index_bus2 = eq_eb.index_bus2.iloc[1:].reset_index().index_bus2
        eq_eb.drop_duplicates(['rdfId'], keep='first', inplace=True)
        if hasattr(self.net, 'sn_mva'):
            eq_eb['sn_mva'] = self.net['sn_mva']
        else:
            eq_eb['sn_mva'] = 1.
        # calculate z base in ohm
        eq_eb['z_base'] = eq_eb.nominalVoltage ** 2 / eq_eb.sn_mva
        eq_eb['rft_pu'] = eq_eb['r'] / eq_eb['z_base']
        eq_eb['xft_pu'] = eq_eb['x'] / eq_eb['z_base']
        eq_eb['rtf_pu'] = eq_eb['r21'] / eq_eb['z_base']
        eq_eb['xtf_pu'] = eq_eb['x21'] / eq_eb['z_base']
        eq_eb['rft0_pu'] = eq_eb['zeroR12'] / eq_eb['z_base']
        eq_eb['xft0_pu'] = eq_eb['zeroX12'] / eq_eb['z_base']
        eq_eb['rtf0_pu'] = eq_eb['zeroR21'] / eq_eb['z_base']
        eq_eb['xtf0_pu'] = eq_eb['zeroX21'] / eq_eb['z_base']
        eq_eb['in_service'] = eq_eb.connected & eq_eb.connected2
        eq_eb.rename(columns={'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'rdfId': sc['o_id'],
                              'index_bus': 'from_bus', 'index_bus2': 'to_bus'}, inplace=True)
        return eq_eb

    def _convert_series_compensators_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting SeriesCompensators.")
        eq_sc = self._prepare_series_compensators_cim16()
        self._copy_to_pp('impedance', eq_sc)
        self.logger.info("Created %s impedance elements in %ss." % (eq_sc.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s impedance elements from SeriesCompensators in %ss." %
                    (eq_sc.index.size, time.time() - time_start)))

    def _prepare_series_compensators_cim16(self) -> pd.DataFrame:
        eq_sc = pd.merge(self.cim['eq']['SeriesCompensator'],
                         self.cim['eq']['BaseVoltage'][['rdfId',
                                                        'nominalVoltage']].rename(columns={'rdfId': 'BaseVoltage'}),
                         how='left', on='BaseVoltage')
        # fill the r21 and x21 values for impedance creation
        eq_sc['r21'] = eq_sc['r'].copy()
        eq_sc['x21'] = eq_sc['x'].copy()
        # set cim type
        eq_sc[sc['o_cl']] = 'SeriesCompensator'

        # add the buses
        eqs_length_before_merge = self.cim['eq']['SeriesCompensator'].index.size
        # until now self.cim['eq']['SeriesCompensator'] looks like:
        #   rdfId   name    r       ...
        #   _x01    bran1   0.056   ...
        #   _x02    bran2   0.471   ...
        # now join with the terminals and bus indexes
        eq_sc = pd.merge(eq_sc, self.bus_merge, how='left', on='rdfId')
        # now eq_sc looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    bran1   0.056   termi025        True        ...
        #   _x01    bran1   0.056   termi223        True        ...
        #   _x02    bran2   0.471   termi154        True        ...
        #   _x02    bran2   0.471   termi199        True        ...
        # if each series compensator got two terminals, reduce back to one row to use fast slicing
        if eq_sc.index.size != eqs_length_before_merge * 2:
            self.logger.error("There is a problem at the SeriesCompensators source data: Not each SeriesCompensator "
                              "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                              (eqs_length_before_merge * 2, eq_sc.index.size))
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="There is a problem at the SeriesCompensators source data: Not each SeriesCompensator "
                        "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                        (eqs_length_before_merge * 2, eq_sc.index.size)))
            dups = eq_sc.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The SeriesCompensator with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The SeriesCompensator data: \n%s" % eq_sc[eq_sc['rdfId'] == rdfId])
                self.report_container.add_log(Report(
                    level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                    message="The SeriesCompensator with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eq_sc = eq_sc[0:0]
        # sort by RDF ID and the sequenceNumber to make sure r12 and r21 are in the correct order
        eq_sc.sort_values(by=['rdfId', 'sequenceNumber'], inplace=True)
        # copy the columns which are needed to reduce the eq_sc to one row per equivalent branch
        eq_sc['rdfId_Terminal2'] = eq_sc['rdfId_Terminal'].copy()
        eq_sc['connected2'] = eq_sc['connected'].copy()
        eq_sc['index_bus2'] = eq_sc['index_bus'].copy()
        eq_sc = eq_sc.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eq_sc.rdfId_Terminal2 = eq_sc.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
        eq_sc.connected2 = eq_sc.connected2.iloc[1:].reset_index().connected2
        eq_sc.index_bus2 = eq_sc.index_bus2.iloc[1:].reset_index().index_bus2
        eq_sc.drop_duplicates(['rdfId'], keep='first', inplace=True)
        if hasattr(self.net, 'sn_mva'):
            eq_sc['sn_mva'] = self.net['sn_mva']
        else:
            eq_sc['sn_mva'] = 1.
        # calculate z base in ohm
        eq_sc['z_base'] = eq_sc.nominalVoltage ** 2 / eq_sc.sn_mva
        eq_sc['rft_pu'] = eq_sc['r'] / eq_sc['z_base']
        eq_sc['xft_pu'] = eq_sc['x'] / eq_sc['z_base']
        eq_sc['rtf_pu'] = eq_sc['r21'] / eq_sc['z_base']
        eq_sc['xtf_pu'] = eq_sc['x21'] / eq_sc['z_base']
        eq_sc['rft0_pu'] = eq_sc['r0'] / eq_sc['z_base']
        eq_sc['xft0_pu'] = eq_sc['x0'] / eq_sc['z_base']
        eq_sc['rtf0_pu'] = eq_sc['r0'] / eq_sc['z_base']
        eq_sc['xtf0_pu'] = eq_sc['x0'] / eq_sc['z_base']
        eq_sc['in_service'] = eq_sc.connected & eq_sc.connected2
        eq_sc.rename(columns={'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'rdfId': sc['o_id'],
                              'index_bus': 'from_bus', 'index_bus2': 'to_bus'}, inplace=True)
        return eq_sc

    def _convert_equivalent_injections_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EquivalentInjections.")
        eqssh_ei = self._prepare_equivalent_injections_cim16()
        # split up to wards and xwards: the wards have no regulation
        eqssh_ei_wards = eqssh_ei.loc[~eqssh_ei.regulationStatus]
        eqssh_ei_xwards = eqssh_ei.loc[eqssh_ei.regulationStatus]
        self._copy_to_pp('ward', eqssh_ei_wards)
        self._copy_to_pp('xward', eqssh_ei_xwards)
        self.logger.info("Created %s wards and %s extended ward elements in %ss." %
                         (eqssh_ei_wards.index.size, eqssh_ei_xwards.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s wards and %s extended ward elements from EquivalentInjections in %ss." %
                    (eqssh_ei_wards.index.size, eqssh_ei_xwards.index.size, time.time() - time_start)))

    def _prepare_equivalent_injections_cim16(self) -> pd.DataFrame:
        eqssh_ei = self._merge_eq_ssh_profile('EquivalentInjection', add_cim_type_column=True)
        eq_base_voltages = pd.concat([self.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']],
                                      self.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']]], sort=False)
        eq_base_voltages.drop_duplicates(subset=['rdfId'], inplace=True)
        eq_base_voltages.rename(columns={'rdfId': 'BaseVoltage'}, inplace=True)
        eqssh_ei = pd.merge(eqssh_ei, eq_base_voltages, how='left', on='BaseVoltage')
        eqssh_ei = pd.merge(eqssh_ei, self.bus_merge, how='left', on='rdfId')
        # maybe the BaseVoltage is not given, also get the nominalVoltage from the buses
        eqssh_ei = pd.merge(eqssh_ei, self.net.bus[['vn_kv']], how='left', left_on='index_bus', right_index=True)
        eqssh_ei.nominalVoltage.fillna(eqssh_ei.vn_kv, inplace=True)
        eqssh_ei['regulationStatus'].fillna(False, inplace=True)
        eqssh_ei['vm_pu'] = eqssh_ei.regulationTarget / eqssh_ei.nominalVoltage
        eqssh_ei.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'connected': 'in_service',
                                 'index_bus': 'bus', 'p': 'ps_mw', 'q': 'qs_mvar', 'r': 'r_ohm', 'x': 'x_ohm'},
                        inplace=True)
        eqssh_ei['pz_mw'] = 0.
        eqssh_ei['qz_mvar'] = 0.
        return eqssh_ei

    def _convert_power_transformers_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting PowerTransformers.")

        eq_power_transformers = self._prepare_power_transformers_cim16()
        # split the power transformers into two and three windings
        power_trafo_counts = eq_power_transformers.PowerTransformer.value_counts()
        power_trafo2w = power_trafo_counts[power_trafo_counts == 2].index.tolist()
        power_trafo3w = power_trafo_counts[power_trafo_counts == 3].index.tolist()

        eq_power_transformers.set_index('PowerTransformer', inplace=True)
        power_trafo2w = eq_power_transformers.loc[power_trafo2w].reset_index()
        power_trafo3w = eq_power_transformers.loc[power_trafo3w].reset_index()

        if power_trafo2w.index.size > 0:
            # process the two winding transformers
            self._create_trafo_characteristics('trafo', power_trafo2w)
            power_trafo2w = self._prepare_trafos_cim16(power_trafo2w)
            self._copy_to_pp('trafo', power_trafo2w)
            self.power_trafo2w = power_trafo2w

        if power_trafo3w.index.size > 0:
            # process the three winding transformers
            self._create_trafo_characteristics('trafo3w', power_trafo3w)
            power_trafo3w = self._prepare_trafo3w_cim16(power_trafo3w)
            self._copy_to_pp('trafo3w', power_trafo3w)
            self.power_trafo3w = power_trafo3w

        self.logger.info("Created %s 2w trafos and %s 3w trafos in %ss." %
                         (power_trafo2w.index.size, power_trafo3w.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s 2w trafos and %s 3w trafos from PowerTransformers in %ss." %
                    (power_trafo2w.index.size, power_trafo3w.index.size, time.time() - time_start)))

    def _create_trafo_characteristics(self, trafo_type, trafo_df_origin):
        if 'id_characteristic' not in trafo_df_origin.columns:
            trafo_df_origin['id_characteristic'] = np.NaN
        if 'characteristic_temp' not in self.net.keys():
            self.net['characteristic_temp'] = pd.DataFrame(columns=['id_characteristic', 'step', 'vk_percent',
                                                                    'vkr_percent', 'vkr_hv_percent', 'vkr_mv_percent',
                                                                    'vkr_lv_percent', 'vk_hv_percent', 'vk_mv_percent',
                                                                    'vk_lv_percent'])
        # get the TablePoints
        ptct = self.cim['eq']['PhaseTapChangerTabular'][['TransformerEnd', 'PhaseTapChangerTable']]
        ptct = pd.merge(ptct, self.cim['eq']['PhaseTapChangerTablePoint'][
            ['PhaseTapChangerTable', 'step', 'r', 'x']], how='left', on='PhaseTapChangerTable')
        # append the ratio tab changers
        ptct_ratio = self.cim['eq']['RatioTapChanger'][['TransformerEnd', 'RatioTapChangerTable']]
        ptct_ratio = pd.merge(ptct_ratio, self.cim['eq']['RatioTapChangerTablePoint'][
            ['RatioTapChangerTable', 'step', 'r', 'x']], how='left', on='RatioTapChangerTable')
        ptct = pd.concat([ptct, ptct_ratio], ignore_index=True, sort=False)
        ptct.rename(columns={'step': 'tabular_step', 'r': 'r_dev', 'x': 'x_dev', 'TransformerEnd': sc['pte_id']},
                    inplace=True)
        ptct.drop(columns=['PhaseTapChangerTable'], inplace=True)
        if trafo_type == 'trafo':
            trafo_df = trafo_df_origin.sort_values(['PowerTransformer', 'endNumber']).reset_index()
            # precessing the transformer data
            # a list of transformer parameters which are used for each transformer winding
            copy_list = ['ratedU', 'r', 'x', sc['pte_id'], 'neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in copy_list:
                # copy the columns which are required for each winding
                trafo_df[one_item + '_lv'] = trafo_df[one_item].copy()
                # cut the first element from the copied columns
                trafo_df[one_item + '_lv'] = trafo_df[one_item + '_lv'].iloc[1:].reset_index()[
                    one_item + '_lv']
            del copy_list, one_item
            fillna_list = ['neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            # just keep one transformer
            trafo_df.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)
            # merge the trafos with the tap changers
            trafo_df = pd.merge(trafo_df, ptct, how='left', on=sc['pte_id'])
            trafo_df = pd.merge(trafo_df, ptct.rename(columns={'tabular_step': 'tabular_step_lv', 'r_dev': 'r_dev_lv',
                                                               'x_dev': 'x_dev_lv',
                                                               sc['pte_id']: sc['pte_id']+'_lv'}),
                                how='left', on=sc['pte_id']+'_lv')
            fillna_list = ['tabular_step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            trafo_df.dropna(subset=['r_dev', 'r_dev_lv'], how='all', inplace=True)
            fillna_list = ['r_dev', 'r_dev_lv', 'x_dev', 'x_dev_lv']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(0, inplace=True)
            # special fix for the case that the impedance is given at the hv side but the tap changer is attached at
            # the lv side (so r_lv = 0 and r_dev_lv > 0):
            trafo_df.loc[(trafo_df['r_dev_lv'] != 0) & (trafo_df['r_lv'] == 0) & (trafo_df['r_dev'] == 0), 'r_dev'] = \
                trafo_df.loc[(trafo_df['r_dev_lv'] != 0) & (trafo_df['r_lv'] == 0) & (trafo_df['r_dev'] == 0),
                             'r_dev_lv']
            trafo_df.loc[(trafo_df['x_dev_lv'] != 0) & (trafo_df['x_lv'] == 0) & (trafo_df['x_dev'] == 0), 'x_dev'] = \
                trafo_df.loc[(trafo_df['x_dev_lv'] != 0) & (trafo_df['x_lv'] == 0) & (trafo_df['x_dev'] == 0),
                             'x_dev_lv']
            trafo_df['r'] = trafo_df['r'] + trafo_df['r'] * trafo_df['r_dev'] / 100
            trafo_df['r_lv'] = trafo_df['r_lv'] * (1 + trafo_df['r_dev_lv'] / 100)
            trafo_df['x'] = trafo_df['x'] + trafo_df['x'] * trafo_df['x_dev'] / 100
            trafo_df['x_lv'] = trafo_df['x_lv'] * (1 + trafo_df['x_dev_lv'] / 100)

            # calculate vkr_percent and vk_percent
            trafo_df['vkr_percent'] = \
                abs(trafo_df.r) * trafo_df.ratedS * 100 / trafo_df.ratedU ** 2 + \
                abs(trafo_df.r_lv) * trafo_df.ratedS * 100 / trafo_df.ratedU_lv ** 2
            trafo_df['vk_percent'] = \
                (abs(trafo_df.r) ** 2 + abs(trafo_df.x) ** 2) ** 0.5 * \
                (trafo_df.ratedS * 1e3) / (10. * trafo_df.ratedU ** 2) + \
                (abs(trafo_df.r_lv) ** 2 + abs(trafo_df.x_lv) ** 2) ** 0.5 * \
                (trafo_df.ratedS * 1e3) / (10. * trafo_df.ratedU_lv ** 2)
            trafo_df['tabular_step'] = trafo_df['tabular_step'].astype(int)
            append_dict = dict({'id_characteristic': [], 'step': [], 'vk_percent': [], 'vkr_percent': []})
        else:
            trafo_df = trafo_df_origin.copy()
            trafo_df = trafo_df.sort_values(['PowerTransformer', 'endNumber']).reset_index()
            # copy the required fields for middle and low voltage
            copy_list = ['ratedS', 'ratedU', 'r', 'x', sc['pte_id'], 'neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in copy_list:
                # copy the columns which are required for each winding
                trafo_df[one_item + '_mv'] = trafo_df[one_item].copy()
                trafo_df[one_item + '_lv'] = trafo_df[one_item].copy()
                # cut the first (or first two) element(s) from the copied columns
                trafo_df[one_item + '_mv'] = trafo_df[one_item + '_mv'].iloc[1:].reset_index()[
                    one_item + '_mv']
                trafo_df[one_item + '_lv'] = trafo_df[one_item + '_lv'].iloc[2:].reset_index()[
                    one_item + '_lv']
            del copy_list, one_item
            fillna_list = ['neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_mv'], inplace=True)
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            # just keep one transformer
            trafo_df.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)
            # merge the trafos with the tap changers
            trafo_df = pd.concat([pd.merge(trafo_df, ptct, how='left', on=sc['pte_id']),
                                  pd.merge(trafo_df,
                                           ptct.rename(columns={'tabular_step': 'tabular_step_mv', 'r_dev': 'r_dev_mv',
                                                                'x_dev': 'x_dev_mv',
                                                                sc['pte_id']: sc['pte_id'] + '_mv'}),
                                           how='left', on=sc['pte_id'] + '_mv'),
                                  pd.merge(trafo_df,
                                           ptct.rename(columns={'tabular_step': 'tabular_step_lv', 'r_dev': 'r_dev_lv',
                                                                'x_dev': 'x_dev_lv',
                                                                sc['pte_id']: sc['pte_id'] + '_lv'}),
                                           how='left', on=sc['pte_id'] + '_lv')
                                  ], ignore_index=True, sort=False)
            # remove elements with mor than one tap changer per trafo
            trafo_df = trafo_df.loc[(~trafo_df.duplicated(subset=['PowerTransformer', 'tabular_step'], keep=False)) | (
                ~trafo_df.RatioTapChangerTable.isna())]
            fillna_list = ['tabular_step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_mv'], inplace=True)
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            trafo_df.dropna(subset=['r_dev', 'r_dev_mv', 'r_dev_lv'], how='all', inplace=True)
            fillna_list = ['r_dev', 'r_dev_mv', 'r_dev_lv', 'x_dev', 'x_dev_mv', 'x_dev_lv']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(0, inplace=True)
            # calculate vkr_percent and vk_percent
            trafo_df['r'] = trafo_df['r'] * (1 + trafo_df['r_dev'] / 100)
            trafo_df['r_mv'] = trafo_df['r_mv'] * (1 + trafo_df['r_dev_mv'] / 100)
            trafo_df['r_lv'] = trafo_df['r_lv'] * (1 + trafo_df['r_dev_lv'] / 100)
            trafo_df['x'] = trafo_df['x'] * (1 + trafo_df['x_dev'] / 100)
            trafo_df['x_mv'] = trafo_df['x_mv'] * (1 + trafo_df['x_dev_mv'] / 100)
            trafo_df['x_lv'] = trafo_df['x_lv'] * (1 + trafo_df['x_dev_lv'] / 100)

            trafo_df['min_s_hvmv'] = trafo_df[["ratedS", "ratedS_mv"]].min(axis=1)
            trafo_df['min_s_mvlv'] = trafo_df[["ratedS_mv", "ratedS_lv"]].min(axis=1)
            trafo_df['min_s_lvhv'] = trafo_df[["ratedS_lv", "ratedS"]].min(axis=1)

            trafo_df['vkr_hv_percent'] = \
                (trafo_df.r + trafo_df.r_mv * (trafo_df.ratedU / trafo_df.ratedU_mv) ** 2) * \
                trafo_df.min_s_hvmv * 100 / trafo_df.ratedU ** 2
            trafo_df['vkr_mv_percent'] = \
                (trafo_df.r_mv + trafo_df.r_lv * (trafo_df.ratedU_mv / trafo_df.ratedU_lv) ** 2) * \
                trafo_df.min_s_mvlv * 100 / trafo_df.ratedU_mv ** 2
            trafo_df['vkr_lv_percent'] = \
                (trafo_df.r_lv + trafo_df.r * (trafo_df.ratedU_lv / trafo_df.ratedU) ** 2) * \
                trafo_df.min_s_lvhv * 100 / trafo_df.ratedU_lv ** 2
            trafo_df['vk_hv_percent'] = \
                ((trafo_df.r + trafo_df.r_mv * (trafo_df.ratedU / trafo_df.ratedU_mv) ** 2) ** 2 +
                 (trafo_df.x + trafo_df.x_mv * (
                         trafo_df.ratedU / trafo_df.ratedU_mv) ** 2) ** 2) ** 0.5 * \
                trafo_df.min_s_hvmv * 100 / trafo_df.ratedU ** 2
            trafo_df['vk_mv_percent'] = \
                ((trafo_df.r_mv + trafo_df.r_lv * (
                        trafo_df.ratedU_mv / trafo_df.ratedU_lv) ** 2) ** 2 +
                 (trafo_df.x_mv + trafo_df.x_lv * (
                         trafo_df.ratedU_mv / trafo_df.ratedU_lv) ** 2) ** 2) ** 0.5 * \
                trafo_df.min_s_mvlv * 100 / trafo_df.ratedU_mv ** 2
            trafo_df['vk_lv_percent'] = \
                ((trafo_df.r_lv + trafo_df.r * (trafo_df.ratedU_lv / trafo_df.ratedU) ** 2) ** 2 +
                 (trafo_df.x_lv + trafo_df.x * (
                         trafo_df.ratedU_lv / trafo_df.ratedU) ** 2) ** 2) ** 0.5 * \
                trafo_df.min_s_lvhv * 100 / trafo_df.ratedU_lv ** 2
            trafo_df['tabular_step'] = trafo_df['tabular_step'].astype(int)
            append_dict = dict({'id_characteristic': [], 'step': [], 'vkr_hv_percent': [], 'vkr_mv_percent': [],
                                'vkr_lv_percent': [], 'vk_hv_percent': [], 'vk_mv_percent': [], 'vk_lv_percent': []})

        def append_row(res_dict, id_c, row, cols):
            res_dict['id_characteristic'].append(id_c)
            res_dict['step'].append(row.tabular_step)
            for variable in ['vkr_percent', 'vk_percent', 'vk_hv_percent', 'vkr_hv_percent', 'vk_mv_percent',
                             'vkr_mv_percent', 'vk_lv_percent', 'vkr_lv_percent']:
                if variable in cols:
                    res_dict[variable].append(getattr(row, variable))

        id_characteristic = self.net['characteristic_temp']['id_characteristic'].max() + 1
        if math.isnan(id_characteristic):
            id_characteristic = 0
        for one_id, one_df in trafo_df.groupby(sc['pte_id']):
            # get next id_characteristic
            if len(append_dict['id_characteristic']) > 0:
                id_characteristic = max(append_dict['id_characteristic']) + 1
            # set the ID at the corresponding transformer
            trafo_df_origin.loc[trafo_df_origin['PowerTransformer'] == trafo_df_origin.loc[
                trafo_df_origin[sc['pte_id']] == one_id, 'PowerTransformer'].values[
                0], 'id_characteristic'] = id_characteristic
            # iterate over the rows and get the desired data
            for one_row in one_df.itertuples():
                # to add only selected characteristic data instead of all available data, disable the next line and
                # uncomment the rest
                append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # if one_row.tabular_step == one_row.highStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # elif one_row.tabular_step == one_row.lowStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # elif one_row.tabular_step == one_row.neutralStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # elif one_row.tabular_step == one_row.step and one_row.step != one_row.highStep \
                #         and one_row.step != one_row.lowStep and one_row.step != one_row.neutralStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
        self.net['characteristic_temp'] = pd.concat([self.net['characteristic_temp'], pd.DataFrame(append_dict)],
                                                    ignore_index=True, sort=False)
        self.net['characteristic_temp']['step'] = self.net['characteristic_temp']['step'].astype(int)
        self.net['trafo_df'] = trafo_df  # todo remove

    def _create_characteristic_object(self, net, trafo_type: str, trafo_id: List, characteristic_df: pd.DataFrame):
        self.logger.info("Adding characteristic object for trafo_type: %s and trafo_id: %s" % (trafo_type, trafo_id))
        for variable in ['vkr_percent', 'vk_percent', 'vk_hv_percent', 'vkr_hv_percent', 'vk_mv_percent',
                         'vkr_mv_percent', 'vk_lv_percent', 'vkr_lv_percent']:
            if variable in characteristic_df.columns:
                pandapower.control.create_trafo_characteristics(net, trafo_type, trafo_id, variable,
                                                                [list(characteristic_df['step'].values)],
                                                                [list(characteristic_df[variable].values)])

    def _prepare_power_transformers_cim16(self) -> pd.DataFrame:
        eq_power_transformers = self.cim['eq']['PowerTransformer'][['rdfId', 'name', 'isPartOfGeneratorUnit']]
        eq_power_transformers[sc['o_cl']] = 'PowerTransformer'
        eq_power_transformer_ends = self.cim['eq']['PowerTransformerEnd'][
            ['rdfId', 'PowerTransformer', 'endNumber', 'Terminal', 'ratedS', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0',
             'phaseAngleClock', 'connectionKind', 'grounded']]

        # merge and append the tap changers
        eqssh_tap_changers = pd.merge(self.cim['eq']['RatioTapChanger'][[
            'rdfId', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement',
            'TapChangerControl']],
                                      self.cim['ssh']['RatioTapChanger'][['rdfId', 'step']], how='left', on='rdfId')
        eqssh_tap_changers[sc['tc']] = 'RatioTapChanger'
        eqssh_tap_changers[sc['tc_id']] = eqssh_tap_changers['rdfId'].copy()
        eqssh_tap_changers_linear = pd.merge(self.cim['eq']['PhaseTapChangerLinear'],
                                             self.cim['ssh']['PhaseTapChangerLinear'], how='left', on='rdfId')
        eqssh_tap_changers_linear['stepVoltageIncrement'] = .001
        eqssh_tap_changers_linear[sc['tc']] = 'PhaseTapChangerLinear'
        eqssh_tap_changers_linear[sc['tc_id']] = eqssh_tap_changers_linear['rdfId'].copy()
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, eqssh_tap_changers_linear], ignore_index=True, sort=False)
        eqssh_tap_changers_async = pd.merge(self.cim['eq']['PhaseTapChangerAsymmetrical'],
                                            self.cim['ssh']['PhaseTapChangerAsymmetrical'], how='left', on='rdfId')
        eqssh_tap_changers_async['stepVoltageIncrement'] = eqssh_tap_changers_async['voltageStepIncrement'][:]
        eqssh_tap_changers_async.drop(columns=['voltageStepIncrement'], inplace=True)
        eqssh_tap_changers_async[sc['tc']] = 'PhaseTapChangerAsymmetrical'
        eqssh_tap_changers_async[sc['tc_id']] = eqssh_tap_changers_async['rdfId'].copy()
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, eqssh_tap_changers_async], ignore_index=True, sort=False)
        eqssh_ratio_tap_changers_sync = pd.merge(self.cim['eq']['PhaseTapChangerSymmetrical'],
                                                 self.cim['ssh']['PhaseTapChangerSymmetrical'], how='left', on='rdfId')
        eqssh_ratio_tap_changers_sync['stepVoltageIncrement'] = eqssh_ratio_tap_changers_sync['voltageStepIncrement']
        eqssh_ratio_tap_changers_sync.drop(columns=['voltageStepIncrement'], inplace=True)
        eqssh_ratio_tap_changers_sync[sc['tc']] = 'PhaseTapChangerSymmetrical'
        eqssh_ratio_tap_changers_sync[sc['tc_id']] = eqssh_ratio_tap_changers_sync['rdfId'].copy()
        eqssh_tap_changers = \
            pd.concat([eqssh_tap_changers, eqssh_ratio_tap_changers_sync], ignore_index=True, sort=False)
        # convert the PhaseTapChangerTabular to one tap changer
        ptct = pd.merge(self.cim['eq']['PhaseTapChangerTabular'][['rdfId', 'TransformerEnd', 'PhaseTapChangerTable',
                                                                  'highStep', 'lowStep', 'neutralStep']],
                        self.cim['ssh']['PhaseTapChangerTabular'][['rdfId', 'step']], how='left', on='rdfId')
        ptct.rename(columns={'step': 'current_step'}, inplace=True)
        ptct = pd.merge(ptct, self.cim['eq']['PhaseTapChangerTablePoint'][
            ['PhaseTapChangerTable', 'step', 'angle', 'ratio']], how='left', on='PhaseTapChangerTable')
        for one_id, one_df in ptct.groupby('TransformerEnd'):
            drop_index = one_df[one_df['step'] != one_df['current_step']].index.values
            keep_index = one_df[one_df['step'] == one_df['current_step']].index.values
            if keep_index.size > 0:
                keep_index = keep_index[0]
            else:
                self.logger.warning("Ignoring PhaseTapChangerTabular with ID: %s. The current tap position is missing "
                                    "in the PhaseTapChangerTablePoints!" % one_id)
                ptct.drop(drop_index, inplace=True)
                continue
            current_step = one_df['current_step'].iloc[0]
            one_df.set_index('step', inplace=True)
            neutral_step = one_df['neutralStep'].iloc[0]
            ptct.drop(drop_index, inplace=True)
            # ptct.loc[keep_index, 'angle'] =
            # one_df.loc[current_step, 'angle'] / max(1, abs(current_step - neutral_step))
            ptct.loc[keep_index, 'angle'] = one_df.loc[current_step, 'angle']  # todo fix if pp supports them
            ptct.loc[keep_index, 'ratio'] = \
                (one_df.loc[current_step, 'ratio'] - 1) * 100 / max(1, abs(current_step - neutral_step))
        ptct.drop(columns=['rdfId', 'PhaseTapChangerTable', 'step'], inplace=True)
        ptct.rename(columns={'current_step': 'step'}, inplace=True)
        # ptct['stepPhaseShiftIncrement'] = ptct['angle'][:]  # todo fix if pp supports them
        ptct['stepVoltageIncrement'] = ptct['ratio'][:]
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, ptct], ignore_index=True, sort=False)
        del eqssh_tap_changers_linear, eqssh_tap_changers_async, eqssh_ratio_tap_changers_sync

        # remove duplicated TapChanger: A Transformer may have one RatioTapChanger and one PhaseTapChanger
        # self.logger.info("eqssh_tap_changers.index.size: %s" % eqssh_tap_changers.index.size)
        # self.logger.info("dups:")
        # for _, one_dup in eqssh_tap_changers[eqssh_tap_changers.duplicated('TransformerEnd', keep=False)].iterrows():
        #     self.logger.info(one_dup)  # no example for testing found
        eqssh_tap_changers.drop_duplicates(subset=['TransformerEnd'], inplace=True)
        # prepare the controllers
        eq_ssh_tap_controllers = self._merge_eq_ssh_profile('TapChangerControl')
        eq_ssh_tap_controllers = \
            eq_ssh_tap_controllers[['rdfId', 'Terminal', 'discrete', 'enabled', 'targetValue', 'targetDeadband']]
        eq_ssh_tap_controllers.rename(columns={'rdfId': 'TapChangerControl'}, inplace=True)
        # first merge with the VoltageLimits
        eq_vl = self.cim['eq']['VoltageLimit'][['OperationalLimitSet', 'OperationalLimitType', 'value']]
        eq_vl = pd.merge(eq_vl, self.cim['eq']['OperationalLimitType'][['rdfId', 'limitType']].rename(
            columns={'rdfId': 'OperationalLimitType'}), how='left', on='OperationalLimitType')
        eq_vl = pd.merge(eq_vl, self.cim['eq']['OperationalLimitSet'][['rdfId', 'Terminal']].rename(
            columns={'rdfId': 'OperationalLimitSet'}), how='left', on='OperationalLimitSet')
        eq_vl = eq_vl[['value', 'limitType', 'Terminal']]
        eq_vl_low = eq_vl.loc[eq_vl['limitType'] == 'lowVoltage'][['value', 'Terminal']].rename(
            columns={'value': 'c_vm_lower_pu'})
        eq_vl_up = eq_vl.loc[eq_vl['limitType'] == 'highVoltage'][['value', 'Terminal']].rename(
            columns={'value': 'c_vm_upper_pu'})
        eq_vl = pd.merge(eq_vl_low, eq_vl_up, how='left', on='Terminal')
        eq_ssh_tap_controllers = pd.merge(eq_ssh_tap_controllers, eq_vl, how='left', on='Terminal')
        eq_ssh_tap_controllers['c_Terminal'] = eq_ssh_tap_controllers['Terminal'][:]
        eq_ssh_tap_controllers.rename(columns={'Terminal': 'rdfId', 'enabled': 'c_in_service',
                                               'targetValue': 'c_vm_set_pu', 'targetDeadband': 'c_tol'}, inplace=True)
        # get the Terminal, ConnectivityNode and bus voltage
        eq_ssh_tap_controllers = \
            pd.merge(eq_ssh_tap_controllers, pd.concat([self.cim['eq']['Terminal'], self.cim['eq_bd']['Terminal']],
                                                       ignore_index=True, sort=False)[
                ['rdfId', 'ConnectivityNode']], how='left', on='rdfId')
        eq_ssh_tap_controllers.drop(columns=['rdfId'], inplace=True)
        eq_ssh_tap_controllers.rename(columns={'ConnectivityNode': sc['o_id']}, inplace=True)
        eq_ssh_tap_controllers = pd.merge(eq_ssh_tap_controllers,
                                          self.net['bus'].reset_index(level=0)[['index', sc['o_id'], 'vn_kv']],
                                          how='left', on=sc['o_id'])
        eq_ssh_tap_controllers.drop(columns=[sc['o_id']], inplace=True)
        eq_ssh_tap_controllers.rename(columns={'index': 'c_bus_id'}, inplace=True)
        eq_ssh_tap_controllers['c_vm_set_pu'] = eq_ssh_tap_controllers['c_vm_set_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_tol'] = eq_ssh_tap_controllers['c_tol'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_vm_lower_pu'] = \
            eq_ssh_tap_controllers['c_vm_lower_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_vm_upper_pu'] = \
            eq_ssh_tap_controllers['c_vm_upper_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers.drop(columns=['vn_kv'], inplace=True)

        eqssh_tap_changers = pd.merge(eqssh_tap_changers, eq_ssh_tap_controllers, how='left', on='TapChangerControl')
        eqssh_tap_changers.rename(columns={'TransformerEnd': sc['pte_id']}, inplace=True)

        eq_power_transformers.rename(columns={'rdfId': 'PowerTransformer'}, inplace=True)
        eq_power_transformer_ends.rename(columns={'rdfId': sc['pte_id']}, inplace=True)
        # add the PowerTransformerEnds
        eq_power_transformers = pd.merge(eq_power_transformers, eq_power_transformer_ends, how='left',
                                         on='PowerTransformer')
        # add the Terminal and bus indexes
        eq_power_transformers = pd.merge(eq_power_transformers, self.bus_merge.drop('rdfId', axis=1),
                                         how='left', left_on='Terminal', right_on='rdfId_Terminal')
        # add the TapChangers
        eq_power_transformers = pd.merge(eq_power_transformers, eqssh_tap_changers, how='left', on=sc['pte_id'])
        return eq_power_transformers

    def _prepare_trafos_cim16(self, power_trafo2w: pd.DataFrame) -> pd.DataFrame:
        power_trafo2w = power_trafo2w.sort_values(['PowerTransformer', 'endNumber']).reset_index()
        # precessing the transformer data
        # a list of transformer parameters which are used for each transformer winding
        copy_list = ['index_bus', 'Terminal', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0', 'neutralStep', 'lowStep',
                     'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step', 'connected',
                     'phaseAngleClock', 'connectionKind', sc['pte_id'], sc['tc'], sc['tc_id'], 'grounded', 'angle']
        for one_item in copy_list:
            # copy the columns which are required for each winding
            power_trafo2w[one_item + '_lv'] = power_trafo2w[one_item].copy()
            # cut the first element from the copied columns
            power_trafo2w[one_item + '_lv'] = power_trafo2w[one_item + '_lv'].iloc[1:].reset_index()[
                one_item + '_lv']
        del copy_list, one_item
        # detect on which winding a tap changer is attached
        power_trafo2w.loc[power_trafo2w['step_lv'].notna(), 'tap_side'] = 'lv'
        power_trafo2w.loc[power_trafo2w['step'].notna(), 'tap_side'] = 'hv'
        fillna_list = ['neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step',
                       sc['tc'], sc['tc_id']]
        for one_item in fillna_list:
            power_trafo2w[one_item].fillna(power_trafo2w[one_item + '_lv'], inplace=True)
        del fillna_list, one_item
        # just keep one transformer
        power_trafo2w.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)

        power_trafo2w['pfe_kw'] = (power_trafo2w.g * power_trafo2w.ratedU ** 2 +
                                   power_trafo2w.g_lv * power_trafo2w.ratedU_lv ** 2) * 1000
        power_trafo2w['vkr_percent'] = \
            abs(power_trafo2w.r) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU ** 2 + \
            abs(power_trafo2w.r_lv) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU_lv ** 2
        power_trafo2w['x_lv_sign'] = np.sign(power_trafo2w['x_lv'])
        power_trafo2w.loc[power_trafo2w['x_lv_sign'] == 0, 'x_lv_sign'] = 1
        power_trafo2w['x_sign'] = np.sign(power_trafo2w['x'])
        power_trafo2w.loc[power_trafo2w['x_sign'] == 0, 'x_sign'] = 1
        power_trafo2w['vk_percent'] = \
            (power_trafo2w.r ** 2 + power_trafo2w.x ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU ** 2) + \
            (power_trafo2w.r_lv ** 2 + power_trafo2w.x_lv ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU_lv ** 2)
        power_trafo2w['vk_percent'] = power_trafo2w['x_lv_sign'] * power_trafo2w['vk_percent']
        power_trafo2w['vk_percent'] = power_trafo2w['x_sign'] * power_trafo2w['vk_percent']
        power_trafo2w['i0_percent'] = \
            (((power_trafo2w.b * power_trafo2w.ratedU ** 2) ** 2 +
              (power_trafo2w.g * power_trafo2w.ratedU ** 2) ** 2) ** .5 +
             ((power_trafo2w.b_lv * power_trafo2w.ratedU_lv ** 2) ** 2 +
              (power_trafo2w.g_lv * power_trafo2w.ratedU_lv ** 2) ** 2) ** .5) * 100 / power_trafo2w.ratedS
        power_trafo2w['vkr0_percent'] = \
            abs(power_trafo2w.r0) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU ** 2 + \
            abs(power_trafo2w.r0_lv) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU_lv ** 2
        power_trafo2w['x0_lv_sign'] = np.sign(power_trafo2w['x0_lv'])
        power_trafo2w.loc[power_trafo2w['x0_lv_sign'] == 0, 'x0_lv_sign'] = 1
        power_trafo2w['x0_sign'] = np.sign(power_trafo2w['x0'])
        power_trafo2w.loc[power_trafo2w['x0_sign'] == 0, 'x0_sign'] = 1
        power_trafo2w['vk0_percent'] = \
            (power_trafo2w.r0 ** 2 + power_trafo2w.x0 ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU ** 2) + \
            (power_trafo2w.r0_lv ** 2 + power_trafo2w.x0_lv ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU_lv ** 2)
        power_trafo2w['vk0_percent'] = power_trafo2w['x0_lv_sign'] * power_trafo2w['vk0_percent']
        power_trafo2w['vk0_percent'] = power_trafo2w['x0_sign'] * power_trafo2w['vk0_percent']
        power_trafo2w['std_type'] = None
        power_trafo2w['df'] = 1.
        # todo remove if pp supports phase shifter
        if power_trafo2w.loc[power_trafo2w['angle'].notna()].index.size > 0:
            self.logger.warning("Modifying angle from 2W transformers. This kind of angle regulation is currently not "
                                "supported by pandapower! Affected transformers: \n%s" %
                                power_trafo2w.loc[power_trafo2w['angle'].notna()])
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="Modifying angle from 2W transformers. This kind of angle regulation is currently not "
                        "supported by pandapower! Affected transformers: \n%s" %
                        power_trafo2w.loc[power_trafo2w['angle'].notna()]))
        power_trafo2w['phaseAngleClock_temp'] = power_trafo2w['phaseAngleClock'].copy()
        power_trafo2w['phaseAngleClock'] = power_trafo2w['angle'] / 30
        power_trafo2w['phaseAngleClock'].fillna(power_trafo2w['angle_lv'] * -1 / 30, inplace=True)
        power_trafo2w['phaseAngleClock'].fillna(power_trafo2w['phaseAngleClock_temp'], inplace=True)
        power_trafo2w['phaseAngleClock_lv'].fillna(0, inplace=True)
        power_trafo2w['shift_degree'] = power_trafo2w['phaseAngleClock'].astype(float).fillna(
            power_trafo2w['phaseAngleClock_lv'].astype(float)) * 30
        power_trafo2w['parallel'] = 1
        power_trafo2w['tap_phase_shifter'] = False
        power_trafo2w['in_service'] = power_trafo2w.connected & power_trafo2w.connected_lv
        power_trafo2w['connectionKind'].fillna('', inplace=True)
        power_trafo2w['connectionKind_lv'].fillna('', inplace=True)
        power_trafo2w.loc[~power_trafo2w['grounded'].astype('bool'), 'connectionKind'] = \
            power_trafo2w.loc[~power_trafo2w['grounded'].astype('bool'), 'connectionKind'].str.replace('n', '')
        power_trafo2w.loc[~power_trafo2w['grounded_lv'].astype('bool'), 'connectionKind_lv'] = \
            power_trafo2w.loc[~power_trafo2w['grounded_lv'].astype('bool'), 'connectionKind_lv'].str.replace('n', '')
        power_trafo2w['vector_group'] = power_trafo2w.connectionKind + power_trafo2w.connectionKind_lv
        power_trafo2w.loc[power_trafo2w['vector_group'] == '', 'vector_group'] = None
        power_trafo2w.rename(columns={
            'PowerTransformer': sc['o_id'], 'Terminal': sc['t_hv'], 'Terminal_lv': sc['t_lv'],
            sc['pte_id']: sc['pte_id_hv'], sc['pte_id']+'_lv': sc['pte_id_lv'], 'index_bus': 'hv_bus',
            'index_bus_lv': 'lv_bus', 'neutralStep': 'tap_neutral', 'lowStep': 'tap_min', 'highStep': 'tap_max',
            'step': 'tap_pos', 'stepVoltageIncrement': 'tap_step_percent', 'stepPhaseShiftIncrement': 'tap_step_degree',
            'isPartOfGeneratorUnit': 'power_station_unit', 'ratedU': 'vn_hv_kv', 'ratedU_lv': 'vn_lv_kv',
            'ratedS': 'sn_mva', 'xground': 'xn_ohm', 'grounded': 'oltc'}, inplace=True)
        return power_trafo2w

    def _prepare_trafo3w_cim16(self, power_trafo3w: pd.DataFrame) -> pd.DataFrame:
        power_trafo3w = power_trafo3w.sort_values(['PowerTransformer', 'endNumber']).reset_index()
        # precessing the transformer data
        # a list of transformer parameters which are used for each transformer winding
        copy_list = ['index_bus', 'Terminal', 'ratedS', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0', 'neutralStep',
                     'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step', 'connected',
                     'angle', 'phaseAngleClock', 'connectionKind', 'grounded', sc['pte_id'], sc['tc'], sc['tc_id']]
        for one_item in copy_list:
            # copy the columns which are required for each winding
            power_trafo3w[one_item + '_mv'] = power_trafo3w[one_item].copy()
            power_trafo3w[one_item + '_lv'] = power_trafo3w[one_item].copy()
            # cut the first (or first two) element(s) from the copied columns
            power_trafo3w[one_item + '_mv'] = power_trafo3w[one_item + '_mv'].iloc[1:].reset_index()[
                one_item + '_mv']
            power_trafo3w[one_item + '_lv'] = power_trafo3w[one_item + '_lv'].iloc[2:].reset_index()[
                one_item + '_lv']
        del copy_list, one_item

        # detect on which winding a tap changer is attached
        power_trafo3w.loc[power_trafo3w['step_lv'].notna(), 'tap_side'] = 'lv'
        power_trafo3w.loc[power_trafo3w['step_mv'].notna(), 'tap_side'] = 'mv'
        power_trafo3w.loc[power_trafo3w['step'].notna(), 'tap_side'] = 'hv'
        fillna_list = ['neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step',
                       sc['tc'], sc['tc_id']]
        for one_item in fillna_list:
            power_trafo3w[one_item].fillna(power_trafo3w[one_item + '_mv'], inplace=True)
            power_trafo3w[one_item].fillna(power_trafo3w[one_item + '_lv'], inplace=True)
        del fillna_list, one_item
        # just keep one transformer
        power_trafo3w.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)

        power_trafo3w['min_s_hvmv'] = power_trafo3w[["ratedS", "ratedS_mv"]].min(axis=1)
        power_trafo3w['min_s_mvlv'] = power_trafo3w[["ratedS_mv", "ratedS_lv"]].min(axis=1)
        power_trafo3w['min_s_lvhv'] = power_trafo3w[["ratedS_lv", "ratedS"]].min(axis=1)
        power_trafo3w['pfe_kw'] = \
            (power_trafo3w.g * power_trafo3w.ratedU ** 2 + power_trafo3w.g_mv * power_trafo3w.ratedU_mv ** 2
             + power_trafo3w.g_lv * power_trafo3w.ratedU_lv ** 2) * 1000
        power_trafo3w['vkr_hv_percent'] = \
            (power_trafo3w.r + power_trafo3w.r_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vkr_mv_percent'] = \
            (power_trafo3w.r_mv + power_trafo3w.r_lv * (power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vkr_lv_percent'] = \
            (power_trafo3w.r_lv + power_trafo3w.r * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['vk_hv_percent'] = \
            ((power_trafo3w.r + power_trafo3w.r_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2 +
             (power_trafo3w.x + power_trafo3w.x_mv * (
                     power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vk_mv_percent'] = \
            ((power_trafo3w.r_mv + power_trafo3w.r_lv * (
                    power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2 +
             (power_trafo3w.x_mv + power_trafo3w.x_lv * (
                     power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vk_lv_percent'] = \
            ((power_trafo3w.r_lv + power_trafo3w.r * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2 +
             (power_trafo3w.x_lv + power_trafo3w.x * (
                     power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['i0_percent'] = \
            (((power_trafo3w.b * power_trafo3w.ratedU ** 2) ** 2 +
              (power_trafo3w.g * power_trafo3w.ratedU ** 2) ** 2) ** .5 +
             ((power_trafo3w.b_mv * power_trafo3w.ratedU_mv ** 2) ** 2 +
              (power_trafo3w.g_mv * power_trafo3w.ratedU_mv ** 2) ** 2) ** .5 +
             ((power_trafo3w.b_lv * power_trafo3w.ratedU_lv ** 2) ** 2 +
              (power_trafo3w.g_lv * power_trafo3w.ratedU_lv ** 2) ** 2) ** .5) * 100 / power_trafo3w.ratedS
        power_trafo3w['vkr0_hv_percent'] = \
            (power_trafo3w.r0 + power_trafo3w.r0_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vkr0_mv_percent'] = \
            (power_trafo3w.r0_mv + power_trafo3w.r0_lv * (power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vkr0_lv_percent'] = \
            (power_trafo3w.r0_lv + power_trafo3w.r0 * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['vk0_hv_percent'] = \
            ((power_trafo3w.r0 + power_trafo3w.r0_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2 +
             (power_trafo3w.x0 + power_trafo3w.x0_mv * (
                     power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vk0_mv_percent'] = \
            ((power_trafo3w.r0_mv + power_trafo3w.r0_lv * (
                    power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2 +
             (power_trafo3w.x0_mv + power_trafo3w.x0_lv * (
                     power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vk0_lv_percent'] = \
            ((power_trafo3w.r0_lv + power_trafo3w.r0 * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2 +
             (power_trafo3w.x0_lv + power_trafo3w.x0 * (
                     power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['std_type'] = None
        # todo remove if pp supports phase shifter
        if power_trafo3w.loc[power_trafo3w['angle_mv'].notna()].index.size > 0:
            self.logger.warning("Modifying angle from 3W transformers. This kind of angle regulation is currently not "
                                "supported by pandapower! Affected transformers: \n%s" %
                                power_trafo3w.loc[power_trafo3w['angle_mv'].notna()])
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="Modifying angle from 3W transformers. This kind of angle regulation is currently not "
                        "supported by pandapower! Affected transformers: \n%s" %
                        power_trafo3w.loc[power_trafo3w['angle_mv'].notna()]))
        power_trafo3w['phaseAngleClock_temp'] = power_trafo3w['phaseAngleClock_mv'].copy()
        power_trafo3w['phaseAngleClock_mv'] = power_trafo3w['angle_mv'] * -1 / 30
        power_trafo3w['phaseAngleClock_mv'].fillna(power_trafo3w['phaseAngleClock_temp'], inplace=True)
        power_trafo3w['phaseAngleClock_mv'].fillna(0, inplace=True)
        power_trafo3w['phaseAngleClock_lv'].fillna(0, inplace=True)
        power_trafo3w['shift_mv_degree'] = power_trafo3w['phaseAngleClock_mv'] * 30
        power_trafo3w['shift_lv_degree'] = power_trafo3w['phaseAngleClock_mv'] * 30
        power_trafo3w['tap_at_star_point'] = False
        power_trafo3w['in_service'] = power_trafo3w.connected & power_trafo3w.connected_mv & power_trafo3w.connected_lv
        power_trafo3w['connectionKind'].fillna('', inplace=True)
        power_trafo3w['connectionKind_mv'].fillna('', inplace=True)
        power_trafo3w['connectionKind_lv'].fillna('', inplace=True)

        power_trafo3w.loc[~power_trafo3w['grounded'].astype('bool'), 'connectionKind'] = \
            power_trafo3w.loc[~power_trafo3w['grounded'].astype('bool'), 'connectionKind'].str.replace('n', '')
        power_trafo3w.loc[~power_trafo3w['grounded_mv'].astype('bool'), 'connectionKind_mv'] = \
            power_trafo3w.loc[~power_trafo3w['grounded_mv'].astype('bool'), 'connectionKind_mv'].str.replace('n', '')
        power_trafo3w.loc[~power_trafo3w['grounded_lv'].astype('bool'), 'connectionKind_lv'] = \
            power_trafo3w.loc[~power_trafo3w['grounded_lv'].astype('bool'), 'connectionKind_lv'].str.replace('n', '')
        power_trafo3w['vector_group'] = \
            power_trafo3w.connectionKind + power_trafo3w.connectionKind_mv + power_trafo3w.connectionKind_lv
        power_trafo3w.loc[power_trafo3w['vector_group'] == '', 'vector_group'] = None
        power_trafo3w.rename(columns={
            'PowerTransformer': sc['o_id'], 'Terminal': sc['t_hv'], 'Terminal_mv': sc['t_mv'],
            'Terminal_lv': sc['t_lv'], sc['pte_id']: sc['pte_id_hv'], sc['pte_id'] + '_mv': sc['pte_id_mv'],
            sc['pte_id'] + '_lv': sc['pte_id_lv'], 'index_bus': 'hv_bus', 'index_bus_mv': 'mv_bus',
            'index_bus_lv': 'lv_bus', 'neutralStep': 'tap_neutral', 'lowStep': 'tap_min', 'highStep': 'tap_max',
            'step': 'tap_pos', 'stepVoltageIncrement': 'tap_step_percent', 'stepPhaseShiftIncrement': 'tap_step_degree',
            'isPartOfGeneratorUnit': 'power_station_unit', 'ratedU': 'vn_hv_kv', 'ratedU_mv': 'vn_mv_kv',
            'ratedU_lv': 'vn_lv_kv', 'ratedS': 'sn_hv_mva', 'ratedS_mv': 'sn_mv_mva', 'ratedS_lv': 'sn_lv_mva'},
            inplace=True)
        return power_trafo3w

    def _create_tap_controller(self, input_df: pd.DataFrame, trafo_type: str):
        if not self.kwargs.get('create_tap_controller', True):
            self.logger.info("Skip creating transformer tap changer controller for transformer type %s." % trafo_type)
            return
        for row_index, row in input_df.loc[input_df['TapChangerControl'].notna()].iterrows():
            trafo_id = self.net[trafo_type].loc[self.net[trafo_type][sc['o_id']] == row[sc['o_id']]].index.values[0]
            trafotype = '2W' if trafo_type == 'trafo' else '3W'
            # get the controlled bus (side), assume "lv" as default
            side = 'lv'
            if sc['t_hv'] in self.net[trafo_type].columns and \
                    row['c_Terminal'] in self.net[trafo_type][sc['t_hv']].values:
                side = 'hv'
            if sc['t_mv'] in self.net[trafo_type].columns and \
                    row['c_Terminal'] in self.net[trafo_type][sc['t_mv']].values:
                side = 'mv'
            if row['discrete']:
                self.logger.info("Creating DiscreteTapControl for transformer %s." % row[sc['o_id']])
                DiscreteTapControl(self.net, trafotype=trafotype, tid=trafo_id, side=side,
                                   tol=row['c_tol'], in_service=row['c_in_service'],
                                   vm_lower_pu=row['c_vm_lower_pu'], vm_upper_pu=row['c_vm_upper_pu'])
            else:
                self.logger.info("Creating ContinuousTapControl for transformer %s." % row[sc['o_id']])
                ContinuousTapControl(self.net, trafotype=trafotype, tid=trafo_id, side=side,
                                     tol=row['c_tol'], in_service=row['c_in_service'], vm_set_pu=row['c_vm_set_pu'])

    def _copy_to_pp(self, pp_type: str, input_df: pd.DataFrame):
        self.logger.debug("Copy %s datasets to pandapower network with type %s" % (input_df.index.size, pp_type))
        if pp_type not in self.net.keys():
            self.logger.warning("Missing pandapower type %s in the pandapower network!" % pp_type)
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="Missing pandapower type %s in the pandapower network!" % pp_type))
            return
        start_index_pp_net = self.net[pp_type].index.size
        self.net[pp_type] = pd.concat([self.net[pp_type], pd.DataFrame(None, index=[list(range(input_df.index.size))])],
                                      ignore_index=True, sort=False)
        for one_attr in self.net[pp_type].columns:
            if one_attr in input_df.columns:
                self.net[pp_type][one_attr][start_index_pp_net:] = input_df[one_attr][:]

    def _add_geo_coordinates_from_gl_cim16(self):
        self.logger.info("Creating the geo coordinates from CGMES GeographicalLocation.")
        time_start = time.time()
        gl_data = pd.merge(self.cim['gl']['PositionPoint'][['Location', 'xPosition', 'yPosition', 'sequenceNumber']],
                           self.cim['gl']['Location'][['rdfId', 'PowerSystemResources']], how='left',
                           left_on='Location', right_on='rdfId')
        gl_data.drop(columns=['Location', 'rdfId'], inplace=True)
        bus_geo = gl_data.rename(columns={'PowerSystemResources': 'Substation'})
        cn = self.cim['eq']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]
        cn = pd.concat([cn, self.cim['eq_bd']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cim['tp']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cim['tp_bd']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn.rename(columns={'rdfId': sc['o_id'], 'ConnectivityNodeContainer': 'rdfId'}, inplace=True)
        cn = pd.merge(cn, self.cim['eq']['VoltageLevel'][['rdfId', 'Substation']], how='left', on='rdfId')
        cn.drop(columns=['rdfId'], inplace=True)
        buses = self.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        buses = pd.merge(buses, cn, how='left', on=sc['o_id'])
        bus_geo = pd.merge(bus_geo, buses, how='inner', on='Substation')
        bus_geo.drop(columns=['Substation'], inplace=True)
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for group_name, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo.sort_values(by='index', inplace=True)
        start_index_pp_net = self.net.bus_geodata.index.size
        self.net.bus_geodata = pd.concat([self.net.bus_geodata, pd.DataFrame(None, index=bus_geo['index'].values)],
                                         ignore_index=False, sort=False)
        self.net.bus_geodata.x[start_index_pp_net:] = bus_geo.xPosition[:]
        self.net.bus_geodata.y[start_index_pp_net:] = bus_geo.yPosition[:]
        self.net.bus_geodata.coords[start_index_pp_net:] = bus_geo.coords[:]

        # the geo coordinates for the lines
        lines = self.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = gl_data.rename(columns={'PowerSystemResources': sc['o_id']})
        line_geo = pd.merge(line_geo, lines, how='inner', on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for group_name, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo.sort_values(by='index', inplace=True)
        # now add the line coordinates
        start_index_pp_net = self.net.line_geodata.index.size
        self.net.line_geodata = pd.concat([self.net.line_geodata, pd.DataFrame(None, index=line_geo['index'].values)],
                                          ignore_index=False, sort=False)
        self.net.line_geodata.coords[start_index_pp_net:] = line_geo.coords[:]

        # now create geo coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(gl_data.rename(columns={'PowerSystemResources': sc['o_id']}),
                                  one_ele_df, how='inner', on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for group_name, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
            # now add the coordinates
            self.net[one_ele]['coords'] = self.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the GL coordinates, needed time: %ss" % (time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the GL coordinates, needed time: %ss" % (time.time() - time_start)))

    def _add_coordinates_from_dl_cim16(self, diagram_name: str = None):
        self.logger.info("Creating the coordinates from CGMES DiagramLayout.")
        time_start = time.time()
        # choose a diagram if it is not given (the first one ascending)
        if diagram_name is None:
            diagram_name = self.cim['dl']['Diagram'].sort_values(by='name')['name'].values[0]
        self.logger.debug("Choosing the geo coordinates from diagram %s" % diagram_name)
        # reduce the source data to the chosen diagram only
        diagram_rdf_id = self.cim['dl']['Diagram']['rdfId'][self.cim['dl']['Diagram']['name'] == diagram_name].values[0]
        dl_do = self.cim['dl']['DiagramObject'][self.cim['dl']['DiagramObject']['Diagram'] == diagram_rdf_id]
        dl_do.rename(columns={'rdfId': 'DiagramObject'}, inplace=True)
        dl_data = pd.merge(dl_do, self.cim['dl']['DiagramObjectPoint'], how='left', on='DiagramObject')
        dl_data.drop(columns=['rdfId', 'Diagram', 'DiagramObject'], inplace=True)
        # the coordinates for the buses
        buses = self.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        bus_geo = pd.merge(dl_data, buses, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for group_name, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo.sort_values(by='index', inplace=True)
        start_index_pp_net = self.net.bus_geodata.index.size
        self.net.bus_geodata = pd.concat([self.net.bus_geodata, pd.DataFrame(None, index=bus_geo['index'].values)],
                                         ignore_index=False, sort=False)
        self.net.bus_geodata.x[start_index_pp_net:] = bus_geo.xPosition[:]
        self.net.bus_geodata.y[start_index_pp_net:] = bus_geo.yPosition[:]
        self.net.bus_geodata.coords[start_index_pp_net:] = bus_geo.coords[:]

        # the coordinates for the lines
        lines = self.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = pd.merge(dl_data, lines, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for group_name, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo.sort_values(by='index', inplace=True)
        # now add the line coordinates
        # if there are no bus geodata in the GL profile the line geodata from DL has higher priority
        if self.net.line_geodata.index.size > 0 and line_geo.index.size > 0:
            self.net.line_geodata = self.net.line_geodata[0:0]
        self.net.line_geodata = pd.concat([self.net.line_geodata, line_geo[['coords', 'index']].set_index('index')],
                                          ignore_index=False, sort=False)

        # now create coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(dl_data, one_ele_df, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for group_name, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
            # now add the coordinates
            self.net[one_ele]['coords'] = self.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the DL coordinates, needed time: %ss" % (time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the DL coordinates, needed time: %ss" % (time.time() - time_start)))

    # noinspection PyShadowingNames
    def convert_to_pp(self, convert_line_to_switch: bool = False, line_r_limit: float = 0.1,
                      line_x_limit: float = 0.1, **kwargs) \
            -> pandapower.auxiliary.pandapowerNet:
        """
        Build the pandapower net.

        :param convert_line_to_switch: Set this parameter to True to enable line -> switch conversion. All lines with a
        resistance lower or equal than line_r_limit or a reactance lower or equal than line_x_limit will become a
        switch. Optional, default: False
        :param line_r_limit: The limit from resistance. Optional, default: 0.1
        :param line_x_limit: The limit from reactance. Optional, default: 0.1
        :return: The pandapower net.
        """
        self.logger.info("Start building the pandapower net.")
        self.report_container.add_log(Report(level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
                                             message="Start building the pandapower net."))

        # create the empty pandapower net and add the additional columns
        self.net = cim_tools.extend_pp_net_cim(self.net, override=False)

        if 'sn_mva' in kwargs.keys():
            self.net['sn_mva'] = kwargs.get('sn_mva')

        # add the CIM IDs to the pandapower network
        for one_prf, one_profile_dict in self.cim.items():
            if 'FullModel' in one_profile_dict.keys() and one_profile_dict['FullModel'].index.size > 0:
                self.net['CGMES'][one_prf] = one_profile_dict['FullModel'].set_index('rdfId').to_dict(orient='index')
        # store the BaseVoltage IDs
        self.net['CGMES']['BaseVoltage'] = \
            pd.concat([self.cim['eq']['BaseVoltage'], self.cim['eq_bd']['BaseVoltage']],
                      sort=False, ignore_index=True)[['rdfId', 'nominalVoltage']]

        # --------- convert busses ---------
        self._convert_connectivity_nodes_cim16()
        # --------- convert external networks ---------
        self._convert_external_network_injections_cim16()
        # --------- convert lines ---------
        self._convert_ac_line_segments_cim16(convert_line_to_switch, line_r_limit, line_x_limit)
        self._convert_dc_line_segments_cim16()
        # --------- convert switches ---------
        self._convert_switches_cim16()
        # --------- convert loads ---------
        self._convert_energy_consumers_cim16()
        self._convert_conform_loads_cim16()
        self._convert_non_conform_loads_cim16()
        self._convert_station_supplies_cim16()
        # --------- convert generators ---------
        self._convert_synchronous_machines_cim16()
        self._convert_asynchronous_machines_cim16()
        self._convert_energy_sources_cim16()
        # --------- convert shunt elements ---------
        self._convert_linear_shunt_compensator_cim16()
        self._convert_nonlinear_shunt_compensator_cim16()
        self._convert_static_var_compensator_cim16()
        # --------- convert impedance elements ---------
        self._convert_equivalent_branches_cim16()
        self._convert_series_compensators_cim16()
        # --------- convert extended ward and ward elements ---------
        self._convert_equivalent_injections_cim16()
        # --------- convert transformers ---------
        self._convert_power_transformers_cim16()

        # create the geo coordinates
        gl_or_dl = str(self.kwargs.get('use_GL_or_DL_profile', 'both')).lower()
        if gl_or_dl == 'gl':
            use_gl_profile = True
            use_dl_profile = False
        elif gl_or_dl == 'dl':
            use_gl_profile = False
            use_dl_profile = True
        else:
            use_gl_profile = True
            use_dl_profile = True
        if self.cim['gl']['Location'].index.size > 0 and self.cim['gl']['PositionPoint'].index.size > 0 and \
                use_gl_profile:
            try:
                self._add_geo_coordinates_from_gl_cim16()
            except Exception as e:
                self.logger.warning("Creating the geo coordinates failed, returning the net without geo coordinates!")
                self.logger.exception(e)
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="Creating the geo coordinates failed, returning the net without geo coordinates!"))
                self.report_container.add_log(Report(
                    level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                    message=traceback.format_exc()))
                self.net.bus_geodata = self.net.bus_geodata[0:0]
                self.net.line_geodata = self.net.line_geodata[0:0]
        if self.cim['dl']['Diagram'].index.size > 0 and self.cim['dl']['DiagramObject'].index.size > 0 and \
                self.cim['dl']['DiagramObjectPoint'].index.size > 0 and self.net.bus_geodata.index.size == 0 and \
                use_dl_profile:
            try:
                self._add_coordinates_from_dl_cim16(diagram_name=kwargs.get('diagram_name', None))
            except Exception as e:
                self.logger.warning("Creating the coordinates failed, returning the net without coordinates!")
                self.logger.exception(e)
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="Creating the coordinates failed, returning the net without coordinates!"))
                self.report_container.add_log(Report(level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                                                     message=traceback.format_exc()))
                self.net.bus_geodata = self.net.bus_geodata[0:0]
                self.net.line_geodata = self.net.line_geodata[0:0]
        self.net = pp_tools.set_pp_col_types(net=self.net)

        # create transformer tap controller
        if self.power_trafo2w.index.size > 0:
            # create transformer tap controller
            self._create_tap_controller(self.power_trafo2w, 'trafo')
            # create the characteristic objects for transformers
            characteristic_df_temp = \
                self.net['characteristic_temp'][['id_characteristic', 'step', 'vk_percent', 'vkr_percent']]
            for trafo_id, trafo_row in self.net.trafo.dropna(subset=['id_characteristic']).iterrows():
                characteristic_df = characteristic_df_temp.loc[
                    characteristic_df_temp['id_characteristic'] == trafo_row['id_characteristic']]
                self._create_characteristic_object(net=self.net, trafo_type='trafo', trafo_id=[trafo_id],
                                                   characteristic_df=characteristic_df)
        if self.power_trafo3w.index.size > 0:
            # create transformer tap controller
            self._create_tap_controller(self.power_trafo3w, 'trafo3w')
            # create the characteristic objects for transformers
            characteristic_df_temp = \
                self.net['characteristic_temp'][['id_characteristic', 'step', 'vkr_hv_percent', 'vkr_mv_percent',
                                                 'vkr_lv_percent', 'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent']]
            for trafo_id, trafo_row in self.net.trafo3w.dropna(subset=['id_characteristic']).iterrows():
                characteristic_df = characteristic_df_temp.loc[
                    characteristic_df_temp['id_characteristic'] == trafo_row['id_characteristic']]
                self._create_characteristic_object(net=self.net, trafo_type='trafo3w', trafo_id=[trafo_id],
                                                   characteristic_df=characteristic_df)

        self.logger.info("Running a power flow.")
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO, message="Running a power flow."))
        try:
            pp.runpp(self.net)
        except Exception as e:
            self.logger.error("Failed running a powerflow.")
            self.logger.exception(e)
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR, message="Failed running a powerflow."))
            self.report_container.add_log(Report(level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION,
                                                 message=traceback.format_exc()))
        else:
            self.logger.info("Power flow solved normal.")
            self.report_container.add_log(Report(
                level=LogLevel.INFO, code=ReportCode.INFO, message="Power flow solved normal."))
        try:
            create_measurements = kwargs.get('create_measurements', None)
            if create_measurements is not None and create_measurements.lower() == 'sv':
                CreateMeasurements(self.net, self.cim).create_measurements_from_sv()
            elif create_measurements is not None and create_measurements.lower() == 'analog':
                CreateMeasurements(self.net, self.cim).create_measurements_from_analog()
            elif create_measurements is not None:
                self.report_container.add_log(Report(
                    level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                    message="Not supported value for argument 'create_measurements', check method signature for"
                            "valid values!"))
                raise ValueError("Not supported value for argument 'create_measurements', check method signature for"
                                 "valid values!")
        except Exception as e:
            self.logger.error("Creating the measurements failed, returning the net without measurements!")
            self.logger.exception(e)
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Creating the measurements failed, returning the net without measurements!"))
            self.report_container.add_log(Report(
                level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                message=traceback.format_exc()))
            self.net.measurement = self.net.measurement[0:0]
        try:
            if kwargs.get('update_assets_from_sv', False):
                CreateMeasurements(self.net, self.cim).update_assets_from_sv()
        except Exception as e:
            self.logger.warning("Updating the assets failed!")
            self.logger.exception(e)
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Updating the assets failed!"))
            self.report_container.add_log(Report(
                level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                message=traceback.format_exc()))
        # a special fix for BB and NB mixed networks:
        # fuse boundary ConnectivityNodes with their TopologicalNodes
        bus_t = self.net.bus.reset_index(level=0, drop=False)
        bus_drop = bus_t.loc[bus_t[sc['o_prf']] == 'eq_bd', ['index', sc['o_id'], 'cim_topnode']]
        bus_drop.rename(columns={'index': 'b1'}, inplace=True)
        bus_drop = pd.merge(bus_drop, bus_t[['index', sc['o_id']]].rename(columns={'index': 'b2', sc['o_id']: 'o_id2'}),
                            how='inner', left_on='cim_topnode', right_on='o_id2')
        if bus_drop.index.size > 0:
            for b1, b2 in bus_drop[['b1', 'b2']].itertuples(index=False):
                self.logger.info("Fusing buses: b1: %s, b2: %s" % (b1, b2))
                pp.fuse_buses(self.net, b1, b2, drop=True, fuse_bus_measurements=True)
        # finally a fix for EquivalentInjections: If an EquivalentInjection is attached to boundary node, check if the
        # network behind this boundary node is attached. In this case, disable the EquivalentInjection.
        ward_t = self.net.ward.copy()
        ward_t['bus_prf'] = ward_t['bus'].map(self.net.bus[[sc['o_prf']]].to_dict().get(sc['o_prf']))
        self.net.ward.loc[(self.net.ward.bus.duplicated(keep=False) &
                           ((ward_t['bus_prf'] == 'eq_bd') | (ward_t['bus_prf'] == 'tp_bd'))), 'in_service'] = False
        xward_t = self.net.xward.copy()
        xward_t['bus_prf'] = xward_t['bus'].map(self.net.bus[[sc['o_prf']]].to_dict().get(sc['o_prf']))
        self.net.xward.loc[(self.net.xward.bus.duplicated(keep=False) &
                            ((xward_t['bus_prf'] == 'eq_bd') | (xward_t['bus_prf'] == 'tp_bd'))), 'in_service'] = False
        self.net['report_container'] = self.report_container
        return self.net

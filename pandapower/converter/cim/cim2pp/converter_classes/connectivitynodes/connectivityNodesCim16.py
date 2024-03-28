import logging
import time
from typing import Tuple

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.connectivityNodesCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class ConnectivityNodesCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_connectivity_nodes_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting ConnectivityNodes / TopologicalNodes.")
        connectivity_nodes, eqssh_terminals = self._prepare_connectivity_nodes_cim16()

        # self._create_busses(connectivity_nodes)
        self.cimConverter.copy_to_pp('bus', connectivity_nodes)

        # a prepared and modified copy of eqssh_terminals to use for lines, switches, loads, sgens and so on
        eqssh_terminals = eqssh_terminals[
            ['rdfId', 'ConductingEquipment', 'ConnectivityNode', 'sequenceNumber', 'connected']].copy()
        eqssh_terminals = eqssh_terminals.rename(columns={'rdfId': 'rdfId_Terminal'})
        eqssh_terminals = eqssh_terminals.rename(columns={'ConductingEquipment': 'rdfId'})
        # buses for merging with assets:
        bus_merge = pd.DataFrame(data=self.cimConverter.net['bus'].loc[:, [sc['o_id'], 'vn_kv']])
        bus_merge = bus_merge.rename(columns={'vn_kv': 'base_voltage_bus'})
        bus_merge = bus_merge.reset_index(level=0)
        bus_merge = bus_merge.rename(columns={'index': 'index_bus', sc['o_id']: 'ConnectivityNode'})
        bus_merge = pd.merge(eqssh_terminals, bus_merge, how='left', on='ConnectivityNode')
        self.cimConverter.bus_merge = bus_merge

        self.logger.info("Created %s busses in %ss" % (connectivity_nodes.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s busses from ConnectivityNodes / TopologicalNodes in %ss" %
                    (connectivity_nodes.index.size, time.time() - time_start)))

    def _prepare_connectivity_nodes_cim16(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # check the model: Bus-Branch or Node-Breaker: In the Bus-Branch model are no ConnectivityNodes
        node_breaker = True if self.cimConverter.cim['eq']['ConnectivityNode'].index.size > 0 else False
        # use this dictionary to store the source profile from the element (normal or boundary profile)
        cn_dict = dict({'eq': {sc['o_prf']: 'eq'}, 'eq_bd': {sc['o_prf']: 'eq_bd'},
                        'tp': {sc['o_prf']: 'tp'}, 'tp_bd': {sc['o_prf']: 'tp_bd'}})
        if node_breaker:
            # Node-Breaker model
            connectivity_nodes = pd.concat([self.cimConverter.cim['eq']['ConnectivityNode'].assign(**cn_dict['eq']),
                                            self.cimConverter.cim['eq_bd']['ConnectivityNode'].assign(
                                                **cn_dict['eq_bd'])],
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
            eq_bay = self.cimConverter.cim['eq']['Bay'].copy()
            eq_bay = eq_bay.rename(columns={'rdfId': 'ConnectivityNodeContainer'})
            eq_bay = pd.merge(self.cimConverter.cim['eq']['ConnectivityNode'][['ConnectivityNodeContainer']], eq_bay,
                              how='inner', on='ConnectivityNodeContainer')
            eq_bay = eq_bay.dropna(subset=['VoltageLevel'])
            eq_bay = pd.merge(eq_bay,
                              self.cimConverter.cim['eq']['VoltageLevel'][['rdfId', 'BaseVoltage', 'Substation']],
                              how='left', left_on='VoltageLevel', right_on='rdfId')
            eq_bay = eq_bay.drop(columns=['VoltageLevel', 'rdfId'])
            eq_bay = eq_bay.rename(columns={'ConnectivityNodeContainer': 'rdfId'})
            # now prepare the substations (3)
            # first get only the needed substation used as ConnectivityNodeContainer in ConnectivityNode
            eq_subs = pd.merge(self.cimConverter.cim['eq']['ConnectivityNode'][['ConnectivityNodeContainer']].rename(
                columns={'ConnectivityNodeContainer': 'Substation'}),
                self.cimConverter.cim['eq']['Substation'][['rdfId']].rename(columns={'rdfId': 'Substation'}),
                how='inner', on='Substation')
            # now merge them with the VoltageLevel
            eq_subs = pd.merge(self.cimConverter.cim['eq']['VoltageLevel'][['rdfId', 'BaseVoltage', 'Substation']],
                               eq_subs,
                               how='inner', on='Substation')
            eq_subs_duplicates = eq_subs[eq_subs.duplicated(['Substation'], keep='first')]
            eq_subs['rdfId'] = eq_subs['Substation']
            if eq_subs_duplicates.index.size > 0:
                self.logger.warning(
                    "More than one VoltageLevel refers to one Substation, maybe the voltages from some buses "
                    "are incorrect, the problematic VoltageLevels and Substations:\n%s" % eq_subs_duplicates)
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="More than one VoltageLevel refers to one Substation, maybe the voltages from some buses "
                            "are incorrect, the problematic VoltageLevels and Substations:\n%s" % eq_subs_duplicates))
            eq_subs = eq_subs.drop_duplicates(['rdfId'], keep='first')
            # now merge the VoltageLevel with the ConnectivityNode
            eq_voltage_levels = self.cimConverter.cim['eq']['VoltageLevel'][['rdfId', 'BaseVoltage', 'Substation']]
            eq_voltage_levels = pd.concat([eq_voltage_levels, eq_bay], ignore_index=True, sort=False)
            eq_voltage_levels = pd.concat([eq_voltage_levels, eq_subs], ignore_index=True, sort=False)
            eq_voltage_levels = eq_voltage_levels.drop_duplicates(['rdfId'], keep='first')
            del eq_bay, eq_subs, eq_subs_duplicates
            eq_substations = self.cimConverter.cim['eq']['Substation'][['rdfId', 'name']]
            eq_substations = eq_substations.rename(columns={'rdfId': 'Substation', 'name': 'name_substation'})
            eq_voltage_levels = pd.merge(eq_voltage_levels, eq_substations, how='left', on='Substation')
            eq_voltage_levels = eq_voltage_levels.drop_duplicates(subset=['rdfId'])
            eq_voltage_levels = eq_voltage_levels.rename(columns={'rdfId': 'ConnectivityNodeContainer'})

            connectivity_nodes = pd.merge(connectivity_nodes, eq_voltage_levels, how='left',
                                          on='ConnectivityNodeContainer')
            connectivity_nodes[sc['sub_id']] = connectivity_nodes['Substation'][:]
            # now prepare the BaseVoltage from the boundary profile at the ConnectivityNode (4)
            eq_bd_cns = pd.merge(self.cimConverter.cim['eq_bd']['ConnectivityNode'][['rdfId']],
                                 self.cimConverter.cim['tp_bd']['ConnectivityNode'][['rdfId', 'TopologicalNode']],
                                 how='inner', on='rdfId')
            # eq_bd_cns.drop(columns=['rdfId'], inplace=True)
            # eq_bd_cns.rename(columns={'TopologicalNode': 'rdfId'}, inplace=True)
            eq_bd_cns = pd.merge(eq_bd_cns,
                                 self.cimConverter.cim['tp_bd']['TopologicalNode'][['rdfId', 'BaseVoltage']].rename(
                                     columns={'rdfId': 'TopologicalNode'}), how='inner', on='TopologicalNode')
            # eq_bd_cns.drop(columns=['TopologicalNode'], inplace=True)
            eq_bd_cns.rename(columns={'BaseVoltage': 'BaseVoltage_2', 'TopologicalNode': 'TopologicalNode_2'},
                             inplace=True)
            connectivity_nodes = pd.merge(connectivity_nodes, eq_bd_cns, how='left', on='rdfId')
            connectivity_nodes['BaseVoltage'].fillna(connectivity_nodes['BaseVoltage_2'], inplace=True)
            connectivity_nodes = connectivity_nodes.drop(columns=['BaseVoltage_2'])
            # check if there is a mix between BB and NB models
            terminals_temp = \
                self.cimConverter.cim['eq']['Terminal'].loc[
                    self.cimConverter.cim['eq']['Terminal']['ConnectivityNode'].isna(), 'rdfId']
            if terminals_temp.index.size > 0:
                terminals_temp = pd.merge(terminals_temp,
                                          self.cimConverter.cim['tp']['Terminal'][['rdfId', 'TopologicalNode']],
                                          how='left', on='rdfId')
                terminals_temp = terminals_temp.drop(columns=['rdfId'])
                terminals_temp = terminals_temp.rename(columns={'TopologicalNode': 'rdfId'})
                terminals_temp = terminals_temp.drop_duplicates(subset=['rdfId'])
                tp_temp = self.cimConverter.cim['tp']['TopologicalNode'][
                    ['rdfId', 'name', 'description', 'BaseVoltage']]
                tp_temp[sc['o_prf']] = 'tp'
                tp_temp = pd.concat(
                    [tp_temp, self.cimConverter.cim['tp_bd']['TopologicalNode'][['rdfId', 'name', 'BaseVoltage']]],
                    sort=False)
                tp_temp[sc['o_prf']].fillna('tp_bd', inplace=True)
                tp_temp[sc['o_cl']] = 'TopologicalNode'
                tp_temp = pd.merge(terminals_temp, tp_temp, how='inner', on='rdfId')
                connectivity_nodes = pd.concat([connectivity_nodes, tp_temp], ignore_index=True, sort=False)
        else:
            # Bus-Branch model
            # concat the TopologicalNodes from the tp and boundary profile and keep the source profile for each element
            # as column using the pandas assign method
            connectivity_nodes = pd.concat([self.cimConverter.cim['tp']['TopologicalNode'].assign(**cn_dict['tp']),
                                            self.cimConverter.cim['tp_bd']['TopologicalNode'].assign(
                                                **cn_dict['tp_bd'])],
                                           ignore_index=True, sort=False)
            connectivity_nodes[sc['o_cl']] = 'TopologicalNode'
            connectivity_nodes['name_substation'] = ''
        # prepare the voltages from the buses
        eq_base_voltages = pd.concat([self.cimConverter.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']],
                                      self.cimConverter.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']]],
                                     ignore_index=True, sort=False)
        eq_base_voltages = eq_base_voltages.drop_duplicates(subset=['rdfId'])
        eq_base_voltages = eq_base_voltages.rename(columns={'rdfId': 'BaseVoltage'})
        # make sure that the BaseVoltage has string datatype
        connectivity_nodes['BaseVoltage'] = connectivity_nodes['BaseVoltage'].astype(str)
        connectivity_nodes = pd.merge(connectivity_nodes, eq_base_voltages, how='left', on='BaseVoltage')
        connectivity_nodes = connectivity_nodes.drop(columns=['BaseVoltage'])
        eqssh_terminals = self.cimConverter.cim['eq']['Terminal'][['rdfId', 'ConnectivityNode', 'ConductingEquipment',
                                                                   'sequenceNumber']]
        eqssh_terminals = \
            pd.concat([eqssh_terminals, self.cimConverter.cim['eq_bd']['Terminal'][['rdfId', 'ConductingEquipment',
                                                                                    'ConnectivityNode',
                                                                                    'sequenceNumber']]],
                      ignore_index=True, sort=False)
        eqssh_terminals = pd.merge(eqssh_terminals, self.cimConverter.cim['ssh']['Terminal'], how='left', on='rdfId')
        eqssh_terminals = pd.merge(eqssh_terminals, self.cimConverter.cim['tp']['Terminal'], how='left', on='rdfId')
        eqssh_terminals['ConnectivityNode'].fillna(eqssh_terminals['TopologicalNode'], inplace=True)
        # concat the DC terminals
        dc_terminals = pd.merge(pd.concat(
            [self.cimConverter.cim['eq']['DCTerminal'], self.cimConverter.cim['eq']['ACDCConverterDCTerminal']],
            ignore_index=True, sort=False),
                                pd.concat([self.cimConverter.cim['ssh']['DCTerminal'],
                                           self.cimConverter.cim['ssh']['ACDCConverterDCTerminal']],
                                          ignore_index=True, sort=False), how='left', on='rdfId')
        dc_terminals = pd.merge(dc_terminals,
                                pd.concat([self.cimConverter.cim['tp']['DCTerminal'],
                                           self.cimConverter.cim['tp']['ACDCConverterDCTerminal']],
                                          ignore_index=True, sort=False), how='left', on='rdfId')
        dc_terminals = dc_terminals.rename(columns={'DCNode': 'ConnectivityNode', 'DCConductingEquipment': 'ConductingEquipment',
                                     'DCTopologicalNode': 'TopologicalNode'})
        eqssh_terminals = pd.concat([eqssh_terminals, dc_terminals], ignore_index=True, sort=False)
        # special fix for concat tp profiles
        eqssh_terminals = eqssh_terminals.drop_duplicates(subset=['rdfId', 'TopologicalNode'])
        eqssh_terminals_temp = eqssh_terminals[['ConnectivityNode', 'TopologicalNode']]
        eqssh_terminals_temp = eqssh_terminals_temp.dropna(subset=['TopologicalNode'])
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
            connectivity_nodes = connectivity_nodes.drop(columns=['TopologicalNode_2'])
        if connectivity_nodes.index.size != connectivity_nodes_size:
            self.logger.warning("There is a problem at the busses!")
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="There is a problem at the busses!"))
            dups = connectivity_nodes.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 1]
            for rdfId, count in dups.items():
                self.logger.warning("The ConnectivityNode with RDF ID %s has %s TopologicalNodes!" % (rdfId, count))
                self.logger.warning("The ConnectivityNode data: \n%s" %
                                    connectivity_nodes[connectivity_nodes['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The ConnectivityNode with RDF ID %s has %s TopologicalNodes!" % (rdfId, count)))
            # raise ValueError("The number of ConnectivityNodes increased after merging with Terminals, number of "
            #                  "ConnectivityNodes before merge: %s, number of ConnectivityNodes after merge: %s" %
            #                  (connectivity_nodes_size, connectivity_nodes.index.size))
            connectivity_nodes = connectivity_nodes.drop_duplicates(subset=['rdfId'], keep='first')
        # add the busbars
        bb = self.cimConverter.cim['eq']['BusbarSection'][['rdfId', 'name']]
        bb = bb.rename(columns={'rdfId': 'busbar_id', 'name': 'busbar_name'})
        bb = pd.merge(bb, self.cimConverter.cim['eq']['Terminal'][['ConnectivityNode', 'ConductingEquipment']].rename(
            columns={'ConnectivityNode': 'rdfId', 'ConductingEquipment': 'busbar_id'}), how='left', on='busbar_id')
        bb = bb.drop_duplicates(subset=['rdfId'], keep='first')
        connectivity_nodes = pd.merge(connectivity_nodes, bb, how='left', on='rdfId')

        connectivity_nodes = connectivity_nodes.rename(columns={'rdfId': sc['o_id'], 'TopologicalNode': sc['ct'], 'nominalVoltage': 'vn_kv',
                                           'name_substation': 'zone'})
        connectivity_nodes['in_service'] = True
        connectivity_nodes['type'] = 'b'
        return connectivity_nodes, eqssh_terminals

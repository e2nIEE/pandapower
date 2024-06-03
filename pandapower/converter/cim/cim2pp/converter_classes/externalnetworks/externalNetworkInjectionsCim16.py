import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.externalNetworkInjectionsCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class ExternalNetworkInjectionsCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_external_network_injections_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting ExternalNetworkInjections.")

        eqssh_eni = self._prepare_external_network_injections_cim16()

        # choose the slack
        eni_ref_prio_min = eqssh_eni.loc[eqssh_eni['enabled'], 'slack_weight'].min()
        # check if the slack is a SynchronousMachine
        sync_machines = self.cimConverter.merge_eq_ssh_profile('SynchronousMachine')
        sync_machines = self.get_voltage_from_controllers(sync_machines)

        sync_ref_prio_min = sync_machines.loc[
            (sync_machines['referencePriority'] > 0) & (sync_machines['enabled']), 'referencePriority'].min()
        if pd.isna(eni_ref_prio_min):
            ref_prio_min = sync_ref_prio_min
        elif pd.isna(sync_ref_prio_min):
            ref_prio_min = eni_ref_prio_min
        else:
            ref_prio_min = min(eni_ref_prio_min, sync_ref_prio_min)

        eni_slacks = eqssh_eni.loc[(eqssh_eni['slack_weight'] == ref_prio_min) & (eqssh_eni['controllable'])]
        eni_gens = eqssh_eni.loc[(eqssh_eni['slack_weight'] != ref_prio_min) & (eqssh_eni['controllable'])]
        eni_sgens = eqssh_eni.loc[~eqssh_eni['controllable']]

        self.cimConverter.copy_to_pp('ext_grid', eni_slacks)
        self.cimConverter.copy_to_pp('gen', eni_gens)
        self.cimConverter.copy_to_pp('sgen', eni_sgens)

        self.logger.info("Created %s external networks, %s generators and %s static generators in %ss" %
                         (eni_slacks.index.size, eni_gens.index.size, eni_sgens.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s external networks, %s generators and %s static generators from "
                    "ExternalNetworkInjections in %ss" %
                    (eni_slacks.index.size, eni_gens.index.size, eni_sgens.index.size, time.time() - time_start)))

    def _prepare_external_network_injections_cim16(self) -> pd.DataFrame:
        if 'sc' in self.cimConverter.cim.keys():
            eni = self.cimConverter.merge_eq_other_profiles(['ssh', 'sc'], 'ExternalNetworkInjection',
                                                        add_cim_type_column=True)
        else:
            eni = self.cimConverter.merge_eq_ssh_profile('ExternalNetworkInjection', add_cim_type_column=True)

        # merge with buses
        eni = pd.merge(eni, self.cimConverter.bus_merge, how='left', on='rdfId')

        # get the voltage from controllers
        eni = self.get_voltage_from_controllers(eni)

        # get slack voltage and angle from SV profile
        eni = pd.merge(eni, self.cimConverter.net.bus[['vn_kv', sc['ct']]],
                       how='left', left_on='index_bus', right_index=True)
        eni = pd.merge(eni, self.cimConverter.cim['sv']['SvVoltage'][['TopologicalNode', 'v', 'angle']],
                       how='left', left_on=sc['ct'], right_on='TopologicalNode')
        eni['controlEnabled'] = eni['controlEnabled'] & eni['enabled']
        eni['vm_pu'] = eni['targetValue'] / eni['vn_kv']  # voltage from regulation
        eni['vm_pu'].fillna(eni['v'] / eni['vn_kv'], inplace=True)  # voltage from measurement
        eni['vm_pu'].fillna(1., inplace=True)  # default voltage
        eni['angle'].fillna(0., inplace=True)  # default angle
        eni['ratedU'] = eni['targetValue'][:]  # targetValue in kV
        eni['ratedU'].fillna(eni['v'], inplace=True)  # v in kV
        eni['ratedU'].fillna(eni['vn_kv'], inplace=True)
        eni['s_sc_max_mva'] = 3 ** .5 * eni['ratedU'] * (eni['maxInitialSymShCCurrent'] / 1e3)
        eni['s_sc_min_mva'] = 3 ** .5 * eni['ratedU'] * (eni['minInitialSymShCCurrent'] / 1e3)
        # get the substations
        eni = pd.merge(eni,
                       self.cimConverter.net.bus[[sc['o_id'], 'zone']].rename({sc['o_id']: 'b_id'}, axis=1),
                       how='left', left_on='ConnectivityNode', right_on='b_id')

        # convert pu generators with prio = 0 to pq generators (PowerFactory does it same)
        eni['referencePriority'].loc[eni['referencePriority'] == 0] = -1
        eni['controlEnabled'].loc[eni['referencePriority'] == -1] = False
        eni['p'] = -eni['p']
        eni['q'] = -eni['q']
        eni['x0x_max'] = ((eni['maxR1ToX1Ratio'] + 1j) /
                          (eni['maxR0ToX0Ratio'] + 1j)).abs() * eni['maxZ0ToZ1Ratio']

        if 'inService' in eni.columns:
            eni['connected'] = eni['connected'] & eni['inService']

        eni.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'zone': sc['sub'],
                            'angle': 'va_degree', 'index_bus': 'bus', 'connected': 'in_service',
                            'minP': 'min_p_mw', 'maxP': 'max_p_mw', 'minQ': 'min_q_mvar', 'maxQ': 'max_q_mvar',
                            'p': 'p_mw', 'q': 'q_mvar', 'controlEnabled': 'controllable',
                            'maxR1ToX1Ratio': 'rx_max', 'minR1ToX1Ratio': 'rx_min', 'maxR0ToX0Ratio': 'r0x0_max',
                            'referencePriority': 'slack_weight'},
                   inplace=True)
        eni['scaling'] = 1.
        eni['type'] = None
        eni['slack'] = False

        return eni

    def get_voltage_from_controllers(self, eqssh_eni):
        regulation_controllers = self.cimConverter.merge_eq_ssh_profile('RegulatingControl')
        regulation_controllers = regulation_controllers.loc[regulation_controllers['mode'] == 'voltage']
        regulation_controllers = regulation_controllers[['rdfId', 'targetValue', 'enabled']]
        regulation_controllers = regulation_controllers.rename(columns={'rdfId': 'RegulatingControl'})
        eqssh_eni = pd.merge(eqssh_eni, regulation_controllers, how='left', on='RegulatingControl')
        return eqssh_eni

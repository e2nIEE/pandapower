import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.synchronousMachinesCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class SynchronousMachinesCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_synchronous_machines_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting SynchronousMachines.")
        eqssh_synchronous_machines = self._prepare_synchronous_machines_cim16()

        # convert the SynchronousMachines with voltage control to gens
        eqssh_sm_gens = eqssh_synchronous_machines.loc[(eqssh_synchronous_machines['mode'] == 'voltage') &
                                                       (eqssh_synchronous_machines['enabled'])]
        self.cimConverter.copy_to_pp('gen', eqssh_sm_gens)
        # now deal with the pq generators
        eqssh_synchronous_machines = eqssh_synchronous_machines.loc[(eqssh_synchronous_machines['mode'] != 'voltage') |
                                                                    (~eqssh_synchronous_machines['enabled'])]
        self.cimConverter.copy_to_pp('sgen', eqssh_synchronous_machines)
        self.logger.info("Created %s gens and %s sgens in %ss." %
                         (eqssh_sm_gens.index.size, eqssh_synchronous_machines.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s gens and %s sgens from SynchronousMachines in %ss." %
                    (eqssh_sm_gens.index.size, eqssh_synchronous_machines.index.size, time.time() - time_start)))

    def _prepare_synchronous_machines_cim16(self) -> pd.DataFrame:
        eq_generating_units = self.cimConverter.cim['eq']['GeneratingUnit'][
            ['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']]
        # a column for the type of the static generator in pandapower
        eq_generating_units['type'] = 'GeneratingUnit'
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['WindGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('WP')
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['HydroGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('Hydro')
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['SolarGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('PV')
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['ThermalGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('Thermal')
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['NuclearGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('Nuclear')
        eq_generating_units = eq_generating_units.rename(columns={'rdfId': 'GeneratingUnit'})

        if 'sc' in self.cimConverter.cim.keys():
            synchronous_machines = self.cimConverter.merge_eq_other_profiles(
            ['ssh', 'sc'], 'SynchronousMachine', add_cim_type_column=True)
        else:
            synchronous_machines = self.cimConverter.merge_eq_ssh_profile('SynchronousMachine', add_cim_type_column=True)

        if 'type' in synchronous_machines.columns:
            synchronous_machines = synchronous_machines.drop(columns=['type'])
        if 'EquipmentContainer' in synchronous_machines.columns:
            synchronous_machines = synchronous_machines.drop(columns=['EquipmentContainer'])
        synchronous_machines = pd.merge(synchronous_machines, eq_generating_units,
                                              how='left', on='GeneratingUnit')
        eqssh_reg_control = self.cimConverter.merge_eq_ssh_profile('RegulatingControl')[
            ['rdfId', 'mode', 'enabled', 'targetValue', 'Terminal']]
        # if voltage is not on connectivity node, topological node will be used
        eqtp_terminals = pd.merge(self.cimConverter.cim['eq']['Terminal'], self.cimConverter.cim['tp']['Terminal'],
                                  how='left', on='rdfId')
        eqtp_terminals.ConnectivityNode = eqtp_terminals.ConnectivityNode.fillna(eqtp_terminals.TopologicalNode)
        eqtp_terminals = eqtp_terminals[['rdfId', 'ConnectivityNode']].rename(
            columns={'rdfId': 'Terminal', 'ConnectivityNode': 'reg_control_cnode'})
        eqssh_reg_control = pd.merge(eqssh_reg_control, eqtp_terminals, how='left', on='Terminal')
        eqssh_reg_control = eqssh_reg_control.drop(columns=['Terminal'])
        # add the voltage from the bus
        eqssh_reg_control = pd.merge(eqssh_reg_control, self.cimConverter.net.bus[['vn_kv', sc['o_id']]].rename(
            columns={sc['o_id']: 'reg_control_cnode'}), how='left', on='reg_control_cnode')
        # merge with RegulatingControl to check if it is a voltage controlled generator
        eqssh_reg_control = eqssh_reg_control.loc[eqssh_reg_control['mode'] == 'voltage']
        synchronous_machines = pd.merge(
            synchronous_machines, eqssh_reg_control.rename(columns={'rdfId': 'RegulatingControl'}),
            how='left', on='RegulatingControl')
        synchronous_machines = pd.merge(synchronous_machines, self.cimConverter.bus_merge, how='left',
                                              on='rdfId')
        synchronous_machines = synchronous_machines.drop_duplicates(['rdfId'], keep='first')
        synchronous_machines['vm_pu'] = synchronous_machines.targetValue / synchronous_machines.vn_kv
        synchronous_machines['vm_pu'].fillna(1., inplace=True)
        synchronous_machines = synchronous_machines.rename(columns={'vn_kv': 'bus_voltage'})
        synchronous_machines['slack'] = False
        # set the slack = True for gens with highest prio
        # get the highest prio from SynchronousMachines
        sync_ref_prio_min = synchronous_machines.loc[
            (synchronous_machines['referencePriority'] > 0) & (synchronous_machines['enabled']),
            'referencePriority'].min()
        # get the highest prio from ExternalNetworkInjection and check if the slack is an ExternalNetworkInjection
        enis = self.cimConverter.merge_eq_ssh_profile('ExternalNetworkInjection')
        regulation_controllers = self.cimConverter.merge_eq_ssh_profile('RegulatingControl')
        regulation_controllers = regulation_controllers.loc[regulation_controllers['mode'] == 'voltage']
        regulation_controllers = regulation_controllers[['rdfId', 'targetValue', 'enabled']]
        regulation_controllers = regulation_controllers.rename(columns={'rdfId': 'RegulatingControl'})
        enis = pd.merge(enis, regulation_controllers, how='left', on='RegulatingControl')

        eni_ref_prio_min = enis.loc[(enis['referencePriority'] > 0) & (enis['enabled']), 'referencePriority'].min()
        if pd.isna(sync_ref_prio_min):
            ref_prio_min = eni_ref_prio_min
        elif pd.isna(eni_ref_prio_min):
            ref_prio_min = sync_ref_prio_min
        else:
            ref_prio_min = min(eni_ref_prio_min, sync_ref_prio_min)

        synchronous_machines.loc[synchronous_machines['referencePriority'] == ref_prio_min, 'slack'] = True
        synchronous_machines['p_mw'] = -synchronous_machines['p']
        synchronous_machines['q_mvar'] = -synchronous_machines['q']
        synchronous_machines['current_source'] = True
        synchronous_machines['sn_mva'] = \
            synchronous_machines['ratedS'].fillna(synchronous_machines['nominalP'])
        # SC data
        synchronous_machines['vn_kv'] = synchronous_machines['ratedU'][:]
        synchronous_machines['rdss_ohm'] = \
            synchronous_machines['r2'] * \
            (synchronous_machines['ratedU'] ** 2 / synchronous_machines['ratedS'])
        synchronous_machines['xdss_pu'] = synchronous_machines['x2'][:]
        synchronous_machines['voltageRegulationRange'].fillna(0., inplace=True)
        synchronous_machines['pg_percent'] = synchronous_machines['voltageRegulationRange']
        synchronous_machines['k'] = (synchronous_machines['ratedS'] * 1e3 / synchronous_machines[
            'ratedU']) / (synchronous_machines['ratedU'] / (
                3 ** .5 * (synchronous_machines['r2'] ** 2 + synchronous_machines['x2'] ** 2)))
        synchronous_machines['rx'] = synchronous_machines['r2'] / synchronous_machines['x2']
        synchronous_machines['scaling'] = 1.
        synchronous_machines['generator_type'] = 'current_source'
        if 'inService' in synchronous_machines.columns:
            synchronous_machines['connected'] = (synchronous_machines['connected']
                                                       & synchronous_machines['inService'])
        synchronous_machines = synchronous_machines.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'],
                                                   'connected': 'in_service', 'index_bus': 'bus',
                                                   'minOperatingP': 'min_p_mw', 'maxOperatingP': 'max_p_mw',
                                                   'minQ': 'min_q_mvar', 'maxQ': 'max_q_mvar',
                                                   'ratedPowerFactor': 'cos_phi'})
        return synchronous_machines

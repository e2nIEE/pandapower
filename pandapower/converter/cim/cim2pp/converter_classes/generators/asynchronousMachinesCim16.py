import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.asynchronousMachinesCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class AsynchronousMachinesCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_asynchronous_machines_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting AsynchronousMachines.")
        eqssh_asynchronous_machines = self._prepare_asynchronous_machines_cim16()
        self.cimConverter.copy_to_pp('motor', eqssh_asynchronous_machines)
        self.logger.info("Created %s motors in %ss." %
                         (eqssh_asynchronous_machines.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s motors from AsynchronousMachines in %ss." %
                    (eqssh_asynchronous_machines.index.size, time.time() - time_start)))

    def _prepare_asynchronous_machines_cim16(self) -> pd.DataFrame:
        eq_generating_units = self.cimConverter.cim['eq']['WindGeneratingUnit'].copy()
        # a column for the type of the static generator in pandapower
        eq_generating_units['type'] = 'WP'
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['GeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'].fillna('GeneratingUnit', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['HydroGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'].fillna('Hydro', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['SolarGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'].fillna('PV', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['ThermalGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'].fillna('Thermal', inplace=True)
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['NuclearGeneratingUnit']],
                                        sort=False)
        eq_generating_units['type'].fillna('Nuclear', inplace=True)
        eq_generating_units = eq_generating_units.rename(columns={'rdfId': 'GeneratingUnit'})
        eqssh_asynchronous_machines = self.cimConverter.merge_eq_ssh_profile('AsynchronousMachine',
                                                                             add_cim_type_column=True)
        # prevent conflict of merging two dataframes each containing column 'name'
        eq_generating_units = eq_generating_units.drop('name', axis=1)
        eqssh_asynchronous_machines = pd.merge(eqssh_asynchronous_machines, eq_generating_units,
                                               how='left', on='GeneratingUnit')
        eqssh_asynchronous_machines = pd.merge(eqssh_asynchronous_machines, self.cimConverter.bus_merge, how='left',
                                               on='rdfId')
        eqssh_asynchronous_machines['p_mw'] = -eqssh_asynchronous_machines['p']
        eqssh_asynchronous_machines['q_mvar'] = -eqssh_asynchronous_machines['q']
        eqssh_asynchronous_machines['current_source'] = True
        eqssh_asynchronous_machines['cos_phi_n'] = eqssh_asynchronous_machines['ratedPowerFactor'][:]
        eqssh_asynchronous_machines['sn_mva'] = \
            eqssh_asynchronous_machines['ratedS'].fillna(eqssh_asynchronous_machines['nominalP'])
        eqssh_asynchronous_machines['generator_type'] = 'async'
        eqssh_asynchronous_machines['loading_percent'] = \
            100 * eqssh_asynchronous_machines['p_mw'] / eqssh_asynchronous_machines['ratedMechanicalPower']
        eqssh_asynchronous_machines = eqssh_asynchronous_machines.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'],
                                                    'connected': 'in_service', 'index_bus': 'bus',
                                                    'rxLockedRotorRatio': 'rx', 'iaIrRatio': 'lrc_pu',
                                                    'ratedPowerFactor': 'cos_phi', 'ratedU': 'vn_kv',
                                                    'efficiency': 'efficiency_n_percent',
                                                    'ratedMechanicalPower': 'pn_mech_mw'})
        eqssh_asynchronous_machines['scaling'] = 1
        eqssh_asynchronous_machines['efficiency_percent'] = 100
        return eqssh_asynchronous_machines

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
        eqssh_generating_units = self.cimConverter.merge_eq_ssh_profile('WindGeneratingUnit')
        # a column for the type of the static generator in pandapower
        eqssh_generating_units['type'] = 'WP'
        eqssh_generating_units = pd.concat([eqssh_generating_units,
                                            self.cimConverter.merge_eq_ssh_profile('GeneratingUnit')],
                                           sort=False)
        eqssh_generating_units['type'].fillna('GeneratingUnit', inplace=True)
        eqssh_generating_units = pd.concat([eqssh_generating_units,
                                            self.cimConverter.merge_eq_ssh_profile('HydroGeneratingUnit')],
                                           sort=False)
        eqssh_generating_units['type'].fillna('Hydro', inplace=True)
        eqssh_generating_units = pd.concat([eqssh_generating_units,
                                            self.cimConverter.merge_eq_ssh_profile('SolarGeneratingUnit')],
                                           sort=False)
        eqssh_generating_units['type'].fillna('PV', inplace=True)
        eqssh_generating_units = pd.concat([eqssh_generating_units,
                                            self.cimConverter.merge_eq_ssh_profile('ThermalGeneratingUnit')],
                                           sort=False)
        eqssh_generating_units['type'].fillna('Thermal', inplace=True)
        eqssh_generating_units = pd.concat([eqssh_generating_units,
                                            self.cimConverter.merge_eq_ssh_profile('NuclearGeneratingUnit')],
                                           sort=False)
        eqssh_generating_units['type'].fillna('Nuclear', inplace=True)
        eqssh_generating_units = eqssh_generating_units.rename(columns={'rdfId': 'GeneratingUnit'})
        if 'sc' in self.cimConverter.cim.keys():
            asynchronous_machines = self.cimConverter.merge_eq_other_profiles(['ssh', 'sc'],
                                                                              'AsynchronousMachine',
                                                                              add_cim_type_column=True)
        else:
            asynchronous_machines = self.cimConverter.merge_eq_ssh_profile('AsynchronousMachine',
                                                                           add_cim_type_column=True)
        # prevent conflict of merging two dataframes each containing column 'name'
        eqssh_generating_units = eqssh_generating_units.drop('name', axis=1)
        asynchronous_machines = pd.merge(asynchronous_machines, eqssh_generating_units,
                                         how='left', suffixes=('_x', '_y'), on='GeneratingUnit')
        asynchronous_machines = pd.merge(asynchronous_machines, self.cimConverter.bus_merge, how='left',
                                         on='rdfId')
        asynchronous_machines['p_mw'] = -asynchronous_machines['p']
        asynchronous_machines['q_mvar'] = -asynchronous_machines['q']
        asynchronous_machines['current_source'] = True
        asynchronous_machines['cos_phi_n'] = asynchronous_machines['ratedPowerFactor'][:]
        asynchronous_machines['sn_mva'] = \
            asynchronous_machines['ratedS'].fillna(asynchronous_machines['nominalP'])
        asynchronous_machines['generator_type'] = 'async'
        asynchronous_machines['loading_percent'] = \
            100 * asynchronous_machines['p_mw'] / asynchronous_machines['ratedMechanicalPower']
        if 'inService_x' in asynchronous_machines.columns:
            asynchronous_machines['connected'] = (asynchronous_machines['connected']
                                                  & asynchronous_machines['inService_x']
                                                  & asynchronous_machines['inService_y'])
        asynchronous_machines = asynchronous_machines.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'],
                                                                      'connected': 'in_service', 'index_bus': 'bus',
                                                                      'rxLockedRotorRatio': 'rx', 'iaIrRatio': 'lrc_pu',
                                                                      'ratedPowerFactor': 'cos_phi', 'ratedU': 'vn_kv',
                                                                      'efficiency': 'efficiency_n_percent',
                                                                      'ratedMechanicalPower': 'pn_mech_mw'})
        asynchronous_machines['scaling'] = 1
        asynchronous_machines['efficiency_percent'] = 100
        return asynchronous_machines

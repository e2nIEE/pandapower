import logging
import time
import math

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.powerElectronicsConnection')

sc = cim_tools.get_pp_net_special_columns_dict()


class PowerElectronicsConnection:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_power_electronics_connection(self):
        time_start = time.time()
        self.logger.info("Start converting PowerElectronicsConnections.")
        power_electronics_connections = self._prepare_power_electronics_connection()
        self.cimConverter.copy_to_pp('sgen', power_electronics_connections)
        self.logger.info(f"Created {power_electronics_connections.index.size} sgens in {time.time() - time_start}s.")
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message=f"Created {power_electronics_connections.index.size} sgens in {time.time() - time_start}s."))

    def _prepare_power_electronics_connection(self) -> pd.DataFrame:
        eq_generating_units = self.cimConverter.cim['eq']['GeneratingUnit'][
            ['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP', 'governorSCD']]
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
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['PhotoVoltaicUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('PV')
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['BatteryUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('Battery')
        eq_generating_units = pd.concat([eq_generating_units, self.cimConverter.cim['eq']['PowerElectronicsWindUnit']],
                                        sort=False)
        eq_generating_units['type'] = eq_generating_units['type'].fillna('WP')
        eq_generating_units = eq_generating_units.rename(columns={'rdfId': 'PowerElectronicsUnit'})
        eq_generating_units = eq_generating_units.drop(columns=['name'])
        eqssh_pecs = self.cimConverter.merge_eq_ssh_profile('PowerElectronicsConnection', add_cim_type_column=True)
        eqssh_pecs = pd.merge(eqssh_pecs, eq_generating_units, how='left', on='PowerElectronicsUnit')
        eqssh_pecs = pd.merge(eqssh_pecs, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_pecs = eqssh_pecs.drop_duplicates(['rdfId'], keep='first')
        eqssh_pecs['controlEnabled'] = eqssh_pecs['controlEnabled'].fillna(False)
        eqssh_pecs['p_mw'] = -eqssh_pecs['p']
        eqssh_pecs['q_mvar'] = -eqssh_pecs['q']
        eqssh_pecs['scaling'] = 1.
        eqssh_pecs['current_source'] = True
        eqssh_pecs['generator_type'] = 'current_source'
        eqssh_pecs['reactive_capability_curve'] = False
        if self.cimConverter.cim_version == '3.0':
           eqssh_pecs['in_service'] = eqssh_pecs.connected & eqssh_pecs.inService
        elif self.cimConverter.cim_version == 'ltds':
           eqssh_pecs['in_service'] = eqssh_pecs.inService
        else:
           eqssh_pecs['in_service'] = eqssh_pecs.connected
        eqssh_pecs = eqssh_pecs.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'index_bus': 'bus',
                                                'controlEnabled': 'controllable'})
        return eqssh_pecs

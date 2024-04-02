import logging
import time
import math

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.energySourceCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class EnergySourceCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_energy_sources_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EnergySources.")
        eqssh_energy_sources = self._prepare_energy_sources_cim16()
        es_slack = eqssh_energy_sources.loc[eqssh_energy_sources.vm_pu.notna()]
        es_sgen = eqssh_energy_sources.loc[eqssh_energy_sources.vm_pu.isna()]
        self.cimConverter.copy_to_pp('ext_grid', es_slack)
        self.cimConverter.copy_to_pp('sgen', es_sgen)
        # self._copy_to_pp('sgen', eqssh_energy_sources)
        self.logger.info("Created %s sgens in %ss." % (eqssh_energy_sources.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s sgens from EnergySources in %ss." %
                    (eqssh_energy_sources.index.size, time.time() - time_start)))

    def _prepare_energy_sources_cim16(self) -> pd.DataFrame:
        eq_energy_scheduling_type = \
            pd.concat([self.cimConverter.cim['eq']['EnergySchedulingType'],
                       self.cimConverter.cim['eq_bd']['EnergySchedulingType']],
                      sort=False)
        eq_energy_scheduling_type = eq_energy_scheduling_type.rename(columns={'rdfId': 'EnergySchedulingType', 'name': 'type'})
        eqssh_energy_sources = self.cimConverter.merge_eq_ssh_profile('EnergySource', add_cim_type_column=True)
        eqssh_energy_sources = pd.merge(eqssh_energy_sources, eq_energy_scheduling_type, how='left',
                                        on='EnergySchedulingType')
        eqssh_energy_sources = pd.merge(eqssh_energy_sources, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_energy_sources = eqssh_energy_sources.drop_duplicates(['rdfId'], keep='first')
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
        eqssh_energy_sources = eqssh_energy_sources.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'connected': 'in_service',
                                             'index_bus': 'bus'})
        return eqssh_energy_sources

import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.equivalentInjectionsCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class EquivalentInjectionsCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_equivalent_injections_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EquivalentInjections.")
        eqssh_ei = self._prepare_equivalent_injections_cim16()
        # split up to wards and xwards: the wards have no regulation
        eqssh_ei_wards = eqssh_ei.loc[~eqssh_ei.regulationStatus]
        eqssh_ei_xwards = eqssh_ei.loc[eqssh_ei.regulationStatus]
        self.cimConverter.copy_to_pp('ward', eqssh_ei_wards)
        self.cimConverter.copy_to_pp('xward', eqssh_ei_xwards)
        self.logger.info("Created %s wards and %s extended ward elements in %ss." %
                         (eqssh_ei_wards.index.size, eqssh_ei_xwards.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s wards and %s extended ward elements from EquivalentInjections in %ss." %
                    (eqssh_ei_wards.index.size, eqssh_ei_xwards.index.size, time.time() - time_start)))

    def _prepare_equivalent_injections_cim16(self) -> pd.DataFrame:
        if 'sc' in self.cimConverter.cim.keys():
            equivalent_injection = self.cimConverter.merge_eq_other_profiles(['ssh', 'sc'], 'EquivalentInjection',
                                                                             add_cim_type_column=True)
        else:
            equivalent_injection = self.cimConverter.merge_eq_ssh_profile('EquivalentInjection',
                                                                          add_cim_type_column=True)
        eq_base_voltages = pd.concat([self.cimConverter.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']],
                                      self.cimConverter.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']]],
                                     sort=False)
        eq_base_voltages = eq_base_voltages.drop_duplicates(subset=['rdfId'])
        eq_base_voltages = eq_base_voltages.rename(columns={'rdfId': 'BaseVoltage'})
        equivalent_injection = pd.merge(equivalent_injection, eq_base_voltages, how='left', on='BaseVoltage')
        equivalent_injection = pd.merge(equivalent_injection, self.cimConverter.bus_merge, how='left', on='rdfId')
        # maybe the BaseVoltage is not given, also get the nominalVoltage from the buses
        equivalent_injection = pd.merge(equivalent_injection, self.cimConverter.net.bus[['vn_kv']], how='left',
                                        left_on='index_bus', right_index=True)
        equivalent_injection.nominalVoltage = equivalent_injection.nominalVoltage.fillna(equivalent_injection.vn_kv)
        equivalent_injection['regulationStatus'].fillna(False, inplace=True)
        equivalent_injection['vm_pu'] = equivalent_injection.regulationTarget / equivalent_injection.nominalVoltage
        if 'inService' in equivalent_injection.columns:
            equivalent_injection['connected'] = equivalent_injection['connected'] & equivalent_injection['inService']
        equivalent_injection.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'connected': 'in_service',
                                             'index_bus': 'bus', 'p': 'ps_mw', 'q': 'qs_mvar'},
                                    inplace=True)
        equivalent_injection['pz_mw'] = 0.
        equivalent_injection['qz_mvar'] = 0.
        equivalent_injection['r_ohm'] = 0.
        equivalent_injection['x_ohm'] = .1
        return equivalent_injection

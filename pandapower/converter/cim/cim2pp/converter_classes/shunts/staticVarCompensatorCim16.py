import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.staticVarCompensatorCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class StaticVarCompensatorCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_static_var_compensator_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting StaticVarCompensator.")
        eq_stat_coms = self._prepare_static_var_compensator_cim16()
        self.cimConverter.copy_to_pp('shunt', eq_stat_coms)
        self.logger.info("Created %s generators in %ss." % (eq_stat_coms.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s generators from StaticVarCompensator in %ss." %
                    (eq_stat_coms.index.size, time.time() - time_start)))

    def _prepare_static_var_compensator_cim16(self) -> pd.DataFrame:
        eq_stat_coms = self.cimConverter.merge_eq_ssh_profile('StaticVarCompensator', True)
        eq_stat_coms = pd.merge(eq_stat_coms, self.cimConverter.bus_merge, how='left', on='rdfId')
        eq_stat_coms = eq_stat_coms.rename(columns={'q': 'q_mvar'})
        # get the active power and reactive power from SV profile
        eq_stat_coms = pd.merge(eq_stat_coms, self.cimConverter.cim['sv']['SvPowerFlow'][['p', 'q', 'Terminal']],
                                how='left', left_on='rdfId_Terminal', right_on='Terminal')
        eq_stat_coms['q_mvar'].fillna(eq_stat_coms['q'], inplace=True)
        eq_stat_coms.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'p': 'p_mw',
                                     'voltageSetPoint': 'vn_kv', 'index_bus': 'bus', 'connected': 'in_service'},
                            inplace=True)
        eq_stat_coms['step'] = 1
        eq_stat_coms['max_step'] = 1
        return eq_stat_coms

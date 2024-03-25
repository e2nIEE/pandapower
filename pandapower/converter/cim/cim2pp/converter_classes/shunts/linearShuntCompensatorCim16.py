import logging
import time

import pandas as pd
import numpy as np

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.linearShuntCompensatorCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class LinearShuntCompensatorCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_linear_shunt_compensator_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting LinearShuntCompensator.")
        eqssh_shunts = self._prepare_linear_shunt_compensator_cim16()
        self.cimConverter.copy_to_pp('shunt', eqssh_shunts)
        self.logger.info("Created %s shunts in %ss." % (eqssh_shunts.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s shunts from LinearShuntCompensator in %ss." %
                    (eqssh_shunts.index.size, time.time() - time_start)))

    def _prepare_linear_shunt_compensator_cim16(self) -> pd.DataFrame:
        eqssh_shunts = self.cimConverter.merge_eq_ssh_profile('LinearShuntCompensator', add_cim_type_column=True)
        eqssh_shunts = pd.merge(eqssh_shunts, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_shunts = eqssh_shunts.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'connected': 'in_service', 'index_bus': 'bus',
            'nomU': 'vn_kv', 'sections': 'step', 'maximumSections': 'max_step'})
        y = eqssh_shunts['gPerSection'] + eqssh_shunts['bPerSection'] * 1j
        s = eqssh_shunts['vn_kv'] ** 2 * np.conj(y)
        eqssh_shunts['p_mw'] = s.values.real
        eqssh_shunts['q_mvar'] = s.values.imag
        return eqssh_shunts

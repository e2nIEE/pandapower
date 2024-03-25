import logging
import time

import pandas as pd
import numpy as np

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.nonLinearShuntCompensatorCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class NonLinearShuntCompensatorCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_nonlinear_shunt_compensator_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting NonlinearShuntCompensator.")
        if self.cimConverter.cim['eq']['NonlinearShuntCompensator'].index.size > 0:
            eqssh_shunts = self._prepare_nonlinear_shunt_compensator_cim16()
            self.cimConverter.copy_to_pp('shunt', eqssh_shunts)
        else:
            eqssh_shunts = pd.DataFrame(None)
        self.logger.info("Created %s shunts in %ss." % (eqssh_shunts.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s shunts from NonlinearShuntCompensator in %ss." %
                    (eqssh_shunts.index.size, time.time() - time_start)))

    def _prepare_nonlinear_shunt_compensator_cim16(self) -> pd.DataFrame:
        eqssh_shunts = self.cimConverter.merge_eq_ssh_profile('NonlinearShuntCompensator', add_cim_type_column=True)
        eqssh_shunts = pd.merge(eqssh_shunts, self.cimConverter.bus_merge, how='left', on='rdfId')

        eqssh_shunts['p'] = float('NaN')
        eqssh_shunts['q'] = float('NaN')
        eqssh_shunts_cols = eqssh_shunts.columns.to_list()
        nscp = self.cimConverter.cim['eq']['NonlinearShuntCompensatorPoint'][
            ['NonlinearShuntCompensator', 'sectionNumber', 'b', 'g']].rename(
            columns={'NonlinearShuntCompensator': 'rdfId'})
        # calculate p and q from b, g, and all the sections
        for i in range(1, int(nscp['sectionNumber'].max()) + 1):
            nscp_t = nscp.loc[nscp['sectionNumber'] == i]
            eqssh_shunts = pd.merge(eqssh_shunts, nscp_t, how='left', on='rdfId')
            y = eqssh_shunts['g'] + eqssh_shunts['b'] * 1j
            s = eqssh_shunts['nomU'] ** 2 * np.conj(y)
            eqssh_shunts['p_temp'] = s.values.real
            eqssh_shunts['q_temp'] = s.values.imag
            if i == 1:
                eqssh_shunts['p'] = eqssh_shunts['p_temp'][:]
                eqssh_shunts['q'] = eqssh_shunts['q_temp'][:]
            else:
                eqssh_shunts.loc[eqssh_shunts['sections'] >= eqssh_shunts['sectionNumber'], 'p'] = \
                    eqssh_shunts['p'] + eqssh_shunts['p_temp']
                eqssh_shunts.loc[eqssh_shunts['sections'] >= eqssh_shunts['sectionNumber'], 'q'] = \
                    eqssh_shunts['q'] + eqssh_shunts['q_temp']
            eqssh_shunts = eqssh_shunts[eqssh_shunts_cols]
        eqssh_shunts = eqssh_shunts.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'connected': 'in_service', 'index_bus': 'bus',
            'nomU': 'vn_kv', 'p': 'p_mw', 'q': 'q_mvar'})
        eqssh_shunts['step'] = 1
        eqssh_shunts['max_step'] = 1
        return eqssh_shunts

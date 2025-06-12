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
            self._create_shunt_characteristic_table(eqssh_shunts)
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
        if 'inService' in eqssh_shunts.columns:
            eqssh_shunts['connected'] = eqssh_shunts['connected'] & eqssh_shunts['inService']
        # added to use the logic of p/q values multiplied by the number of steps
        # this is only correct for the current step values
        eqssh_shunts['p'] = eqssh_shunts['p'] / eqssh_shunts['sections']
        eqssh_shunts['q'] = eqssh_shunts['q'] / eqssh_shunts['sections']
        eqssh_shunts = eqssh_shunts.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'connected': 'in_service', 'index_bus': 'bus',
            'nomU': 'vn_kv', 'p': 'p_mw', 'q': 'q_mvar', 'sections': 'step', 'maximumSections': 'max_step'})
        return eqssh_shunts

    def _create_shunt_characteristic_table(self, eqssh_shunts):
        if 'id_characteristic_table' not in eqssh_shunts.columns:
            eqssh_shunts['id_characteristic_table'] = np.nan
        if 'shunt_characteristic_table' not in self.cimConverter.net.keys():
            self.cimConverter.net['shunt_characteristic_table'] = pd.DataFrame(
                columns=['id_characteristic', 'step', 'q_mvar', 'p_mw'])
        char_temp = eqssh_shunts.drop(columns=['p_mw', 'q_mvar', 'step'])
        char_temp['p'] = float('NaN')
        char_temp['q'] = float('NaN')
        # get the NonlinearShuntCompensatorPoints
        nscp = self.cimConverter.cim['eq']['NonlinearShuntCompensatorPoint'][
            ['NonlinearShuntCompensator', 'sectionNumber', 'b', 'g']].rename(
            columns={'NonlinearShuntCompensator': sc['o_id']})
        char_temp = pd.merge(char_temp, nscp, how='left', on=sc['o_id'])
        # calculate p & q from b & g for all sections
        y = char_temp['g'] + char_temp['b'] * 1j
        s = char_temp['vn_kv'] ** 2 * np.conj(y)
        char_temp['p_temp'] = s.values.real
        char_temp['q_temp'] = s.values.imag
        # calculate cumulative sums for all sections
        char_temp = char_temp.sort_values(by=[sc['o_id'], 'sectionNumber'])
        char_temp['p'] = char_temp.groupby(sc['o_id'])['p_temp'].cumsum()
        char_temp['q'] = char_temp.groupby(sc['o_id'])['q_temp'].cumsum()
        char_temp = char_temp.rename(columns={'p': 'p_mw', 'q': 'q_mvar', 'sectionNumber': 'step'})
        char_temp['step'] = char_temp['step'].astype(int)
        # assign id_characteristic
        char_temp['id_characteristic'] = pd.factorize(char_temp[sc['o_id']])[0]

        # set the id_characteristic at the corresponding shunt
        id_char_dict = char_temp.drop_duplicates(sc['o_id']).set_index(sc['o_id'])['id_characteristic'].to_dict()
        eqssh_shunts['id_characteristic_table'] = eqssh_shunts[sc['o_id']].map(id_char_dict).astype('Int64')

        # create step_dependency_table flag
        if 'step_dependency_table' not in eqssh_shunts.columns:
            eqssh_shunts["step_dependency_table"] = False
        # set step_dependency_table as True for all non-linear shunt compensators
        eqssh_shunts.loc[eqssh_shunts['id_characteristic_table'].notna(), 'step_dependency_table'] = True

        # populate shunt_characteristic_temp table
        self.cimConverter.net['shunt_characteristic_table'] = \
            char_temp[['id_characteristic', 'step', 'q_mvar', 'p_mw']]

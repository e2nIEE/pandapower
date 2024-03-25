import logging
import time

import pandas as pd
import numpy as np

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.acLineSegmentsCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class AcLineSegmentsCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_ac_line_segments_cim16(self, convert_line_to_switch, line_r_limit, line_x_limit):
        time_start = time.time()
        self.logger.info("Start converting ACLineSegments.")
        eq_ac_line_segments = self._prepare_ac_line_segments_cim16(convert_line_to_switch, line_r_limit, line_x_limit)

        # now create the lines and the switches
        # -------- lines --------
        if 'line' in eq_ac_line_segments['kindOfType'].values:
            line_df = eq_ac_line_segments.loc[eq_ac_line_segments['kindOfType'] == 'line']
            line_df = line_df.rename(columns={
                'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'],
                'index_bus': 'from_bus', 'index_bus2': 'to_bus', 'length': 'length_km',
                'shortCircuitEndTemperature': 'endtemp_degree'})
            line_df[sc['o_cl']] = 'ACLineSegment'
            line_df['in_service'] = line_df.connected & line_df.connected2
            line_df['r_ohm_per_km'] = abs(line_df.r) / line_df.length_km
            line_df['x_ohm_per_km'] = abs(line_df.x) / line_df.length_km
            line_df['c_nf_per_km'] = abs(line_df.bch) / (2 * 50 * np.pi * line_df.length_km) * 1e9
            line_df['g_us_per_km'] = abs(line_df.gch) * 1e6 / line_df.length_km
            line_df['r0_ohm_per_km'] = abs(line_df.r0) / line_df.length_km
            line_df['x0_ohm_per_km'] = abs(line_df.x0) / line_df.length_km
            line_df['c0_nf_per_km'] = abs(line_df.b0ch) / (2 * 50 * np.pi * line_df.length_km) * 1e9
            line_df['g0_us_per_km'] = abs(line_df.g0ch) * 1e6 / line_df.length_km
            line_df['parallel'] = 1
            line_df['df'] = 1.
            line_df['type'] = None
            line_df['std_type'] = None
            self.cimConverter.copy_to_pp('line', line_df)
        else:
            line_df = pd.DataFrame(None)
        # -------- switches --------
        if 'switch' in eq_ac_line_segments['kindOfType'].values:
            switch_df = eq_ac_line_segments.loc[eq_ac_line_segments['kindOfType'] == 'switch']

            switch_df = switch_df.rename(columns={
                'rdfId': sc['o_id'], 'index_bus': 'bus', 'index_bus2': 'element', 'rdfId_Terminal': sc['t_bus'],
                'rdfId_Terminal2': sc['t_ele']})
            switch_df['et'] = 'b'
            switch_df['type'] = None
            switch_df['z_ohm'] = 0
            if switch_df.index.size > 0:
                switch_df['closed'] = switch_df.connected & switch_df.connected2
            self.cimConverter.copy_to_pp('switch', switch_df)
        else:
            switch_df = pd.DataFrame(None)

        self.logger.info("Created %s lines and %s switches in %ss" %
                         (line_df.index.size, switch_df.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s lines and %s switches from ACLineSegments in %ss" %
                    (line_df.index.size, switch_df.index.size, time.time() - time_start)))

    def _prepare_ac_line_segments_cim16(self, convert_line_to_switch, line_r_limit, line_x_limit) -> pd.DataFrame:
        line_length_before_merge = self.cimConverter.cim['eq']['ACLineSegment'].index.size
        # until now self.cim['eq']['ACLineSegment'] looks like:
        #   rdfId   name    r       ...
        #   _x01    line1   0.056   ...
        #   _x02    line2   0.471   ...
        # now join with the terminals
        eq_ac_line_segments = pd.merge(self.cimConverter.cim['eq']['ACLineSegment'], self.cimConverter.bus_merge,
                                       how='left', on='rdfId')
        eq_ac_line_segments[sc['o_cl']] = 'ACLineSegment'
        # now eq_ac_line_segments looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    line1   0.056   termi025        True        ...
        #   _x01    line1   0.056   termi223        True        ...
        #   _x02    line2   0.471   termi154        True        ...
        #   _x02    line2   0.471   termi199        True        ...
        # if each switch got two terminals, reduce back to one line to use fast slicing
        if eq_ac_line_segments.index.size != line_length_before_merge * 2:
            self.logger.error("Error processing the ACLineSegments, there is a problem with Terminals in the source "
                              "data!")
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Error processing the ACLineSegments, there is a problem with Terminals in the source data!"))
            dups = eq_ac_line_segments.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The ACLineSegment with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The ACLineSegment data: \n%s" %
                                    eq_ac_line_segments[eq_ac_line_segments['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The ACLineSegment with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eq_ac_line_segments = eq_ac_line_segments[0:0]
        eq_ac_line_segments.reset_index(inplace=True)
        # now merge with OperationalLimitSets and CurrentLimits
        eq_operational_limit_sets = self.cimConverter.cim['eq']['OperationalLimitSet'][['rdfId', 'Terminal']]
        eq_operational_limit_sets = eq_operational_limit_sets.rename(columns={'rdfId': 'rdfId_OperationalLimitSet',
                                                  'Terminal': 'rdfId_Terminal'})
        eq_ac_line_segments = pd.merge(eq_ac_line_segments, eq_operational_limit_sets, how='left',
                                       on='rdfId_Terminal')
        eq_current_limits = self.cimConverter.cim['eq']['CurrentLimit'][['rdfId', 'OperationalLimitSet', 'value']]
        eq_current_limits = eq_current_limits.rename(columns={'rdfId': 'rdfId_CurrentLimit',
                                          'OperationalLimitSet': 'rdfId_OperationalLimitSet'})
        eq_ac_line_segments = pd.merge(eq_ac_line_segments, eq_current_limits, how='left',
                                       on='rdfId_OperationalLimitSet')
        eq_ac_line_segments.value = eq_ac_line_segments.value.astype(float)
        # sort by rdfId, sequenceNumber and value. value is max_i_ka, choose the lowest one if more than one is
        # given (A line may have more than one max_i_ka in CIM, different modes e.g. normal)
        eq_ac_line_segments = eq_ac_line_segments.sort_values(by=['rdfId', 'sequenceNumber', 'value'])
        eq_ac_line_segments = eq_ac_line_segments.drop_duplicates(['rdfId', 'rdfId_Terminal'], keep='first')

        # copy the columns which are needed to reduce the eq_ac_line_segments to one row per line
        eq_ac_line_segments['rdfId_Terminal2'] = eq_ac_line_segments['rdfId_Terminal'].copy()
        eq_ac_line_segments['connected2'] = eq_ac_line_segments['connected'].copy()
        eq_ac_line_segments['index_bus2'] = eq_ac_line_segments['index_bus'].copy()
        eq_ac_line_segments['value2'] = eq_ac_line_segments['value'].copy()
        eq_ac_line_segments = eq_ac_line_segments.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eq_ac_line_segments.rdfId_Terminal2 = eq_ac_line_segments.rdfId_Terminal2.iloc[
                                              1:].reset_index().rdfId_Terminal2
        eq_ac_line_segments.connected2 = eq_ac_line_segments.connected2.iloc[1:].reset_index().connected2
        eq_ac_line_segments.index_bus2 = eq_ac_line_segments.index_bus2.iloc[1:].reset_index().index_bus2
        eq_ac_line_segments.value2 = eq_ac_line_segments.value2.iloc[1:].reset_index().value2
        eq_ac_line_segments = eq_ac_line_segments.drop_duplicates(['rdfId'], keep='first')
        # get the max_i_ka
        eq_ac_line_segments['max_i_ka'] = eq_ac_line_segments['value'].fillna(eq_ac_line_segments['value2']) * 1e-3

        # filter if line or switches will be added
        eq_ac_line_segments['kindOfType'] = 'line'
        if convert_line_to_switch:
            eq_ac_line_segments.loc[(abs(eq_ac_line_segments['r']) <= line_r_limit) |
                                    (abs(eq_ac_line_segments['x']) <= line_x_limit), 'kindOfType'] = 'switch'
        return eq_ac_line_segments

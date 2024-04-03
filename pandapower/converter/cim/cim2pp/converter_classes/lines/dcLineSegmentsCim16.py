import logging
import time
from typing import Dict, List

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.acLineSegmentsCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class DcLineSegmentsCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_dc_line_segments_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting DCLineSegments.")
        eq_dc_line_segments = self._prepare_dc_line_segments_cim16()

        self.cimConverter.copy_to_pp('dcline', eq_dc_line_segments)

        self.logger.info("Created %s DC lines in %ss" % (eq_dc_line_segments.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s DC lines from DCLineSegments in %ss" %
                    (eq_dc_line_segments.index.size, time.time() - time_start)))

    def _prepare_dc_line_segments_cim16(self) -> pd.DataFrame:
        line_length_before_merge = self.cimConverter.cim['eq']['DCLineSegment'].index.size
        # until now self.cim['eq']['DCLineSegment'] looks like:
        #   rdfId   name    ...
        #   _x01    line1   ...
        #   _x02    line2   ...
        # now join with the terminals
        dc_line_segments = pd.merge(self.cimConverter.cim['eq']['DCLineSegment'], self.cimConverter.bus_merge,
                                    how='left', on='rdfId')
        dc_line_segments = dc_line_segments[['rdfId', 'name', 'ConnectivityNode', 'sequenceNumber']]
        dc_line_segments[sc['o_cl']] = 'DCLineSegment'
        # now dc_line_segments looks like:
        #   rdfId   name    rdfId_Terminal  connected   ...
        #   _x01    line1   termi025        True        ...
        #   _x01    line1   termi223        True        ...
        #   _x02    line2   termi154        True        ...
        #   _x02    line2   termi199        True        ...
        # if each switch got two terminals, reduce back to one line to use fast slicing
        if dc_line_segments.index.size != line_length_before_merge * 2:
            self.logger.error("Error processing the DCLineSegments, there is a problem with Terminals in the source "
                              "data!")
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Error processing the DCLineSegments, there is a problem with Terminals in the source data!"))
            return pd.DataFrame(None)
        dc_line_segments.reset_index(inplace=True)

        # now merge with the Converters
        converters = pd.merge(
            pd.concat([self.cimConverter.cim['eq']['CsConverter'], self.cimConverter.cim['eq']['VsConverter']],
                      ignore_index=True, sort=False),
            pd.concat([self.cimConverter.cim['ssh']['CsConverter'], self.cimConverter.cim['ssh']['VsConverter']],
                      ignore_index=True, sort=False),
            how='left', on='rdfId')
        if 'name' in converters.columns:
            converters = converters.drop(columns=['name'])
        # merge with the terminals
        converters = pd.merge(converters, self.cimConverter.bus_merge, how='left', on='rdfId')
        converters = converters.drop(columns=['sequenceNumber'])
        converters = converters.rename(columns={'rdfId': 'converters'})
        converter_terminals = pd.concat(
            [self.cimConverter.cim['eq']['Terminal'], self.cimConverter.cim['eq_bd']['Terminal']],
            ignore_index=True, sort=False)
        converter_terminals = converter_terminals[['rdfId']].rename(columns={'rdfId': 'rdfId_Terminal'})
        converters_t = pd.merge(converters, converter_terminals, how='inner', on='rdfId_Terminal')

        dc_line_segments = pd.merge(dc_line_segments, converters[['converters', 'ConnectivityNode']],
                                    how='left', on='ConnectivityNode')
        # get the missing converters (maybe there is a switch or something else between the line and the converter)
        t = self.cimConverter.cim['eq']['DCTerminal'][['DCNode', 'DCConductingEquipment', 'sequenceNumber']]
        t = t.rename(columns={'DCNode': 'ConnectivityNode', 'DCConductingEquipment': 'ConductingEquipment'})

        def search_converter(cn_ids: Dict[str, str], visited_cns: List[str]) -> str:
            new_cn_dict = dict()
            for one_cn, from_dev in cn_ids.items():
                # get the Terminals
                t_temp = t.loc[t['ConnectivityNode'] == one_cn, :]
                for _, one_t in t_temp.iterrows():
                    # prevent running backwards
                    if one_t['ConductingEquipment'] == from_dev:
                        continue
                    ids_temp = t.loc[(t['ConductingEquipment'] == one_t['ConductingEquipment']) & (
                            t['sequenceNumber'] != one_t['sequenceNumber'])]['ConnectivityNode'].values
                    for id_temp in ids_temp:
                        # check if the ConnectivityNode has a converter
                        if id_temp in converters['ConnectivityNode'].values:
                            # found the converter
                            return converters.loc[converters['ConnectivityNode'] == id_temp, 'converters'].values[0]
                        if id_temp not in visited_cns:
                            new_cn_dict[id_temp] = one_t['ConductingEquipment']
            if len(list(new_cn_dict.keys())) > 0:
                visited_cns.extend(list(cn_ids.keys()))
                return search_converter(cn_ids=new_cn_dict, visited_cns=visited_cns)

        for row_index, row in dc_line_segments[dc_line_segments['converters'].isna()].iterrows():
            conv = search_converter(cn_ids=dict({row['ConnectivityNode']: row['rdfId']}),
                                    visited_cns=[row['ConnectivityNode']])
            dc_line_segments.loc[row_index, 'converters'] = conv
            if conv is None:
                self.logger.warning("Problem with converting tht DC line %s: No ACDC converter found, maybe the DC "
                                    "part is too complex to reduce it to pandapower requirements!")
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="Error processing the DCLineSegments, there is a problem with Terminals in the source "
                            "data!"))
        dc_line_segments = dc_line_segments.drop(columns=['ConnectivityNode'])
        dc_line_segments = pd.merge(dc_line_segments, converters_t, how='left', on='converters')
        dc_line_segments['targetUpcc'].fillna(dc_line_segments['base_voltage_bus'], inplace=True)

        # copy the columns which are needed to reduce the dc_line_segments to one row per line
        dc_line_segments = dc_line_segments.sort_values(by=['rdfId', 'sequenceNumber'])
        # a list of DC line parameters which are used for each DC line end
        dc_line_segments.reset_index(inplace=True)
        copy_list = ['index_bus', 'rdfId_Terminal', 'connected', 'p', 'ratedUdc', 'targetUpcc', 'base_voltage_bus']
        for one_item in copy_list:
            # copy the columns which are required for each line end
            dc_line_segments[one_item + '2'] = dc_line_segments[one_item].copy()
            # cut the first element from the copied columns
            dc_line_segments[one_item + '2'] = dc_line_segments[one_item + '2'].iloc[1:].reset_index()[one_item + '2']
        del copy_list, one_item
        dc_line_segments = dc_line_segments.drop_duplicates(['rdfId'], keep='first')
        dc_line_segments = pd.merge(dc_line_segments,
                                    pd.DataFrame(dc_line_segments.pivot_table(index=['converters'], aggfunc='size'),
                                                 columns=['converter_dups']), how='left', on='converters')
        dc_line_segments['loss_mw'] = \
            abs(abs(dc_line_segments['p']) - abs(dc_line_segments['p2'])) / dc_line_segments['converter_dups']
        dc_line_segments['p_mw'] = dc_line_segments['p'] / dc_line_segments['converter_dups']
        dc_line_segments['loss_percent'] = 0
        dc_line_segments['vm_from_pu'] = dc_line_segments['targetUpcc'] / dc_line_segments['base_voltage_bus']
        dc_line_segments['vm_to_pu'] = dc_line_segments['targetUpcc2'] / dc_line_segments['base_voltage_bus2']
        dc_line_segments['in_service'] = dc_line_segments.connected & dc_line_segments.connected2
        dc_line_segments = dc_line_segments.rename(columns={
            'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'index_bus': 'from_bus',
            'index_bus2': 'to_bus'})

        return dc_line_segments

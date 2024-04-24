import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.seriesCompensatorsCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class SeriesCompensatorsCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_series_compensators_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting SeriesCompensators.")
        eq_sc = self._prepare_series_compensators_cim16()
        self.cimConverter.copy_to_pp('impedance', eq_sc)
        self.logger.info("Created %s impedance elements in %ss." % (eq_sc.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s impedance elements from SeriesCompensators in %ss." %
                    (eq_sc.index.size, time.time() - time_start)))

    def _prepare_series_compensators_cim16(self) -> pd.DataFrame:
        eq_sc = pd.merge(self.cimConverter.cim['eq']['SeriesCompensator'],
                         self.cimConverter.cim['eq']['BaseVoltage'][['rdfId',
                                                                     'nominalVoltage']].rename(
                             columns={'rdfId': 'BaseVoltage'}),
                         how='left', on='BaseVoltage')
        # fill the r21 and x21 values for impedance creation
        eq_sc['r21'] = eq_sc['r'].copy()
        eq_sc['x21'] = eq_sc['x'].copy()
        # set cim type
        eq_sc[sc['o_cl']] = 'SeriesCompensator'

        # add the buses
        eqs_length_before_merge = self.cimConverter.cim['eq']['SeriesCompensator'].index.size
        # until now self.cim['eq']['SeriesCompensator'] looks like:
        #   rdfId   name    r       ...
        #   _x01    bran1   0.056   ...
        #   _x02    bran2   0.471   ...
        # now join with the terminals and bus indexes
        eq_sc = pd.merge(eq_sc, self.cimConverter.bus_merge, how='left', on='rdfId')
        # now eq_sc looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    bran1   0.056   termi025        True        ...
        #   _x01    bran1   0.056   termi223        True        ...
        #   _x02    bran2   0.471   termi154        True        ...
        #   _x02    bran2   0.471   termi199        True        ...
        # if each series compensator got two terminals, reduce back to one row to use fast slicing
        if eq_sc.index.size != eqs_length_before_merge * 2:
            self.logger.error("There is a problem at the SeriesCompensators source data: Not each SeriesCompensator "
                              "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                              (eqs_length_before_merge * 2, eq_sc.index.size))
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="There is a problem at the SeriesCompensators source data: Not each SeriesCompensator "
                        "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                        (eqs_length_before_merge * 2, eq_sc.index.size)))
            dups = eq_sc.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The SeriesCompensator with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The SeriesCompensator data: \n%s" % eq_sc[eq_sc['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                    message="The SeriesCompensator with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eq_sc = eq_sc[0:0]
        # sort by RDF ID and the sequenceNumber to make sure r12 and r21 are in the correct order
        eq_sc = eq_sc.sort_values(by=['rdfId', 'sequenceNumber'])
        # copy the columns which are needed to reduce the eq_sc to one row per equivalent branch
        eq_sc['rdfId_Terminal2'] = eq_sc['rdfId_Terminal'].copy()
        eq_sc['connected2'] = eq_sc['connected'].copy()
        eq_sc['index_bus2'] = eq_sc['index_bus'].copy()
        eq_sc = eq_sc.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eq_sc.rdfId_Terminal2 = eq_sc.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
        eq_sc.connected2 = eq_sc.connected2.iloc[1:].reset_index().connected2
        eq_sc.index_bus2 = eq_sc.index_bus2.iloc[1:].reset_index().index_bus2
        eq_sc = eq_sc.drop_duplicates(['rdfId'], keep='first')
        if hasattr(self.cimConverter.net, 'sn_mva'):
            eq_sc['sn_mva'] = self.cimConverter.net['sn_mva']
        else:
            eq_sc['sn_mva'] = 1.
        # calculate z base in ohm
        eq_sc['z_base'] = eq_sc.nominalVoltage ** 2 / eq_sc.sn_mva
        eq_sc['rft_pu'] = eq_sc['r'] / eq_sc['z_base']
        eq_sc['xft_pu'] = eq_sc['x'] / eq_sc['z_base']
        eq_sc['rtf_pu'] = eq_sc['r21'] / eq_sc['z_base']
        eq_sc['xtf_pu'] = eq_sc['x21'] / eq_sc['z_base']
        eq_sc['rft0_pu'] = eq_sc['r0'] / eq_sc['z_base']
        eq_sc['xft0_pu'] = eq_sc['x0'] / eq_sc['z_base']
        eq_sc['rtf0_pu'] = eq_sc['r0'] / eq_sc['z_base']
        eq_sc['xtf0_pu'] = eq_sc['x0'] / eq_sc['z_base']
        eq_sc['in_service'] = eq_sc.connected & eq_sc.connected2
        eq_sc = eq_sc.rename(columns={'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'rdfId': sc['o_id'],
                              'index_bus': 'from_bus', 'index_bus2': 'to_bus'})
        return eq_sc

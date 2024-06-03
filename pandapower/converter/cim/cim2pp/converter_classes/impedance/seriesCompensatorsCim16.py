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
        if 'sc' in self.cimConverter.cim.keys():
            ser_comp = self.cimConverter.merge_eq_sc_profile('SeriesCompensator')
        else:
            ser_comp = self.cimConverter.cim['eq']['SeriesCompensator']

        ser_comp = pd.merge(ser_comp,
                            self.cimConverter.cim['eq']['BaseVoltage'][['rdfId','nominalVoltage']].rename(
                                columns={'rdfId': 'BaseVoltage'}),
                            how='left', on='BaseVoltage')
        # fill the r21 and x21 values for impedance creation
        ser_comp['r21'] = ser_comp['r'].copy()
        ser_comp['x21'] = ser_comp['x'].copy()
        # set cim type
        ser_comp[sc['o_cl']] = 'SeriesCompensator'

        # add the buses
        eqs_length_before_merge = self.cimConverter.cim['eq']['SeriesCompensator'].index.size
        # until now self.cim['eq']['SeriesCompensator'] looks like:
        #   rdfId   name    r       ...
        #   _x01    bran1   0.056   ...
        #   _x02    bran2   0.471   ...
        # now join with the terminals and bus indexes
        ser_comp = pd.merge(ser_comp, self.cimConverter.bus_merge, how='left', on='rdfId')
        # now ser_comp looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    bran1   0.056   termi025        True        ...
        #   _x01    bran1   0.056   termi223        True        ...
        #   _x02    bran2   0.471   termi154        True        ...
        #   _x02    bran2   0.471   termi199        True        ...
        # if each series compensator got two terminals, reduce back to one row to use fast slicing
        if ser_comp.index.size != eqs_length_before_merge * 2:
            self.logger.error("There is a problem at the SeriesCompensators source data: Not each SeriesCompensator "
                              "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                              (eqs_length_before_merge * 2, ser_comp.index.size))
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="There is a problem at the SeriesCompensators source data: Not each SeriesCompensator "
                        "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                        (eqs_length_before_merge * 2, ser_comp.index.size)))
            dups = ser_comp.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The SeriesCompensator with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The SeriesCompensator data: \n%s" % ser_comp[ser_comp['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                    message="The SeriesCompensator with RDF ID %s has %s Terminals!" % (rdfId, count)))
            ser_comp = ser_comp[0:0]
        # sort by RDF ID and the sequenceNumber to make sure r12 and r21 are in the correct order
        ser_comp = ser_comp.sort_values(by=['rdfId', 'sequenceNumber'])
        # copy the columns which are needed to reduce the ser_comp to one row per equivalent branch
        ser_comp['rdfId_Terminal2'] = ser_comp['rdfId_Terminal'].copy()
        ser_comp['connected2'] = ser_comp['connected'].copy()
        ser_comp['index_bus2'] = ser_comp['index_bus'].copy()
        ser_comp = ser_comp.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        ser_comp.rdfId_Terminal2 = ser_comp.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
        ser_comp.connected2 = ser_comp.connected2.iloc[1:].reset_index().connected2
        ser_comp.index_bus2 = ser_comp.index_bus2.iloc[1:].reset_index().index_bus2
        ser_comp = ser_comp.drop_duplicates(['rdfId'], keep='first')
        if hasattr(self.cimConverter.net, 'sn_mva'):
            ser_comp['sn_mva'] = self.cimConverter.net['sn_mva']
        else:
            ser_comp['sn_mva'] = 1.
        # calculate z base in ohm
        ser_comp['z_base'] = ser_comp.nominalVoltage ** 2 / ser_comp.sn_mva
        ser_comp['rft_pu'] = ser_comp['r'] / ser_comp['z_base']
        ser_comp['xft_pu'] = ser_comp['x'] / ser_comp['z_base']
        ser_comp['rtf_pu'] = ser_comp['r21'] / ser_comp['z_base']
        ser_comp['xtf_pu'] = ser_comp['x21'] / ser_comp['z_base']
        ser_comp['rft0_pu'] = ser_comp['r0'] / ser_comp['z_base']
        ser_comp['xft0_pu'] = ser_comp['x0'] / ser_comp['z_base']
        ser_comp['rtf0_pu'] = ser_comp['r0'] / ser_comp['z_base']
        ser_comp['xtf0_pu'] = ser_comp['x0'] / ser_comp['z_base']
        ser_comp['in_service'] = ser_comp.connected & ser_comp.connected2
        ser_comp = ser_comp.rename(columns={'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'],
                                            'rdfId': sc['o_id'], 'index_bus': 'from_bus', 'index_bus2': 'to_bus'})
        return ser_comp

import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.equivalentBranchesCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class EquivalentBranchesCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_equivalent_branches_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EquivalentBranches.")
        eq_eb = self._prepare_equivalent_branches_cim16()
        self.cimConverter.copy_to_pp('impedance', eq_eb)
        self.logger.info("Created %s impedance elements in %ss." % (eq_eb.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s impedance elements from EquivalentBranches in %ss." %
                    (eq_eb.index.size, time.time() - time_start)))

    def _prepare_equivalent_branches_cim16(self) -> pd.DataFrame:
        eq_eb = pd.merge(self.cimConverter.cim['eq']['EquivalentBranch'],
                         pd.concat([self.cimConverter.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']].rename(
                             columns={'rdfId': 'BaseVoltage'}),
                             self.cimConverter.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']].rename(
                                 columns={'rdfId': 'BaseVoltage'})], ignore_index=True, sort=False),
                         how='left', on='BaseVoltage')
        # the r21 and x21 are optional if they are equal to r and x, so fill up missing values
        eq_eb['r21'].fillna(eq_eb['r'], inplace=True)
        eq_eb['x21'].fillna(eq_eb['x'], inplace=True)
        eq_eb['zeroR21'].fillna(eq_eb['zeroR12'], inplace=True)
        eq_eb['zeroX21'].fillna(eq_eb['zeroX12'], inplace=True)
        # set cim type
        eq_eb[sc['o_cl']] = 'EquivalentBranch'

        # add the buses
        eqb_length_before_merge = self.cimConverter.cim['eq']['EquivalentBranch'].index.size
        # until now self.cim['eq']['EquivalentBranch'] looks like:
        #   rdfId   name    r       ...
        #   _x01    bran1   0.056   ...
        #   _x02    bran2   0.471   ...
        # now join with the terminals and bus indexes
        eq_eb = pd.merge(eq_eb, self.cimConverter.bus_merge, how='left', on='rdfId')
        # now eq_eb looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    bran1   0.056   termi025        True        ...
        #   _x01    bran1   0.056   termi223        True        ...
        #   _x02    bran2   0.471   termi154        True        ...
        #   _x02    bran2   0.471   termi199        True        ...
        # if each equivalent branch got two terminals, reduce back to one row to use fast slicing
        if eq_eb.index.size != eqb_length_before_merge * 2:
            self.logger.error("There is a problem at the EquivalentBranches source data: Not each EquivalentBranch "
                              "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                              (eqb_length_before_merge * 2, eq_eb.index.size))
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="There is a problem at the EquivalentBranches source data: Not each EquivalentBranch "
                        "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                        (eqb_length_before_merge * 2, eq_eb.index.size)))
            dups = eq_eb.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The EquivalentBranch with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The EquivalentBranch data: \n%s" % eq_eb[eq_eb['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The EquivalentBranch with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eq_eb = eq_eb[0:0]
        # sort by RDF ID and the sequenceNumber to make sure r12 and r21 are in the correct order
        eq_eb = eq_eb.sort_values(by=['rdfId', 'sequenceNumber'])
        # copy the columns which are needed to reduce the eq_eb to one row per equivalent branch
        eq_eb['rdfId_Terminal2'] = eq_eb['rdfId_Terminal'].copy()
        eq_eb['connected2'] = eq_eb['connected'].copy()
        eq_eb['index_bus2'] = eq_eb['index_bus'].copy()
        eq_eb = eq_eb.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eq_eb.rdfId_Terminal2 = eq_eb.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
        eq_eb.connected2 = eq_eb.connected2.iloc[1:].reset_index().connected2
        eq_eb.index_bus2 = eq_eb.index_bus2.iloc[1:].reset_index().index_bus2
        eq_eb = eq_eb.drop_duplicates(['rdfId'], keep='first')
        if hasattr(self.cimConverter.net, 'sn_mva'):
            eq_eb['sn_mva'] = self.cimConverter.net['sn_mva']
        else:
            eq_eb['sn_mva'] = 1.
        # calculate z base in ohm
        eq_eb['z_base'] = eq_eb.nominalVoltage ** 2 / eq_eb.sn_mva
        eq_eb['rft_pu'] = eq_eb['r'] / eq_eb['z_base']
        eq_eb['xft_pu'] = eq_eb['x'] / eq_eb['z_base']
        eq_eb['rtf_pu'] = eq_eb['r21'] / eq_eb['z_base']
        eq_eb['xtf_pu'] = eq_eb['x21'] / eq_eb['z_base']
        eq_eb['rft0_pu'] = eq_eb['zeroR12'] / eq_eb['z_base']
        eq_eb['xft0_pu'] = eq_eb['zeroX12'] / eq_eb['z_base']
        eq_eb['rtf0_pu'] = eq_eb['zeroR21'] / eq_eb['z_base']
        eq_eb['xtf0_pu'] = eq_eb['zeroX21'] / eq_eb['z_base']
        eq_eb['in_service'] = eq_eb.connected & eq_eb.connected2
        eq_eb = eq_eb.rename(columns={'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'rdfId': sc['o_id'],
                              'index_bus': 'from_bus', 'index_bus2': 'to_bus'})
        return eq_eb

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
        if 'sc' in self.cimConverter.cim.keys():
            eqb = self.cimConverter.merge_eq_sc_profile('EquivalentBranch')
        else:
            eqb = self.cimConverter.cim['eq']['EquivalentBranch']
        eqb = pd.merge(eqb,
                       pd.concat([self.cimConverter.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']].rename(
                           columns={'rdfId': 'BaseVoltage'}),
                           self.cimConverter.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']].rename(
                               columns={'rdfId': 'BaseVoltage'})], ignore_index=True, sort=False),
                       how='left', on='BaseVoltage')
        # the r21 and x21 are optional if they are equal to r and x, so fill up missing values
        eqb['r21'].fillna(eqb['r'], inplace=True)
        eqb['x21'].fillna(eqb['x'], inplace=True)
        eqb['zeroR21'].fillna(eqb['zeroR12'], inplace=True)
        eqb['zeroX21'].fillna(eqb['zeroX12'], inplace=True)
        # set cim type
        eqb[sc['o_cl']] = 'EquivalentBranch'

        # add the buses
        eqb_length_before_merge = self.cimConverter.cim['eq']['EquivalentBranch'].index.size
        # until now self.cim['eq']['EquivalentBranch'] looks like:
        #   rdfId   name    r       ...
        #   _x01    bran1   0.056   ...
        #   _x02    bran2   0.471   ...
        # now join with the terminals and bus indexes
        eqb = pd.merge(eqb, self.cimConverter.bus_merge, how='left', on='rdfId')
        # now eqb looks like:
        #   rdfId   name    r       rdfId_Terminal  connected   ...
        #   _x01    bran1   0.056   termi025        True        ...
        #   _x01    bran1   0.056   termi223        True        ...
        #   _x02    bran2   0.471   termi154        True        ...
        #   _x02    bran2   0.471   termi199        True        ...
        # if each equivalent branch got two terminals, reduce back to one row to use fast slicing
        if eqb.index.size != eqb_length_before_merge * 2:
            self.logger.error("There is a problem at the EquivalentBranches source data: Not each EquivalentBranch "
                              "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                              (eqb_length_before_merge * 2, eqb.index.size))
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="There is a problem at the EquivalentBranches source data: Not each EquivalentBranch "
                        "has two Terminals, %s Terminals should be given but there are %s Terminals available" %
                        (eqb_length_before_merge * 2, eqb.index.size)))
            dups = eqb.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The EquivalentBranch with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The EquivalentBranch data: \n%s" % eqb[eqb['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The EquivalentBranch with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eqb = eqb[0:0]
        # sort by RDF ID and the sequenceNumber to make sure r12 and r21 are in the correct order
        eqb = eqb.sort_values(by=['rdfId', 'sequenceNumber'])
        # copy the columns which are needed to reduce the eqb to one row per equivalent branch
        eqb['rdfId_Terminal2'] = eqb['rdfId_Terminal'].copy()
        eqb['connected2'] = eqb['connected'].copy()
        eqb['index_bus2'] = eqb['index_bus'].copy()
        eqb = eqb.reset_index()
        # here is where the magic happens: just remove the first value from the copied columns, reset the index
        # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
        eqb.rdfId_Terminal2 = eqb.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
        eqb.connected2 = eqb.connected2.iloc[1:].reset_index().connected2
        eqb.index_bus2 = eqb.index_bus2.iloc[1:].reset_index().index_bus2
        eqb = eqb.drop_duplicates(['rdfId'], keep='first')
        if hasattr(self.cimConverter.net, 'sn_mva'):
            eqb['sn_mva'] = self.cimConverter.net['sn_mva']
        else:
            eqb['sn_mva'] = 1.
        # calculate z base in ohm
        eqb['z_base'] = eqb.nominalVoltage ** 2 / eqb.sn_mva
        eqb['rft_pu'] = eqb['r'] / eqb['z_base']
        eqb['xft_pu'] = eqb['x'] / eqb['z_base']
        eqb['rtf_pu'] = eqb['r21'] / eqb['z_base']
        eqb['xtf_pu'] = eqb['x21'] / eqb['z_base']
        eqb['rft0_pu'] = eqb['zeroR12'] / eqb['z_base']
        eqb['xft0_pu'] = eqb['zeroX12'] / eqb['z_base']
        eqb['rtf0_pu'] = eqb['zeroR21'] / eqb['z_base']
        eqb['xtf0_pu'] = eqb['zeroX21'] / eqb['z_base']
        eqb['in_service'] = eqb.connected & eqb.connected2
        eqb = eqb.rename(columns={'rdfId_Terminal': sc['t_from'], 'rdfId_Terminal2': sc['t_to'], 'rdfId': sc['o_id'],
                                  'index_bus': 'from_bus', 'index_bus2': 'to_bus'})
        return eqb

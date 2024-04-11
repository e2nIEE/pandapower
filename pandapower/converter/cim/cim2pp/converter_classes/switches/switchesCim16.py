import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.switchesCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class SwitchesCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_switches_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting Breakers, Disconnectors, LoadBreakSwitches and Switches.")
        eqssh_switches = self._prepare_switches_cim16()
        self.cimConverter.copy_to_pp('switch', eqssh_switches)
        self.logger.info("Created %s switches in %ss." % (eqssh_switches.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s switches from Breakers, Disconnectors, LoadBreakSwitches and Switches in %ss." %
                    (eqssh_switches.index.size, time.time() - time_start)))

    def _prepare_switches_cim16(self) -> pd.DataFrame:
        eqssh_switches = self.cimConverter.merge_eq_ssh_profile('Breaker', add_cim_type_column=True)
        eqssh_switches['type'] = 'CB'
        start_index_cim_net = eqssh_switches.index.size
        eqssh_switches = \
            pd.concat(
                [eqssh_switches, self.cimConverter.merge_eq_ssh_profile('Disconnector', add_cim_type_column=True)],
                ignore_index=True, sort=False)
        eqssh_switches.type[start_index_cim_net:] = 'DS'
        start_index_cim_net = eqssh_switches.index.size
        eqssh_switches = \
            pd.concat(
                [eqssh_switches, self.cimConverter.merge_eq_ssh_profile('LoadBreakSwitch', add_cim_type_column=True)],
                ignore_index=True, sort=False)
        eqssh_switches.type[start_index_cim_net:] = 'LBS'
        start_index_cim_net = eqssh_switches.index.size
        # switches needs to be the last which getting appended because of class inherit problem in jpa
        eqssh_switches = pd.concat(
            [eqssh_switches, self.cimConverter.merge_eq_ssh_profile('Switch', add_cim_type_column=True)],
            ignore_index=True, sort=False)
        eqssh_switches.type[start_index_cim_net:] = 'LS'
        # drop all duplicates to fix class inherit problem in jpa
        eqssh_switches = eqssh_switches.drop_duplicates(subset=['rdfId'], keep='first')
        switch_length_before_merge = eqssh_switches.index.size
        # until now eqssh_switches looks like:
        #   rdfId   name    open    ...
        #   _x01    switch1 True    ...
        #   _x02    switch2 False   ...
        # now join with the terminals
        eqssh_switches = pd.merge(eqssh_switches, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_switches = eqssh_switches.sort_values(by=['rdfId', 'sequenceNumber'])
        eqssh_switches.reset_index(inplace=True)
        # copy the columns which are needed to reduce the eqssh_switches to one line per switch
        eqssh_switches['rdfId_Terminal2'] = eqssh_switches['rdfId_Terminal'].copy()
        eqssh_switches['connected2'] = eqssh_switches['connected'].copy()
        eqssh_switches['index_bus2'] = eqssh_switches['index_bus'].copy()
        if eqssh_switches.index.size == switch_length_before_merge * 2:
            # here is where the magic happens: just remove the first value from the copied columns, reset the index
            # and replace the old column with the cut one. At least just remove the duplicates on column rdfId
            eqssh_switches.rdfId_Terminal2 = eqssh_switches.rdfId_Terminal2.iloc[1:].reset_index().rdfId_Terminal2
            eqssh_switches.connected2 = eqssh_switches.connected2.iloc[1:].reset_index().connected2
            eqssh_switches.index_bus2 = eqssh_switches.index_bus2.iloc[1:].reset_index().index_bus2
            eqssh_switches = eqssh_switches.drop_duplicates(subset=['rdfId'], keep='first')
        else:
            self.logger.error("Something went wrong at switches, seems like that terminals for connection with "
                              "connectivity nodes are missing!")
            self.cimConverter.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Something went wrong at switches, seems like that terminals for connection with "
                        "connectivity nodes are missing!"))
            dups = eqssh_switches.pivot_table(index=['rdfId'], aggfunc='size')
            dups = dups.loc[dups != 2]
            for rdfId, count in dups.items():
                self.logger.warning("The switch with RDF ID %s has %s Terminals!" % (rdfId, count))
                self.logger.warning("The switch data: \n%s" % eqssh_switches[eqssh_switches['rdfId'] == rdfId])
                self.cimConverter.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="The switch with RDF ID %s has %s Terminals!" % (rdfId, count)))
            eqssh_switches = eqssh_switches[0:0]
        eqssh_switches = eqssh_switches.rename(columns={'rdfId': sc['o_id'], 'index_bus': 'bus', 'index_bus2': 'element',
                                       'rdfId_Terminal': sc['t_bus'], 'rdfId_Terminal2': sc['t_ele']})
        eqssh_switches['et'] = 'b'
        eqssh_switches['z_ohm'] = 0
        if eqssh_switches.index.size > 0:
            eqssh_switches['closed'] = ~eqssh_switches.open & eqssh_switches.connected & eqssh_switches.connected2
        return eqssh_switches

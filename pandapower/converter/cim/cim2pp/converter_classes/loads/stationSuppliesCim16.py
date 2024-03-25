import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.stationSuppliesCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class StationSuppliesCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_station_supplies_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting StationSupplies.")
        eqssh_station_supplies = self._prepare_station_supplies_cim16()
        self.cimConverter.copy_to_pp('load', eqssh_station_supplies)
        self.logger.info("Created %s loads in %ss." % (eqssh_station_supplies.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from StationSupplies in %ss." %
                    (eqssh_station_supplies.index.size, time.time() - time_start)))

    def _prepare_station_supplies_cim16(self) -> pd.DataFrame:
        eqssh_station_supplies = self.cimConverter.merge_eq_ssh_profile('StationSupply', add_cim_type_column=True)
        eqssh_station_supplies = pd.merge(eqssh_station_supplies, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_station_supplies = eqssh_station_supplies.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                               'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'})
        eqssh_station_supplies['const_i_percent'] = 0.
        eqssh_station_supplies['const_z_percent'] = 0.
        eqssh_station_supplies['scaling'] = 1.
        eqssh_station_supplies['type'] = None
        return eqssh_station_supplies

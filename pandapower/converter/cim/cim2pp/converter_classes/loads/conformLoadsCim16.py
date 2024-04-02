import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.conformLoadsCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class ConformLoadsCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_conform_loads_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting ConformLoads.")
        eqssh_conform_loads = self._prepare_conform_loads_cim16()
        self.cimConverter.copy_to_pp('load', eqssh_conform_loads)
        self.logger.info("Created %s loads in %ss." % (eqssh_conform_loads.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from ConformLoads in %ss." %
                    (eqssh_conform_loads.index.size, time.time() - time_start)))

    def _prepare_conform_loads_cim16(self) -> pd.DataFrame:
        eqssh_conform_loads = self.cimConverter.merge_eq_ssh_profile('ConformLoad', add_cim_type_column=True)
        eqssh_conform_loads = pd.merge(eqssh_conform_loads, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_conform_loads = eqssh_conform_loads.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                            'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'})
        eqssh_conform_loads['const_i_percent'] = 0.
        eqssh_conform_loads['const_z_percent'] = 0.
        eqssh_conform_loads['scaling'] = 1.
        eqssh_conform_loads['type'] = None
        return eqssh_conform_loads

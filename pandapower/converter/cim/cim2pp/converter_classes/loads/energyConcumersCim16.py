import logging
import time

import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.energyConsumersCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class EnergyConsumersCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_energy_consumers_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EnergyConsumers.")
        eqssh_energy_consumers = self._prepare_energy_consumers_cim16()
        self.cimConverter.copy_to_pp('load', eqssh_energy_consumers)
        self.logger.info("Created %s loads in %ss." % (eqssh_energy_consumers.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s loads from EnergyConsumers in %ss." %
                    (eqssh_energy_consumers.index.size, time.time() - time_start)))

    def _prepare_energy_consumers_cim16(self) -> pd.DataFrame:
        eqssh_energy_consumers = self.cimConverter.merge_eq_ssh_profile('EnergyConsumer', add_cim_type_column=True)
        eqssh_energy_consumers = pd.merge(eqssh_energy_consumers, self.cimConverter.bus_merge, how='left', on='rdfId')
        eqssh_energy_consumers = eqssh_energy_consumers.rename(columns={'rdfId': sc['o_id'], 'rdfId_Terminal': sc['t'], 'index_bus': 'bus',
                                               'connected': 'in_service', 'p': 'p_mw', 'q': 'q_mvar'})
        eqssh_energy_consumers['const_i_percent'] = 0.
        eqssh_energy_consumers['const_z_percent'] = 0.
        eqssh_energy_consumers['scaling'] = 1.
        eqssh_energy_consumers['type'] = None
        return eqssh_energy_consumers

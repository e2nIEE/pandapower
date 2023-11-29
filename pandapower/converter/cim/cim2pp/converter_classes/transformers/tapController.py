import logging
from typing import List

import pandas as pd

import pandapower.auxiliary
from pandapower.control.controller.trafo.ContinuousTapControl import ContinuousTapControl
from pandapower.control.controller.trafo.DiscreteTapControl import DiscreteTapControl
from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net

logger = logging.getLogger('cim.cim2pp.converter_classes.tapController')

sc = cim_tools.get_pp_net_special_columns_dict()


class TapController:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def create_tap_controller_for_power_transformers(self):
        if self.cimConverter.power_trafo2w.index.size > 0:
            # create transformer tap controller
            self._create_tap_controller(self.cimConverter.power_trafo2w, 'trafo')
            # create the characteristic objects for transformers
            characteristic_df_temp = \
                self.cimConverter.net['characteristic_temp'][['id_characteristic', 'step', 'vk_percent', 'vkr_percent']]
            for trafo_id, trafo_row in self.cimConverter.net.trafo.dropna(subset=['id_characteristic']).iterrows():
                characteristic_df = characteristic_df_temp.loc[
                    characteristic_df_temp['id_characteristic'] == trafo_row['id_characteristic']]
                self._create_characteristic_object(net=self.cimConverter.net, trafo_type='trafo', trafo_id=[trafo_id],
                                                   characteristic_df=characteristic_df)
        if self.cimConverter.power_trafo3w.index.size > 0:
            # create transformer tap controller
            self._create_tap_controller(self.cimConverter.power_trafo3w, 'trafo3w')
            # create the characteristic objects for transformers
            characteristic_df_temp = \
                self.cimConverter.net['characteristic_temp'][
                    ['id_characteristic', 'step', 'vkr_hv_percent', 'vkr_mv_percent',
                     'vkr_lv_percent', 'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent']]
            for trafo_id, trafo_row in self.cimConverter.net.trafo3w.dropna(subset=['id_characteristic']).iterrows():
                characteristic_df = characteristic_df_temp.loc[
                    characteristic_df_temp['id_characteristic'] == trafo_row['id_characteristic']]
                self._create_characteristic_object(net=self.cimConverter.net, trafo_type='trafo3w', trafo_id=[trafo_id],
                                                   characteristic_df=characteristic_df)

    def _create_characteristic_object(self, net, trafo_type: str, trafo_id: List, characteristic_df: pd.DataFrame):
        self.logger.info("Adding characteristic object for trafo_type: %s and trafo_id: %s" % (trafo_type, trafo_id))
        for variable in ['vkr_percent', 'vk_percent', 'vk_hv_percent', 'vkr_hv_percent', 'vk_mv_percent',
                         'vkr_mv_percent', 'vk_lv_percent', 'vkr_lv_percent']:
            if variable in characteristic_df.columns:
                pandapower.control.create_trafo_characteristics(net, trafo_type, trafo_id, variable,
                                                                [characteristic_df['step'].to_list()],
                                                                [characteristic_df[variable].to_list()])

    def _create_tap_controller(self, input_df: pd.DataFrame, trafo_type: str):
        if not self.cimConverter.kwargs.get('create_tap_controller', True):
            self.logger.info("Skip creating transformer tap changer controller for transformer type %s." % trafo_type)
            return
        for _, row in input_df.loc[input_df['TapChangerControl'].notna()].iterrows():
            trafo_id = self.cimConverter.net[trafo_type].loc[
                self.cimConverter.net[trafo_type][sc['o_id']] == row[sc['o_id']]].index.values[0]
            trafotype = '2W' if trafo_type == 'trafo' else '3W'
            # get the controlled bus (side), assume "lv" as default
            side = 'lv'
            if sc['t_hv'] in self.cimConverter.net[trafo_type].columns and \
                    row['c_Terminal'] in self.cimConverter.net[trafo_type][sc['t_hv']].values:
                side = 'hv'
            if sc['t_mv'] in self.cimConverter.net[trafo_type].columns and \
                    row['c_Terminal'] in self.cimConverter.net[trafo_type][sc['t_mv']].values:
                side = 'mv'
            if row['discrete']:
                self.logger.info("Creating DiscreteTapControl for transformer %s." % row[sc['o_id']])
                DiscreteTapControl(self.cimConverter.net, trafotype=trafotype, tid=trafo_id, side=side,
                                   tol=row['c_tol'], in_service=row['c_in_service'],
                                   vm_lower_pu=row['c_vm_lower_pu'], vm_upper_pu=row['c_vm_upper_pu'])
            else:
                self.logger.info("Creating ContinuousTapControl for transformer %s." % row[sc['o_id']])
                ContinuousTapControl(self.cimConverter.net, trafotype=trafotype, tid=trafo_id, side=side,
                                     tol=row['c_tol'], in_service=row['c_in_service'], vm_set_pu=row['c_vm_set_pu'])

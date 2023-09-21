# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import math
from typing import Dict, List
import traceback
import pandapower as pp
import pandapower.auxiliary
from pandapower.control.controller.trafo.ContinuousTapControl import ContinuousTapControl
from pandapower.control.controller.trafo.DiscreteTapControl import DiscreteTapControl
import pandas as pd
import numpy as np
import time
from .convert_measurements import CreateMeasurements
from .. import cim_tools
from .. import pp_tools
from .. import cim_classes
from ..other_classes import ReportContainer, Report, LogLevel, ReportCode
logger = logging.getLogger('cim.cim2pp.build_pp_net')

pd.set_option('display.max_columns', 900)
pd.set_option('display.max_rows', 90000)
sc = cim_tools.get_pp_net_special_columns_dict()


class CimConverter:

    def __init__(self, cim_parser: cim_classes.CimParser, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cim_parser: cim_classes.CimParser = cim_parser
        self.kwargs = kwargs
        self.cim: Dict[str, Dict[str, pd.DataFrame]] = self.cim_parser.get_cim_dict()
        self.net: pandapower.auxiliary.pandapowerNet = pp.create_empty_network()
        self.bus_merge: pd.DataFrame = pd.DataFrame()
        self.power_trafo2w: pd.DataFrame = pd.DataFrame()
        self.power_trafo3w: pd.DataFrame = pd.DataFrame()
        self.report_container: ReportContainer = cim_parser.get_report_container()

    def merge_eq_ssh_profile(self, cim_type: str, add_cim_type_column: bool = False) -> pd.DataFrame:
        df = pd.merge(self.cim['eq'][cim_type], self.cim['ssh'][cim_type], how='left', on='rdfId')
        if add_cim_type_column:
            df[sc['o_cl']] = cim_type
        return df

    def _convert_equivalent_injections_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting EquivalentInjections.")
        eqssh_ei = self._prepare_equivalent_injections_cim16()
        # split up to wards and xwards: the wards have no regulation
        eqssh_ei_wards = eqssh_ei.loc[~eqssh_ei.regulationStatus]
        eqssh_ei_xwards = eqssh_ei.loc[eqssh_ei.regulationStatus]
        self.copy_to_pp('ward', eqssh_ei_wards)
        self.copy_to_pp('xward', eqssh_ei_xwards)
        self.logger.info("Created %s wards and %s extended ward elements in %ss." %
                         (eqssh_ei_wards.index.size, eqssh_ei_xwards.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s wards and %s extended ward elements from EquivalentInjections in %ss." %
                    (eqssh_ei_wards.index.size, eqssh_ei_xwards.index.size, time.time() - time_start)))

    def _prepare_equivalent_injections_cim16(self) -> pd.DataFrame:
        eqssh_ei = self.merge_eq_ssh_profile('EquivalentInjection', add_cim_type_column=True)
        eq_base_voltages = pd.concat([self.cim['eq']['BaseVoltage'][['rdfId', 'nominalVoltage']],
                                      self.cim['eq_bd']['BaseVoltage'][['rdfId', 'nominalVoltage']]], sort=False)
        eq_base_voltages.drop_duplicates(subset=['rdfId'], inplace=True)
        eq_base_voltages.rename(columns={'rdfId': 'BaseVoltage'}, inplace=True)
        eqssh_ei = pd.merge(eqssh_ei, eq_base_voltages, how='left', on='BaseVoltage')
        eqssh_ei = pd.merge(eqssh_ei, self.bus_merge, how='left', on='rdfId')
        # maybe the BaseVoltage is not given, also get the nominalVoltage from the buses
        eqssh_ei = pd.merge(eqssh_ei, self.net.bus[['vn_kv']], how='left', left_on='index_bus', right_index=True)
        eqssh_ei.nominalVoltage.fillna(eqssh_ei.vn_kv, inplace=True)
        eqssh_ei['regulationStatus'].fillna(False, inplace=True)
        eqssh_ei['vm_pu'] = eqssh_ei.regulationTarget / eqssh_ei.nominalVoltage
        eqssh_ei.rename(columns={'rdfId_Terminal': sc['t'], 'rdfId': sc['o_id'], 'connected': 'in_service',
                                 'index_bus': 'bus', 'p': 'ps_mw', 'q': 'qs_mvar'},
                        inplace=True)
        eqssh_ei['pz_mw'] = 0.
        eqssh_ei['qz_mvar'] = 0.
        eqssh_ei['r_ohm'] = 0.
        eqssh_ei['x_ohm'] = .1
        return eqssh_ei

    def _convert_power_transformers_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting PowerTransformers.")

        eq_power_transformers = self._prepare_power_transformers_cim16()
        # split the power transformers into two and three windings
        power_trafo_counts = eq_power_transformers.PowerTransformer.value_counts()
        power_trafo2w = power_trafo_counts[power_trafo_counts == 2].index.tolist()
        power_trafo3w = power_trafo_counts[power_trafo_counts == 3].index.tolist()

        eq_power_transformers.set_index('PowerTransformer', inplace=True)
        power_trafo2w = eq_power_transformers.loc[power_trafo2w].reset_index()
        power_trafo3w = eq_power_transformers.loc[power_trafo3w].reset_index()

        if power_trafo2w.index.size > 0:
            # process the two winding transformers
            self._create_trafo_characteristics('trafo', power_trafo2w)
            power_trafo2w = self._prepare_trafos_cim16(power_trafo2w)
            self.copy_to_pp('trafo', power_trafo2w)
            self.power_trafo2w = power_trafo2w

        if power_trafo3w.index.size > 0:
            # process the three winding transformers
            self._create_trafo_characteristics('trafo3w', power_trafo3w)
            power_trafo3w = self._prepare_trafo3w_cim16(power_trafo3w)
            self.copy_to_pp('trafo3w', power_trafo3w)
            self.power_trafo3w = power_trafo3w

        self.logger.info("Created %s 2w trafos and %s 3w trafos in %ss." %
                         (power_trafo2w.index.size, power_trafo3w.index.size, time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s 2w trafos and %s 3w trafos from PowerTransformers in %ss." %
                    (power_trafo2w.index.size, power_trafo3w.index.size, time.time() - time_start)))

    def _create_trafo_characteristics(self, trafo_type, trafo_df_origin):
        if 'id_characteristic' not in trafo_df_origin.columns:
            trafo_df_origin['id_characteristic'] = np.NaN
        if 'characteristic_temp' not in self.net.keys():
            self.net['characteristic_temp'] = pd.DataFrame(columns=['id_characteristic', 'step', 'vk_percent',
                                                                    'vkr_percent', 'vkr_hv_percent', 'vkr_mv_percent',
                                                                    'vkr_lv_percent', 'vk_hv_percent', 'vk_mv_percent',
                                                                    'vk_lv_percent'])
        # get the TablePoints
        ptct = self.cim['eq']['PhaseTapChangerTabular'][['TransformerEnd', 'PhaseTapChangerTable']]
        ptct = pd.merge(ptct, self.cim['eq']['PhaseTapChangerTablePoint'][
            ['PhaseTapChangerTable', 'step', 'r', 'x']], how='left', on='PhaseTapChangerTable')
        # append the ratio tab changers
        ptct_ratio = self.cim['eq']['RatioTapChanger'][['TransformerEnd', 'RatioTapChangerTable']]
        ptct_ratio = pd.merge(ptct_ratio, self.cim['eq']['RatioTapChangerTablePoint'][
            ['RatioTapChangerTable', 'step', 'r', 'x']], how='left', on='RatioTapChangerTable')
        ptct = pd.concat([ptct, ptct_ratio], ignore_index=True, sort=False)
        ptct.rename(columns={'step': 'tabular_step', 'r': 'r_dev', 'x': 'x_dev', 'TransformerEnd': sc['pte_id']},
                    inplace=True)
        ptct.drop(columns=['PhaseTapChangerTable'], inplace=True)
        if trafo_type == 'trafo':
            trafo_df = trafo_df_origin.sort_values(['PowerTransformer', 'endNumber']).reset_index()
            # precessing the transformer data
            # a list of transformer parameters which are used for each transformer winding
            copy_list = ['ratedU', 'r', 'x', sc['pte_id'], 'neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in copy_list:
                # copy the columns which are required for each winding
                trafo_df[one_item + '_lv'] = trafo_df[one_item].copy()
                # cut the first element from the copied columns
                trafo_df[one_item + '_lv'] = trafo_df[one_item + '_lv'].iloc[1:].reset_index()[
                    one_item + '_lv']
            del copy_list, one_item
            fillna_list = ['neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            # just keep one transformer
            trafo_df.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)
            # merge the trafos with the tap changers
            trafo_df = pd.merge(trafo_df, ptct, how='left', on=sc['pte_id'])
            trafo_df = pd.merge(trafo_df, ptct.rename(columns={'tabular_step': 'tabular_step_lv', 'r_dev': 'r_dev_lv',
                                                               'x_dev': 'x_dev_lv',
                                                               sc['pte_id']: sc['pte_id']+'_lv'}),
                                how='left', on=sc['pte_id']+'_lv')
            fillna_list = ['tabular_step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            trafo_df.dropna(subset=['r_dev', 'r_dev_lv'], how='all', inplace=True)
            fillna_list = ['r_dev', 'r_dev_lv', 'x_dev', 'x_dev_lv']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(0, inplace=True)
            # special fix for the case that the impedance is given at the hv side but the tap changer is attached at
            # the lv side (so r_lv = 0 and r_dev_lv > 0):
            trafo_df.loc[(trafo_df['r_dev_lv'] != 0) & (trafo_df['r_lv'] == 0) & (trafo_df['r_dev'] == 0), 'r_dev'] = \
                trafo_df.loc[(trafo_df['r_dev_lv'] != 0) & (trafo_df['r_lv'] == 0) & (trafo_df['r_dev'] == 0),
                             'r_dev_lv']
            trafo_df.loc[(trafo_df['x_dev_lv'] != 0) & (trafo_df['x_lv'] == 0) & (trafo_df['x_dev'] == 0), 'x_dev'] = \
                trafo_df.loc[(trafo_df['x_dev_lv'] != 0) & (trafo_df['x_lv'] == 0) & (trafo_df['x_dev'] == 0),
                             'x_dev_lv']
            trafo_df['r'] = trafo_df['r'] + trafo_df['r'] * trafo_df['r_dev'] / 100
            trafo_df['r_lv'] = trafo_df['r_lv'] * (1 + trafo_df['r_dev_lv'] / 100)
            trafo_df['x'] = trafo_df['x'] + trafo_df['x'] * trafo_df['x_dev'] / 100
            trafo_df['x_lv'] = trafo_df['x_lv'] * (1 + trafo_df['x_dev_lv'] / 100)

            # calculate vkr_percent and vk_percent
            trafo_df['vkr_percent'] = \
                abs(trafo_df.r) * trafo_df.ratedS * 100 / trafo_df.ratedU ** 2 + \
                abs(trafo_df.r_lv) * trafo_df.ratedS * 100 / trafo_df.ratedU_lv ** 2
            trafo_df['vk_percent'] = \
                (abs(trafo_df.r) ** 2 + abs(trafo_df.x) ** 2) ** 0.5 * \
                (trafo_df.ratedS * 1e3) / (10. * trafo_df.ratedU ** 2) + \
                (abs(trafo_df.r_lv) ** 2 + abs(trafo_df.x_lv) ** 2) ** 0.5 * \
                (trafo_df.ratedS * 1e3) / (10. * trafo_df.ratedU_lv ** 2)
            trafo_df['tabular_step'] = trafo_df['tabular_step'].astype(int)
            append_dict = dict({'id_characteristic': [], 'step': [], 'vk_percent': [], 'vkr_percent': []})
        else:
            trafo_df = trafo_df_origin.copy()
            trafo_df = trafo_df.sort_values(['PowerTransformer', 'endNumber']).reset_index()
            # copy the required fields for middle and low voltage
            copy_list = ['ratedS', 'ratedU', 'r', 'x', sc['pte_id'], 'neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in copy_list:
                # copy the columns which are required for each winding
                trafo_df[one_item + '_mv'] = trafo_df[one_item].copy()
                trafo_df[one_item + '_lv'] = trafo_df[one_item].copy()
                # cut the first (or first two) element(s) from the copied columns
                trafo_df[one_item + '_mv'] = trafo_df[one_item + '_mv'].iloc[1:].reset_index()[
                    one_item + '_mv']
                trafo_df[one_item + '_lv'] = trafo_df[one_item + '_lv'].iloc[2:].reset_index()[
                    one_item + '_lv']
            del copy_list, one_item
            fillna_list = ['neutralStep', 'lowStep', 'highStep', 'step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_mv'], inplace=True)
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            # just keep one transformer
            trafo_df.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)
            # merge the trafos with the tap changers
            trafo_df = pd.concat([pd.merge(trafo_df, ptct, how='left', on=sc['pte_id']),
                                  pd.merge(trafo_df,
                                           ptct.rename(columns={'tabular_step': 'tabular_step_mv', 'r_dev': 'r_dev_mv',
                                                                'x_dev': 'x_dev_mv',
                                                                sc['pte_id']: sc['pte_id'] + '_mv'}),
                                           how='left', on=sc['pte_id'] + '_mv'),
                                  pd.merge(trafo_df,
                                           ptct.rename(columns={'tabular_step': 'tabular_step_lv', 'r_dev': 'r_dev_lv',
                                                                'x_dev': 'x_dev_lv',
                                                                sc['pte_id']: sc['pte_id'] + '_lv'}),
                                           how='left', on=sc['pte_id'] + '_lv')
                                  ], ignore_index=True, sort=False)
            # remove elements with mor than one tap changer per trafo
            trafo_df = trafo_df.loc[(~trafo_df.duplicated(subset=['PowerTransformer', 'tabular_step'], keep=False)) | (
                ~trafo_df.RatioTapChangerTable.isna())]
            # remove elements with mor than one tap changer per trafo
            trafo_df = trafo_df.loc[(~trafo_df.duplicated(subset=['PowerTransformer', 'tabular_step'], keep=False)) | (
                ~trafo_df.RatioTapChangerTable.isna())]
            fillna_list = ['tabular_step']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(trafo_df[one_item + '_mv'], inplace=True)
                trafo_df[one_item].fillna(trafo_df[one_item + '_lv'], inplace=True)
            del fillna_list, one_item
            trafo_df.dropna(subset=['r_dev', 'r_dev_mv', 'r_dev_lv'], how='all', inplace=True)
            fillna_list = ['r_dev', 'r_dev_mv', 'r_dev_lv', 'x_dev', 'x_dev_mv', 'x_dev_lv']
            for one_item in fillna_list:
                trafo_df[one_item].fillna(0, inplace=True)
            # calculate vkr_percent and vk_percent
            trafo_df['r'] = trafo_df['r'] * (1 + trafo_df['r_dev'] / 100)
            trafo_df['r_mv'] = trafo_df['r_mv'] * (1 + trafo_df['r_dev_mv'] / 100)
            trafo_df['r_lv'] = trafo_df['r_lv'] * (1 + trafo_df['r_dev_lv'] / 100)
            trafo_df['x'] = trafo_df['x'] * (1 + trafo_df['x_dev'] / 100)
            trafo_df['x_mv'] = trafo_df['x_mv'] * (1 + trafo_df['x_dev_mv'] / 100)
            trafo_df['x_lv'] = trafo_df['x_lv'] * (1 + trafo_df['x_dev_lv'] / 100)

            trafo_df['min_s_hvmv'] = trafo_df[["ratedS", "ratedS_mv"]].min(axis=1)
            trafo_df['min_s_mvlv'] = trafo_df[["ratedS_mv", "ratedS_lv"]].min(axis=1)
            trafo_df['min_s_lvhv'] = trafo_df[["ratedS_lv", "ratedS"]].min(axis=1)

            trafo_df['vkr_hv_percent'] = \
                (trafo_df.r + trafo_df.r_mv * (trafo_df.ratedU / trafo_df.ratedU_mv) ** 2) * \
                trafo_df.min_s_hvmv * 100 / trafo_df.ratedU ** 2
            trafo_df['vkr_mv_percent'] = \
                (trafo_df.r_mv + trafo_df.r_lv * (trafo_df.ratedU_mv / trafo_df.ratedU_lv) ** 2) * \
                trafo_df.min_s_mvlv * 100 / trafo_df.ratedU_mv ** 2
            trafo_df['vkr_lv_percent'] = \
                (trafo_df.r_lv + trafo_df.r * (trafo_df.ratedU_lv / trafo_df.ratedU) ** 2) * \
                trafo_df.min_s_lvhv * 100 / trafo_df.ratedU_lv ** 2
            trafo_df['vk_hv_percent'] = \
                ((trafo_df.r + trafo_df.r_mv * (trafo_df.ratedU / trafo_df.ratedU_mv) ** 2) ** 2 +
                 (trafo_df.x + trafo_df.x_mv * (
                         trafo_df.ratedU / trafo_df.ratedU_mv) ** 2) ** 2) ** 0.5 * \
                trafo_df.min_s_hvmv * 100 / trafo_df.ratedU ** 2
            trafo_df['vk_mv_percent'] = \
                ((trafo_df.r_mv + trafo_df.r_lv * (
                        trafo_df.ratedU_mv / trafo_df.ratedU_lv) ** 2) ** 2 +
                 (trafo_df.x_mv + trafo_df.x_lv * (
                         trafo_df.ratedU_mv / trafo_df.ratedU_lv) ** 2) ** 2) ** 0.5 * \
                trafo_df.min_s_mvlv * 100 / trafo_df.ratedU_mv ** 2
            trafo_df['vk_lv_percent'] = \
                ((trafo_df.r_lv + trafo_df.r * (trafo_df.ratedU_lv / trafo_df.ratedU) ** 2) ** 2 +
                 (trafo_df.x_lv + trafo_df.x * (
                         trafo_df.ratedU_lv / trafo_df.ratedU) ** 2) ** 2) ** 0.5 * \
                trafo_df.min_s_lvhv * 100 / trafo_df.ratedU_lv ** 2
            trafo_df['tabular_step'] = trafo_df['tabular_step'].astype(int)
            append_dict = dict({'id_characteristic': [], 'step': [], 'vkr_hv_percent': [], 'vkr_mv_percent': [],
                                'vkr_lv_percent': [], 'vk_hv_percent': [], 'vk_mv_percent': [], 'vk_lv_percent': []})

        def append_row(res_dict, id_c, row, cols):
            res_dict['id_characteristic'].append(id_c)
            res_dict['step'].append(row.tabular_step)
            for variable in ['vkr_percent', 'vk_percent', 'vk_hv_percent', 'vkr_hv_percent', 'vk_mv_percent',
                             'vkr_mv_percent', 'vk_lv_percent', 'vkr_lv_percent']:
                if variable in cols:
                    res_dict[variable].append(getattr(row, variable))

        id_characteristic = self.net['characteristic_temp']['id_characteristic'].max() + 1
        if math.isnan(id_characteristic):
            id_characteristic = 0
        for one_id, one_df in trafo_df.groupby(sc['pte_id']):
            # get next id_characteristic
            if len(append_dict['id_characteristic']) > 0:
                id_characteristic = max(append_dict['id_characteristic']) + 1
            # set the ID at the corresponding transformer
            trafo_df_origin.loc[trafo_df_origin['PowerTransformer'] == trafo_df_origin.loc[
                trafo_df_origin[sc['pte_id']] == one_id, 'PowerTransformer'].values[
                0], 'id_characteristic'] = id_characteristic
            # iterate over the rows and get the desired data
            for one_row in one_df.itertuples():
                # to add only selected characteristic data instead of all available data, disable the next line and
                # uncomment the rest
                append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # if one_row.tabular_step == one_row.highStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # elif one_row.tabular_step == one_row.lowStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # elif one_row.tabular_step == one_row.neutralStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
                # elif one_row.tabular_step == one_row.step and one_row.step != one_row.highStep \
                #         and one_row.step != one_row.lowStep and one_row.step != one_row.neutralStep:
                #     append_row(append_dict, id_characteristic, one_row, one_df.columns)
        self.net['characteristic_temp'] = pd.concat([self.net['characteristic_temp'], pd.DataFrame(append_dict)],
                                                    ignore_index=True, sort=False)
        self.net['characteristic_temp']['step'] = self.net['characteristic_temp']['step'].astype(int)

    def _create_characteristic_object(self, net, trafo_type: str, trafo_id: List, characteristic_df: pd.DataFrame):
        self.logger.info("Adding characteristic object for trafo_type: %s and trafo_id: %s" % (trafo_type, trafo_id))
        for variable in ['vkr_percent', 'vk_percent', 'vk_hv_percent', 'vkr_hv_percent', 'vk_mv_percent',
                         'vkr_mv_percent', 'vk_lv_percent', 'vkr_lv_percent']:
            if variable in characteristic_df.columns:
                pandapower.control.create_trafo_characteristics(net, trafo_type, trafo_id, variable,
                                                                [list(characteristic_df['step'].values)],
                                                                [list(characteristic_df[variable].values)])

    def _prepare_power_transformers_cim16(self) -> pd.DataFrame:
        eq_power_transformers = self.cim['eq']['PowerTransformer'][['rdfId', 'name', 'isPartOfGeneratorUnit']]
        eq_power_transformers[sc['o_cl']] = 'PowerTransformer'
        eq_power_transformer_ends = self.cim['eq']['PowerTransformerEnd'][
            ['rdfId', 'PowerTransformer', 'endNumber', 'Terminal', 'ratedS', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0',
             'phaseAngleClock', 'connectionKind', 'grounded']]

        # merge and append the tap changers
        eqssh_tap_changers = pd.merge(self.cim['eq']['RatioTapChanger'][[
            'rdfId', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement',
            'TapChangerControl']],
                                      self.cim['ssh']['RatioTapChanger'][['rdfId', 'step']], how='left', on='rdfId')
        eqssh_tap_changers[sc['tc']] = 'RatioTapChanger'
        eqssh_tap_changers[sc['tc_id']] = eqssh_tap_changers['rdfId'].copy()
        eqssh_tap_changers_linear = pd.merge(self.cim['eq']['PhaseTapChangerLinear'],
                                             self.cim['ssh']['PhaseTapChangerLinear'], how='left', on='rdfId')
        eqssh_tap_changers_linear['stepVoltageIncrement'] = .001
        eqssh_tap_changers_linear[sc['tc']] = 'PhaseTapChangerLinear'
        eqssh_tap_changers_linear[sc['tc_id']] = eqssh_tap_changers_linear['rdfId'].copy()
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, eqssh_tap_changers_linear], ignore_index=True, sort=False)
        eqssh_tap_changers_async = pd.merge(self.cim['eq']['PhaseTapChangerAsymmetrical'],
                                            self.cim['ssh']['PhaseTapChangerAsymmetrical'], how='left', on='rdfId')
        eqssh_tap_changers_async['stepVoltageIncrement'] = eqssh_tap_changers_async['voltageStepIncrement'][:]
        eqssh_tap_changers_async.drop(columns=['voltageStepIncrement'], inplace=True)
        eqssh_tap_changers_async[sc['tc']] = 'PhaseTapChangerAsymmetrical'
        eqssh_tap_changers_async[sc['tc_id']] = eqssh_tap_changers_async['rdfId'].copy()
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, eqssh_tap_changers_async], ignore_index=True, sort=False)
        eqssh_ratio_tap_changers_sync = pd.merge(self.cim['eq']['PhaseTapChangerSymmetrical'],
                                                 self.cim['ssh']['PhaseTapChangerSymmetrical'], how='left', on='rdfId')
        eqssh_ratio_tap_changers_sync['stepVoltageIncrement'] = eqssh_ratio_tap_changers_sync['voltageStepIncrement']
        eqssh_ratio_tap_changers_sync.drop(columns=['voltageStepIncrement'], inplace=True)
        eqssh_ratio_tap_changers_sync[sc['tc']] = 'PhaseTapChangerSymmetrical'
        eqssh_ratio_tap_changers_sync[sc['tc_id']] = eqssh_ratio_tap_changers_sync['rdfId'].copy()
        eqssh_tap_changers = \
            pd.concat([eqssh_tap_changers, eqssh_ratio_tap_changers_sync], ignore_index=True, sort=False)
        # convert the PhaseTapChangerTabular to one tap changer
        ptct = pd.merge(self.cim['eq']['PhaseTapChangerTabular'][['rdfId', 'TransformerEnd', 'PhaseTapChangerTable',
                                                                  'highStep', 'lowStep', 'neutralStep']],
                        self.cim['ssh']['PhaseTapChangerTabular'][['rdfId', 'step']], how='left', on='rdfId')
        ptct.rename(columns={'step': 'current_step'}, inplace=True)
        ptct = pd.merge(ptct, self.cim['eq']['PhaseTapChangerTablePoint'][
            ['PhaseTapChangerTable', 'step', 'angle', 'ratio']], how='left', on='PhaseTapChangerTable')
        for one_id, one_df in ptct.groupby('TransformerEnd'):
            drop_index = one_df[one_df['step'] != one_df['current_step']].index.values
            keep_index = one_df[one_df['step'] == one_df['current_step']].index.values
            if keep_index.size > 0:
                keep_index = keep_index[0]
            else:
                self.logger.warning("Ignoring PhaseTapChangerTabular with ID: %s. The current tap position is missing "
                                    "in the PhaseTapChangerTablePoints!" % one_id)
                ptct.drop(drop_index, inplace=True)
                continue
            current_step = one_df['current_step'].iloc[0]
            one_df.set_index('step', inplace=True)
            neutral_step = one_df['neutralStep'].iloc[0]
            ptct.drop(drop_index, inplace=True)
            # ptct.loc[keep_index, 'angle'] =
            # one_df.loc[current_step, 'angle'] / max(1, abs(current_step - neutral_step))
            ptct.loc[keep_index, 'angle'] = one_df.loc[current_step, 'angle']  # todo fix if pp supports them
            ptct.loc[keep_index, 'ratio'] = \
                (one_df.loc[current_step, 'ratio'] - 1) * 100 / max(1, abs(current_step - neutral_step))
        ptct.drop(columns=['rdfId', 'PhaseTapChangerTable', 'step'], inplace=True)
        ptct.rename(columns={'current_step': 'step'}, inplace=True)
        # ptct['stepPhaseShiftIncrement'] = ptct['angle'][:]  # todo fix if pp supports them
        ptct['stepVoltageIncrement'] = ptct['ratio'][:]
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, ptct], ignore_index=True, sort=False)
        del eqssh_tap_changers_linear, eqssh_tap_changers_async, eqssh_ratio_tap_changers_sync

        # remove duplicated TapChanger: A Transformer may have one RatioTapChanger and one PhaseTapChanger
        # self.logger.info("eqssh_tap_changers.index.size: %s" % eqssh_tap_changers.index.size)
        # self.logger.info("dups:")
        # for _, one_dup in eqssh_tap_changers[eqssh_tap_changers.duplicated('TransformerEnd', keep=False)].iterrows():
        #     self.logger.info(one_dup)  # no example for testing found
        eqssh_tap_changers.drop_duplicates(subset=['TransformerEnd'], inplace=True)
        # prepare the controllers
        eq_ssh_tap_controllers = self.merge_eq_ssh_profile('TapChangerControl')
        eq_ssh_tap_controllers = \
            eq_ssh_tap_controllers[['rdfId', 'Terminal', 'discrete', 'enabled', 'targetValue', 'targetDeadband']]
        eq_ssh_tap_controllers.rename(columns={'rdfId': 'TapChangerControl'}, inplace=True)
        # first merge with the VoltageLimits
        eq_vl = self.cim['eq']['VoltageLimit'][['OperationalLimitSet', 'OperationalLimitType', 'value']]
        eq_vl = pd.merge(eq_vl, self.cim['eq']['OperationalLimitType'][['rdfId', 'limitType']].rename(
            columns={'rdfId': 'OperationalLimitType'}), how='left', on='OperationalLimitType')
        eq_vl = pd.merge(eq_vl, self.cim['eq']['OperationalLimitSet'][['rdfId', 'Terminal']].rename(
            columns={'rdfId': 'OperationalLimitSet'}), how='left', on='OperationalLimitSet')
        eq_vl = eq_vl[['value', 'limitType', 'Terminal']]
        eq_vl_low = eq_vl.loc[eq_vl['limitType'] == 'lowVoltage'][['value', 'Terminal']].rename(
            columns={'value': 'c_vm_lower_pu'})
        eq_vl_up = eq_vl.loc[eq_vl['limitType'] == 'highVoltage'][['value', 'Terminal']].rename(
            columns={'value': 'c_vm_upper_pu'})
        eq_vl = pd.merge(eq_vl_low, eq_vl_up, how='left', on='Terminal')
        eq_ssh_tap_controllers = pd.merge(eq_ssh_tap_controllers, eq_vl, how='left', on='Terminal')
        eq_ssh_tap_controllers['c_Terminal'] = eq_ssh_tap_controllers['Terminal'][:]
        eq_ssh_tap_controllers.rename(columns={'Terminal': 'rdfId', 'enabled': 'c_in_service',
                                               'targetValue': 'c_vm_set_pu', 'targetDeadband': 'c_tol'}, inplace=True)
        # get the Terminal, ConnectivityNode and bus voltage
        eq_ssh_tap_controllers = \
            pd.merge(eq_ssh_tap_controllers, pd.concat([self.cim['eq']['Terminal'], self.cim['eq_bd']['Terminal']],
                                                       ignore_index=True, sort=False)[
                ['rdfId', 'ConnectivityNode']], how='left', on='rdfId')
        eq_ssh_tap_controllers.drop(columns=['rdfId'], inplace=True)
        eq_ssh_tap_controllers.rename(columns={'ConnectivityNode': sc['o_id']}, inplace=True)
        eq_ssh_tap_controllers = pd.merge(eq_ssh_tap_controllers,
                                          self.net['bus'].reset_index(level=0)[['index', sc['o_id'], 'vn_kv']],
                                          how='left', on=sc['o_id'])
        eq_ssh_tap_controllers.drop(columns=[sc['o_id']], inplace=True)
        eq_ssh_tap_controllers.rename(columns={'index': 'c_bus_id'}, inplace=True)
        eq_ssh_tap_controllers['c_vm_set_pu'] = eq_ssh_tap_controllers['c_vm_set_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_tol'] = eq_ssh_tap_controllers['c_tol'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_vm_lower_pu'] = \
            eq_ssh_tap_controllers['c_vm_lower_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_vm_upper_pu'] = \
            eq_ssh_tap_controllers['c_vm_upper_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers.drop(columns=['vn_kv'], inplace=True)

        eqssh_tap_changers = pd.merge(eqssh_tap_changers, eq_ssh_tap_controllers, how='left', on='TapChangerControl')
        eqssh_tap_changers.rename(columns={'TransformerEnd': sc['pte_id']}, inplace=True)

        eq_power_transformers.rename(columns={'rdfId': 'PowerTransformer'}, inplace=True)
        eq_power_transformer_ends.rename(columns={'rdfId': sc['pte_id']}, inplace=True)
        # add the PowerTransformerEnds
        eq_power_transformers = pd.merge(eq_power_transformers, eq_power_transformer_ends, how='left',
                                         on='PowerTransformer')
        # add the Terminal and bus indexes
        eq_power_transformers = pd.merge(eq_power_transformers, self.bus_merge.drop('rdfId', axis=1),
                                         how='left', left_on='Terminal', right_on='rdfId_Terminal')
        # add the TapChangers
        eq_power_transformers = pd.merge(eq_power_transformers, eqssh_tap_changers, how='left', on=sc['pte_id'])
        return eq_power_transformers

    def _prepare_trafos_cim16(self, power_trafo2w: pd.DataFrame) -> pd.DataFrame:
        power_trafo2w = power_trafo2w.sort_values(['PowerTransformer', 'endNumber']).reset_index()
        # precessing the transformer data
        # a list of transformer parameters which are used for each transformer winding
        copy_list = ['index_bus', 'Terminal', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0', 'neutralStep', 'lowStep',
                     'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step', 'connected',
                     'phaseAngleClock', 'connectionKind', sc['pte_id'], sc['tc'], sc['tc_id'], 'grounded', 'angle']
        for one_item in copy_list:
            # copy the columns which are required for each winding
            power_trafo2w[one_item + '_lv'] = power_trafo2w[one_item].copy()
            # cut the first element from the copied columns
            power_trafo2w[one_item + '_lv'] = power_trafo2w[one_item + '_lv'].iloc[1:].reset_index()[
                one_item + '_lv']
        del copy_list, one_item
        # detect on which winding a tap changer is attached
        power_trafo2w.loc[power_trafo2w['step_lv'].notna(), 'tap_side'] = 'lv'
        power_trafo2w.loc[power_trafo2w['step'].notna(), 'tap_side'] = 'hv'
        fillna_list = ['neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step',
                       sc['tc'], sc['tc_id']]
        for one_item in fillna_list:
            power_trafo2w[one_item].fillna(power_trafo2w[one_item + '_lv'], inplace=True)
        del fillna_list, one_item
        # just keep one transformer
        power_trafo2w.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)

        power_trafo2w['pfe_kw'] = (power_trafo2w.g * power_trafo2w.ratedU ** 2 +
                                   power_trafo2w.g_lv * power_trafo2w.ratedU_lv ** 2) * 1000
        power_trafo2w['vkr_percent'] = \
            abs(power_trafo2w.r) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU ** 2 + \
            abs(power_trafo2w.r_lv) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU_lv ** 2
        power_trafo2w['x_lv_sign'] = np.sign(power_trafo2w['x_lv'])
        power_trafo2w.loc[power_trafo2w['x_lv_sign'] == 0, 'x_lv_sign'] = 1
        power_trafo2w['x_sign'] = np.sign(power_trafo2w['x'])
        power_trafo2w.loc[power_trafo2w['x_sign'] == 0, 'x_sign'] = 1
        power_trafo2w['vk_percent'] = \
            (power_trafo2w.r ** 2 + power_trafo2w.x ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU ** 2) + \
            (power_trafo2w.r_lv ** 2 + power_trafo2w.x_lv ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU_lv ** 2)
        power_trafo2w['vk_percent'] = power_trafo2w['x_lv_sign'] * power_trafo2w['vk_percent']
        power_trafo2w['vk_percent'] = power_trafo2w['x_sign'] * power_trafo2w['vk_percent']
        power_trafo2w['i0_percent'] = \
            (((power_trafo2w.b * power_trafo2w.ratedU ** 2) ** 2 +
              (power_trafo2w.g * power_trafo2w.ratedU ** 2) ** 2) ** .5 +
             ((power_trafo2w.b_lv * power_trafo2w.ratedU_lv ** 2) ** 2 +
              (power_trafo2w.g_lv * power_trafo2w.ratedU_lv ** 2) ** 2) ** .5) * 100 / power_trafo2w.ratedS
        power_trafo2w['vkr0_percent'] = \
            abs(power_trafo2w.r0) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU ** 2 + \
            abs(power_trafo2w.r0_lv) * power_trafo2w.ratedS * 100 / power_trafo2w.ratedU_lv ** 2
        power_trafo2w['x0_lv_sign'] = np.sign(power_trafo2w['x0_lv'])
        power_trafo2w.loc[power_trafo2w['x0_lv_sign'] == 0, 'x0_lv_sign'] = 1
        power_trafo2w['x0_sign'] = np.sign(power_trafo2w['x0'])
        power_trafo2w.loc[power_trafo2w['x0_sign'] == 0, 'x0_sign'] = 1
        power_trafo2w['vk0_percent'] = \
            (power_trafo2w.r0 ** 2 + power_trafo2w.x0 ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU ** 2) + \
            (power_trafo2w.r0_lv ** 2 + power_trafo2w.x0_lv ** 2) ** 0.5 * \
            (power_trafo2w.ratedS * 1e3) / (10. * power_trafo2w.ratedU_lv ** 2)
        power_trafo2w['vk0_percent'] = power_trafo2w['x0_lv_sign'] * power_trafo2w['vk0_percent']
        power_trafo2w['vk0_percent'] = power_trafo2w['x0_sign'] * power_trafo2w['vk0_percent']
        power_trafo2w['std_type'] = None
        power_trafo2w['df'] = 1.
        # todo remove if pp supports phase shifter
        if power_trafo2w.loc[power_trafo2w['angle'].notna()].index.size > 0:
            self.logger.warning("Modifying angle from 2W transformers. This kind of angle regulation is currently not "
                                "supported by pandapower! Affected transformers: \n%s" %
                                power_trafo2w.loc[power_trafo2w['angle'].notna()])
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="Modifying angle from 2W transformers. This kind of angle regulation is currently not "
                        "supported by pandapower! Affected transformers: \n%s" %
                        power_trafo2w.loc[power_trafo2w['angle'].notna()]))
        power_trafo2w['phaseAngleClock_temp'] = power_trafo2w['phaseAngleClock'].copy()
        power_trafo2w['phaseAngleClock'] = power_trafo2w['angle'] / 30
        power_trafo2w['phaseAngleClock'].fillna(power_trafo2w['angle_lv'] * -1 / 30, inplace=True)
        power_trafo2w['phaseAngleClock'].fillna(power_trafo2w['phaseAngleClock_temp'], inplace=True)
        power_trafo2w['phaseAngleClock_lv'].fillna(0, inplace=True)
        power_trafo2w['shift_degree'] = power_trafo2w['phaseAngleClock'].astype(float).fillna(
            power_trafo2w['phaseAngleClock_lv'].astype(float)) * 30
        power_trafo2w['parallel'] = 1
        power_trafo2w['tap_phase_shifter'] = False
        power_trafo2w['in_service'] = power_trafo2w.connected & power_trafo2w.connected_lv
        power_trafo2w['connectionKind'].fillna('', inplace=True)
        power_trafo2w['connectionKind_lv'].fillna('', inplace=True)
        power_trafo2w.loc[~power_trafo2w['grounded'].astype('bool'), 'connectionKind'] = \
            power_trafo2w.loc[~power_trafo2w['grounded'].astype('bool'), 'connectionKind'].str.replace('n', '')
        power_trafo2w.loc[~power_trafo2w['grounded_lv'].astype('bool'), 'connectionKind_lv'] = \
            power_trafo2w.loc[~power_trafo2w['grounded_lv'].astype('bool'), 'connectionKind_lv'].str.replace('n', '')
        power_trafo2w['vector_group'] = power_trafo2w.connectionKind + power_trafo2w.connectionKind_lv
        power_trafo2w.loc[power_trafo2w['vector_group'] == '', 'vector_group'] = None
        power_trafo2w.rename(columns={
            'PowerTransformer': sc['o_id'], 'Terminal': sc['t_hv'], 'Terminal_lv': sc['t_lv'],
            sc['pte_id']: sc['pte_id_hv'], sc['pte_id']+'_lv': sc['pte_id_lv'], 'index_bus': 'hv_bus',
            'index_bus_lv': 'lv_bus', 'neutralStep': 'tap_neutral', 'lowStep': 'tap_min', 'highStep': 'tap_max',
            'step': 'tap_pos', 'stepVoltageIncrement': 'tap_step_percent', 'stepPhaseShiftIncrement': 'tap_step_degree',
            'isPartOfGeneratorUnit': 'power_station_unit', 'ratedU': 'vn_hv_kv', 'ratedU_lv': 'vn_lv_kv',
            'ratedS': 'sn_mva', 'xground': 'xn_ohm', 'grounded': 'oltc'}, inplace=True)
        return power_trafo2w

    def _prepare_trafo3w_cim16(self, power_trafo3w: pd.DataFrame) -> pd.DataFrame:
        power_trafo3w = power_trafo3w.sort_values(['PowerTransformer', 'endNumber']).reset_index()
        # precessing the transformer data
        # a list of transformer parameters which are used for each transformer winding
        copy_list = ['index_bus', 'Terminal', 'ratedS', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0', 'neutralStep',
                     'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step', 'connected',
                     'angle', 'phaseAngleClock', 'connectionKind', 'grounded', sc['pte_id'], sc['tc'], sc['tc_id']]
        for one_item in copy_list:
            # copy the columns which are required for each winding
            power_trafo3w[one_item + '_mv'] = power_trafo3w[one_item].copy()
            power_trafo3w[one_item + '_lv'] = power_trafo3w[one_item].copy()
            # cut the first (or first two) element(s) from the copied columns
            power_trafo3w[one_item + '_mv'] = power_trafo3w[one_item + '_mv'].iloc[1:].reset_index()[
                one_item + '_mv']
            power_trafo3w[one_item + '_lv'] = power_trafo3w[one_item + '_lv'].iloc[2:].reset_index()[
                one_item + '_lv']
        del copy_list, one_item

        # detect on which winding a tap changer is attached
        power_trafo3w.loc[power_trafo3w['step_lv'].notna(), 'tap_side'] = 'lv'
        power_trafo3w.loc[power_trafo3w['step_mv'].notna(), 'tap_side'] = 'mv'
        power_trafo3w.loc[power_trafo3w['step'].notna(), 'tap_side'] = 'hv'
        fillna_list = ['neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step',
                       sc['tc'], sc['tc_id']]
        for one_item in fillna_list:
            power_trafo3w[one_item].fillna(power_trafo3w[one_item + '_mv'], inplace=True)
            power_trafo3w[one_item].fillna(power_trafo3w[one_item + '_lv'], inplace=True)
        del fillna_list, one_item
        # just keep one transformer
        power_trafo3w.drop_duplicates(subset=['PowerTransformer'], keep='first', inplace=True)

        power_trafo3w['min_s_hvmv'] = power_trafo3w[["ratedS", "ratedS_mv"]].min(axis=1)
        power_trafo3w['min_s_mvlv'] = power_trafo3w[["ratedS_mv", "ratedS_lv"]].min(axis=1)
        power_trafo3w['min_s_lvhv'] = power_trafo3w[["ratedS_lv", "ratedS"]].min(axis=1)
        power_trafo3w['pfe_kw'] = \
            (power_trafo3w.g * power_trafo3w.ratedU ** 2 + power_trafo3w.g_mv * power_trafo3w.ratedU_mv ** 2
             + power_trafo3w.g_lv * power_trafo3w.ratedU_lv ** 2) * 1000
        power_trafo3w['vkr_hv_percent'] = \
            (power_trafo3w.r + power_trafo3w.r_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vkr_mv_percent'] = \
            (power_trafo3w.r_mv + power_trafo3w.r_lv * (power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vkr_lv_percent'] = \
            (power_trafo3w.r_lv + power_trafo3w.r * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['vk_hv_percent'] = \
            ((power_trafo3w.r + power_trafo3w.r_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2 +
             (power_trafo3w.x + power_trafo3w.x_mv * (
                     power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vk_mv_percent'] = \
            ((power_trafo3w.r_mv + power_trafo3w.r_lv * (
                    power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2 +
             (power_trafo3w.x_mv + power_trafo3w.x_lv * (
                     power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vk_lv_percent'] = \
            ((power_trafo3w.r_lv + power_trafo3w.r * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2 +
             (power_trafo3w.x_lv + power_trafo3w.x * (
                     power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['i0_percent'] = \
            (((power_trafo3w.b * power_trafo3w.ratedU ** 2) ** 2 +
              (power_trafo3w.g * power_trafo3w.ratedU ** 2) ** 2) ** .5 +
             ((power_trafo3w.b_mv * power_trafo3w.ratedU_mv ** 2) ** 2 +
              (power_trafo3w.g_mv * power_trafo3w.ratedU_mv ** 2) ** 2) ** .5 +
             ((power_trafo3w.b_lv * power_trafo3w.ratedU_lv ** 2) ** 2 +
              (power_trafo3w.g_lv * power_trafo3w.ratedU_lv ** 2) ** 2) ** .5) * 100 / power_trafo3w.ratedS
        power_trafo3w['vkr0_hv_percent'] = \
            (power_trafo3w.r0 + power_trafo3w.r0_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vkr0_mv_percent'] = \
            (power_trafo3w.r0_mv + power_trafo3w.r0_lv * (power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vkr0_lv_percent'] = \
            (power_trafo3w.r0_lv + power_trafo3w.r0 * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['vk0_hv_percent'] = \
            ((power_trafo3w.r0 + power_trafo3w.r0_mv * (power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2 +
             (power_trafo3w.x0 + power_trafo3w.x0_mv * (
                     power_trafo3w.ratedU / power_trafo3w.ratedU_mv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_hvmv * 100 / power_trafo3w.ratedU ** 2
        power_trafo3w['vk0_mv_percent'] = \
            ((power_trafo3w.r0_mv + power_trafo3w.r0_lv * (
                    power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2 +
             (power_trafo3w.x0_mv + power_trafo3w.x0_lv * (
                     power_trafo3w.ratedU_mv / power_trafo3w.ratedU_lv) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_mvlv * 100 / power_trafo3w.ratedU_mv ** 2
        power_trafo3w['vk0_lv_percent'] = \
            ((power_trafo3w.r0_lv + power_trafo3w.r0 * (power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2 +
             (power_trafo3w.x0_lv + power_trafo3w.x0 * (
                     power_trafo3w.ratedU_lv / power_trafo3w.ratedU) ** 2) ** 2) ** 0.5 * \
            power_trafo3w.min_s_lvhv * 100 / power_trafo3w.ratedU_lv ** 2
        power_trafo3w['std_type'] = None
        # todo remove if pp supports phase shifter
        if power_trafo3w.loc[power_trafo3w['angle_mv'].notna()].index.size > 0:
            self.logger.warning("Modifying angle from 3W transformers. This kind of angle regulation is currently not "
                                "supported by pandapower! Affected transformers: \n%s" %
                                power_trafo3w.loc[power_trafo3w['angle_mv'].notna()])
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="Modifying angle from 3W transformers. This kind of angle regulation is currently not "
                        "supported by pandapower! Affected transformers: \n%s" %
                        power_trafo3w.loc[power_trafo3w['angle_mv'].notna()]))
        power_trafo3w['phaseAngleClock_temp'] = power_trafo3w['phaseAngleClock_mv'].copy()
        power_trafo3w['phaseAngleClock_mv'] = power_trafo3w['angle_mv'] * -1 / 30
        power_trafo3w['phaseAngleClock_mv'].fillna(power_trafo3w['phaseAngleClock_temp'], inplace=True)
        power_trafo3w['phaseAngleClock_mv'].fillna(0, inplace=True)
        power_trafo3w['phaseAngleClock_lv'].fillna(0, inplace=True)
        power_trafo3w['shift_mv_degree'] = power_trafo3w['phaseAngleClock_mv'] * 30
        power_trafo3w['shift_lv_degree'] = power_trafo3w['phaseAngleClock_mv'] * 30
        power_trafo3w['tap_at_star_point'] = False
        power_trafo3w['in_service'] = power_trafo3w.connected & power_trafo3w.connected_mv & power_trafo3w.connected_lv
        power_trafo3w['connectionKind'].fillna('', inplace=True)
        power_trafo3w['connectionKind_mv'].fillna('', inplace=True)
        power_trafo3w['connectionKind_lv'].fillna('', inplace=True)

        power_trafo3w.loc[~power_trafo3w['grounded'].astype('bool'), 'connectionKind'] = \
            power_trafo3w.loc[~power_trafo3w['grounded'].astype('bool'), 'connectionKind'].str.replace('n', '')
        power_trafo3w.loc[~power_trafo3w['grounded_mv'].astype('bool'), 'connectionKind_mv'] = \
            power_trafo3w.loc[~power_trafo3w['grounded_mv'].astype('bool'), 'connectionKind_mv'].str.replace('n', '')
        power_trafo3w.loc[~power_trafo3w['grounded_lv'].astype('bool'), 'connectionKind_lv'] = \
            power_trafo3w.loc[~power_trafo3w['grounded_lv'].astype('bool'), 'connectionKind_lv'].str.replace('n', '')
        power_trafo3w['vector_group'] = \
            power_trafo3w.connectionKind + power_trafo3w.connectionKind_mv + power_trafo3w.connectionKind_lv
        power_trafo3w.loc[power_trafo3w['vector_group'] == '', 'vector_group'] = None
        power_trafo3w.rename(columns={
            'PowerTransformer': sc['o_id'], 'Terminal': sc['t_hv'], 'Terminal_mv': sc['t_mv'],
            'Terminal_lv': sc['t_lv'], sc['pte_id']: sc['pte_id_hv'], sc['pte_id'] + '_mv': sc['pte_id_mv'],
            sc['pte_id'] + '_lv': sc['pte_id_lv'], 'index_bus': 'hv_bus', 'index_bus_mv': 'mv_bus',
            'index_bus_lv': 'lv_bus', 'neutralStep': 'tap_neutral', 'lowStep': 'tap_min', 'highStep': 'tap_max',
            'step': 'tap_pos', 'stepVoltageIncrement': 'tap_step_percent', 'stepPhaseShiftIncrement': 'tap_step_degree',
            'isPartOfGeneratorUnit': 'power_station_unit', 'ratedU': 'vn_hv_kv', 'ratedU_mv': 'vn_mv_kv',
            'ratedU_lv': 'vn_lv_kv', 'ratedS': 'sn_hv_mva', 'ratedS_mv': 'sn_mv_mva', 'ratedS_lv': 'sn_lv_mva'},
            inplace=True)
        return power_trafo3w

    def _create_tap_controller(self, input_df: pd.DataFrame, trafo_type: str):
        if not self.kwargs.get('create_tap_controller', True):
            self.logger.info("Skip creating transformer tap changer controller for transformer type %s." % trafo_type)
            return
        for row_index, row in input_df.loc[input_df['TapChangerControl'].notna()].iterrows():
            trafo_id = self.net[trafo_type].loc[self.net[trafo_type][sc['o_id']] == row[sc['o_id']]].index.values[0]
            trafotype = '2W' if trafo_type == 'trafo' else '3W'
            # get the controlled bus (side), assume "lv" as default
            side = 'lv'
            if sc['t_hv'] in self.net[trafo_type].columns and \
                    row['c_Terminal'] in self.net[trafo_type][sc['t_hv']].values:
                side = 'hv'
            if sc['t_mv'] in self.net[trafo_type].columns and \
                    row['c_Terminal'] in self.net[trafo_type][sc['t_mv']].values:
                side = 'mv'
            if row['discrete']:
                self.logger.info("Creating DiscreteTapControl for transformer %s." % row[sc['o_id']])
                DiscreteTapControl(self.net, trafotype=trafotype, tid=trafo_id, side=side,
                                   tol=row['c_tol'], in_service=row['c_in_service'],
                                   vm_lower_pu=row['c_vm_lower_pu'], vm_upper_pu=row['c_vm_upper_pu'])
            else:
                self.logger.info("Creating ContinuousTapControl for transformer %s." % row[sc['o_id']])
                ContinuousTapControl(self.net, trafotype=trafotype, tid=trafo_id, side=side,
                                     tol=row['c_tol'], in_service=row['c_in_service'], vm_set_pu=row['c_vm_set_pu'])

    def copy_to_pp(self, pp_type: str, input_df: pd.DataFrame):
        self.logger.debug("Copy %s datasets to pandapower network with type %s" % (input_df.index.size, pp_type))
        if pp_type not in self.net.keys():
            self.logger.warning("Missing pandapower type %s in the pandapower network!" % pp_type)
            self.report_container.add_log(Report(
                level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                message="Missing pandapower type %s in the pandapower network!" % pp_type))
            return
        start_index_pp_net = self.net[pp_type].index.size
        self.net[pp_type] = pd.concat([self.net[pp_type], pd.DataFrame(None, index=[list(range(input_df.index.size))])],
                                      ignore_index=True, sort=False)
        for one_attr in self.net[pp_type].columns:
            if one_attr in input_df.columns:
                self.net[pp_type][one_attr][start_index_pp_net:] = input_df[one_attr][:]

    def _add_geo_coordinates_from_gl_cim16(self):
        self.logger.info("Creating the geo coordinates from CGMES GeographicalLocation.")
        time_start = time.time()
        gl_data = pd.merge(self.cim['gl']['PositionPoint'][['Location', 'xPosition', 'yPosition', 'sequenceNumber']],
                           self.cim['gl']['Location'][['rdfId', 'PowerSystemResources']], how='left',
                           left_on='Location', right_on='rdfId')
        gl_data.drop(columns=['Location', 'rdfId'], inplace=True)
        bus_geo = gl_data.rename(columns={'PowerSystemResources': 'Substation'})
        cn = self.cim['eq']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]
        cn = pd.concat([cn, self.cim['eq_bd']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cim['tp']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cim['tp_bd']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn.rename(columns={'rdfId': sc['o_id'], 'ConnectivityNodeContainer': 'rdfId'}, inplace=True)
        cn = pd.merge(cn, self.cim['eq']['VoltageLevel'][['rdfId', 'Substation']], how='left', on='rdfId')
        cn.drop(columns=['rdfId'], inplace=True)
        buses = self.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        buses = pd.merge(buses, cn, how='left', on=sc['o_id'])
        bus_geo = pd.merge(bus_geo, buses, how='inner', on='Substation')
        bus_geo.drop(columns=['Substation'], inplace=True)
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for group_name, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo.sort_values(by='index', inplace=True)
        start_index_pp_net = self.net.bus_geodata.index.size
        self.net.bus_geodata = pd.concat([self.net.bus_geodata, pd.DataFrame(None, index=bus_geo['index'].values)],
                                         ignore_index=False, sort=False)
        self.net.bus_geodata.x[start_index_pp_net:] = bus_geo.xPosition[:]
        self.net.bus_geodata.y[start_index_pp_net:] = bus_geo.yPosition[:]
        self.net.bus_geodata.coords[start_index_pp_net:] = bus_geo.coords[:]
        # reduce to max two coordinates for buses (see pandapower documentation for details)
        self.net.bus_geodata['coords_length'] = self.net.bus_geodata['coords'].apply(len)
        self.net.bus_geodata.loc[self.net.bus_geodata['coords_length'] == 1, 'coords'] = np.nan
        self.net.bus_geodata['coords'] = self.net.bus_geodata.apply(
            lambda row: [row['coords'][0], row['coords'][-1]] if row['coords_length'] > 2 else row['coords'], axis=1)
        if 'coords_length' in self.net.bus_geodata.columns:
            self.net.bus_geodata.drop(columns=['coords_length'], inplace=True)

        # the geo coordinates for the lines
        lines = self.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = gl_data.rename(columns={'PowerSystemResources': sc['o_id']})
        line_geo = pd.merge(line_geo, lines, how='inner', on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for group_name, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo.sort_values(by='index', inplace=True)
        # now add the line coordinates
        start_index_pp_net = self.net.line_geodata.index.size
        self.net.line_geodata = pd.concat([self.net.line_geodata, pd.DataFrame(None, index=line_geo['index'].values)],
                                          ignore_index=False, sort=False)
        self.net.line_geodata.coords[start_index_pp_net:] = line_geo.coords[:]

        # now create geo coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(gl_data.rename(columns={'PowerSystemResources': sc['o_id']}),
                                  one_ele_df, how='inner', on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for group_name, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
            # now add the coordinates
            self.net[one_ele]['coords'] = self.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the GL coordinates, needed time: %ss" % (time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the GL coordinates, needed time: %ss" % (time.time() - time_start)))

    def _add_coordinates_from_dl_cim16(self, diagram_name: str = None):
        self.logger.info("Creating the coordinates from CGMES DiagramLayout.")
        time_start = time.time()
        # choose a diagram if it is not given (the first one ascending)
        if diagram_name is None:
            diagram_name = self.cim['dl']['Diagram'].sort_values(by='name')['name'].values[0]
        self.logger.debug("Choosing the geo coordinates from diagram %s" % diagram_name)
        if diagram_name != 'all':
            # reduce the source data to the chosen diagram only
            diagram_rdf_id = \
                self.cim['dl']['Diagram']['rdfId'][self.cim['dl']['Diagram']['name'] == diagram_name].values[0]
            dl_do = self.cim['dl']['DiagramObject'][self.cim['dl']['DiagramObject']['Diagram'] == diagram_rdf_id]
            dl_do.rename(columns={'rdfId': 'DiagramObject'}, inplace=True)
        else:
            dl_do = self.cim['dl']['DiagramObject'].copy()
            dl_do.rename(columns={'rdfId': 'DiagramObject'}, inplace=True)
        dl_data = pd.merge(dl_do, self.cim['dl']['DiagramObjectPoint'], how='left', on='DiagramObject')
        dl_data.drop(columns=['rdfId', 'Diagram', 'DiagramObject'], inplace=True)
        # the coordinates for the buses
        buses = self.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        bus_geo = pd.merge(dl_data, buses, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for group_name, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo.sort_values(by='index', inplace=True)
        start_index_pp_net = self.net.bus_geodata.index.size
        self.net.bus_geodata = pd.concat([self.net.bus_geodata, pd.DataFrame(None, index=bus_geo['index'].values)],
                                         ignore_index=False, sort=False)
        self.net.bus_geodata.x[start_index_pp_net:] = bus_geo.xPosition[:]
        self.net.bus_geodata.y[start_index_pp_net:] = bus_geo.yPosition[:]
        self.net.bus_geodata.coords[start_index_pp_net:] = bus_geo.coords[:]
        # reduce to max two coordinates for buses (see pandapower documentation for details)
        self.net.bus_geodata['coords_length'] = self.net.bus_geodata['coords'].apply(len)
        self.net.bus_geodata.loc[self.net.bus_geodata['coords_length'] == 1, 'coords'] = np.nan
        self.net.bus_geodata['coords'] = self.net.bus_geodata.apply(
            lambda row: [row['coords'][0], row['coords'][-1]] if row['coords_length'] > 2 else row['coords'], axis=1)
        if 'coords_length' in self.net.bus_geodata.columns:
            self.net.bus_geodata.drop(columns=['coords_length'], inplace=True)

        # the coordinates for the lines
        lines = self.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = pd.merge(dl_data, lines, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for group_name, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo.sort_values(by='index', inplace=True)
        # now add the line coordinates
        # if there are no bus geodata in the GL profile the line geodata from DL has higher priority
        if self.net.line_geodata.index.size > 0 and line_geo.index.size > 0:
            self.net.line_geodata = self.net.line_geodata[0:0]
        self.net.line_geodata = pd.concat([self.net.line_geodata, line_geo[['coords', 'index']].set_index('index')],
                                          ignore_index=False, sort=False)

        # now create coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(dl_data, one_ele_df, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for group_name, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
            # now add the coordinates
            self.net[one_ele]['coords'] = self.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the DL coordinates, needed time: %ss" % (time.time() - time_start))
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the DL coordinates, needed time: %ss" % (time.time() - time_start)))

    # noinspection PyShadowingNames
    def convert_to_pp(self, convert_line_to_switch: bool = False, line_r_limit: float = 0.1,
                      line_x_limit: float = 0.1, **kwargs) \
            -> pandapower.auxiliary.pandapowerNet:
        """
        Build the pandapower net.

        :param convert_line_to_switch: Set this parameter to True to enable line -> switch conversion. All lines with a
        resistance lower or equal than line_r_limit or a reactance lower or equal than line_x_limit will become a
        switch. Optional, default: False
        :param line_r_limit: The limit from resistance. Optional, default: 0.1
        :param line_x_limit: The limit from reactance. Optional, default: 0.1
        :return: The pandapower net.
        """
        self.logger.info("Start building the pandapower net.")
        self.report_container.add_log(Report(level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
                                             message="Start building the pandapower net."))

        # create the empty pandapower net and add the additional columns
        self.net = cim_tools.extend_pp_net_cim(self.net, override=False)

        if 'sn_mva' in kwargs.keys():
            self.net['sn_mva'] = kwargs.get('sn_mva')

        # add the CIM IDs to the pandapower network
        for one_prf, one_profile_dict in self.cim.items():
            if 'FullModel' in one_profile_dict.keys() and one_profile_dict['FullModel'].index.size > 0:
                self.net['CGMES'][one_prf] = one_profile_dict['FullModel'].set_index('rdfId').to_dict(orient='index')
        # store the BaseVoltage IDs
        self.net['CGMES']['BaseVoltage'] = \
            pd.concat([self.cim['eq']['BaseVoltage'], self.cim['eq_bd']['BaseVoltage']],
                      sort=False, ignore_index=True)[['rdfId', 'nominalVoltage']]

        # --------- convert busses ---------
        from .converter_classes.connectivitynodes import connectivityNodesCim16
        connectivityNodesCim16.ConnectivityNodesCim16(cimConverter=self).convert_connectivity_nodes_cim16()
        # self._convert_connectivity_nodes_cim16()
        # --------- convert external networks ---------
        from .converter_classes.externalnetworks import externalNetworkInjectionsCim16
        externalNetworkInjectionsCim16.ExternalNetworkInjectionsCim16(cimConverter=self).convert_external_network_injections_cim16()
        # self._convert_external_network_injections_cim16()
        # --------- convert lines ---------
        from .converter_classes.lines import acLineSegmentsCim16
        acLineSegmentsCim16.AcLineSegmentsCim16(cimConverter=self).convert_ac_line_segments_cim16(convert_line_to_switch, line_r_limit, line_x_limit)
        # self._convert_ac_line_segments_cim16(convert_line_to_switch, line_r_limit, line_x_limit)
        from .converter_classes.lines import dcLineSegmentsCim16
        dcLineSegmentsCim16.DcLineSegmentsCim16(cimConverter=self).convert_dc_line_segments_cim16()
        # self._convert_dc_line_segments_cim16()
        # --------- convert switches ---------
        from .converter_classes.switches import switchesCim16
        switchesCim16.SwitchesCim16(cimConverter=self).convert_switches_cim16()
        # self._convert_switches_cim16()
        # --------- convert loads ---------
        from .converter_classes.loads import energyConcumersCim16
        energyConcumersCim16.EnergyConsumersCim16(cimConverter=self).convert_energy_consumers_cim16()
        # self._convert_energy_consumers_cim16()
        from.converter_classes.loads import conformLoadsCim16
        conformLoadsCim16.ConformLoadsCim16(cimConverter=self).convert_conform_loads_cim16()
        # self._convert_conform_loads_cim16()
        from .converter_classes.loads import nonConformLoadsCim16
        nonConformLoadsCim16.NonConformLoadsCim16(cimConverter=self).convert_non_conform_loads_cim16()
        # self._convert_non_conform_loads_cim16()
        from .converter_classes.loads import stationSuppliesCim16
        stationSuppliesCim16.StationSuppliesCim16(cimConverter=self).convert_station_supplies_cim16()
        # self._convert_station_supplies_cim16()
        # --------- convert generators ---------
        from .converter_classes.generators import synchronousMachinesCim16
        synchronousMachinesCim16.SynchronousMachinesCim16(cimConverter=self).convert_synchronous_machines_cim16()
        # self._convert_synchronous_machines_cim16()
        from .converter_classes.generators import asynchronousMachinesCim16
        asynchronousMachinesCim16.AsynchronousMachinesCim16(cimConverter=self).convert_asynchronous_machines_cim16()
        # self._convert_asynchronous_machines_cim16()
        from .converter_classes.generators import energySourcesCim16
        energySourcesCim16.EnergySourceCim16(cimConverter=self).convert_energy_sources_cim16()
        # self._convert_energy_sources_cim16()
        # --------- convert shunt elements ---------
        from .converter_classes.shunts import linearShuntCompensatorCim16
        linearShuntCompensatorCim16.LinearShuntCompensatorCim16(cimConverter=self).convert_linear_shunt_compensator_cim16()
        # self._convert_linear_shunt_compensator_cim16()
        from .converter_classes.shunts import nonLinearShuntCompensatorCim16
        nonLinearShuntCompensatorCim16.NonLinearShuntCompensatorCim16(cimConverter=self).convert_nonlinear_shunt_compensator_cim16()
        # self._convert_nonlinear_shunt_compensator_cim16()
        from .converter_classes.shunts import staticVarCompensatorCim16
        staticVarCompensatorCim16.StaticVarCompensatorCim16(cimConverter=self).convert_static_var_compensator_cim16()
        # self._convert_static_var_compensator_cim16()
        # --------- convert impedance elements ---------
        from .converter_classes.impedance import equivalentBranchesCim16
        equivalentBranchesCim16.EquivalentBranchesCim16(cimConverter=self).convert_equivalent_branches_cim16()
        # self._convert_equivalent_branches_cim16()
        from .converter_classes.impedance import seriesCompensatorsCim16
        seriesCompensatorsCim16.SeriesCompensatorsCim16(cimConverter=self).convert_series_compensators_cim16()
        # self._convert_series_compensators_cim16()
        # --------- convert extended ward and ward elements ---------
        self._convert_equivalent_injections_cim16()
        # --------- convert transformers ---------
        self._convert_power_transformers_cim16()

        # create the geo coordinates
        gl_or_dl = str(self.kwargs.get('use_GL_or_DL_profile', 'both')).lower()
        if gl_or_dl == 'gl':
            use_gl_profile = True
            use_dl_profile = False
        elif gl_or_dl == 'dl':
            use_gl_profile = False
            use_dl_profile = True
        else:
            use_gl_profile = True
            use_dl_profile = True
        if self.cim['gl']['Location'].index.size > 0 and self.cim['gl']['PositionPoint'].index.size > 0 and \
                use_gl_profile:
            try:
                self._add_geo_coordinates_from_gl_cim16()
            except Exception as e:
                self.logger.warning("Creating the geo coordinates failed, returning the net without geo coordinates!")
                self.logger.exception(e)
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="Creating the geo coordinates failed, returning the net without geo coordinates!"))
                self.report_container.add_log(Report(
                    level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                    message=traceback.format_exc()))
                self.net.bus_geodata = self.net.bus_geodata[0:0]
                self.net.line_geodata = self.net.line_geodata[0:0]
        if self.cim['dl']['Diagram'].index.size > 0 and self.cim['dl']['DiagramObject'].index.size > 0 and \
                self.cim['dl']['DiagramObjectPoint'].index.size > 0 and self.net.bus_geodata.index.size == 0 and \
                use_dl_profile:
            try:
                self._add_coordinates_from_dl_cim16(diagram_name=kwargs.get('diagram_name', None))
            except Exception as e:
                self.logger.warning("Creating the coordinates failed, returning the net without coordinates!")
                self.logger.exception(e)
                self.report_container.add_log(Report(
                    level=LogLevel.WARNING, code=ReportCode.WARNING_CONVERTING,
                    message="Creating the coordinates failed, returning the net without coordinates!"))
                self.report_container.add_log(Report(level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                                                     message=traceback.format_exc()))
                self.net.bus_geodata = self.net.bus_geodata[0:0]
                self.net.line_geodata = self.net.line_geodata[0:0]
        self.net = pp_tools.set_pp_col_types(net=self.net)

        # create transformer tap controller
        if self.power_trafo2w.index.size > 0:
            # create transformer tap controller
            self._create_tap_controller(self.power_trafo2w, 'trafo')
            # create the characteristic objects for transformers
            characteristic_df_temp = \
                self.net['characteristic_temp'][['id_characteristic', 'step', 'vk_percent', 'vkr_percent']]
            for trafo_id, trafo_row in self.net.trafo.dropna(subset=['id_characteristic']).iterrows():
                characteristic_df = characteristic_df_temp.loc[
                    characteristic_df_temp['id_characteristic'] == trafo_row['id_characteristic']]
                self._create_characteristic_object(net=self.net, trafo_type='trafo', trafo_id=[trafo_id],
                                                   characteristic_df=characteristic_df)
        if self.power_trafo3w.index.size > 0:
            # create transformer tap controller
            self._create_tap_controller(self.power_trafo3w, 'trafo3w')
            # create the characteristic objects for transformers
            characteristic_df_temp = \
                self.net['characteristic_temp'][['id_characteristic', 'step', 'vkr_hv_percent', 'vkr_mv_percent',
                                                 'vkr_lv_percent', 'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent']]
            for trafo_id, trafo_row in self.net.trafo3w.dropna(subset=['id_characteristic']).iterrows():
                characteristic_df = characteristic_df_temp.loc[
                    characteristic_df_temp['id_characteristic'] == trafo_row['id_characteristic']]
                self._create_characteristic_object(net=self.net, trafo_type='trafo3w', trafo_id=[trafo_id],
                                                   characteristic_df=characteristic_df)

        self.logger.info("Running a power flow.")
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO, message="Running a power flow."))
        try:
            pp.runpp(self.net)
        except Exception as e:
            self.logger.error("Failed running a powerflow.")
            self.logger.exception(e)
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR, message="Failed running a powerflow."))
            self.report_container.add_log(Report(level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION,
                                                 message=traceback.format_exc()))
        else:
            self.logger.info("Power flow solved normal.")
            self.report_container.add_log(Report(
                level=LogLevel.INFO, code=ReportCode.INFO, message="Power flow solved normal."))
        try:
            create_measurements = kwargs.get('create_measurements', None)
            if create_measurements is not None and create_measurements.lower() == 'sv':
                CreateMeasurements(self.net, self.cim).create_measurements_from_sv()
            elif create_measurements is not None and create_measurements.lower() == 'analog':
                CreateMeasurements(self.net, self.cim).create_measurements_from_analog()
            elif create_measurements is not None:
                self.report_container.add_log(Report(
                    level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                    message="Not supported value for argument 'create_measurements', check method signature for"
                            "valid values!"))
                raise ValueError("Not supported value for argument 'create_measurements', check method signature for"
                                 "valid values!")
        except Exception as e:
            self.logger.error("Creating the measurements failed, returning the net without measurements!")
            self.logger.exception(e)
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Creating the measurements failed, returning the net without measurements!"))
            self.report_container.add_log(Report(
                level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                message=traceback.format_exc()))
            self.net.measurement = self.net.measurement[0:0]
        try:
            if kwargs.get('update_assets_from_sv', False):
                CreateMeasurements(self.net, self.cim).update_assets_from_sv()
        except Exception as e:
            self.logger.warning("Updating the assets failed!")
            self.logger.exception(e)
            self.report_container.add_log(Report(
                level=LogLevel.ERROR, code=ReportCode.ERROR_CONVERTING,
                message="Updating the assets failed!"))
            self.report_container.add_log(Report(
                level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION_CONVERTING,
                message=traceback.format_exc()))
        # a special fix for BB and NB mixed networks:
        # fuse boundary ConnectivityNodes with their TopologicalNodes
        bus_t = self.net.bus.reset_index(level=0, drop=False)
        bus_drop = bus_t.loc[bus_t[sc['o_prf']] == 'eq_bd', ['index', sc['o_id'], 'cim_topnode']]
        bus_drop.rename(columns={'index': 'b1'}, inplace=True)
        bus_drop = pd.merge(bus_drop, bus_t[['index', sc['o_id']]].rename(columns={'index': 'b2', sc['o_id']: 'o_id2'}),
                            how='inner', left_on='cim_topnode', right_on='o_id2')
        if bus_drop.index.size > 0:
            for b1, b2 in bus_drop[['b1', 'b2']].itertuples(index=False):
                self.logger.info("Fusing buses: b1: %s, b2: %s" % (b1, b2))
                pp.fuse_buses(self.net, b1, b2, drop=True, fuse_bus_measurements=True)
        # finally a fix for EquivalentInjections: If an EquivalentInjection is attached to boundary node, check if the
        # network behind this boundary node is attached. In this case, disable the EquivalentInjection.
        ward_t = self.net.ward.copy()
        ward_t['bus_prf'] = ward_t['bus'].map(self.net.bus[[sc['o_prf']]].to_dict().get(sc['o_prf']))
        self.net.ward.loc[(self.net.ward.bus.duplicated(keep=False) &
                           ((ward_t['bus_prf'] == 'eq_bd') | (ward_t['bus_prf'] == 'tp_bd'))), 'in_service'] = False
        xward_t = self.net.xward.copy()
        xward_t['bus_prf'] = xward_t['bus'].map(self.net.bus[[sc['o_prf']]].to_dict().get(sc['o_prf']))
        self.net.xward.loc[(self.net.xward.bus.duplicated(keep=False) &
                            ((xward_t['bus_prf'] == 'eq_bd') | (xward_t['bus_prf'] == 'tp_bd'))), 'in_service'] = False
        self.net['report_container'] = self.report_container
        return self.net

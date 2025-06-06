import logging
import math
import time

import numpy as np
import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.powerTransformersCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class PowerTransformersCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def convert_power_transformers_cim16(self):
        time_start = time.time()
        self.logger.info("Start converting PowerTransformers.")

        power_transformers = self._prepare_power_transformers_cim16()
        # split the power transformers into two and three windings
        power_trafo_counts = power_transformers.PowerTransformer.value_counts()
        power_trafo2w = power_trafo_counts[power_trafo_counts == 2].index.tolist()
        power_trafo3w = power_trafo_counts[power_trafo_counts == 3].index.tolist()

        power_transformers = power_transformers.set_index('PowerTransformer')
        power_trafo2w = power_transformers.loc[power_trafo2w].reset_index()
        power_trafo3w = power_transformers.loc[power_trafo3w].reset_index()

        if power_trafo2w.index.size > 0:
            # process the two winding transformers
            self._create_trafo_characteristic_table('trafo', power_trafo2w)
            power_trafo2w = self._prepare_trafos_cim16(power_trafo2w)
            self.cimConverter.copy_to_pp('trafo', power_trafo2w)
            self.cimConverter.power_trafo2w = power_trafo2w

        if power_trafo3w.index.size > 0:
            # process the three winding transformers
            self._create_trafo_characteristic_table('trafo3w', power_trafo3w)
            power_trafo3w = self._prepare_trafo3w_cim16(power_trafo3w)
            self.cimConverter.copy_to_pp('trafo3w', power_trafo3w)
            self.cimConverter.power_trafo3w = power_trafo3w

        self.logger.info("Created %s 2w trafos and %s 3w trafos in %ss." %
                         (power_trafo2w.index.size, power_trafo3w.index.size, time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created %s 2w trafos and %s 3w trafos from PowerTransformers in %ss." %
                    (power_trafo2w.index.size, power_trafo3w.index.size, time.time() - time_start)))

    def _create_trafo_characteristic_table(self, trafo_type, trafo_df_origin):
        if 'id_characteristic_table' not in trafo_df_origin.columns:
            trafo_df_origin['id_characteristic_table'] = pd.Series(pd.NA, dtype="Int64")
        if 'trafo_characteristic_table' not in self.cimConverter.net.keys():
            self.cimConverter.net['trafo_characteristic_table'] = pd.DataFrame(
                columns=['id_characteristic', 'step', 'voltage_ratio', 'angle_deg', 'vk_percent',
                         'vkr_percent', 'vkr_hv_percent', 'vkr_mv_percent',
                         'vkr_lv_percent', 'vk_hv_percent', 'vk_mv_percent',
                         'vk_lv_percent'])
        # get the TablePoints
        ptct = self.cimConverter.cim['eq']['PhaseTapChangerTabular'][['TransformerEnd', 'PhaseTapChangerTable']]
        ptct = pd.merge(ptct, self.cimConverter.cim['eq']['PhaseTapChangerTablePoint'][
            ['PhaseTapChangerTable', 'step', 'r', 'x', 'ratio', 'angle']], how='left', on='PhaseTapChangerTable')
        # append the ratio tap changers
        ptct_ratio = self.cimConverter.cim['eq']['RatioTapChanger'][['TransformerEnd', 'RatioTapChangerTable']]
        ptct_ratio = pd.merge(ptct_ratio, self.cimConverter.cim['eq']['RatioTapChangerTablePoint'][
            ['RatioTapChangerTable', 'step', 'r', 'x', 'ratio']], how='left', on='RatioTapChangerTable')
        ptct = pd.concat([ptct, ptct_ratio], ignore_index=True, sort=False)
        ptct = ptct.rename(columns={'step': 'tabular_step', 'r': 'r_dev', 'x': 'x_dev', 'TransformerEnd': sc['pte_id'],
                                    'ratio': 'ratio_dev', 'angle': 'angle_dev'})
        if trafo_type == 'trafo':
            trafo_df = trafo_df_origin.sort_values(['PowerTransformer', 'endNumber']).reset_index()
            # processing the transformer data
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
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_lv'])
            del fillna_list, one_item
            # just keep one transformer
            trafo_df = trafo_df.drop_duplicates(subset=['PowerTransformer'], keep='first')
            # merge the trafos with the tap changers
            trafo_df = pd.merge(trafo_df, ptct, how='left', on=sc['pte_id'])
            trafo_df = pd.merge(trafo_df, ptct.rename(columns={'tabular_step': 'tabular_step_lv', 'r_dev': 'r_dev_lv',
                                                               'x_dev': 'x_dev_lv', 'ratio_dev': 'ratio_dev_lv',
                                                               'angle_dev': 'angle_dev_lv',
                                                               sc['pte_id']: sc['pte_id'] + '_lv'}),
                                how='left', on=sc['pte_id'] + '_lv')
            fillna_list = ['tabular_step']
            for one_item in fillna_list:
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_lv'])
            del fillna_list, one_item
            trafo_df = trafo_df.dropna(subset=['r_dev', 'r_dev_lv'], how='all')
            fillna_list = ['r_dev', 'r_dev_lv', 'x_dev', 'x_dev_lv']
            for one_item in fillna_list:
                trafo_df[one_item] = trafo_df[one_item].fillna(0)
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
            fillna_list = ['ratio_dev', 'angle_dev']
            for one_item in fillna_list:
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_lv'])
            trafo_df['ratio_dev'] = trafo_df['ratio_dev'].fillna(1.)
            trafo_df['angle_dev'] = trafo_df['angle_dev'].fillna(0)
            trafo_df = trafo_df.rename(columns={'ratio_dev': 'voltage_ratio', 'angle_dev': 'angle_deg'})

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
            append_dict = dict({'id_characteristic': [], 'step': [], 'voltage_ratio': [], 'angle_deg': [],
                                'vk_percent': [], 'vkr_percent': []})
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
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_mv'])
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_lv'])
            del fillna_list, one_item
            # just keep one transformer
            trafo_df = trafo_df.drop_duplicates(subset=['PowerTransformer'], keep='first')
            # merge the trafos with the tap changers
            trafo_df = pd.concat([pd.merge(trafo_df, ptct, how='left', on=sc['pte_id']),
                                  pd.merge(trafo_df,
                                           ptct.rename(columns={'tabular_step': 'tabular_step_mv', 'r_dev': 'r_dev_mv',
                                                                'x_dev': 'x_dev_mv', 'ratio_dev': 'ratio_dev_mv',
                                                                'angle_dev': 'angle_dev_mv',
                                                                sc['pte_id']: sc['pte_id'] + '_mv'}),
                                           how='left', on=sc['pte_id'] + '_mv'),
                                  pd.merge(trafo_df,
                                           ptct.rename(columns={'tabular_step': 'tabular_step_lv', 'r_dev': 'r_dev_lv',
                                                                'x_dev': 'x_dev_lv', 'ratio_dev': 'ratio_dev_lv',
                                                                'angle_dev': 'angle_dev_lv',
                                                                sc['pte_id']: sc['pte_id'] + '_lv'}),
                                           how='left', on=sc['pte_id'] + '_lv')
                                  ], ignore_index=True, sort=False)
            # remove elements with more than one tap changer per trafo
            # todo: for multiple tap changers the deletion of tap entries must be changed
            trafo_df = trafo_df.loc[(~trafo_df.duplicated(subset=['PowerTransformer', 'tabular_step'], keep=False)) | (
               ~trafo_df.RatioTapChangerTable.isna()) | (~trafo_df.PhaseTapChangerTable.isna())]
            fillna_list = ['tabular_step']
            for one_item in fillna_list:
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_mv'])
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_lv'])
            del fillna_list, one_item
            trafo_df = trafo_df.dropna(subset=['r_dev', 'r_dev_mv', 'r_dev_lv'], how='all')
            fillna_list = ['r_dev', 'r_dev_mv', 'r_dev_lv', 'x_dev', 'x_dev_mv', 'x_dev_lv']
            for one_item in fillna_list:
                trafo_df[one_item] = trafo_df[one_item].fillna(0)
            fillna_list = ['ratio_dev', 'angle_dev']
            for one_item in fillna_list:
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_mv'])
                trafo_df[one_item] = trafo_df[one_item].fillna(trafo_df[one_item + '_lv'])
            trafo_df['ratio_dev'] = trafo_df['ratio_dev'].fillna(1.)
            trafo_df['angle_dev'] = trafo_df['angle_dev'].fillna(0)
            trafo_df = trafo_df.rename(columns={'ratio_dev': 'voltage_ratio', 'angle_dev': 'angle_deg'})

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
            append_dict = dict({'id_characteristic': [], 'step': [], 'voltage_ratio': [], 'angle_deg': [],
                                'vkr_hv_percent': [], 'vkr_mv_percent': [], 'vkr_lv_percent': [], 'vk_hv_percent': [],
                                'vk_mv_percent': [], 'vk_lv_percent': []})

        def append_row(res_dict, id_c, row, cols):
            res_dict['id_characteristic'].append(id_c)
            res_dict['step'].append(row.tabular_step)
            for variable in ['voltage_ratio', 'angle_deg', 'vkr_percent', 'vk_percent', 'vk_hv_percent',
                             'vkr_hv_percent', 'vk_mv_percent', 'vkr_mv_percent', 'vk_lv_percent', 'vkr_lv_percent']:
                if variable in cols:
                    res_dict[variable].append(getattr(row, variable))

        id_characteristic = self.cimConverter.net['trafo_characteristic_table']['id_characteristic'].max() + 1
        if math.isnan(id_characteristic):
            id_characteristic = 0
        for one_id, one_df in trafo_df.groupby(sc['pte_id']):
            # get next id_characteristic
            if len(append_dict['id_characteristic']) > 0:
                id_characteristic = max(append_dict['id_characteristic']) + 1
            # set the ID at the corresponding transformer
            trafo_df_origin.loc[trafo_df_origin['PowerTransformer'] == trafo_df_origin.loc[
                trafo_df_origin[sc['pte_id']] == one_id, 'PowerTransformer'].values[
                0], 'id_characteristic_table'] = id_characteristic
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

        # create tap_dependency_table flag
        if 'tap_dependency_table' not in trafo_df_origin.columns:
            trafo_df_origin['tap_dependency_table'] = False
        # set tap_dependency_table as True for respective trafos
        trafo_df_origin.loc[trafo_df_origin['id_characteristic_table'].notna(), 'tap_dependency_table'] = True

        self.cimConverter.net['trafo_characteristic_table'] = pd.concat(
            [self.cimConverter.net['trafo_characteristic_table'], pd.DataFrame(append_dict)],
            ignore_index=True, sort=False)
        self.cimConverter.net['trafo_characteristic_table']['step'] = \
            self.cimConverter.net['trafo_characteristic_table']['step'].astype(int)

    def _prepare_power_transformers_cim16(self) -> pd.DataFrame:
        if 'sc' in self.cimConverter.cim.keys():
            power_transformers = self.cimConverter.merge_eq_sc_profile('PowerTransformer')
        else:
            power_transformers = self.cimConverter.cim['eq']['PowerTransformer']
        power_transformers = power_transformers[['rdfId', 'name', 'description', 'isPartOfGeneratorUnit']]
        power_transformers[sc['o_cl']] = 'PowerTransformer'

        if 'sc' in self.cimConverter.cim.keys():
            power_transformer_ends = self.cimConverter.merge_eq_sc_profile('PowerTransformerEnd')
        else:
            power_transformer_ends = self.cimConverter.cim['eq']['PowerTransformerEnd']
        power_transformer_ends = power_transformer_ends[
            ['rdfId', 'PowerTransformer', 'endNumber', 'Terminal', 'ratedS', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0',
             'phaseAngleClock', 'connectionKind', 'grounded']]

        # merge and append the tap changers
        eqssh_tap_changers = pd.merge(self.cimConverter.cim['eq']['RatioTapChanger'][[
            'rdfId', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement',
            'TapChangerControl']],
                                      self.cimConverter.cim['ssh']['RatioTapChanger'][['rdfId', 'step']], how='left',
                                      on='rdfId')
        eqssh_tap_changers[sc['tc']] = 'RatioTapChanger'
        eqssh_tap_changers['tap_changer_type'] = "Ratio"  # Ratio/Asymmetrical phase shifter
        eqssh_tap_changers[sc['tc_id']] = eqssh_tap_changers['rdfId'].copy()
        # todo: check correct implementation for PhaseTapChangerLinear tap changers
        eqssh_tap_changers_linear = pd.merge(self.cimConverter.cim['eq']['PhaseTapChangerLinear'],
                                             self.cimConverter.cim['ssh']['PhaseTapChangerLinear'], how='left',
                                             on='rdfId')
        eqssh_tap_changers_linear['stepVoltageIncrement'] = .001
        eqssh_tap_changers_linear[sc['tc']] = 'PhaseTapChangerLinear'
        eqssh_tap_changers_linear['tap_changer_type'] = "Ideal"  # Ideal phase shifter
        eqssh_tap_changers_linear[sc['tc_id']] = eqssh_tap_changers_linear['rdfId'].copy()
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, eqssh_tap_changers_linear], ignore_index=True, sort=False)
        # todo: check correct implementation for PhaseTapChangerAsymmetrical tap changers
        eqssh_tap_changers_async = pd.merge(self.cimConverter.cim['eq']['PhaseTapChangerAsymmetrical'],
                                            self.cimConverter.cim['ssh']['PhaseTapChangerAsymmetrical'], how='left',
                                            on='rdfId')
        eqssh_tap_changers_async['stepVoltageIncrement'] = eqssh_tap_changers_async['voltageStepIncrement'][:]
        eqssh_tap_changers_async = eqssh_tap_changers_async.drop(columns=['voltageStepIncrement'])
        eqssh_tap_changers_async[sc['tc']] = 'PhaseTapChangerAsymmetrical'
        eqssh_tap_changers_async['tap_changer_type'] = "Ratio"  # Ratio/Asymmetrical phase shifter
        eqssh_tap_changers_async[sc['tc_id']] = eqssh_tap_changers_async['rdfId'].copy()
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, eqssh_tap_changers_async], ignore_index=True, sort=False)
        # todo: check correct implementation for PhaseTapChangerSymmetrical tap changers
        eqssh_ratio_tap_changers_sync = pd.merge(self.cimConverter.cim['eq']['PhaseTapChangerSymmetrical'],
                                                 self.cimConverter.cim['ssh']['PhaseTapChangerSymmetrical'], how='left',
                                                 on='rdfId')
        eqssh_ratio_tap_changers_sync['stepVoltageIncrement'] = eqssh_ratio_tap_changers_sync['voltageStepIncrement']
        eqssh_ratio_tap_changers_sync = eqssh_ratio_tap_changers_sync.drop(columns=['voltageStepIncrement'])
        eqssh_ratio_tap_changers_sync[sc['tc']] = 'PhaseTapChangerSymmetrical'
        eqssh_ratio_tap_changers_sync['tap_changer_type'] = "Symmetrical"  # Symmetrical phase shifter
        eqssh_ratio_tap_changers_sync[sc['tc_id']] = eqssh_ratio_tap_changers_sync['rdfId'].copy()
        eqssh_tap_changers = \
            pd.concat([eqssh_tap_changers, eqssh_ratio_tap_changers_sync], ignore_index=True, sort=False)
        # convert the PhaseTapChangerTabular to one tap changer
        ptct = pd.merge(
            self.cimConverter.cim['eq']['PhaseTapChangerTabular'][['rdfId', 'TransformerEnd', 'PhaseTapChangerTable',
                                                                   'highStep', 'lowStep', 'neutralStep']],
            self.cimConverter.cim['ssh']['PhaseTapChangerTabular'][['rdfId', 'step']], how='left', on='rdfId')
        ptct = ptct.rename(columns={'step': 'current_step'})
        ptct = pd.merge(ptct, self.cimConverter.cim['eq']['PhaseTapChangerTablePoint'][
            ['PhaseTapChangerTable', 'step', 'angle', 'ratio']], how='left', on='PhaseTapChangerTable')
        for one_id, one_df in ptct.groupby('TransformerEnd'):
            drop_index = one_df[one_df['step'] != one_df['current_step']].index.values
            keep_index = one_df[one_df['step'] == one_df['current_step']].index.values
            if keep_index.size > 0:
                keep_index = keep_index[0]
            else:
                self.logger.warning("Ignoring PhaseTapChangerTabular with ID: %s. The current tap position is missing "
                                    "in the PhaseTapChangerTablePoints!" % one_id)
                ptct = ptct.drop(drop_index)
                continue
            one_df = one_df.set_index('step')
            neutral_step = one_df['neutralStep'].iloc[0]
            ptct = ptct.drop(drop_index)
            # keep the angle and ratio based on neutral tap position (to populate tap_step_percent and tap_step_degree)
            ptct.loc[keep_index, 'angle'] = one_df.loc[neutral_step, 'angle']
            ptct.loc[keep_index, 'ratio'] = (one_df.loc[neutral_step, 'ratio'] - 1) * 100
        ptct[sc['tc']] = 'PhaseTapChangerTabular'
        ptct[sc['tc_id']] = ptct['rdfId'].copy()
        ptct = ptct.drop(columns=['rdfId', 'PhaseTapChangerTable', 'step'])
        ptct = ptct.rename(columns={'current_step': 'step'})
        ptct['stepPhaseShiftIncrement'] = ptct['angle'][:]
        ptct['stepVoltageIncrement'] = ptct['ratio'][:]
        ptct['tap_changer_type'] = "Tabular"  # PhaseTapChangerTabular
        eqssh_tap_changers = pd.concat([eqssh_tap_changers, ptct], ignore_index=True, sort=False)
        del eqssh_tap_changers_linear, eqssh_tap_changers_async, eqssh_ratio_tap_changers_sync

        # remove duplicated TapChanger: A Transformer may have one RatioTapChanger and one PhaseTapChanger
        # self.logger.info("eqssh_tap_changers.index.size: %s" % eqssh_tap_changers.index.size)
        # self.logger.info("dups:")
        # for _, one_dup in eqssh_tap_changers[eqssh_tap_changers.duplicated('TransformerEnd', keep=False)].iterrows():
        #     self.logger.info(one_dup)  # no example for testing found
        eqssh_tap_changers = eqssh_tap_changers.drop_duplicates(subset=['TransformerEnd'])
        # prepare the controllers
        eq_ssh_tap_controllers = self.cimConverter.merge_eq_ssh_profile('TapChangerControl')
        eq_ssh_tap_controllers = \
            eq_ssh_tap_controllers[['rdfId', 'Terminal', 'discrete', 'enabled', 'targetValue', 'targetDeadband']]
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.rename(columns={'rdfId': 'TapChangerControl'})
        # first merge with the VoltageLimits
        if 'VoltageLimit' in self.cimConverter.cim['ssh'].keys():
            vl = self.cimConverter.merge_eq_ssh_profile('VoltageLimit')[['OperationalLimitSet', 'OperationalLimitType',
                                                                         'value']]
        else:
            vl = self.cimConverter.cim['eq']['VoltageLimit'][['OperationalLimitSet', 'OperationalLimitType', 'value']]
        opl_kind = 'kind' if 'kind' in self.cimConverter.cim['eq']['OperationalLimitType'].columns else 'limitType'
        vl = pd.merge(vl, self.cimConverter.cim['eq']['OperationalLimitType'][['rdfId', opl_kind]].rename(
            columns={'rdfId': 'OperationalLimitType'}), how='left', on='OperationalLimitType')
        vl = pd.merge(vl, self.cimConverter.cim['eq']['OperationalLimitSet'][['rdfId', 'Terminal']].rename(
            columns={'rdfId': 'OperationalLimitSet'}), how='left', on='OperationalLimitSet')
        vl = vl[['value', opl_kind, 'Terminal']]
        vl_low = vl.loc[vl[opl_kind] == 'lowVoltage'][['value', 'Terminal']].rename(
            columns={'value': 'c_vm_lower_pu'})
        vl_up = vl.loc[vl[opl_kind] == 'highVoltage'][['value', 'Terminal']].rename(
            columns={'value': 'c_vm_upper_pu'})
        vl = pd.merge(vl_low, vl_up, how='left', on='Terminal')
        eq_ssh_tap_controllers = pd.merge(eq_ssh_tap_controllers, vl, how='left', on='Terminal')
        eq_ssh_tap_controllers['c_Terminal'] = eq_ssh_tap_controllers['Terminal'][:]
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.rename(columns={'Terminal': 'rdfId', 'enabled': 'c_in_service',
                                               'targetValue': 'c_vm_set_pu', 'targetDeadband': 'c_tol'})
        # get the Terminal, ConnectivityNode and bus voltage
        eq_ssh_tap_controllers = \
            pd.merge(eq_ssh_tap_controllers,
                     pd.concat([self.cimConverter.cim['eq']['Terminal'], self.cimConverter.cim['eq_bd']['Terminal']],
                               ignore_index=True, sort=False)[
                         ['rdfId', 'ConnectivityNode']], how='left', on='rdfId')
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.drop(columns=['rdfId'])
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.rename(columns={'ConnectivityNode': sc['o_id']})
        eq_ssh_tap_controllers = pd.merge(eq_ssh_tap_controllers,
                                          self.cimConverter.net['bus'].reset_index(level=0)[
                                              ['index', sc['o_id'], 'vn_kv']],
                                          how='left', on=sc['o_id'])
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.drop(columns=[sc['o_id']])
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.rename(columns={'index': 'c_bus_id'})
        eq_ssh_tap_controllers['c_vm_set_pu'] = eq_ssh_tap_controllers['c_vm_set_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_tol'] = eq_ssh_tap_controllers['c_tol'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_vm_lower_pu'] = \
            eq_ssh_tap_controllers['c_vm_lower_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers['c_vm_upper_pu'] = \
            eq_ssh_tap_controllers['c_vm_upper_pu'] / eq_ssh_tap_controllers['vn_kv']
        eq_ssh_tap_controllers = eq_ssh_tap_controllers.drop(columns=['vn_kv'])

        eqssh_tap_changers = pd.merge(eqssh_tap_changers, eq_ssh_tap_controllers, how='left', on='TapChangerControl')
        eqssh_tap_changers = eqssh_tap_changers.rename(columns={'TransformerEnd': sc['pte_id']})

        power_transformers = power_transformers.rename(columns={'rdfId': 'PowerTransformer'})
        power_transformer_ends = power_transformer_ends.rename(columns={'rdfId': sc['pte_id']})
        # add the PowerTransformerEnds
        power_transformers = pd.merge(power_transformers, power_transformer_ends, how='left',
                                         on='PowerTransformer')
        # add the Terminal and bus indexes
        power_transformers = pd.merge(power_transformers, self.cimConverter.bus_merge.drop('rdfId', axis=1),
                                         how='left', left_on='Terminal', right_on='rdfId_Terminal')
        # add the TapChangers
        power_transformers = pd.merge(power_transformers, eqssh_tap_changers, how='left', on=sc['pte_id'])
        return power_transformers

    def _prepare_trafos_cim16(self, power_trafo2w: pd.DataFrame) -> pd.DataFrame:
        power_trafo2w = power_trafo2w.sort_values(['PowerTransformer', 'endNumber']).reset_index()
        # precessing the transformer data
        # a list of transformer parameters which are used for each transformer winding
        copy_list = ['index_bus', 'Terminal', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0', 'neutralStep', 'lowStep',
                     'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step', 'connected',
                     'phaseAngleClock', 'connectionKind', sc['pte_id'], sc['tc'], sc['tc_id'], 'grounded', 'angle',
                     'tap_changer_type']
        for one_item in copy_list:
            # copy the columns which are required for each winding
            power_trafo2w[one_item + '_lv'] = power_trafo2w[one_item].copy()
            # cut the first element from the copied columns
            power_trafo2w[one_item + '_lv'] = power_trafo2w[one_item + '_lv'].iloc[1:].reset_index()[
                one_item + '_lv']
        del copy_list, one_item
        # detect on which winding a tap changer is attached
        power_trafo2w['tap_side'] = None
        power_trafo2w.loc[power_trafo2w['step_lv'].notna(), 'tap_side'] = 'lv'
        power_trafo2w.loc[power_trafo2w['step'].notna(), 'tap_side'] = 'hv'
        fillna_list = ['neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step',
                       sc['tc'], sc['tc_id'], 'tap_changer_type']
        for one_item in fillna_list:
            power_trafo2w[one_item] = power_trafo2w[one_item].fillna(power_trafo2w[one_item + '_lv'])
        del fillna_list, one_item
        # just keep one transformer
        power_trafo2w = power_trafo2w.drop_duplicates(subset=['PowerTransformer'], keep='first')

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
        power_trafo2w.loc[power_trafo2w['phaseAngleClock'] == 0, 'phaseAngleClock'] = np.nan
        power_trafo2w['phaseAngleClock_lv'] = power_trafo2w['phaseAngleClock_lv'].fillna(0)
        power_trafo2w['shift_degree'] = power_trafo2w['phaseAngleClock'].astype(float).fillna(
            power_trafo2w['phaseAngleClock_lv'].astype(float)) * 30
        power_trafo2w['parallel'] = 1
        power_trafo2w['in_service'] = power_trafo2w.connected & power_trafo2w.connected_lv
        power_trafo2w['connectionKind'] = power_trafo2w['connectionKind'].fillna('')
        power_trafo2w['connectionKind_lv'] = power_trafo2w['connectionKind_lv'].fillna('')
        power_trafo2w['grounded'] = power_trafo2w['grounded'].fillna(True)
        power_trafo2w['grounded_lv'] = power_trafo2w['grounded_lv'].fillna(True)
        # power_trafo2w.loc[~power_trafo2w['grounded'].astype('bool'), 'connectionKind'] = \
        #     power_trafo2w.loc[~power_trafo2w['grounded'].astype('bool'), 'connectionKind'].str.replace('n', '')
        # power_trafo2w.loc[~power_trafo2w['grounded_lv'].astype('bool'), 'connectionKind_lv'] = \
        #     power_trafo2w.loc[~power_trafo2w['grounded_lv'].astype('bool'), 'connectionKind_lv'].str.replace('n', '')
        power_trafo2w['vector_group'] = power_trafo2w.connectionKind.str.upper() + power_trafo2w.connectionKind_lv.str.lower()
        power_trafo2w.loc[power_trafo2w['vector_group'] == '', 'vector_group'] = None
        power_trafo2w = power_trafo2w.rename(columns={
            'PowerTransformer': sc['o_id'], 'Terminal': sc['t_hv'], 'Terminal_lv': sc['t_lv'],
            sc['pte_id']: sc['pte_id_hv'], sc['pte_id'] + '_lv': sc['pte_id_lv'], 'index_bus': 'hv_bus',
            'index_bus_lv': 'lv_bus', 'neutralStep': 'tap_neutral', 'lowStep': 'tap_min', 'highStep': 'tap_max',
            'step': 'tap_pos', 'stepVoltageIncrement': 'tap_step_percent', 'stepPhaseShiftIncrement': 'tap_step_degree',
            'isPartOfGeneratorUnit': 'power_station_unit', 'ratedU': 'vn_hv_kv', 'ratedU_lv': 'vn_lv_kv',
            'ratedS': 'sn_mva', 'xground': 'xn_ohm', 'grounded': 'oltc'})
        return power_trafo2w

    def _prepare_trafo3w_cim16(self, power_trafo3w: pd.DataFrame) -> pd.DataFrame:
        power_trafo3w = power_trafo3w.sort_values(['PowerTransformer', 'endNumber']).reset_index()
        # precessing the transformer data
        # a list of transformer parameters which are used for each transformer winding
        copy_list = ['index_bus', 'Terminal', 'ratedS', 'ratedU', 'r', 'x', 'b', 'g', 'r0', 'x0', 'neutralStep',
                     'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step', 'connected',
                     'angle', 'phaseAngleClock', 'connectionKind', 'grounded', sc['pte_id'], sc['tc'], sc['tc_id'],
                     'tap_changer_type']
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
        power_trafo3w['tap_side'] = None
        power_trafo3w.loc[power_trafo3w['step_lv'].notna(), 'tap_side'] = 'lv'
        power_trafo3w.loc[power_trafo3w['step_mv'].notna(), 'tap_side'] = 'mv'
        power_trafo3w.loc[power_trafo3w['step'].notna(), 'tap_side'] = 'hv'
        fillna_list = ['neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement', 'stepPhaseShiftIncrement', 'step',
                       sc['tc'], sc['tc_id'], 'tap_changer_type']
        for one_item in fillna_list:
            power_trafo3w[one_item] = power_trafo3w[one_item].fillna(power_trafo3w[one_item + '_mv'])
            power_trafo3w[one_item] = power_trafo3w[one_item].fillna(power_trafo3w[one_item + '_lv'])
        del fillna_list, one_item
        # just keep one transformer
        power_trafo3w = power_trafo3w.drop_duplicates(subset=['PowerTransformer'], keep='first')

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
        power_trafo3w['phaseAngleClock_mv'] = power_trafo3w['phaseAngleClock_mv'].fillna(0)
        power_trafo3w['phaseAngleClock_lv'] = power_trafo3w['phaseAngleClock_lv'].fillna(0)
        power_trafo3w['shift_mv_degree'] = power_trafo3w['phaseAngleClock_mv'].astype(float) * 30
        power_trafo3w['shift_lv_degree'] = power_trafo3w['phaseAngleClock_mv'].astype(float) * 30
        power_trafo3w['tap_at_star_point'] = False
        power_trafo3w['in_service'] = power_trafo3w.connected & power_trafo3w.connected_mv & power_trafo3w.connected_lv
        power_trafo3w['connectionKind'] = power_trafo3w['connectionKind'].fillna('')
        power_trafo3w['connectionKind_mv'] = power_trafo3w['connectionKind_mv'].fillna('')
        power_trafo3w['connectionKind_lv'] = power_trafo3w['connectionKind_lv'].fillna('')
        power_trafo3w['grounded'] = power_trafo3w['grounded'].fillna(True)
        power_trafo3w['grounded_mv'] = power_trafo3w['grounded_mv'].fillna(True)
        power_trafo3w['grounded_lv'] = power_trafo3w['grounded_lv'].fillna(True)

        # power_trafo3w.loc[~power_trafo3w['grounded'].astype('bool'), 'connectionKind'] = \
        #     power_trafo3w.loc[~power_trafo3w['grounded'].astype('bool'), 'connectionKind'].str.replace('n', '')
        # power_trafo3w.loc[~power_trafo3w['grounded_mv'].astype('bool'), 'connectionKind_mv'] = \
        #     power_trafo3w.loc[~power_trafo3w['grounded_mv'].astype('bool'), 'connectionKind_mv'].str.replace('n', '')
        # power_trafo3w.loc[~power_trafo3w['grounded_lv'].astype('bool'), 'connectionKind_lv'] = \
        #     power_trafo3w.loc[~power_trafo3w['grounded_lv'].astype('bool'), 'connectionKind_lv'].str.replace('n', '')
        power_trafo3w['vector_group'] = power_trafo3w.connectionKind.str.upper() + power_trafo3w.connectionKind_mv.str.lower() + power_trafo3w.connectionKind_lv.str.lower()
        power_trafo3w.loc[power_trafo3w['vector_group'] == '', 'vector_group'] = None
        power_trafo3w = power_trafo3w.rename(columns={
            'PowerTransformer': sc['o_id'], 'Terminal': sc['t_hv'], 'Terminal_mv': sc['t_mv'],
            'Terminal_lv': sc['t_lv'], sc['pte_id']: sc['pte_id_hv'], sc['pte_id'] + '_mv': sc['pte_id_mv'],
            sc['pte_id'] + '_lv': sc['pte_id_lv'], 'index_bus': 'hv_bus', 'index_bus_mv': 'mv_bus',
            'index_bus_lv': 'lv_bus', 'neutralStep': 'tap_neutral', 'lowStep': 'tap_min', 'highStep': 'tap_max',
            'step': 'tap_pos', 'stepVoltageIncrement': 'tap_step_percent', 'stepPhaseShiftIncrement': 'tap_step_degree',
            'isPartOfGeneratorUnit': 'power_station_unit', 'ratedU': 'vn_hv_kv', 'ratedU_mv': 'vn_mv_kv',
            'ratedU_lv': 'vn_lv_kv', 'ratedS': 'sn_hv_mva', 'ratedS_mv': 'sn_mv_mva', 'ratedS_lv': 'sn_lv_mva'})
        return power_trafo3w

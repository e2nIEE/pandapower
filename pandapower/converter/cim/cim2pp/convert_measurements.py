# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import time
from typing import Dict
import pandas as pd
import numpy as np
import pandapower.auxiliary
from .. import cim_tools


class CreateMeasurements:

    def __init__(self, net: pandapower.auxiliary.pandapowerNet, cim: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.net = net
        self.cim = cim

    def _set_measurement_element_datatype(self):
        self.net.measurement.element = self.net.measurement.element.astype(np.sctypeDict.get("UInt32"))

    def _copy_to_measurement(self, input_df: pd.DataFrame):
        pp_type = 'measurement'
        self.logger.debug("Copy %s datasets to pandapower network with type %s" % (input_df.index.size, pp_type))
        if pp_type not in self.net.keys():
            self.logger.warning("Missing pandapower type %s in the pandapower network!" % pp_type)
            return
        start_index_pp_net = self.net[pp_type].index.size
        self.net[pp_type] = pd.concat([self.net[pp_type], pd.DataFrame(None, index=[list(range(input_df.index.size))])],
                                      ignore_index=True, sort=False)
        for one_attr in self.net[pp_type].columns:
            if one_attr in input_df.columns:
                self.net[pp_type][one_attr][start_index_pp_net:] = input_df[one_attr][:]

    def create_measurements_from_analog(self):
        self.logger.info("------------------------- Creating measurements from Analog -------------------------")
        time_start = time.time()
        sc = cim_tools.get_pp_net_special_columns_dict()
        # join the Analogs with the AnalogValues and MeasurementValueSources
        analogs = pd.merge(
            self.cim['eq']['Analog'][['rdfId', 'measurementType', 'unitSymbol', 'unitMultiplier', 'Terminal',
                                      'PowerSystemResource', 'positiveFlowIn']],
            self.cim['eq']['AnalogValue'][['sensorAccuracy', 'MeasurementValueSource', 'Analog', 'value']],
            how='inner', left_on='rdfId', right_on='Analog')
        analogs = analogs.drop(columns=['rdfId', 'Analog'])
        analogs = pd.merge(analogs, self.cim['eq']['MeasurementValueSource'], how='left',
                           left_on='MeasurementValueSource',
                           right_on='rdfId')
        analogs = analogs.drop(columns=['rdfId', 'MeasurementValueSource'])
        # collect all the assets (line, trafo, trafo3w) and its connections
        assets = pd.DataFrame(None, columns=['element_type', 'side'])
        append_dict = dict({'line': {'from_bus': 'from', 'to_bus': 'to'},
                            'trafo': {'hv_bus': 'hv', 'lv_bus': 'lv'},
                            'trafo3w': {'hv_bus': 'hv', 'mv_bus': 'mv', 'lv_bus': 'lv'}})
        for element_type, sides in append_dict.items():
            for side_name, side in sides.items():
                temp = self.net[element_type][[sc['o_id'], side_name, sc[side_name]]]. \
                    reset_index().rename(columns={side_name: 'bus', sc[side_name]: 'terminal_asset'})
                temp['element_type'] = element_type
                temp['side'] = side
                assets = pd.concat([assets, temp], sort=False)
        assets = assets.rename(columns={'index': 'element'})
        # now join the analogs with the assets
        psr = pd.merge(analogs, assets, how='inner', left_on='PowerSystemResource', right_on=sc['o_id'])
        # keep only entries which are associated to the terminal from the asset
        psr = psr.loc[psr.Terminal == psr.terminal_asset]
        # remove the PhaseVoltage measurements
        psr = psr.loc[psr.measurementType != 'PhaseVoltage']
        psr['measurement_type'] = psr.unitSymbol.map({'W': 'p', 'VAr': 'q', 'A': 'i', 'V': 'v'})
        # change the sign if need
        psr['value'].loc[~psr['positiveFlowIn']] = psr.loc[~psr['positiveFlowIn']]['value'] * (-1)
        # convert all amperes to ka
        psr['value'].loc[psr['measurement_type'] == 'i'] = psr.loc[psr['measurement_type'] == 'i']['value'] / 1e3
        # move the voltage measurements to the buses
        psr = pd.merge(psr, self.net.bus[['vn_kv']], how='inner', left_on='bus', right_index=True)
        temp = psr.loc[psr['measurement_type'] == 'v']
        temp['value'] = temp['value'] / temp['vn_kv']
        temp['std_dev'] = temp['sensorAccuracy'] / temp['vn_kv']
        temp['element_type'] = 'bus'
        temp['element'] = temp['bus']
        temp['side'] = None
        psr.loc[psr['measurement_type'] == 'v'] = temp

        self._copy_to_measurement(psr)

        # set the element from float to default uint32
        self._set_measurement_element_datatype()

        self.logger.info("Needed time for creating the measurements: %ss" % (time.time() - time_start))

    def create_measurements_from_sv(self):
        self.logger.info("--------------------------- Creating measurements from SV ---------------------------")
        time_start = time.time()
        sc = cim_tools.get_pp_net_special_columns_dict()
        # get the measurements from the sv profile and set the Terminal as index
        sv_powerflow = self.cim['sv']['SvPowerFlow'][['Terminal', 'p', 'q']]
        sv_powerflow = sv_powerflow.set_index('Terminal')

        # ---------------------------------------measure: bus v---------------------------------------------------
        busses_temp = self.net.bus[['name', 'vn_kv', sc['ct']]].copy()
        busses_temp = busses_temp.reset_index(level=0)
        busses_temp = busses_temp.rename(columns={'index': 'element', sc['ct']: 'TopologicalNode'})
        sv_sv_voltages = pd.merge(self.cim['sv']['SvVoltage'][['TopologicalNode', 'v']], busses_temp,
                                  how='left', on='TopologicalNode')
        # drop all the rows mit vn_kv == np.NaN (no measurements available for that bus)
        sv_sv_voltages = sv_sv_voltages.dropna(subset=['vn_kv'])
        sv_sv_voltages.reset_index(inplace=True)
        if 'index' in sv_sv_voltages.columns:
            sv_sv_voltages = sv_sv_voltages.drop(['index'], axis=1)
        # value -> voltage ()
        sv_sv_voltages['value'] = sv_sv_voltages.v / sv_sv_voltages.vn_kv
        sv_sv_voltages['value'].replace(0, np.nan, inplace=True)
        # drop all the rows mit value == np.NaN
        sv_sv_voltages = sv_sv_voltages.dropna(subset=['value'])
        sv_sv_voltages.reset_index(inplace=True)
        sv_sv_voltages['value_stddev'] = sv_sv_voltages.value * 0.001
        sv_sv_voltages['vn_kv_stddev'] = 0.1 / sv_sv_voltages.vn_kv
        sv_sv_voltages['std_dev'] = sv_sv_voltages[["value_stddev", "vn_kv_stddev"]].max(axis=1)
        sv_sv_voltages['measurement_type'] = 'v'
        sv_sv_voltages['element_type'] = 'bus'
        sv_sv_voltages['side'] = None

        self._copy_to_measurement(sv_sv_voltages)

        # ---------------------------------------measure: line---------------------------------------------------
        sigma_line = 0.03
        line_temp = self.net.line[['name', 'from_bus', 'to_bus', sc['t_from'], sc['t_to']]].copy()
        line_temp['p_from'] = \
            pd.merge(line_temp[sc['t_from']], sv_powerflow['p'], left_on=sc['t_from'], right_index=True)['p']
        line_temp['p_to'] = \
            pd.merge(line_temp[sc['t_to']], sv_powerflow['p'], left_on=sc['t_to'], right_index=True)['p']
        line_temp['q_from'] = \
            pd.merge(line_temp[sc['t_from']], sv_powerflow['q'], left_on=sc['t_from'], right_index=True)['q']
        line_temp['q_to'] = \
            pd.merge(line_temp[sc['t_to']], sv_powerflow['q'], left_on=sc['t_to'], right_index=True)['q']

        line_temp = line_temp.dropna(subset=['p_from', 'p_to', 'q_from', 'q_to'], thresh=4)

        line_temp['stddev_line_from_p'] = abs(line_temp.p_from) * sigma_line + 1.
        line_temp['stddev_line_to_p'] = abs(line_temp.p_to) * sigma_line + 1.
        line_temp['stddev_line_from_q'] = abs(line_temp.q_from) * sigma_line + 1.
        line_temp['stddev_line_to_q'] = abs(line_temp.q_to) * sigma_line + 1.
        line_temp['element_type'] = 'line'
        line_temp['element'] = line_temp.index[:]
        # copy the data into the measurement dataframe
        # ---------------------------------------measure: line p from---------------------------------------------------
        line_temp['measurement_type'] = 'p'
        line_temp['value'] = line_temp.p_from
        line_temp['std_dev'] = line_temp.stddev_line_from_p
        line_temp['side'] = line_temp.from_bus
        self._copy_to_measurement(line_temp)
        # ---------------------------------------measure: line p to---------------------------------------------------
        line_temp['value'] = line_temp.p_to
        line_temp['std_dev'] = line_temp.stddev_line_to_p
        line_temp['side'] = line_temp.to_bus
        self._copy_to_measurement(line_temp)
        # ---------------------------------------measure: line q from---------------------------------------------------
        line_temp['measurement_type'] = 'q'
        line_temp['value'] = line_temp.q_from
        line_temp['std_dev'] = line_temp.stddev_line_from_q
        line_temp['side'] = line_temp.from_bus
        self._copy_to_measurement(line_temp)
        # ---------------------------------------measure: line q to---------------------------------------------------
        line_temp['value'] = line_temp.q_to
        line_temp['std_dev'] = line_temp.stddev_line_to_q
        line_temp['side'] = line_temp.to_bus
        self._copy_to_measurement(line_temp)

        # ---------------------------------------measure: trafo---------------------------------------------------
        sigma_trafo = 0.03
        trafo_temp = self.net.trafo[['name', 'hv_bus', 'lv_bus', sc['t_hv'], sc['t_lv']]].copy()
        # if trafo_temp.index.size > 0:
        trafo_temp['p_hv'] = \
            pd.merge(trafo_temp[sc['t_hv']], sv_powerflow['p'], left_on=sc['t_hv'], right_index=True)['p']
        trafo_temp['p_lv'] = \
            pd.merge(trafo_temp[sc['t_lv']], sv_powerflow['p'], left_on=sc['t_lv'], right_index=True)['p']
        trafo_temp['q_hv'] = \
            pd.merge(trafo_temp[sc['t_hv']], sv_powerflow['q'], left_on=sc['t_hv'], right_index=True)['q']
        trafo_temp['q_lv'] = \
            pd.merge(trafo_temp[sc['t_lv']], sv_powerflow['q'], left_on=sc['t_lv'], right_index=True)['q']

        trafo_temp = trafo_temp.dropna(subset=['p_hv', 'p_lv', 'q_hv', 'q_lv'], thresh=4)

        trafo_temp['stddev_trafo_hv_p'] = abs(trafo_temp.p_hv) * sigma_trafo + 1.
        trafo_temp['stddev_trafo_lv_p'] = abs(trafo_temp.p_lv) * sigma_trafo + 1.
        trafo_temp['stddev_trafo_hv_q'] = abs(trafo_temp.q_hv) * sigma_trafo + 1.
        trafo_temp['stddev_trafo_lv_q'] = abs(trafo_temp.q_lv) * sigma_trafo + 1.
        trafo_temp['element_type'] = 'trafo'
        trafo_temp['element'] = trafo_temp.index[:]
        # copy the data into the measurement dataframe
        # ---------------------------------------measure: trafo p hv---------------------------------------------------
        trafo_temp['measurement_type'] = 'p'
        trafo_temp['value'] = trafo_temp.p_hv
        trafo_temp['std_dev'] = trafo_temp.stddev_trafo_hv_p
        trafo_temp['side'] = trafo_temp.hv_bus
        self._copy_to_measurement(trafo_temp)
        # ---------------------------------------measure: trafo p lv---------------------------------------------------
        trafo_temp['value'] = trafo_temp.p_lv
        trafo_temp['std_dev'] = trafo_temp.stddev_trafo_lv_p
        trafo_temp['side'] = trafo_temp.lv_bus
        self._copy_to_measurement(trafo_temp)
        # ---------------------------------------measure: trafo q hv---------------------------------------------------
        trafo_temp['measurement_type'] = 'q'
        trafo_temp['value'] = trafo_temp.q_hv
        trafo_temp['std_dev'] = trafo_temp.stddev_trafo_hv_q
        trafo_temp['side'] = trafo_temp.hv_bus
        self._copy_to_measurement(trafo_temp)
        # ---------------------------------------measure: trafo q lv---------------------------------------------------
        trafo_temp['value'] = trafo_temp.q_lv
        trafo_temp['std_dev'] = trafo_temp.stddev_trafo_lv_q
        trafo_temp['side'] = trafo_temp.lv_bus
        self._copy_to_measurement(trafo_temp)

        # ---------------------------------------measure: trafo3w---------------------------------------------------
        sigma_trafo3w = 0.03
        trafo3w_temp = self.net.trafo3w[
            ['name', 'hv_bus', 'mv_bus', 'lv_bus', sc['t_hv'], sc['t_mv'], sc['t_lv']]].copy()
        trafo3w_temp['p_hv'] = \
            pd.merge(trafo3w_temp[sc['t_hv']], sv_powerflow['p'], left_on=sc['t_hv'], right_index=True)['p']
        trafo3w_temp['p_mv'] = \
            pd.merge(trafo3w_temp[sc['t_mv']], sv_powerflow['p'], left_on=sc['t_mv'], right_index=True)['p']
        trafo3w_temp['p_lv'] = \
            pd.merge(trafo3w_temp[sc['t_lv']], sv_powerflow['p'], left_on=sc['t_lv'], right_index=True)['p']
        trafo3w_temp['q_hv'] = \
            pd.merge(trafo3w_temp[sc['t_hv']], sv_powerflow['q'], left_on=sc['t_hv'], right_index=True)['q']
        trafo3w_temp['q_mv'] = \
            pd.merge(trafo3w_temp[sc['t_mv']], sv_powerflow['q'], left_on=sc['t_mv'], right_index=True)['q']
        trafo3w_temp['q_lv'] = \
            pd.merge(trafo3w_temp[sc['t_lv']], sv_powerflow['q'], left_on=sc['t_lv'], right_index=True)['q']

        trafo3w_temp = trafo3w_temp.dropna(subset=['p_hv', 'p_mv', 'p_lv', 'q_hv', 'q_mv', 'q_lv'], thresh=6)

        trafo3w_temp['stddev_trafo_hv_p'] = abs(trafo3w_temp.p_hv) * sigma_trafo3w + 1.
        trafo3w_temp['stddev_trafo_mv_p'] = abs(trafo3w_temp.p_mv) * sigma_trafo3w + 1.
        trafo3w_temp['stddev_trafo_lv_p'] = abs(trafo3w_temp.p_lv) * sigma_trafo3w + 1.
        trafo3w_temp['stddev_trafo_hv_q'] = abs(trafo3w_temp.q_hv) * sigma_trafo3w + 1.
        trafo3w_temp['stddev_trafo_mv_q'] = abs(trafo3w_temp.q_mv) * sigma_trafo3w + 1.
        trafo3w_temp['stddev_trafo_lv_q'] = abs(trafo3w_temp.q_lv) * sigma_trafo3w + 1.
        trafo3w_temp['element_type'] = 'trafo3w'
        trafo3w_temp['element'] = trafo3w_temp.index[:]
        # copy the data into the measurement dataframe
        # ---------------------------------------measure: trafo3w p hv---------------------------------------------
        trafo3w_temp['measurement_type'] = 'p'
        trafo3w_temp['value'] = trafo3w_temp.p_hv
        trafo3w_temp['std_dev'] = trafo3w_temp.stddev_trafo_hv_p
        trafo3w_temp['side'] = trafo3w_temp.hv_bus
        self._copy_to_measurement(trafo3w_temp)
        # ---------------------------------------measure: trafo3w p mv---------------------------------------------
        trafo3w_temp['value'] = trafo3w_temp.p_mv
        trafo3w_temp['std_dev'] = trafo3w_temp.stddev_trafo_mv_p
        trafo3w_temp['side'] = trafo3w_temp.mv_bus
        self._copy_to_measurement(trafo3w_temp)
        # ---------------------------------------measure: trafo3w p lv---------------------------------------------
        trafo3w_temp['value'] = trafo3w_temp.p_lv
        trafo3w_temp['std_dev'] = trafo3w_temp.stddev_trafo_lv_p
        trafo3w_temp['side'] = trafo3w_temp.lv_bus
        self._copy_to_measurement(trafo3w_temp)
        # ---------------------------------------measure: trafo3w q hv---------------------------------------------
        trafo3w_temp['measurement_type'] = 'q'
        trafo3w_temp['value'] = trafo3w_temp.q_hv
        trafo3w_temp['std_dev'] = trafo3w_temp.stddev_trafo_hv_q
        trafo3w_temp['side'] = trafo3w_temp.hv_bus
        self._copy_to_measurement(trafo3w_temp)
        # ---------------------------------------measure: trafo3w q mv---------------------------------------------
        trafo3w_temp['value'] = trafo3w_temp.q_mv
        trafo3w_temp['std_dev'] = trafo3w_temp.stddev_trafo_mv_q
        trafo3w_temp['side'] = trafo3w_temp.mv_bus
        self._copy_to_measurement(trafo3w_temp)
        # ---------------------------------------measure: trafo3w q lv---------------------------------------------
        trafo3w_temp['value'] = trafo3w_temp.q_lv
        trafo3w_temp['std_dev'] = trafo3w_temp.stddev_trafo_lv_q
        trafo3w_temp['side'] = trafo3w_temp.lv_bus
        self._copy_to_measurement(trafo3w_temp)

        # remove NaN values
        self.net.measurement = self.net.measurement.dropna(subset=['value'])
        # set the element from float to default uint32
        self._set_measurement_element_datatype()

        self.logger.info("Needed time for creating the measurements: %ss" % (time.time() - time_start))

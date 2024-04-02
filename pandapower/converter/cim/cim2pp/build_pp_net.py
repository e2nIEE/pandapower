# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import logging
import traceback
from typing import Dict

import pandas as pd

import pandapower as pp
import pandapower.auxiliary
from .convert_measurements import CreateMeasurements
from .. import cim_classes
from .. import cim_tools
from .. import pp_tools
from ..other_classes import ReportContainer, Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.build_pp_net')

pd.set_option('display.max_columns', 900)
pd.set_option('display.max_rows', 90000)
sc = cim_tools.get_pp_net_special_columns_dict()


class CimConverter:

    def __init__(self, cim_parser: cim_classes.CimParser, converter_classes: Dict, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cim_parser: cim_classes.CimParser = cim_parser
        self.kwargs = kwargs
        self.cim: Dict[str, Dict[str, pd.DataFrame]] = self.cim_parser.get_cim_dict()
        self.net: pandapower.auxiliary.pandapowerNet = pp.create_empty_network()
        self.bus_merge: pd.DataFrame = pd.DataFrame()
        self.power_trafo2w: pd.DataFrame = pd.DataFrame()
        self.power_trafo3w: pd.DataFrame = pd.DataFrame()
        self.report_container: ReportContainer = cim_parser.get_report_container()
        self.classes_dict = converter_classes

    def merge_eq_ssh_profile(self, cim_type: str, add_cim_type_column: bool = False) -> pd.DataFrame:
        df = pd.merge(self.cim['eq'][cim_type], self.cim['ssh'][cim_type], how='left', on='rdfId')
        if add_cim_type_column:
            df[sc['o_cl']] = cim_type
        return df

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
        self.classes_dict['ConnectivityNodesCim16'](cimConverter=self).convert_connectivity_nodes_cim16()
        # --------- convert external networks ---------
        self.classes_dict['externalNetworkInjectionsCim16'](
            cimConverter=self).convert_external_network_injections_cim16()
        # --------- convert lines ---------
        self.classes_dict['acLineSegmentsCim16'](cimConverter=self).convert_ac_line_segments_cim16(
            convert_line_to_switch, line_r_limit, line_x_limit)
        self.classes_dict['dcLineSegmentsCim16'](cimConverter=self).convert_dc_line_segments_cim16()
        # --------- convert switches ---------
        self.classes_dict['switchesCim16'](cimConverter=self).convert_switches_cim16()
        # --------- convert loads ---------
        self.classes_dict['energyConcumersCim16'](cimConverter=self).convert_energy_consumers_cim16()
        self.classes_dict['conformLoadsCim16'](cimConverter=self).convert_conform_loads_cim16()
        self.classes_dict['nonConformLoadsCim16'](cimConverter=self).convert_non_conform_loads_cim16()
        self.classes_dict['stationSuppliesCim16'](cimConverter=self).convert_station_supplies_cim16()
        # --------- convert generators ---------
        self.classes_dict['synchronousMachinesCim16'](cimConverter=self).convert_synchronous_machines_cim16()
        self.classes_dict['asynchronousMachinesCim16'](cimConverter=self).convert_asynchronous_machines_cim16()
        self.classes_dict['energySourcesCim16'](cimConverter=self).convert_energy_sources_cim16()
        # --------- convert shunt elements ---------
        self.classes_dict['linearShuntCompensatorCim16'](cimConverter=self).convert_linear_shunt_compensator_cim16()
        self.classes_dict['nonLinearShuntCompensatorCim16'](
            cimConverter=self).convert_nonlinear_shunt_compensator_cim16()
        self.classes_dict['staticVarCompensatorCim16'](cimConverter=self).convert_static_var_compensator_cim16()
        # --------- convert impedance elements ---------
        self.classes_dict['equivalentBranchesCim16'](cimConverter=self).convert_equivalent_branches_cim16()
        self.classes_dict['seriesCompensatorsCim16'](cimConverter=self).convert_series_compensators_cim16()
        # --------- convert extended ward and ward elements ---------
        self.classes_dict['equivalentInjectionsCim16'](cimConverter=self).convert_equivalent_injections_cim16()
        # --------- convert transformers ---------
        self.classes_dict['powerTransformersCim16'](cimConverter=self).convert_power_transformers_cim16()

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
                self.classes_dict['geoCoordinatesFromGLCim16'](cimConverter=self).add_geo_coordinates_from_gl_cim16()
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
                self.classes_dict['coordinatesFromDLCim16'](cimConverter=self).add_coordinates_from_dl_cim16(
                    diagram_name=kwargs.get('diagram_name', None))
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
        self.classes_dict['tapController'](cimConverter=self).create_tap_controller_for_power_transformers()

        self.logger.info("Running a power flow.")
        self.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO, message="Running a power flow."))
        if kwargs.get('run_powerflow', False):
            try:
                pp.runpp(self.net)
            except Exception as e:
                self.logger.error("Failed running a powerflow.")
                self.logger.exception(e)
                self.report_container.add_log(Report(
                    level=LogLevel.ERROR, code=ReportCode.ERROR, message="Failed running a powerflow."))
                self.report_container.add_log(Report(level=LogLevel.EXCEPTION, code=ReportCode.EXCEPTION,
                                                     message=traceback.format_exc()))
                if not kwargs.get('ignore_errors', True):
                    raise e
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
            if not kwargs.get('ignore_errors', True):
                raise e
        # a special fix for BB and NB mixed networks:
        # fuse boundary ConnectivityNodes with their TopologicalNodes
        bus_t = self.net.bus.reset_index(level=0, drop=False)
        bus_drop = bus_t.loc[bus_t[sc['o_prf']] == 'eq_bd', ['index', sc['o_id'], 'cim_topnode']]
        bus_drop = bus_drop.rename(columns={'index': 'b1'})
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

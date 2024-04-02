# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from __future__ import annotations
import logging
import os
import re
import tempfile
import zipfile
from typing import Dict, List
import pandas as pd
import numpy as np
import xml.etree.ElementTree
import xml.etree.cElementTree as xmlET
from .other_classes import ReportContainer, Report, LogLevel, ReportCode
from .cim_tools import get_cim16_schema


class CimParser:

    def __init__(self, cim: Dict[str, Dict[str, pd.DataFrame]] = None):
        """
        This class parses CIM files and loads its content to a dictionary of
        CIM profile (dict) -> CIM element type (str) -> CIM elements (DataFrame)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cim: Dict[str, Dict[str, pd.DataFrame]] = cim if cim is not None else self.get_cim_data_structure()
        self.file_names: Dict[str, str] = dict()
        self.report_container = ReportContainer()

    def parse_files(self, file_list: List[str] or str = None, encoding: str = 'utf-8', prepare_cim_net: bool = False,
                    set_data_types: bool = False) -> CimParser:
        """
        Parse CIM XML files from a storage.

        :param file_list: The path to the CGMES files as a list. Note: The files need a FullModel to parse the
        CGMES profile. Optional, default: None.
        :param encoding: The encoding from the files. Optional, default: utf-8
        :param prepare_cim_net: Set this parameter to True to prepare the parsed cim data according to the
        CimConverter. Optional, default: False
        :param set_data_types: Set this parameter to True to set the cim data types at the parsed data. Optional,
        default: False
        :return: Self
        """
        self.logger.info("Start parsing CIM files.")
        self.report_container.add_log(Report(level=LogLevel.INFO, code=ReportCode.INFO_PARSING,
                                             message="CIM parser starts parsing CIM files."))
        if file_list is not None:
            if isinstance(file_list, list):
                for file in file_list:
                    self._parse_source_file(file=file, output=self.cim, encoding=encoding)
            else:
                self._parse_source_file(file=file_list, output=self.cim, encoding=encoding)

        if prepare_cim_net:
            self.prepare_cim_net()
        if set_data_types:
            self.set_cim_data_types()
        self.logger.info("Finished parsing CIM files.")
        self.report_container.add_log(Report(level=LogLevel.INFO, code=ReportCode.INFO_PARSING,
                                      message="CIM parser finished parsing CIM files."))
        return self

    def set_cim_data_types(self) -> CimParser:
        """
        Set the data types from the columns from the DataFrames for each CIM element type and profile. Note: Currently
        only elements required for the CGMES converter are set.
        """
        self.logger.info("Setting the cim data types.")
        default_values = dict(
            {'positiveFlowIn': True, 'connected': True, 'length': 1., 'sections': 1, 'maximumSections': 1,
             'referencePriority': 999999, 'gch': 0., 'g0ch': 0.})  # todo check gch g0ch sections maximumSections
        to_bool = dict({'True': True, 'true': True, 'TRUE': True, True: True,
                        'False': False, 'false': False, 'FALSE': False, False: False,
                        'nan': False, 'NaN': False, 'NAN': False, 'Nan': False, np.NaN: False})
        float_type = float
        int_type = pd.Int64Dtype()
        bool_type = pd.BooleanDtype()
        data_types_map = dict({'Float': float_type, 'Integer': int_type, 'Boolean': bool_type})
        cim_16_schema = get_cim16_schema()
        for profile in self.cim.keys():
            for cim_element_type, item in self.cim[profile].items():
                for col in item.columns:
                    # skip elements which are not available in the schema like FullModel
                    if cim_element_type not in cim_16_schema[profile]:
                        self.logger.debug("Skipping CIM element type %s from profile %s." % (cim_element_type, profile))
                        continue
                    if col in cim_16_schema[profile][cim_element_type]['fields'].keys() and \
                            'data_type_prim' in cim_16_schema[profile][cim_element_type]['fields'][col].keys():
                        data_type_col_str = cim_16_schema[profile][cim_element_type]['fields'][col]['data_type_prim']
                        if data_type_col_str in data_types_map.keys():
                            data_type_col = data_types_map[data_type_col_str]
                        else:
                            continue
                        self.logger.debug("Setting data type of %s from CIM element %s as type %s" %
                                          (col, cim_element_type, data_type_col_str))
                        if col in default_values.keys():  # todo deprecated due to repair function?
                            self.cim[profile][cim_element_type][col] = self.cim[profile][cim_element_type][col].fillna(value=default_values[col])
                        if data_type_col == bool_type:
                            self.cim[profile][cim_element_type][col] = \
                                self.cim[profile][cim_element_type][col].map(to_bool)
                        try:
                            # special fix for integers:
                            if data_type_col == int_type:
                                self.cim[profile][cim_element_type][col] = \
                                    self.cim[profile][cim_element_type][col].astype(float_type)
                            self.cim[profile][cim_element_type][col] = \
                                self.cim[profile][cim_element_type][col].astype(data_type_col)
                        except Exception as e:
                            self.logger.warning("Couldn't set the datatype to %s for field %s at CIM type %s in "
                                                "profile %s!" % (data_type_col_str, col, cim_element_type, profile))
                            self.logger.warning("This may be harmless if the data is not need by the converter. "
                                                "Message: %s" % e)
        self.logger.info("Finished setting the cim data types.")
        self.report_container.add_log(Report(level=LogLevel.INFO, code=ReportCode.INFO_PARSING,
                                             message="CIM parser set the data types from the CIM data."))
        return self

    def prepare_cim_net(self) -> CimParser:
        """
        Make sure that the cim dictionaries only consists of valid DataFrames for each cim element type and append
        missing columns (not set but required CIM fields).
        """
        self.logger.info("Start preparing the cim data.")
        cim_data_structure = self.get_cim_data_structure()
        for profile in list(self.cim.keys()):
            if profile not in cim_data_structure.keys():
                # this profile is not used by the converter, drop it
                del self.cim[profile]
                continue
            for cim_element_type in list(self.cim[profile].keys()):
                # check if the CIM element type is a pd.DataFrame
                if not isinstance(self.cim[profile][cim_element_type], pd.DataFrame):
                    if profile in cim_data_structure.keys() and cim_element_type in cim_data_structure[profile].keys():
                        # replace the cim element type with the default empty DataFrame from the cim_data_structure
                        self.cim[profile][cim_element_type] = cim_data_structure[profile][cim_element_type]
                    else:
                        # this cim element type is not used by the converter, drop it
                        del self.cim[profile][cim_element_type]
                    self.logger.warning("%s isn't a DataFrame! The data won't be used!" % cim_element_type)

        # append missing columns to the CIM net
        for profile in cim_data_structure.keys():
            if profile not in self.cim.keys():
                self.cim[profile] = cim_data_structure[profile]
                continue
            for cim_element_type, item in cim_data_structure[profile].items():
                if cim_element_type not in self.cim[profile].keys():
                    self.cim[profile][cim_element_type] = cim_data_structure[profile][cim_element_type]
                    continue
                for column in item.columns:
                    if column not in self.cim[profile][cim_element_type].columns:
                        self.logger.info("Adding missing column %s to CIM element %s" % (column, cim_element_type))
                        self.cim[profile][cim_element_type][column] = np.NaN

        # now remove columns which are not needed by the converter (to avoid renaming problems when merging DataFrames)
        for profile in cim_data_structure.keys():
            for cim_element_type in cim_data_structure[profile].keys():
                self.cim[profile][cim_element_type] = \
                    self.cim[profile][cim_element_type][cim_data_structure[profile][cim_element_type].columns]
        self.logger.info("Finished preparing the cim data.")
        self.report_container.add_log(Report(level=LogLevel.INFO, code=ReportCode.INFO_PARSING,
                                             message="CIM parser finished preparing the CIM data."))
        return self

    def get_cim_data_structure(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
           Get the cim data structure used by the converter.
           :return Dict[str, Dict[str, pd.DataFrame]]: The cim data structure used by the converter.
           """
        self.logger.debug("Returning the CIM data structure.")
        return dict({
            'eq': {
                'ControlArea': pd.DataFrame(columns=['rdfId', 'name', 'type']),
                'TieFlow': pd.DataFrame(columns=['rdfId', 'Terminal', 'ControlArea', 'positiveFlowIn']),
                'ConnectivityNode': pd.DataFrame(columns=['rdfId', 'name', 'description', 'ConnectivityNodeContainer']),
                'Bay': pd.DataFrame(columns=['rdfId', 'VoltageLevel']),
                'BusbarSection': pd.DataFrame(columns=['rdfId', 'name']),
                'Substation': pd.DataFrame(columns=['rdfId', 'name', 'Region']),
                'GeographicalRegion': pd.DataFrame(columns=['rdfId', 'name']),
                'SubGeographicalRegion': pd.DataFrame(columns=['rdfId', 'name', 'Region']),
                'VoltageLevel': pd.DataFrame(columns=['rdfId', 'name', 'shortName', 'BaseVoltage', 'Substation']),
                'BaseVoltage': pd.DataFrame(columns=['rdfId', 'name', 'nominalVoltage']),
                'ExternalNetworkInjection': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'minP', 'maxP', 'minQ', 'maxQ', 'BaseVoltage', 'EquipmentContainer',
                    'RegulatingControl', 'governorSCD', 'maxInitialSymShCCurrent', 'minInitialSymShCCurrent',
                    'maxR1ToX1Ratio', 'minR1ToX1Ratio', 'maxR0ToX0Ratio', 'maxZ0ToZ1Ratio']),
                'ACLineSegment': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'length', 'r', 'x', 'bch', 'gch', 'r0', 'x0', 'b0ch', 'g0ch',
                    'shortCircuitEndTemperature', 'BaseVoltage']),
                'Terminal': pd.DataFrame(columns=[
                    'rdfId', 'name', 'ConnectivityNode', 'ConductingEquipment', 'sequenceNumber']),
                'OperationalLimitSet': pd.DataFrame(columns=['rdfId', 'name', 'Terminal']),
                'OperationalLimitType': pd.DataFrame(columns=['rdfId', 'name', 'limitType']),
                'CurrentLimit': pd.DataFrame(columns=[
                    'rdfId', 'name', 'OperationalLimitSet', 'OperationalLimitType', 'value']),
                'VoltageLimit': pd.DataFrame(columns=[
                    'rdfId', 'name', 'OperationalLimitSet', 'OperationalLimitType', 'value']),
                'DCNode': pd.DataFrame(columns=['rdfId', 'name', 'DCEquipmentContainer']),
                'DCEquipmentContainer': pd.DataFrame(columns=['rdfId', 'name']),
                'DCConverterUnit': pd.DataFrame(columns=['rdfId', 'name', 'Substation', 'operationMode']),
                'DCLineSegment': pd.DataFrame(columns=['rdfId', 'name', 'description', 'EquipmentContainer']),
                'CsConverter': pd.DataFrame(columns=['rdfId', 'BaseVoltage', 'ratedUdc']),
                'VsConverter': pd.DataFrame(columns=['rdfId', 'name', 'BaseVoltage', 'EquipmentContainer', 'ratedUdc']),
                'DCTerminal': pd.DataFrame(columns=[
                    'rdfId', 'name', 'DCNode', 'DCConductingEquipment', 'sequenceNumber']),
                'ACDCConverterDCTerminal': pd.DataFrame(columns=[
                    'rdfId', 'name', 'DCNode', 'DCConductingEquipment', 'sequenceNumber']),
                'Breaker': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'EquipmentContainer', 'normalOpen', 'retained']),
                'Disconnector': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'EquipmentContainer', 'normalOpen', 'retained']),
                'Switch': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'EquipmentContainer', 'normalOpen', 'retained']),
                'LoadBreakSwitch': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'EquipmentContainer', 'normalOpen', 'retained']),
                'EnergyConsumer': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'BaseVoltage', 'EquipmentContainer']),
                'ConformLoad': pd.DataFrame(columns=['rdfId', 'name', 'description']),
                'NonConformLoad': pd.DataFrame(columns=['rdfId', 'name', 'description']),
                'StationSupply': pd.DataFrame(columns=['rdfId', 'name', 'description', 'BaseVoltage']),
                'GeneratingUnit': pd.DataFrame(columns=[
                    'rdfId', 'name', 'nominalP', 'initialP', 'minOperatingP', 'maxOperatingP', 'EquipmentContainer']),
                'WindGeneratingUnit': pd.DataFrame(columns=['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']),
                'HydroGeneratingUnit': pd.DataFrame(columns=['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']),
                'SolarGeneratingUnit': pd.DataFrame(columns=['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']),
                'ThermalGeneratingUnit': pd.DataFrame(columns=['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']),
                'NuclearGeneratingUnit': pd.DataFrame(columns=['rdfId', 'nominalP', 'minOperatingP', 'maxOperatingP']),
                'RegulatingControl': pd.DataFrame(columns=['rdfId', 'name', 'mode', 'Terminal']),
                'SynchronousMachine': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'GeneratingUnit', 'EquipmentContainer', 'ratedU', 'ratedS', 'type',
                    'r2', 'x2', 'ratedPowerFactor', 'voltageRegulationRange', 'minQ', 'maxQ', 'RegulatingControl']),
                'AsynchronousMachine': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'GeneratingUnit', 'ratedS', 'ratedU', 'ratedPowerFactor',
                    'rxLockedRotorRatio', 'iaIrRatio', 'efficiency', 'ratedMechanicalPower']),
                'EnergySource': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'nominalVoltage', 'EnergySchedulingType', 'BaseVoltage',
                    'EquipmentContainer', 'voltageAngle', 'voltageMagnitude']),
                'EnergySchedulingType': pd.DataFrame(columns=['rdfId', 'name']),
                'StaticVarCompensator': pd.DataFrame(columns=['rdfId', 'name', 'description', 'voltageSetPoint']),
                'PowerTransformer': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'EquipmentContainer', 'isPartOfGeneratorUnit']),
                'PowerTransformerEnd': pd.DataFrame(columns=[
                    'rdfId', 'name', 'PowerTransformer', 'endNumber', 'Terminal', 'ratedS', 'ratedU',
                    'r', 'x', 'r0', 'x0', 'b', 'g', 'BaseVoltage', 'phaseAngleClock', 'connectionKind', 'grounded',
                    'xground']),
                'TapChangerControl': pd.DataFrame(columns=['rdfId', 'name', 'mode', 'Terminal']),
                'RatioTapChanger': pd.DataFrame(columns=[
                    'rdfId', 'name', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'stepVoltageIncrement',
                    'neutralU', 'normalStep', 'ltcFlag', 'tculControlMode', 'TapChangerControl',
                    'RatioTapChangerTable']),
                'PhaseTapChangerLinear': pd.DataFrame(columns=[
                    'rdfId', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'stepPhaseShiftIncrement',
                    'TapChangerControl']),
                'PhaseTapChangerAsymmetrical': pd.DataFrame(columns=[
                    'rdfId', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'voltageStepIncrement',
                    'TapChangerControl']),
                'PhaseTapChangerSymmetrical': pd.DataFrame(columns=[
                    'rdfId', 'TransformerEnd', 'neutralStep', 'lowStep', 'highStep', 'voltageStepIncrement',
                    'TapChangerControl']),
                'PhaseTapChangerTabular': pd.DataFrame(columns=[
                    'rdfId', 'TransformerEnd', 'PhaseTapChangerTable', 'highStep', 'lowStep', 'neutralStep',
                    'TapChangerControl']),
                'PhaseTapChangerTablePoint': pd.DataFrame(columns=[
                    'rdfId', 'PhaseTapChangerTable', 'step', 'angle', 'ratio', 'r', 'x']),
                'RatioTapChangerTable': pd.DataFrame(columns=['rdfId', 'TransformerEnd', 'RatioTapChangerTable',
                                                              'highStep', 'lowStep', 'neutralStep']),
                'RatioTapChangerTablePoint': pd.DataFrame(columns=['rdfId', 'RatioTapChangerTable', 'step',
                                                                   'r', 'x', 'ratio']),
                'LinearShuntCompensator': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'nomU', 'gPerSection', 'bPerSection', 'maximumSections']),
                'NonlinearShuntCompensator': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'nomU', 'maximumSections']),
                'NonlinearShuntCompensatorPoint': pd.DataFrame(columns=[
                    'rdfId', 'description', 'NonlinearShuntCompensator', 'sectionNumber', 'b', 'g']),
                'EquivalentBranch': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'BaseVoltage', 'r', 'x', 'r21', 'x21', 'zeroR12', 'zeroR21',
                    'zeroX12', 'zeroX21']),
                'EquivalentInjection': pd.DataFrame(columns=['rdfId', 'name', 'description', 'BaseVoltage', 'r', 'x']),
                'SeriesCompensator': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'BaseVoltage', 'r', 'x', 'r0', 'x0']),
                'Analog': pd.DataFrame(columns=[
                    'rdfId', 'name', 'measurementType', 'unitSymbol', 'unitMultiplier', 'Terminal',
                    'PowerSystemResource', 'positiveFlowIn']),
                'AnalogValue': pd.DataFrame(columns=[
                    'rdfId', 'name', 'sensorAccuracy', 'MeasurementValueSource', 'Analog', 'value']),
                'MeasurementValueSource': pd.DataFrame(columns=['rdfId', 'name'])
            },
            'eq_bd': {
                'ConnectivityNode': pd.DataFrame(columns=['rdfId', 'name', 'ConnectivityNodeContainer']),
                'BaseVoltage': pd.DataFrame(columns=['rdfId', 'name', 'nominalVoltage']),
                'Terminal': pd.DataFrame(
                    columns=['rdfId', 'ConnectivityNode', 'ConductingEquipment', 'sequenceNumber']),
                'EnergySource': pd.DataFrame(columns=['rdfId', 'name', 'nominalVoltage', 'EnergySchedulingType']),
                'EnergySchedulingType': pd.DataFrame(columns=['rdfId', 'name'])
            },
            'ssh': {
                'ControlArea': pd.DataFrame(columns=['rdfId', 'netInterchange']),
                'ExternalNetworkInjection': pd.DataFrame(columns=[
                    'rdfId', 'p', 'q', 'referencePriority', 'controlEnabled']),
                'Terminal': pd.DataFrame(columns=['rdfId', 'connected']),
                'DCTerminal': pd.DataFrame(columns=['rdfId', 'connected']),
                'ACDCConverterDCTerminal': pd.DataFrame(columns=['rdfId', 'connected']),
                'CsConverter': pd.DataFrame(columns=['rdfId', 'p', 'q']),
                'VsConverter': pd.DataFrame(columns=[
                    'rdfId', 'p', 'q', 'targetUpcc', 'droop', 'droopCompensation', 'qShare', 'targetUdc', 'targetPpcc',
                    'targetQpcc', 'pPccControl', 'qPccControl']),
                'Breaker': pd.DataFrame(columns=['rdfId', 'open']),
                'Disconnector': pd.DataFrame(columns=['rdfId', 'open']),
                'Switch': pd.DataFrame(columns=['rdfId', 'open']),
                'LoadBreakSwitch': pd.DataFrame(columns=['rdfId', 'open']),
                'EnergyConsumer': pd.DataFrame(columns=['rdfId', 'p', 'q']),
                'ConformLoad': pd.DataFrame(columns=['rdfId', 'p', 'q']),
                'NonConformLoad': pd.DataFrame(columns=['rdfId', 'p', 'q']),
                'StationSupply': pd.DataFrame(columns=['rdfId', 'p', 'q']),
                'RegulatingControl': pd.DataFrame(columns=[
                    'rdfId', 'discrete', 'enabled', 'targetValue', 'targetValueUnitMultiplier']),
                'SynchronousMachine': pd.DataFrame(columns=[
                    'rdfId', 'p', 'q', 'referencePriority', 'operatingMode', 'controlEnabled']),
                'AsynchronousMachine': pd.DataFrame(columns=['rdfId', 'p', 'q']),
                'EnergySource': pd.DataFrame(columns=['rdfId', 'activePower', 'reactivePower']),
                'StaticVarCompensator': pd.DataFrame(columns=['rdfId', 'q']),
                'TapChangerControl': pd.DataFrame(columns=[
                    'rdfId', 'discrete', 'enabled', 'targetValue', 'targetValueUnitMultiplier', 'targetDeadband']),
                'RatioTapChanger': pd.DataFrame(columns=['rdfId', 'step', 'controlEnabled']),
                'PhaseTapChangerLinear': pd.DataFrame(columns=['rdfId', 'step']),
                'PhaseTapChangerAsymmetrical': pd.DataFrame(columns=['rdfId', 'step']),
                'PhaseTapChangerSymmetrical': pd.DataFrame(columns=['rdfId', 'step']),
                'PhaseTapChangerTabular': pd.DataFrame(columns=['rdfId', 'step']),
                'LinearShuntCompensator': pd.DataFrame(columns=['rdfId', 'controlEnabled', 'sections']),
                'NonlinearShuntCompensator': pd.DataFrame(columns=['rdfId', 'controlEnabled', 'sections']),
                'EquivalentInjection': pd.DataFrame(columns=['rdfId', 'regulationTarget', 'regulationStatus', 'p', 'q'])
            },
            'sv': {
                'SvVoltage': pd.DataFrame(columns=['rdfId', 'TopologicalNode', 'v', 'angle']),
                'SvPowerFlow': pd.DataFrame(columns=['rdfId', 'Terminal', 'p', 'q']),
                'SvShuntCompensatorSections': pd.DataFrame(columns=['rdfId', 'ShuntCompensator', 'sections']),
                'SvTapStep': pd.DataFrame(columns=['rdfId', 'TapChanger', 'position'])
            },
            'tp': {
                'TopologicalNode': pd.DataFrame(columns=[
                    'rdfId', 'name', 'description', 'ConnectivityNodeContainer', 'BaseVoltage']),
                'DCTopologicalNode': pd.DataFrame(columns=['rdfId', 'name', 'DCEquipmentContainer']),
                'ConnectivityNode': pd.DataFrame(columns=['rdfId', 'TopologicalNode']),
                'Terminal': pd.DataFrame(columns=['rdfId', 'TopologicalNode']),
                'DCTerminal': pd.DataFrame(columns=['rdfId', 'DCTopologicalNode']),
                'ACDCConverterDCTerminal': pd.DataFrame(columns=['rdfId', 'DCTopologicalNode'])
            },
            'tp_bd': {
                'TopologicalNode': pd.DataFrame(columns=['rdfId', 'name', 'ConnectivityNodeContainer', 'BaseVoltage']),
                'ConnectivityNode': pd.DataFrame(columns=['rdfId', 'TopologicalNode'])
            },
            'dl': {
                'Diagram': pd.DataFrame(columns=['rdfId', 'name']),
                'DiagramObject': pd.DataFrame(columns=['rdfId', 'IdentifiedObject', 'Diagram', 'name']),
                'DiagramObjectPoint': pd.DataFrame(columns=[
                    'rdfId', 'sequenceNumber', 'xPosition', 'yPosition', 'DiagramObject'])},
            'gl': {
                'CoordinateSystem': pd.DataFrame(columns=['rdfId', 'name', 'crsUrn']),
                'Location': pd.DataFrame(columns=['rdfId', 'PowerSystemResources', 'CoordinateSystem']),
                'PositionPoint': pd.DataFrame(columns=['rdfId', 'Location', 'sequenceNumber', 'xPosition', 'yPosition'])
            }})

    def _parse_element(self, element, parsed=None):
        if parsed is None:
            parsed = dict()
        for key in element.keys():
            combined_key = element.tag + '-' + key
            if combined_key not in parsed:
                parsed[combined_key] = element.attrib.get(key)
            else:
                if not isinstance(parsed[combined_key], list):
                    parsed[combined_key] = [parsed[combined_key]]
                parsed[combined_key].append(element.attrib.get(key))
        if element.tag not in parsed and element.text is not None and element.text.strip(' \t\n\r'):
            parsed[element.tag] = element.text
        for child in list(element):
            self._parse_element(child, parsed)
        return parsed

    def _get_df(self, items):
        return pd.DataFrame([self._parse_element(child) for child in iter(items)])

    def _get_cgmes_profile_from_xml(self, root: xml.etree.ElementTree.Element, ignore_errors: bool = False,
                                    default_profile: str = 'unknown') -> str:
        """
        Get the CGMES profile from the XML file.

        :param root: The root element from the XML tree
        :param ignore_errors: Ignore errors and return a profile version if possible. If no profile is readable,
        the content from the parameter default will be returned. Optional, default: False
        :param default_profile: The default profile name which will be returned if ignore_errors is set to True.
        Optional, default: 'unknown'
        :return: The profile in short from: 'eq' for Equipment, 'eq_bd' for EquipmentBoundary,
        'ssh' for SteadyStateHypothesis, 'sv' for StateVariables,
        'tp' for Topology, 'tp_bd' for TopologyBoundary
        """
        element_types = pd.Series([ele.tag for ele in list(root)])
        element_types.drop_duplicates(inplace=True)
        full_model = element_types.str.find('FullModel')
        if full_model.max() >= 0:
            full_model = element_types[full_model >= 0].values[0]
        else:
            full_model = 'FullModel'
        full_model_profile = full_model[:-9] + 'Model.profile'
        full_model_df = self._get_df(root.findall('.//' + full_model))
        if full_model_df.index.size == 0 and ignore_errors:
            self.logger.warning("The FullModel is not given in the XML tree, returning %s" % default_profile)
            return default_profile
        elif full_model_df.index.size == 0:
            raise Exception("The FullModel is not given in the XML tree.")
        if full_model_df.index.size > 1 and ignore_errors:
            self.logger.warning("It is more than one FullModel given, returning the profile from the first FullModel.")
        elif full_model_df.index.size > 1:
            raise Exception("It is more than one FullModel given.")
        if full_model_profile not in full_model_df.columns and ignore_errors:
            self.logger.warning("The profile is not given in the FullModel, returning %s" % default_profile)
            return default_profile
        elif full_model_profile not in full_model_df.columns:
            raise Exception("The profile is not given in the FullModel.")
        profile_list = full_model_df[full_model_profile].values[0]
        if not isinstance(profile_list, list):
            profile_list = [profile_list]
        for one_profile in profile_list:
            if '/EquipmentCore/' in one_profile or '/EquipmentOperation/' in one_profile or \
                    '/EquipmentShortCircuit/' in one_profile:
                return 'eq'
            elif '/SteadyStateHypothesis/' in one_profile:
                return 'ssh'
            elif '/StateVariables/' in one_profile:
                return 'sv'
            elif '/Topology/' in one_profile:
                return 'tp'
            elif '/DiagramLayout/' in one_profile:
                return 'dl'
            elif '/GeographicalLocation/' in one_profile:
                return 'gl'
            elif '/EquipmentBoundary/' in one_profile or '/EquipmentBoundaryOperation/' in one_profile:
                return 'eq_bd'
            elif '/TopologyBoundary/' in one_profile:
                return 'tp_bd'
        if ignore_errors:
            self.logger.warning("The CGMES profile could not be parsed from the XML, returning %s" % default_profile)
            self.report_container.add_log(Report(level=LogLevel.ERROR, code=ReportCode.ERROR_PARSING,
                                                 message="The CGMES profile could not be parsed from the XML, "
                                                         "returning %s" % default_profile))
            return default_profile
        else:
            self.report_container.add_log(Report(level=LogLevel.ERROR, code=ReportCode.ERROR_PARSING,
                                                 message="The CGMES profile could not be parsed from the XML."))
            raise Exception("The CGMES profile could not be parsed from the XML.")

    def _parse_source_file(self, file: str, output: dict, encoding: str, profile_name: str = None):
        self.logger.info("Parsing file: %s" % file)
        if not self._check_file(file):
            return
        # check if the file is a zip archive
        if file.lower().endswith('.zip'):
            # extract the zip in a temporary folder and delete it later
            temp_dir = tempfile.TemporaryDirectory()
            temp_dir_path = os.path.realpath(temp_dir.name)
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(temp_dir_path)
            # parse the extracted CIM files
            for temp_file in os.listdir(temp_dir_path):
                temp_file = os.path.join(temp_dir_path, temp_file)
                if os.path.isfile(temp_file):
                    self._parse_source_file(temp_file, output=output, encoding=encoding)
                elif os.path.isdir(temp_file):
                    for sub_temp_file in os.listdir(temp_file):
                        sub_temp_file = os.path.join(temp_file, sub_temp_file)
                        self._parse_source_file(sub_temp_file, output=output, encoding=encoding)
            temp_dir.cleanup()
            del temp_dir, temp_dir_path
            return
        with open(file, mode='r', encoding=encoding, errors='ignore') as f:
            cim_str = f.read()
        xml_tree = xmlET.fromstring(cim_str)
        if profile_name is None:
            prf = self._get_cgmes_profile_from_xml(xml_tree)
        else:
            prf = profile_name
        self.file_names[prf] = file
        # get all CIM elements to parse
        element_types = pd.Series([ele.tag for ele in list(xml_tree)])
        element_types.drop_duplicates(inplace=True)
        prf_content: Dict[str, pd.DataFrame] = dict()
        ns_dict = dict()
        if prf not in ns_dict.keys():
            ns_dict[prf] = dict()
        for _, element_type in element_types.items():
            element_type_c = re.sub('{.*}', '', element_type)
            prf_content[element_type_c] = self._get_df(xml_tree.findall(element_type))
            # rename the columns (remove the namespaces)
            if element_type_c not in ns_dict[prf].keys():
                ns_dict[prf][element_type_c] = dict()
            for col in prf_content[element_type_c].columns:
                col_new = re.sub('[{].*?[}]', '', col)
                col_new = col_new.split('.')[-1]
                if col_new.endswith('-resource'):
                    col_new = col_new[:-9]
                    # remove the first character of each value if col_new is a CGMES class, e.g. Terminal
                    # other wise remove the namespace from the literals (e.g. col_new is unitMultiplier, then the
                    # value is like http://iec.ch/TC57/2013/CIM-schema-cim16#UnitMultiplier.M
                    if col_new[0].isupper():
                        prf_content[element_type_c][col] = prf_content[element_type_c][col].str[1:]
                    elif prf_content[element_type_c][col].index.size > 0:
                        # get the namespace from the literal, Note: get the largest string because some values could
                        # be nan
                        name_space = \
                            prf_content[element_type_c][col].values[prf_content[element_type_c][col].str.len().idxmax()]
                        # remove the namespace from the literal
                        prf_content[element_type_c][col] = \
                            prf_content[element_type_c][col].str[name_space.rfind('.')+1:]
                elif col_new.endswith('-about'):
                    col_new = 'rdfId'
                    prf_content[element_type_c][col] = prf_content[element_type_c][col].str[1:]
                elif col_new.endswith('-ID'):
                    col_new = 'rdfId'
                ns_dict[prf][element_type_c][col] = col_new
            prf_content[element_type_c] = prf_content[element_type_c].rename(columns={**ns_dict[prf][element_type_c]})
        if prf not in output.keys():
            output[prf] = prf_content
        else:
            for ele, df in prf_content.items():
                if ele not in output[prf].keys():
                    output[prf][ele] = pd.DataFrame()
                output[prf][ele] = pd.concat([output[prf][ele], prf_content[ele]], ignore_index=True, sort=False)

    def _check_file(self, file: str) -> bool:
        if not os.path.isfile(file):
            self.logger.error("%s is not a valid file!" % file)
            self.report_container.add_log(Report(level=LogLevel.ERROR, code=ReportCode.ERROR_PARSING,
                                                 message="%s is not a valid file!" % file))
            return False
        elif file.lower().endswith('xml') or file.lower().endswith('rdf') or file.lower().endswith('zip'):
            return True
        else:
            return False

    def get_cim_dict(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        return self.cim

    def set_cim_dict(self, cim: Dict[str, Dict[str, pd.DataFrame]]):
        self.cim = cim

    def get_file_names(self) -> Dict[str, str]:
        return self.file_names

    def get_report_container(self) -> ReportContainer:
        return self.report_container

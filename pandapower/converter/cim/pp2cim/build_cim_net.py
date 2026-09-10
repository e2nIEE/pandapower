# -*- coding: utf-8 -*-

"""Reconstruction of the CIM/CGMES data structure from a pandapower net.

:class:`PpToCimConverter` walks the pandapower element tables and rebuilds the
corresponding CIM objects (TopologicalNode/ConnectivityNode, Terminals,
ACLineSegment, PowerTransformer, loads, generation, switches, shunts, tap
changers, operational limits, power-flow results, coordinates) profile by
profile. It relies on the CGMES identifiers preserved by the importer
(``origin_id``/``origin_class`` and the terminal/end/container references) so
that an imported net round-trips back to equivalent CGMES.
"""
# The converter is intentionally kept in one module: it mirrors the CGMES schema
# class-by-class, so a single cohesive file is easier to follow than an arbitrary split.
# pylint: disable=too-many-lines
from __future__ import annotations
import collections
import json
import logging
import math
import uuid
from typing import Dict, List

import numpy as np
import pandas as pd

from pandapower.auxiliary import pandapowerNet
from .. import cim_tools
from ..cim_classes import CimParser

sc = cim_tools.get_pp_net_special_columns_dict()

SUPPORTED_CGMES_VERSIONS = ('2.4.15', '3.0')

# Special net column carrying the CGMES RegulatingControl.enabled flag for an element.
REGULATING_CONTROL_ENABLED = 'RegulatingControl.enabled'

# CGMES profile URIs used to synthesize a FullModel header when the net does not carry one
# (i.e. the net was not imported from CGMES). For round-trip the headers come from net['CGMES'].
PROFILE_URI = {
    '2.4.15': {
        'eq': 'http://entsoe.eu/CIM/EquipmentCore/3/1',
        'ssh': 'http://entsoe.eu/CIM/SteadyStateHypothesis/1/1',
        'sv': 'http://entsoe.eu/CIM/StateVariables/4/1',
        'tp': 'http://entsoe.eu/CIM/Topology/4/1',
        'dl': 'http://entsoe.eu/CIM/DiagramLayout/3/1',
        'gl': 'http://entsoe.eu/CIM/GeographicalLocation/2/1',
    },
    '3.0': {
        'eq': 'http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0',
        'ssh': 'http://iec.ch/TC57/ns/CIM/SteadyStateHypothesis-EU/3.0',
        'sv': 'http://iec.ch/TC57/ns/CIM/StateVariables-EU/3.0',
        'tp': 'http://iec.ch/TC57/ns/CIM/Topology-EU/3.0',
        'dl': 'http://iec.ch/TC57/ns/CIM/DiagramLayout-EU/3.0',
        'gl': 'http://iec.ch/TC57/ns/CIM/GeographicalLocation-EU/3.0',
    },
}


def _new_uuid() -> str:
    return '_' + str(uuid.uuid4())


class PpToCimConverter:
    """
    Builds a CIM data structure (profile -> CIM element type -> DataFrame) from a pandapower net,
    reusing the CGMES identifiers preserved by the cim2pp converter (origin_id, terminals, topology).

    The result can be serialized to RDF/XML by :class:`pandapower.converter.cim.cim_writer.CimWriter`.

    Note: unlike the importer (which splits each element family into its own module under
    ``cim2pp/converter_classes``), the exporter keeps all conversions in a single class on purpose:
    they share a lot of cross-element state (terminals, base voltages, regulating controls, tap
    changers and operational limits are accumulated across several element types and emitted once at
    the end), which a per-class layout would make awkward.
    """

    def __init__(self, net: pandapowerNet, cgmes_version: str = '2.4.15', **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        if cgmes_version not in SUPPORTED_CGMES_VERSIONS:
            raise ValueError(f"Unsupported CGMES version {cgmes_version!r}, "
                             f"expected one of {SUPPORTED_CGMES_VERSIONS}")
        self.net = net
        self.cgmes_version = cgmes_version
        self.kwargs = kwargs
        # net['CGMES'] (FullModel headers + BaseVoltage), present when the net was imported from CGMES
        self._cgmes = net['CGMES'] if isinstance(net.get('CGMES', None), dict) else {}
        # a mutable, empty copy of the canonical CIM data structure (correct columns per class)
        blueprint = CimParser(cgmes_version=cgmes_version).get_cim_data_structure()
        self.cim: Dict[str, Dict[str, pd.DataFrame]] = {
            profile: {cls: df.copy() for cls, df in classes.items()} for profile, classes in blueprint.items()}

        # accumulators for the terminals shared across all converted elements
        self._terminals_eq: List[dict] = []
        self._terminals_tp: List[dict] = []
        self._terminals_ssh: List[dict] = []
        # accumulators for RegulatingControls (shared by ENIs and SynchronousMachines)
        self._reg_control_eq: List[dict] = []
        self._reg_control_ssh: List[dict] = []
        # accumulators for tap changers, keyed by CIM class (RatioTapChanger, PhaseTapChanger*)
        self._tc_eq: Dict[str, List[dict]] = collections.defaultdict(list)
        self._tc_ssh: Dict[str, List[dict]] = collections.defaultdict(list)
        # accumulators for tabular phase tap changers (PhaseTapChangerTabular + Table + TablePoints)
        self._tabular_eq: List[dict] = []
        self._tabular_ssh: List[dict] = []
        self._tabular_tables: List[dict] = []
        self._tabular_points: List[dict] = []
        # accumulators for table-based RatioTapChangers (RatioTapChangerTable + TablePoints)
        self._ratio_tables: List[dict] = []
        self._ratio_points: List[dict] = []
        # per-step characteristic rows (voltage_ratio / angle) grouped by id_characteristic
        self._char_by_id = self._index_characteristic_table()
        # accumulators for current limits (OperationalLimitSet + CurrentLimit + OperationalLimitType)
        self._op_limit_sets: List[dict] = []
        self._current_limits: List[dict] = []
        self._op_limit_type_rows: List[dict] = []
        # (limitType, acceptableDuration) -> OperationalLimitType rdfId (de-duplicates the types)
        self._op_limit_types: Dict[tuple, str] = {}

        # nominalVoltage -> BaseVoltage rdfId, built from net['CGMES']['BaseVoltage']
        self._voltage_to_bv: Dict[float, str] = {}
        # container objects rebuilt from the buses (de-duplicated by id). VoltageLevels carry the
        # BaseVoltage that the importer uses to give node-breaker buses their voltage, and both are
        # also needed to attach bus geo in the GL profile.
        self._voltage_levels: Dict[str, dict] = {}
        self._substations: Dict[str, dict] = {}
        self._sub_regions: Dict[str, dict] = {}
        self._geo_regions: Dict[str, dict] = {}

    # ------------------------------------------------------------------ orchestration

    def convert_to_cim(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run all element converters and return the CIM data structure.

        :return: The CIM data structure (profile -> CIM element type -> DataFrame).
        """
        self.logger.info("Start converting the pandapower net to CIM.")
        self._create_full_model()
        self._create_base_voltages()
        self._convert_buses()
        self._convert_lines()
        self._convert_loads()
        self._convert_external_network_injections()
        self._convert_synchronous_machines()
        self._convert_energy_sources()
        self._convert_switches()
        self._convert_shunts()
        self._convert_impedances()
        self._convert_power_transformers()
        self._convert_power_transformers_3w()
        self._convert_equivalent_injections()
        self._convert_state_variables()
        self._convert_diagram_layout()
        self._convert_geographical_location()
        self._finalize_terminals()
        self._finalize_regulating_controls()
        self._finalize_tap_changers()
        self._finalize_operational_limits()
        self.logger.info("Finished converting the pandapower net to CIM.")
        return self.cim

    # ------------------------------------------------------------------ headers / base voltages

    def _create_full_model(self):
        for profile in ['eq', 'ssh', 'tp', 'sv']:
            rows = self._full_model_rows(profile, synthesize=profile in ('eq', 'ssh', 'tp'))
            if rows:
                self.cim[profile]['FullModel'] = pd.DataFrame(rows)

    def _full_model_rows(self, profile: str, synthesize: bool) -> List[dict]:
        """Build FullModel header rows for a profile, reusing net['CGMES'] when available and
        otherwise synthesizing a minimal header so the parser can re-detect the profile."""
        cgmes = self._cgmes
        if profile in cgmes and isinstance(cgmes[profile], dict) and len(cgmes[profile]) > 0:
            rows = []
            for rdf_id, fields in cgmes[profile].items():
                row = {'rdfId': rdf_id}
                row.update(fields)
                rows.append(row)
            return rows
        if synthesize:
            return [{'rdfId': str(uuid.uuid4()),
                     'profile': PROFILE_URI.get(self.cgmes_version, {}).get(profile)}]
        return []

    def _create_base_voltages(self):
        bv = self._cgmes.get('BaseVoltage')
        records = []
        if isinstance(bv, pd.DataFrame) and not bv.empty:
            for _, row in bv.iterrows():
                nominal = float(row['nominalVoltage'])
                records.append({'rdfId': row['rdfId'], 'nominalVoltage': nominal})
                self._voltage_to_bv.setdefault(round(nominal, 6), row['rdfId'])
        # add a BaseVoltage for every bus voltage that is not covered yet
        for vn_kv in self.net.bus['vn_kv'].dropna().unique():
            key = round(float(vn_kv), 6)
            if key not in self._voltage_to_bv:
                bv_id = _new_uuid()
                self._voltage_to_bv[key] = bv_id
                records.append({'rdfId': bv_id, 'nominalVoltage': float(vn_kv)})
        self._set('eq', 'BaseVoltage', records)

    def _base_voltage_id(self, vn_kv) -> str | None:
        if pd.isna(vn_kv):
            return None
        return self._voltage_to_bv.get(round(float(vn_kv), 6))

    # ------------------------------------------------------------------ buses / topology

    def _convert_buses(self):
        tp_nodes, eq_cns, tp_cns = [], [], []
        for _, bus in self.net.bus.iterrows():
            origin_id = bus.get(sc['o_id'])
            origin_class = bus.get(sc['o_cl'])
            if pd.isna(origin_id):
                origin_id = _new_uuid()
            bv_id = self._base_voltage_id(bus.get('vn_kv'))
            name = bus.get('name')
            description = bus.get('description')
            cnc_id = bus.get(sc['cnc_id'])
            self._add_bus_containers(bus, cnc_id, bv_id)
            if origin_class == 'ConnectivityNode':
                # node-breaker: the bus is a ConnectivityNode pointing to its TopologicalNode
                tn_id = bus.get(sc['ct'])
                eq_cns.append({'rdfId': origin_id, 'name': name, 'description': description,
                               'ConnectivityNodeContainer': cnc_id})
                if not pd.isna(tn_id):
                    tp_nodes.append({'rdfId': tn_id, 'name': name, 'description': description,
                                     'BaseVoltage': bv_id, 'ConnectivityNodeContainer': cnc_id})
                    tp_cns.append({'rdfId': origin_id, 'TopologicalNode': tn_id})
            else:
                # bus-branch: the bus is a TopologicalNode
                tp_nodes.append({'rdfId': origin_id, 'name': name, 'description': description,
                                 'BaseVoltage': bv_id, 'ConnectivityNodeContainer': cnc_id})
        self._set('tp', 'TopologicalNode', tp_nodes)
        self._set('eq', 'ConnectivityNode', eq_cns)
        self._set('tp', 'ConnectivityNode', tp_cns)
        self._set('eq', 'VoltageLevel', list(self._voltage_levels.values()))
        self._set('eq', 'Substation', list(self._substations.values()))
        self._set('eq', 'SubGeographicalRegion', list(self._sub_regions.values()))
        self._set('eq', 'GeographicalRegion', list(self._geo_regions.values()))

    def _add_bus_containers(self, bus, cnc_id, bv_id):
        # Rebuild the container hierarchy a bus belongs to: VoltageLevel (carrying the BaseVoltage) ->
        # Substation -> SubGeographicalRegion -> GeographicalRegion. The importer resolves a
        # node-breaker bus's voltage through ConnectivityNode -> VoltageLevel -> BaseVoltage, so these
        # must exist independently of the geo profile, and emitting the regions keeps the
        # Substation.Region reference resolvable.
        sub_id = bus.get(sc['sub_id'])
        sgr_id = bus.get('SubGeographicalRegion_id')
        gr_id = bus.get('GeographicalRegion_id')
        if not pd.isna(cnc_id):
            self._voltage_levels.setdefault(cnc_id, {
                'rdfId': cnc_id, 'name': bus.get('zone'), 'BaseVoltage': bv_id,
                'Substation': self._none_if_na(sub_id)})
        if not pd.isna(sub_id):
            self._substations.setdefault(sub_id, {
                'rdfId': sub_id, 'name': bus.get('zone'), 'Region': self._none_if_na(sgr_id)})
        if not pd.isna(sgr_id):
            self._sub_regions.setdefault(sgr_id, {
                'rdfId': sgr_id, 'name': bus.get('SubGeographicalRegion_name'),
                'Region': self._none_if_na(gr_id)})
        if not pd.isna(gr_id):
            self._geo_regions.setdefault(gr_id, {
                'rdfId': gr_id, 'name': bus.get('GeographicalRegion_name')})

    def _node_ref(self, bus_idx):
        """Return (connectivity_node_id, topological_node_id) for a pandapower bus index."""
        if pd.isna(bus_idx) or bus_idx not in self.net.bus.index:
            return None, None
        bus = self.net.bus.loc[bus_idx]
        origin_id = bus.get(sc['o_id'])
        if bus.get(sc['o_cl']) == 'ConnectivityNode':
            return origin_id, bus.get(sc['ct'])
        return None, origin_id  # bus-branch: only a TopologicalNode

    def _add_terminal(self, terminal_id, conducting_equipment, sequence_number, bus_idx, connected, name=None):
        if pd.isna(terminal_id):
            terminal_id = _new_uuid()
        cn_id, tn_id = self._node_ref(bus_idx)
        self._terminals_eq.append({'rdfId': terminal_id, 'name': name, 'ConnectivityNode': cn_id,
                                   'ConductingEquipment': conducting_equipment, 'sequenceNumber': sequence_number})
        self._terminals_tp.append({'rdfId': terminal_id, 'TopologicalNode': tn_id})
        self._terminals_ssh.append({'rdfId': terminal_id, 'connected': bool(connected)})
        return terminal_id

    def _finalize_terminals(self):
        self._set('eq', 'Terminal', self._terminals_eq)
        self._set('tp', 'Terminal', self._terminals_tp)
        self._set('ssh', 'Terminal', self._terminals_ssh)

    # ------------------------------------------------------------------ branches / injections

    def _convert_lines(self):
        eq_rows = []
        for _, line in self.net.line.iterrows():
            if line.get(sc['o_cl']) not in (None, 'ACLineSegment') and not pd.isna(line.get(sc['o_cl'])):
                # only ACLineSegment-derived lines are exported here (e.g. DCLineSegment is skipped)
                continue
            origin_id = self._id_or_new(line.get(sc['o_id']))
            length = float(line['length_km'])
            in_service = bool(line['in_service'])
            eq_rows.append({
                'rdfId': origin_id, 'name': line.get('name'), 'description': line.get('description'),
                'length': length,
                'r': line['r_ohm_per_km'] * length, 'x': line['x_ohm_per_km'] * length,
                'bch': line['c_nf_per_km'] * (2 * 50 * math.pi * length) / 1e9,
                'gch': line.get('g_us_per_km', 0.0) * length / 1e6,
                'r0': line.get('r0_ohm_per_km', np.nan) * length,
                'x0': line.get('x0_ohm_per_km', np.nan) * length,
                'b0ch': line.get('c0_nf_per_km', np.nan) * (2 * 50 * math.pi * length) / 1e9,
                'g0ch': line.get('g0_us_per_km', 0.0) * length / 1e6,
                'shortCircuitEndTemperature': line.get('endtemp_degree'),
                'EquipmentContainer': line.get('EquipmentContainer_id'),
            })
            self._add_terminal(line.get(sc['t_from']), origin_id, 1, line['from_bus'], in_service)
            self._add_terminal(line.get(sc['t_to']), origin_id, 2, line['to_bus'], in_service)
            max_i_ka = line.get('max_i_ka')
            self._add_current_limit(line.get(sc['t_from']),
                                    max_i_ka * 1e3 if not pd.isna(max_i_ka) else np.nan)
        self._set('eq', 'ACLineSegment', eq_rows)
        # emit a minimal Line container for each referenced EquipmentContainer that is not already a
        # Substation/VoltageLevel, so the ACLineSegment.EquipmentContainer reference resolves
        line_containers = {row['EquipmentContainer'] for row in eq_rows
                           if not pd.isna(row.get('EquipmentContainer'))
                           and row['EquipmentContainer'] not in self._voltage_levels
                           and row['EquipmentContainer'] not in self._substations}
        if line_containers:
            self.cim['eq'].setdefault('Line', pd.DataFrame(columns=['rdfId', 'name']))
            self._set('eq', 'Line', [{'rdfId': cid} for cid in line_containers])

    def _add_current_limit(self, terminal_id, value_a, limit_type='patl', acceptable_duration=None):
        # Reconstruct an OperationalLimitSet + CurrentLimit on the given terminal (value in Amperes).
        # OperationalLimitTypes are de-duplicated by (limitType, acceptableDuration).
        if terminal_id is None or pd.isna(terminal_id) or pd.isna(value_a):
            return
        limit_type = 'patl' if (limit_type is None or pd.isna(limit_type)) else limit_type
        duration = None if (acceptable_duration is None or pd.isna(acceptable_duration)) else acceptable_duration
        key = (limit_type, duration)
        if key not in self._op_limit_types:
            olt_id = _new_uuid()
            self._op_limit_types[key] = olt_id
            self._op_limit_type_rows.append({'rdfId': olt_id, 'name': limit_type, 'limitType': limit_type,
                                             'acceptableDuration': duration})
        ols_id = _new_uuid()
        self._op_limit_sets.append({'rdfId': ols_id, 'Terminal': terminal_id})
        self._current_limits.append({'rdfId': _new_uuid(), 'OperationalLimitSet': ols_id,
                                     'OperationalLimitType': self._op_limit_types[key], 'value': value_a})

    def _add_side_current_limit(self, trafo, terminal_id, side):
        # add the CurrentLimit a transformer carries on the given winding side (hv / mv / lv)
        self._add_current_limit(terminal_id, trafo.get(f'CurrentLimit.value_{side}'),
                                trafo.get(f'OperationalLimitType.limitType_{side}'),
                                trafo.get(f'OperationalLimitType.acceptableDuration_{side}'))

    def _finalize_operational_limits(self):
        self._set('eq', 'OperationalLimitType', self._op_limit_type_rows)
        self._set('eq', 'OperationalLimitSet', self._op_limit_sets)
        self._set('eq', 'CurrentLimit', self._current_limits)

    def _convert_loads(self):
        # EnergyConsumer / ConformLoad / NonConformLoad / StationSupply all map to the load table
        by_class = collections.defaultdict(list)
        ssh_rows = collections.defaultdict(list)
        for _, load in self.net.load.iterrows():
            origin_class = load.get(sc['o_cl'])
            if pd.isna(origin_class):
                origin_class = 'EnergyConsumer'
            origin_id = self._id_or_new(load.get(sc['o_id']))
            by_class[origin_class].append({
                'rdfId': origin_id, 'name': load.get('name'), 'description': load.get('description')})
            ssh_rows[origin_class].append({
                'rdfId': origin_id, 'p': float(load['p_mw']), 'q': float(load['q_mvar'])})
            self._add_terminal(load.get(sc['t']), origin_id, 1, load['bus'], bool(load['in_service']))
        for cls, rows in by_class.items():
            self._set('eq', cls, rows)
            self._set('ssh', cls, ssh_rows[cls])

    def _convert_external_network_injections(self):
        # An ExternalNetworkInjection may be imported as ext_grid, gen or sgen depending on its
        # control settings, so scan all three tables for ENI-derived rows.
        eq_rows, ssh_rows = [], []
        for table in ('ext_grid', 'gen', 'sgen'):
            for _, ext in self.net[table].iterrows():
                origin_class = ext.get(sc['o_cl'])
                # ext_grids default to an ExternalNetworkInjection when no origin is known (e.g. a
                # hand-built net); gen/sgen are only exported here when explicitly an ENI.
                if origin_class != 'ExternalNetworkInjection' and not (
                        table == 'ext_grid' and (origin_class is None or pd.isna(origin_class))):
                    continue
                origin_id = self._id_or_new(ext.get(sc['o_id']))
                terminal_id = self._add_terminal(ext.get(sc['t']), origin_id, 1, ext['bus'],
                                                 bool(ext['in_service']))
                reg_id = self._add_regulating_control(ext, terminal_id)
                eq_rows.append({
                    'rdfId': origin_id, 'name': ext.get('name'), 'description': ext.get('description'),
                    'minP': ext.get('min_p_mw'), 'maxP': ext.get('max_p_mw'),
                    'minQ': ext.get('min_q_mvar'), 'maxQ': ext.get('max_q_mvar'),
                    'RegulatingControl': reg_id})
                ssh_rows.append({
                    'rdfId': origin_id, 'p': self._neg(ext.get('p_mw')), 'q': self._neg(ext.get('q_mvar')),
                    'referencePriority': ext.get('referencePriority'),
                    'controlEnabled': self._as_bool(ext.get(REGULATING_CONTROL_ENABLED))})
        self._set('eq', 'ExternalNetworkInjection', eq_rows)
        self._set('ssh', 'ExternalNetworkInjection', ssh_rows)

    def _convert_synchronous_machines(self):
        # SynchronousMachine -> gen (voltage control) or sgen. Reconstruct the linked GeneratingUnit
        # and (when voltage controlled) the RegulatingControl so the importer reproduces the split.
        eq_sm, ssh_sm, eq_gu = [], [], []
        for table in ('gen', 'sgen'):
            for _, machine in self.net[table].iterrows():
                if machine.get(sc['o_cl']) != 'SynchronousMachine':
                    continue
                origin_id = self._id_or_new(machine.get(sc['o_id']))
                gu_id = _new_uuid()
                terminal_id = self._add_terminal(machine.get(sc['t']), origin_id, 1, machine['bus'],
                                                 bool(machine['in_service']))
                reg_id = self._add_regulating_control(machine, terminal_id)
                rated_s = machine.get('sn_mva')
                eq_gu.append({'rdfId': gu_id, 'name': machine.get('name'),
                              'nominalP': rated_s,
                              'minOperatingP': machine.get('min_p_mw'),
                              'maxOperatingP': machine.get('max_p_mw'),
                              'governorSCD': machine.get('governorSCD')})
                eq_sm.append({'rdfId': origin_id, 'name': machine.get('name'),
                              'description': machine.get('description'),
                              'GeneratingUnit': gu_id, 'ratedS': rated_s, 'ratedU': machine.get('vn_kv'),
                              'ratedPowerFactor': machine.get('cos_phi'),
                              'minQ': machine.get('min_q_mvar'), 'maxQ': machine.get('max_q_mvar'),
                              'RegulatingControl': reg_id})
                ssh_sm.append({'rdfId': origin_id, 'p': self._neg(machine.get('p_mw')),
                               'q': self._neg(machine.get('q_mvar')),
                               'referencePriority': machine.get('referencePriority'),
                               'controlEnabled': self._as_bool(machine.get(REGULATING_CONTROL_ENABLED))})
        self._set('eq', 'GeneratingUnit', eq_gu)
        self._set('eq', 'SynchronousMachine', eq_sm)
        self._set('ssh', 'SynchronousMachine', ssh_sm)

    def _convert_energy_sources(self):
        # EnergySource -> sgen, or ext_grid when it carries a voltage set point (the importer routes
        # EnergySources with a vm_pu to ext_grid and the rest to sgen). Setting voltageMagnitude /
        # voltageAngle reproduces that split.
        eq_rows, ssh_rows = [], []
        for table in ('sgen', 'ext_grid'):
            for _, es in self.net[table].iterrows():
                if es.get(sc['o_cl']) != 'EnergySource':
                    continue
                origin_id = self._id_or_new(es.get(sc['o_id']))
                eq_rows.append(self._energy_source_eq_row(es, origin_id))
                ssh_rows.append({'rdfId': origin_id, 'activePower': self._neg(es.get('p_mw')),
                                 'reactivePower': self._neg(es.get('q_mvar'))})
                self._add_terminal(es.get(sc['t']), origin_id, 1, es['bus'], bool(es['in_service']))
        self._set('eq', 'EnergySource', eq_rows)
        self._set('ssh', 'EnergySource', ssh_rows)

    def _energy_source_eq_row(self, es, origin_id) -> dict:
        # Build the EQ EnergySource row; a voltage set point (reproducing the importer's ext_grid
        # split) is only emitted when both the per-unit voltage and the bus base voltage are known.
        vn_kv = self.net.bus.loc[es['bus'], 'vn_kv'] if es['bus'] in self.net.bus.index else np.nan
        row = {'rdfId': origin_id, 'name': es.get('name'), 'description': es.get('description')}
        vm_pu = es.get('vm_pu')
        if pd.isna(vm_pu) or pd.isna(vn_kv):
            return row
        row['voltageMagnitude'] = vm_pu * vn_kv
        row['nominalVoltage'] = vn_kv
        va_degree = es.get('va_degree')
        if not pd.isna(va_degree):
            row['voltageAngle'] = va_degree * math.pi / 180
        return row

    def _add_regulating_control(self, row, terminal_id) -> str | None:
        """Reconstruct a RegulatingControl from the preserved RegulatingControl.* columns. Returns
        the new RegulatingControl rdfId, or None when the element has no regulation info."""
        mode = row.get('RegulatingControl.mode')
        enabled = row.get(REGULATING_CONTROL_ENABLED)
        target = row.get('RegulatingControl.targetValue')
        if pd.isna(mode) and pd.isna(enabled) and pd.isna(target):
            return None
        reg_id = _new_uuid()
        self._reg_control_eq.append({'rdfId': reg_id, 'mode': self._none_if_na(mode),
                                     'Terminal': terminal_id})
        self._reg_control_ssh.append({'rdfId': reg_id, 'enabled': self._as_bool(enabled),
                                      'targetValue': self._none_if_na(target)})
        return reg_id

    def _finalize_regulating_controls(self):
        self._set('eq', 'RegulatingControl', self._reg_control_eq)
        self._set('ssh', 'RegulatingControl', self._reg_control_ssh)

    def _convert_switches(self):
        # Breaker / Disconnector / LoadBreakSwitch / Switch all map to the pandapower switch table
        # (with et == 'b'). The switch connects two buses via two terminals.
        eq_by_class = collections.defaultdict(list)
        ssh_by_class = collections.defaultdict(list)
        for _, switch in self.net.switch.iterrows():
            if switch.get('et') != 'b':
                continue  # only bus-bus switches originate from CGMES switching devices
            origin_class = switch.get(sc['o_cl'])
            if pd.isna(origin_class):
                origin_class = 'Breaker'
            origin_id = self._id_or_new(switch.get(sc['o_id']))
            closed = bool(switch['closed'])
            in_ka = switch.get('in_ka')
            eq_by_class[origin_class].append({
                'rdfId': origin_id, 'name': switch.get('name'), 'description': switch.get('description'),
                'normalOpen': not closed, 'ratedCurrent': in_ka * 1e3 if not pd.isna(in_ka) else np.nan})
            ssh_by_class[origin_class].append({'rdfId': origin_id, 'open': not closed})
            # element is a bus (et == 'b')
            self._add_terminal(switch.get(sc['t_bus']), origin_id, 1, switch['bus'], closed)
            self._add_terminal(switch.get(sc['t_ele']), origin_id, 2, switch['element'], closed)
        for cls, rows in eq_by_class.items():
            self._set('eq', cls, rows)
            self._set('ssh', cls, ssh_by_class[cls])

    def _convert_shunts(self):
        # LinearShuntCompensator, NonlinearShuntCompensator and StaticVarCompensator map to the
        # shunt table.
        lin_eq, lin_ssh, svc_eq, svc_ssh = [], [], [], []
        nonlin_eq, nonlin_ssh, nonlin_points = [], [], []
        for _, shunt in self.net.shunt.iterrows():
            origin_class = shunt.get(sc['o_cl'])
            origin_id = self._id_or_new(shunt.get(sc['o_id']))
            vn_kv = shunt.get('vn_kv')
            connected = bool(shunt['in_service'])
            # complex per-section admittance: s = vn^2 * conj(g + b*1j) -> g = p/vn^2, b = -q/vn^2
            g_per_section = shunt['p_mw'] / vn_kv ** 2 if vn_kv else np.nan
            b_per_section = -shunt['q_mvar'] / vn_kv ** 2 if vn_kv else np.nan
            if origin_class == 'StaticVarCompensator':
                svc_eq.append({'rdfId': origin_id, 'name': shunt.get('name'),
                               'description': shunt.get('description'), 'voltageSetPoint': vn_kv,
                               'sVCControlMode': shunt.get('sVCControlMode')})
                svc_ssh.append({'rdfId': origin_id, 'q': float(shunt['q_mvar'])})
            elif origin_class == 'NonlinearShuntCompensator':
                eq_row, ssh_row, points = self._nonlinear_shunt_rows(
                    shunt, origin_id, vn_kv, g_per_section, b_per_section)
                nonlin_eq.append(eq_row)
                nonlin_ssh.append(ssh_row)
                nonlin_points.extend(points)
            elif origin_class in (None, 'LinearShuntCompensator') or pd.isna(origin_class):
                lin_eq.append({'rdfId': origin_id, 'name': shunt.get('name'),
                               'description': shunt.get('description'), 'nomU': vn_kv,
                               'gPerSection': g_per_section, 'bPerSection': b_per_section,
                               'maximumSections': shunt.get('max_step'),
                               'normalSections': shunt.get('step')})
                lin_ssh.append({'rdfId': origin_id, 'sections': shunt.get('step')})
            else:
                self.logger.warning("Skipping shunt %s: origin_class %s not supported yet.",
                                    origin_id, origin_class)
                continue
            self._add_terminal(shunt.get(sc['t']), origin_id, 1, shunt['bus'], connected)
        self._set('eq', 'LinearShuntCompensator', lin_eq)
        self._set('ssh', 'LinearShuntCompensator', lin_ssh)
        self._set('eq', 'StaticVarCompensator', svc_eq)
        self._set('ssh', 'StaticVarCompensator', svc_ssh)
        self._set('eq', 'NonlinearShuntCompensator', nonlin_eq)
        self._set('ssh', 'NonlinearShuntCompensator', nonlin_ssh)
        self._set('eq', 'NonlinearShuntCompensatorPoint', nonlin_points)

    def _nonlinear_shunt_rows(self, shunt, origin_id, vn_kv, g_per_section, b_per_section):
        # The per-section points are not preserved on the net; reconstruct uniform points so the
        # importer's aggregate (sum over active sections / sections) reproduces p_mw/q_mvar.
        max_step = shunt.get('max_step')
        sections = int(max_step) if not pd.isna(max_step) else int(shunt.get('step', 1) or 1)
        eq_row = {'rdfId': origin_id, 'name': shunt.get('name'),
                  'description': shunt.get('description'), 'nomU': vn_kv, 'maximumSections': max_step}
        ssh_row = {'rdfId': origin_id, 'sections': shunt.get('step')}
        points = [{'rdfId': _new_uuid(), 'NonlinearShuntCompensator': origin_id,
                   'sectionNumber': section, 'g': g_per_section, 'b': b_per_section}
                  for section in range(1, sections + 1)]
        return eq_row, ssh_row, points

    def _convert_impedances(self):
        # impedance table holds SeriesCompensator and EquivalentBranch elements. Both store their
        # per-unit values referred to z_base = vn_kv**2 / sn_mva (vn from the from-bus).
        sc_rows, eb_rows = [], []
        for _, imp in self.net.impedance.iterrows():
            origin_class = imp.get(sc['o_cl'])
            origin_id = self._id_or_new(imp.get(sc['o_id']))
            from_bus = imp['from_bus']
            vn_kv = self.net.bus.loc[from_bus, 'vn_kv'] if from_bus in self.net.bus.index else np.nan
            sn_mva = imp.get('sn_mva')
            z_base = (vn_kv ** 2) / sn_mva if not pd.isna(vn_kv) and sn_mva else np.nan
            bv_id = self._base_voltage_id(vn_kv)
            in_service = bool(imp['in_service'])
            if origin_class == 'EquivalentBranch':
                eb_rows.append({'rdfId': origin_id, 'name': imp.get('name'),
                                'description': imp.get('description'), 'BaseVoltage': bv_id,
                                'r': imp['rft_pu'] * z_base, 'x': imp['xft_pu'] * z_base,
                                'r21': imp.get('rtf_pu', np.nan) * z_base,
                                'x21': imp.get('xtf_pu', np.nan) * z_base,
                                'zeroR12': imp.get('rft0_pu', np.nan) * z_base,
                                'zeroX12': imp.get('xft0_pu', np.nan) * z_base,
                                'zeroR21': imp.get('rtf0_pu', np.nan) * z_base,
                                'zeroX21': imp.get('xtf0_pu', np.nan) * z_base})
            elif origin_class in (None, 'SeriesCompensator') or pd.isna(origin_class):
                sc_rows.append({'rdfId': origin_id, 'name': imp.get('name'),
                                'description': imp.get('description'), 'BaseVoltage': bv_id,
                                'r': imp['rft_pu'] * z_base, 'x': imp['xft_pu'] * z_base,
                                'r0': imp.get('rft0_pu', np.nan) * z_base,
                                'x0': imp.get('xft0_pu', np.nan) * z_base})
            else:
                self.logger.warning("Skipping impedance %s: origin_class %s not supported.",
                                    origin_id, origin_class)
                continue
            self._add_terminal(imp.get(sc['t_from']), origin_id, 1, from_bus, in_service)
            self._add_terminal(imp.get(sc['t_to']), origin_id, 2, imp['to_bus'], in_service)
        self._set('eq', 'SeriesCompensator', sc_rows)
        self._set('eq', 'EquivalentBranch', eb_rows)

    def _convert_power_transformers(self):
        # 2-winding PowerTransformer only (first slice). All series impedance and the magnetizing
        # branch are placed on the HV end (endNumber 1); the LV end (endNumber 2) is left ideal.
        # This reproduces the importer's vk/vkr/i0/pfe (which sum both ends). Tap changers and
        # 3-winding transformers are not reversed yet.
        pt_rows, end_rows = [], []
        for _, trafo in self.net.trafo.iterrows():
            if trafo.get(sc['o_cl']) not in (None, 'PowerTransformer') and not pd.isna(trafo.get(sc['o_cl'])):
                continue
            origin_id = self._id_or_new(trafo.get(sc['o_id']))
            sn = float(trafo['sn_mva'])
            vn_hv = float(trafo['vn_hv_kv'])
            vn_lv = float(trafo['vn_lv_kv'])
            in_service = bool(trafo['in_service'])
            # series impedance (positive and zero sequence) and magnetizing branch on the HV side
            r_hv, x_hv = self._impedance_from_vk(trafo['vk_percent'], trafo['vkr_percent'], vn_hv, sn)
            r0_hv, x0_hv = self._impedance_from_vk(self._safe(trafo.get('vk0_percent')),
                                                  self._safe(trafo.get('vkr0_percent')), vn_hv, sn)
            g_hv, b_hv = self._magnetizing_branch(vn_hv, sn, trafo.get('pfe_kw'), trafo.get('i0_percent'))
            clock = self._phase_angle_clock(trafo.get('shift_degree'))

            pte_hv = self._id_or_new(trafo.get(sc['pte_id_hv']))
            pte_lv = self._id_or_new(trafo.get(sc['pte_id_lv']))
            conn_hv, conn_lv = self._split_vector_group(trafo.get('vector_group'), 2)
            pt_rows.append({'rdfId': origin_id, 'name': trafo.get('name'),
                            'description': trafo.get('description'),
                            'isPartOfGeneratorUnit': self._as_bool(trafo.get('power_station_unit'))})
            end_rows.append({'rdfId': pte_hv, 'name': trafo.get('name'), 'PowerTransformer': origin_id,
                             'endNumber': 1, 'Terminal': trafo.get(sc['t_hv']), 'ratedS': sn, 'ratedU': vn_hv,
                             'r': r_hv, 'x': x_hv, 'b': b_hv, 'g': g_hv, 'r0': r0_hv, 'x0': x0_hv,
                             'BaseVoltage': self._base_voltage_id(vn_hv), 'phaseAngleClock': clock,
                             'connectionKind': conn_hv})
            end_rows.append({'rdfId': pte_lv, 'name': trafo.get('name'), 'PowerTransformer': origin_id,
                             'endNumber': 2, 'Terminal': trafo.get(sc['t_lv']), 'ratedS': sn, 'ratedU': vn_lv,
                             'r': 0.0, 'x': 0.0, 'b': 0.0, 'g': 0.0, 'r0': 0.0, 'x0': 0.0,
                             'BaseVoltage': self._base_voltage_id(vn_lv), 'phaseAngleClock': 0,
                             'connectionKind': conn_lv})
            self._add_terminal(trafo.get(sc['t_hv']), origin_id, 1, trafo['hv_bus'], in_service)
            self._add_terminal(trafo.get(sc['t_lv']), origin_id, 2, trafo['lv_bus'], in_service)
            self._add_tap_changer(trafo, {'hv': pte_hv, 'lv': pte_lv})
            for side in ('hv', 'lv'):
                self._add_side_current_limit(trafo, trafo.get(sc[f't_{side}']), side)
        self._set('eq', 'PowerTransformer', pt_rows)
        self._set('eq', 'PowerTransformerEnd', end_rows)

    def _add_tap_changer(self, trafo, pte_by_side: Dict[str, str]):
        # Reconstruct the tap changer on the end given by tap_side. RatioTapChanger and all phase tap
        # changers (Linear/Asymmetrical/Symmetrical/Tabular) are supported.
        tc_class = trafo.get(sc['tc'])
        pte_id = pte_by_side.get(trafo.get('tap_side'))
        if pte_id is None or pd.isna(tc_class):
            return
        tc_id = self._id_or_new(trafo.get(sc['tc_id']))
        common = {'rdfId': tc_id, 'TransformerEnd': pte_id, 'neutralStep': trafo.get('tap_neutral'),
                  'lowStep': trafo.get('tap_min'), 'highStep': trafo.get('tap_max')}
        if tc_class == 'PhaseTapChangerTabular':
            self._add_tabular_tap_changer(tc_id, common, trafo)
            return
        if tc_class == 'RatioTapChanger':
            # a table-based RatioTapChanger also carries a per-step ratio characteristic; rebuild its
            # RatioTapChangerTable so the non-linear per-step ratio survives (in addition to the
            # linear stepVoltageIncrement, which pandapower keeps as tap_step_percent)
            table_id = self._ratio_table_id(trafo)
            self._tc_eq[tc_class].append({**common, 'stepVoltageIncrement': trafo.get('tap_step_percent'),
                                          'RatioTapChangerTable': table_id})
        elif tc_class == 'PhaseTapChangerLinear':
            self._tc_eq[tc_class].append({**common, 'stepPhaseShiftIncrement': trafo.get('tap_step_degree')})
        elif tc_class == 'PhaseTapChangerAsymmetrical':
            self._tc_eq[tc_class].append({**common, 'voltageStepIncrement': trafo.get('tap_step_percent'),
                                          'windingConnectionAngle': trafo.get('tap_step_degree')})
        elif tc_class == 'PhaseTapChangerSymmetrical':
            self._tc_eq[tc_class].append({**common, 'voltageStepIncrement': trafo.get('tap_step_percent')})
        else:
            return  # unknown tap changer type
        self._tc_ssh[tc_class].append({'rdfId': tc_id, 'step': trafo.get('tap_pos')})

    def _add_tabular_tap_changer(self, tc_id, common, trafo):
        # A tabular phase tap changer defines a per-step ratio/angle table. pandapower flattens it
        # into net['trafo_characteristic_table'] (one row per step). Rebuild the PhaseTapChangerTable
        # and its TablePoints from that so the importer recovers the same per-step ratio/angle, hence
        # the same transformer ratio at the operating tap. (The per-step impedance deviation is not
        # reconstructed; the base impedance is exported on the windings and treated as tap-independent.)
        rows = self._char_by_id.get(trafo.get('id_characteristic_table'))
        if rows is None or rows.empty:
            return  # no characteristic available -> cannot reconstruct the tap changer
        table_id = _new_uuid()
        self._tabular_tables.append({'rdfId': table_id})
        self._tabular_eq.append({**common, 'PhaseTapChangerTable': table_id})
        self._tabular_ssh.append({'rdfId': tc_id, 'step': trafo.get('tap_pos')})
        self._tabular_points.extend(
            self._table_points(table_id, rows, trafo, 'PhaseTapChangerTable', with_angle=True))

    def _ratio_table_id(self, trafo):
        # build a RatioTapChangerTable + per-step points from the characteristic, returning its id
        # (or None for a purely linear RatioTapChanger without a characteristic table)
        rows = self._char_by_id.get(trafo.get('id_characteristic_table'))
        if rows is None or rows.empty:
            return None
        table_id = _new_uuid()
        self._ratio_tables.append({'rdfId': table_id})
        self._ratio_points.extend(
            self._table_points(table_id, rows, trafo, 'RatioTapChangerTable', with_angle=False))
        return table_id

    def _index_characteristic_table(self):
        ct = self.net.get('trafo_characteristic_table')
        if not isinstance(ct, pd.DataFrame) or ct.empty or 'id_characteristic' not in ct.columns:
            return {}
        cols = [c for c in ['step', 'voltage_ratio', 'angle_deg', 'vk_percent', 'vkr_percent']
                if c in ct.columns]
        return {cid: g[cols] for cid, g in ct.dropna(subset=['step']).groupby('id_characteristic')}

    def _table_points(self, table_id, rows, trafo, table_key, with_angle):
        """Build tap-changer table-point rows from the per-step characteristic. The per-step ratio
        (and, for phase changers, angle) are taken directly; the per-step impedance deviation r/x is
        reconstructed for two-winding transformers (percent space), and left at 0 for three-winding
        ones (the per-winding recombination is not inverted)."""
        vk_base, vkr_base = trafo.get('vk_percent'), trafo.get('vkr_percent')
        two_winding = ('vk_percent' in rows.columns and not pd.isna(vk_base) and not pd.isna(vkr_base)
                       and vk_base and abs(vk_base) >= abs(vkr_base))
        vkx_base = math.sqrt(vk_base ** 2 - vkr_base ** 2) if two_winding else 0.0
        points = []
        for _, point in rows.iterrows():
            r_dev = x_dev = 0.0
            if two_winding and not pd.isna(point.get('vk_percent')) and vkr_base and vkx_base:
                vk, vkr = float(point['vk_percent']), float(point['vkr_percent'])
                r_dev = (vkr / vkr_base - 1) * 100
                vkx = math.sqrt(max(vk ** 2 - vkr ** 2, 0.0))
                x_dev = (vkx / vkx_base - 1) * 100
            row = {'rdfId': _new_uuid(), table_key: table_id, 'step': int(point['step']),
                   'ratio': float(point['voltage_ratio']), 'r': r_dev, 'x': x_dev}
            if with_angle:
                row['angle'] = float(point['angle_deg'])
            points.append(row)
        return points

    def _finalize_tap_changers(self):
        for tc_class in self._tc_eq:
            self._set('eq', tc_class, self._tc_eq[tc_class])
            self._set('ssh', tc_class, self._tc_ssh[tc_class])
        # tabular phase tap changers (the PhaseTapChangerTable class is not in the blueprint)
        if self._tabular_eq:
            self.cim['eq'].setdefault('PhaseTapChangerTable', pd.DataFrame(columns=['rdfId']))
            self._set('eq', 'PhaseTapChangerTable', self._tabular_tables)
            self._set('eq', 'PhaseTapChangerTabular', self._tabular_eq)
            self._set('ssh', 'PhaseTapChangerTabular', self._tabular_ssh)
            self._set('eq', 'PhaseTapChangerTablePoint', self._tabular_points)
        # tables for table-based RatioTapChangers
        self._set('eq', 'RatioTapChangerTable', self._ratio_tables)
        self._set('eq', 'RatioTapChangerTablePoint', self._ratio_points)

    def _convert_power_transformers_3w(self):
        # 3-winding PowerTransformer. The pandapower per-winding-pair short-circuit values are the
        # importer's recombination of the three CGMES end impedances (each referred to its own end's
        # voltage). We invert that pairwise system to recover the per-end r/x (positive and zero
        # sequence); the magnetizing branch is placed on the HV end.
        pt_rows, end_rows = [], []
        for _, trafo in self.net.trafo3w.iterrows():
            if trafo.get(sc['o_cl']) not in (None, 'PowerTransformer') and not pd.isna(trafo.get(sc['o_cl'])):
                continue
            origin_id = self._id_or_new(trafo.get(sc['o_id']))
            u = {'hv': float(trafo['vn_hv_kv']), 'mv': float(trafo['vn_mv_kv']), 'lv': float(trafo['vn_lv_kv'])}
            s = {'hv': float(trafo['sn_hv_mva']), 'mv': float(trafo['sn_mv_mva']), 'lv': float(trafo['sn_lv_mva'])}
            in_service = bool(trafo['in_service'])
            # pairwise scaling factors and voltage-ratio couplings used by the importer
            a_hv = min(s['hv'], s['mv']) * 100 / u['hv'] ** 2
            a_mv = min(s['mv'], s['lv']) * 100 / u['mv'] ** 2
            a_lv = min(s['lv'], s['hv']) * 100 / u['lv'] ** 2
            k_hvmv = (u['hv'] / u['mv']) ** 2
            k_mvlv = (u['mv'] / u['lv']) ** 2
            k_lvhv = (u['lv'] / u['hv']) ** 2

            r = self._solve_3w(trafo['vkr_hv_percent'] / a_hv, trafo['vkr_mv_percent'] / a_mv,
                               trafo['vkr_lv_percent'] / a_lv, k_hvmv, k_mvlv, k_lvhv)
            x = self._solve_3w(*self._x_targets(trafo, 'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent',
                                                a_hv, a_mv, a_lv, r, k_hvmv, k_mvlv, k_lvhv),
                               k_hvmv, k_mvlv, k_lvhv)
            r0 = self._solve_3w(self._safe(trafo.get('vkr0_hv_percent')) / a_hv,
                                self._safe(trafo.get('vkr0_mv_percent')) / a_mv,
                                self._safe(trafo.get('vkr0_lv_percent')) / a_lv, k_hvmv, k_mvlv, k_lvhv)
            x0 = self._solve_3w(*self._x_targets(trafo, 'vk0_hv_percent', 'vk0_mv_percent', 'vk0_lv_percent',
                                                 a_hv, a_mv, a_lv, r0, k_hvmv, k_mvlv, k_lvhv),
                                k_hvmv, k_mvlv, k_lvhv)
            # magnetizing branch on the HV end
            g_hv, b_hv = self._magnetizing_branch(u['hv'], s['hv'], trafo.get('pfe_kw'),
                                                  trafo.get('i0_percent'))
            clock = {'hv': 0,
                     'mv': self._phase_angle_clock(trafo.get('shift_mv_degree')),
                     'lv': self._phase_angle_clock(trafo.get('shift_lv_degree'))}
            pte = {'hv': self._id_or_new(trafo.get(sc['pte_id_hv'])),
                   'mv': self._id_or_new(trafo.get(sc['pte_id_mv'])),
                   'lv': self._id_or_new(trafo.get(sc['pte_id_lv']))}
            term = {'hv': trafo.get(sc['t_hv']), 'mv': trafo.get(sc['t_mv']), 'lv': trafo.get(sc['t_lv'])}
            bus = {'hv': trafo['hv_bus'], 'mv': trafo['mv_bus'], 'lv': trafo['lv_bus']}

            pt_rows.append({'rdfId': origin_id, 'name': trafo.get('name'),
                            'description': trafo.get('description'),
                            'isPartOfGeneratorUnit': self._as_bool(trafo.get('power_station_unit'))})
            for n, side in enumerate(('hv', 'mv', 'lv'), start=1):
                end_rows.append({'rdfId': pte[side], 'name': trafo.get('name'), 'PowerTransformer': origin_id,
                                 'endNumber': n, 'Terminal': term[side], 'ratedS': s[side], 'ratedU': u[side],
                                 'r': r[side], 'x': x[side], 'r0': r0[side], 'x0': x0[side],
                                 'b': b_hv if side == 'hv' else 0.0, 'g': g_hv if side == 'hv' else 0.0,
                                 'BaseVoltage': self._base_voltage_id(u[side]), 'phaseAngleClock': clock[side]})
                self._add_terminal(term[side], origin_id, n, bus[side], in_service)
                self._add_side_current_limit(trafo, term[side], side)
            self._add_tap_changer(trafo, pte)
        self._set('eq', 'PowerTransformer', pt_rows)
        self._set('eq', 'PowerTransformerEnd', end_rows)

    def _x_targets(self, trafo, vk_hv_col, vk_mv_col, vk_lv_col, a_hv, a_mv, a_lv, r, k_hvmv, k_mvlv, k_lvhv):
        # given the solved per-end r values, recover the per-pair reactance sums from the vk values
        def bx(vk_col, a, pair_r):
            return self._reactance(self._safe(trafo.get(vk_col)) / a, pair_r)
        bx1 = bx(vk_hv_col, a_hv, r['hv'] + k_hvmv * r['mv'])
        bx2 = bx(vk_mv_col, a_mv, r['mv'] + k_mvlv * r['lv'])
        bx3 = bx(vk_lv_col, a_lv, r['lv'] + k_lvhv * r['hv'])
        return bx1, bx2, bx3

    @staticmethod
    def _solve_3w(b1, b2, b3, k_hvmv, k_mvlv, k_lvhv):
        # solve the linear system:
        #   v_hv + k_hvmv*v_mv         = b1
        #          v_mv + k_mvlv*v_lv  = b2
        #   k_lvhv*v_hv +        v_lv  = b3
        v_mv = (k_lvhv * b1 + b2 / k_mvlv - b3) / (k_lvhv * k_hvmv + 1 / k_mvlv)
        v_hv = b1 - k_hvmv * v_mv
        v_lv = (b2 - v_mv) / k_mvlv
        return {'hv': v_hv, 'mv': v_mv, 'lv': v_lv}

    # ------------------------------------------------------------------ transformer impedance helpers

    def _impedance_from_vk(self, vk, vkr, vn_kv, sn_mva):
        """Recover the (r, x) ohm impedance of a winding from the per-unit short-circuit values.
        vk_percent is signed (it carries the sign of x), so its magnitude gives z and its sign x."""
        base = vn_kv ** 2 / (sn_mva * 100)  # %-to-ohm factor at this winding's voltage/rating
        r = vkr * base  # vkr_percent is computed from abs(r), so it is non-negative
        z = abs(vk) * base
        return r, math.copysign(self._reactance(z, r), vk)

    @staticmethod
    def _reactance(z, r):
        """Reactance magnitude from impedance and resistance: sqrt(z^2 - r^2), floored at 0."""
        return math.sqrt(max(z ** 2 - r ** 2, 0.0))

    def _magnetizing_branch(self, vn_kv, sn_mva, pfe_kw, i0_percent):
        """Recover the (g, b) magnetizing admittance on a winding from pfe_kw / i0_percent."""
        if not vn_kv:
            return 0.0, 0.0
        g = self._safe(pfe_kw) / (vn_kv ** 2 * 1000)
        i0_s = self._safe(i0_percent) * sn_mva / 100
        b_sq = i0_s ** 2 - (g * vn_kv ** 2) ** 2
        b = math.sqrt(b_sq) / vn_kv ** 2 if b_sq > 0 else 0.0
        return g, b

    def _phase_angle_clock(self, shift_degree):
        """CIM phaseAngleClock (multiples of 30 degrees) from a pandapower shift in degrees."""
        return self._safe(shift_degree) / 30

    def _convert_equivalent_injections(self):
        # EquivalentInjection -> ward (regulationStatus False) or xward (regulationStatus True)
        eq_rows, ssh_rows = [], []
        for table, regulation_status in (('ward', False), ('xward', True)):
            df = self.net[table]
            for _, row in df.iterrows():
                origin_id = self._id_or_new(row.get(sc['o_id']))
                eq_rows.append({'rdfId': origin_id, 'name': row.get('name'),
                                'description': row.get('description'),
                                'regulationCapability': regulation_status})
                ssh_row = {'rdfId': origin_id, 'p': float(row['ps_mw']), 'q': float(row['qs_mvar']),
                           'regulationStatus': regulation_status}
                if regulation_status:
                    vn_kv = self.net.bus.loc[row['bus'], 'vn_kv'] if row['bus'] in self.net.bus.index else np.nan
                    ssh_row['regulationTarget'] = row.get('vm_pu') * vn_kv if not pd.isna(vn_kv) else np.nan
                ssh_rows.append(ssh_row)
                self._add_terminal(row.get(sc['t']), origin_id, 1, row['bus'], bool(row['in_service']))
        self._set('eq', 'EquivalentInjection', eq_rows)
        self._set('ssh', 'EquivalentInjection', ssh_rows)

    # ------------------------------------------------------------------ state variables (SV)

    def _convert_state_variables(self):
        # The SV profile holds power-flow results and is only exported for a solved net.
        if self.net.res_bus.empty:
            self.logger.info("No power-flow results (res_bus is empty); skipping the SV profile.")
            # drop the synthesized SV FullModel header so the empty profile is not written
            self.cim['sv'].pop('FullModel', None)
            return
        self._convert_sv_voltage()
        self._convert_sv_tap_steps()
        self._convert_sv_shunt_sections()

    def _convert_sv_voltage(self):
        rows, seen = [], set()
        res_bus = self.net.res_bus
        for bus_idx, bus in self.net.bus.iterrows():
            if bus_idx not in res_bus.index:
                continue
            res = res_bus.loc[bus_idx]
            if pd.isna(res['vm_pu']):
                continue
            _, tn_id = self._node_ref(bus_idx)
            # SvVoltage references a TopologicalNode; skip buses without one (and de-duplicate, as
            # node-breaker buses joined by closed switches share a single TopologicalNode)
            if tn_id is None or pd.isna(tn_id) or tn_id in seen:
                continue
            seen.add(tn_id)
            rows.append({'rdfId': _new_uuid(), 'TopologicalNode': tn_id,
                         'v': res['vm_pu'] * bus['vn_kv'], 'angle': res['va_degree']})
        self._set('sv', 'SvVoltage', rows)

    def _convert_sv_tap_steps(self):
        rows = []
        for table in ('trafo', 'trafo3w'):
            for _, trafo in self.net[table].iterrows():
                if trafo.get(sc['tc']) != 'RatioTapChanger':
                    continue  # only tap changers that were actually exported
                tc_id = trafo.get(sc['tc_id'])
                pos = trafo.get('tap_pos')
                if pd.isna(tc_id) or pd.isna(pos):
                    continue
                rows.append({'rdfId': _new_uuid(), 'TapChanger': tc_id, 'position': pos})
        self._set('sv', 'SvTapStep', rows)

    def _convert_sv_shunt_sections(self):
        rows = []
        for _, shunt in self.net.shunt.iterrows():
            if shunt.get(sc['o_cl']) not in (None, 'LinearShuntCompensator') and not pd.isna(shunt.get(sc['o_cl'])):
                continue
            rows.append({'rdfId': _new_uuid(), 'ShuntCompensator': shunt.get(sc['o_id']),
                         'sections': shunt.get('step')})
        self._set('sv', 'SvShuntCompensatorSections', rows)

    # ------------------------------------------------------------------ diagram layout (DL)

    # pandapower element tables whose diagram coordinates are exported (Point, except the branch
    # elements which carry a LineString).
    _DIAGRAM_TABLES = ('bus', 'line', 'trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen',
                       'gen', 'impedance', 'shunt', 'ward', 'xward')
    _DIAGRAM_LINE_TABLES = ('line', 'dcline', 'impedance')

    def _iter_geo(self, column, tables):
        """Yield (row, origin_id, coords) for each element in `tables` carrying a parseable GeoJSON
        coordinate string in `column` (shared by the DL and GL exporters)."""
        for table in tables:
            df = self.net[table]
            if column not in df.columns:
                continue
            for _, row in df.iterrows():
                coords = self._parse_geojson(row.get(column))
                origin_id = row.get(sc['o_id'])
                if coords is not None and not pd.isna(origin_id):
                    yield row, origin_id, coords

    def _convert_diagram_layout(self):
        # Export the 'diagram' coordinates (DiagramLayout profile). DiagramObject.IdentifiedObject
        # points directly at each element's rdfId, so no container hierarchy is needed.
        diagram_id = _new_uuid()
        diagram_objects, points = [], []
        for row, origin_id, coords in self._iter_geo('diagram', self._DIAGRAM_TABLES):
            do_id = _new_uuid()
            diagram_objects.append({'rdfId': do_id, 'IdentifiedObject': origin_id,
                                    'Diagram': diagram_id, 'name': row.get('name')})
            for seq, (x, y) in enumerate(coords, start=1):
                points.append({'rdfId': _new_uuid(), 'DiagramObject': do_id,
                               'sequenceNumber': seq, 'xPosition': x, 'yPosition': y})
        if not diagram_objects:
            return
        self.cim['dl']['FullModel'] = pd.DataFrame(self._full_model_rows('dl', synthesize=True))
        self._set('dl', 'Diagram', [{'rdfId': diagram_id, 'name': 'pandapower'}])
        self._set('dl', 'DiagramObject', diagram_objects)
        self._set('dl', 'DiagramObjectPoint', points)

    # ------------------------------------------------------------------ geographical location (GL)

    # GL point elements (single coordinate) and line elements (LineString); buses are handled
    # separately because their geo is attached to the Substation, not the node directly.
    _GL_POINT_TABLES = ('trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'shunt',
                        'ward', 'xward')
    _GL_LINE_TABLES = ('line', 'impedance')

    def _convert_geographical_location(self):
        # Export the 'geo' coordinates (GeographicalLocation profile). Element geo maps directly to a
        # Location; bus geo is attached to the Substation (reconstructing the VoltageLevel/Substation
        # container hierarchy that the importer needs to map it back to the bus).
        cs_id = _new_uuid()
        locations, points = [], []

        # bus geo is attached to the Substation (the Substation/VoltageLevel containers are already
        # rebuilt in _convert_buses); emit one Location per Substation
        if 'geo' in self.net.bus.columns:
            seen_substations = set()
            for _, bus in self.net.bus.iterrows():
                coords = self._parse_geojson(bus.get('geo'))
                sub_id = bus.get(sc['sub_id'])
                if coords is None or pd.isna(sub_id) or sub_id in seen_substations:
                    continue
                seen_substations.add(sub_id)
                self._append_location(locations, points, cs_id, sub_id, coords)

        for _, origin_id, coords in self._iter_geo('geo', self._GL_POINT_TABLES + self._GL_LINE_TABLES):
            self._append_location(locations, points, cs_id, origin_id, coords)

        if not locations:
            return
        self.cim['gl']['FullModel'] = pd.DataFrame(self._full_model_rows('gl', synthesize=True))
        self._set('gl', 'CoordinateSystem',
                  [{'rdfId': cs_id, 'name': 'WGS84', 'crsUrn': 'urn:ogc:def:crs:EPSG::4326'}])
        self._set('gl', 'Location', locations)
        self._set('gl', 'PositionPoint', points)

    @staticmethod
    def _append_location(locations, points, cs_id, psr_id, coords):
        loc_id = _new_uuid()
        locations.append({'rdfId': loc_id, 'PowerSystemResources': psr_id, 'CoordinateSystem': cs_id})
        for seq, (x, y) in enumerate(coords, start=1):
            points.append({'rdfId': _new_uuid(), 'Location': loc_id, 'sequenceNumber': seq,
                           'xPosition': x, 'yPosition': y})

    @staticmethod
    def _none_if_na(value):
        return None if value is None or pd.isna(value) else value

    @staticmethod
    def _parse_geojson(value):
        """Return a list of (x, y) coordinate tuples from a GeoJSON Point/LineString string, or None."""
        if value is None or not isinstance(value, str) or pd.isna(value):
            return None
        try:
            geo = json.loads(value)
        except (ValueError, TypeError):
            return None
        coords = geo.get('coordinates')
        if not coords:
            return None
        if geo.get('type') == 'Point':
            return [(coords[0], coords[1])]
        return [(point[0], point[1]) for point in coords]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _id_or_new(value):
        # NaN is truthy, so "value or _new_uuid()" would keep a NaN id; use an explicit check.
        return _new_uuid() if value is None or pd.isna(value) else value

    @staticmethod
    def _split_vector_group(vector_group, windings: int):
        """Split a pandapower vector_group back into per-winding CIM connectionKind values.

        The importer builds it as the HV connectionKind upper-cased followed by the LV (and MV)
        connectionKind lower-cased, e.g. 'YNyn' -> HV 'Yn', LV 'Yn'. Only the unambiguous two-winding
        case is reversed (the HV part is the leading upper-case run); for three windings the split is
        ambiguous, so connectionKind is left unset.
        """
        none = tuple([None] * windings)
        if vector_group is None or not isinstance(vector_group, str) or pd.isna(vector_group):
            return none
        if windings != 2:
            return none
        split = next((i for i, ch in enumerate(vector_group) if ch.islower()), None)
        if not split:  # split is None (no lower-case part) or 0 (no leading HV part) -> can't split
            return none
        return vector_group[:split].capitalize(), vector_group[split:].capitalize()

    @staticmethod
    def _safe(value):
        # return 0.0 for NaN/None numeric values (used in transformer impedance reconstruction)
        return 0.0 if value is None or pd.isna(value) else float(value)

    @staticmethod
    def _neg(value):
        # CGMES uses the load sign convention for injections (p = -p_mw)
        return -value if value is not None and not pd.isna(value) else value

    @staticmethod
    def _as_bool(value):
        if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
            return None
        return bool(value)

    def _set(self, profile: str, cls: str, rows: List[dict]):
        """Append rows (list of dicts) to the CIM DataFrame for the given profile/class, keeping the
        blueprint columns."""
        if not rows:
            return
        template = self.cim[profile][cls]
        new_df = pd.DataFrame(rows)
        # keep only known columns, in blueprint order; create missing ones as NaN
        for col in template.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[list(template.columns)]
        self.cim[profile][cls] = pd.concat([template, new_df], ignore_index=True, sort=False) \
            if not template.empty else new_df

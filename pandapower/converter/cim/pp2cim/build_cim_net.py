# -*- coding: utf-8 -*-
from __future__ import annotations
import collections
import logging
import math
import uuid
from typing import Dict, List

import numpy as np
import pandas as pd

from pandapower.auxiliary import pandapowerNet
from .. import cim_tools
from ..cim_classes import CimParser

logger = logging.getLogger('cim.pp2cim.build_cim_net')

sc = cim_tools.get_pp_net_special_columns_dict()

# CGMES profile URIs used to synthesize a FullModel header when the net does not carry one
# (i.e. the net was not imported from CGMES). For round-trip the headers come from net['CGMES'].
PROFILE_URI = {
    '2.4.15': {
        'eq': 'http://entsoe.eu/CIM/EquipmentCore/3/1',
        'ssh': 'http://entsoe.eu/CIM/SteadyStateHypothesis/1/1',
        'sv': 'http://entsoe.eu/CIM/StateVariables/4/1',
        'tp': 'http://entsoe.eu/CIM/Topology/4/1',
    },
    '3.0': {
        'eq': 'http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0',
        'ssh': 'http://iec.ch/TC57/ns/CIM/SteadyStateHypothesis-EU/3.0',
        'sv': 'http://iec.ch/TC57/ns/CIM/StateVariables-EU/3.0',
        'tp': 'http://iec.ch/TC57/ns/CIM/Topology-EU/3.0',
    },
}


def _new_uuid() -> str:
    return '_' + str(uuid.uuid4())


class PpToCimConverter:
    """
    Builds a CIM data structure (profile -> CIM element type -> DataFrame) from a pandapower net,
    reusing the CGMES identifiers preserved by the cim2pp converter (origin_id, terminals, topology).

    The result can be serialized to RDF/XML by :class:`pandapower.converter.cim.cim_writer.CimWriter`.
    """

    def __init__(self, net: pandapowerNet, cgmes_version: str = '2.4.15', **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.net = net
        self.cgmes_version = cgmes_version
        self.kwargs = kwargs
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

        # nominalVoltage -> BaseVoltage rdfId, built from net['CGMES']['BaseVoltage']
        self._voltage_to_bv: Dict[float, str] = {}

    # ------------------------------------------------------------------ orchestration

    def convert_to_cim(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        self.logger.info("Start converting the pandapower net to CIM.")
        self._create_full_model()
        self._create_base_voltages()
        self._convert_buses()
        self._convert_lines()
        self._convert_loads()
        self._convert_external_network_injections()
        self._convert_synchronous_machines()
        self._convert_switches()
        self._convert_shunts()
        self._convert_impedances()
        self._convert_power_transformers()
        self._convert_power_transformers_3w()
        self._convert_equivalent_injections()
        self._convert_state_variables()
        self._finalize_terminals()
        self._finalize_regulating_controls()
        self._finalize_tap_changers()
        self.logger.info("Finished converting the pandapower net to CIM.")
        return self.cim

    # ------------------------------------------------------------------ headers / base voltages

    def _create_full_model(self):
        cgmes = self.net.get('CGMES', {}) if isinstance(self.net.get('CGMES', None), dict) else {}
        for profile in ['eq', 'ssh', 'tp', 'sv']:
            rows = []
            if profile in cgmes and isinstance(cgmes[profile], dict) and len(cgmes[profile]) > 0:
                for rdf_id, fields in cgmes[profile].items():
                    row = {'rdfId': rdf_id}
                    row.update(fields)
                    rows.append(row)
            elif profile in ('eq', 'ssh', 'tp'):
                # synthesize a minimal header so the parser can re-detect the profile on re-import
                rows.append({'rdfId': str(uuid.uuid4()),
                             'profile': PROFILE_URI.get(self.cgmes_version, {}).get(profile)})
            if rows:
                self.cim[profile]['FullModel'] = pd.DataFrame(rows)

    def _create_base_voltages(self):
        bv = self.net.get('CGMES', {}).get('BaseVoltage') if isinstance(self.net.get('CGMES', None), dict) else None
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
        for idx, line in self.net.line.iterrows():
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
        self._set('eq', 'ACLineSegment', eq_rows)

    def _convert_loads(self):
        # EnergyConsumer / ConformLoad / NonConformLoad / StationSupply all map to the load table
        by_class: Dict[str, List[dict]] = {}
        ssh_rows: Dict[str, List[dict]] = {}
        for idx, load in self.net.load.iterrows():
            origin_class = load.get(sc['o_cl'])
            if pd.isna(origin_class):
                origin_class = 'EnergyConsumer'
            origin_id = self._id_or_new(load.get(sc['o_id']))
            by_class.setdefault(origin_class, []).append({
                'rdfId': origin_id, 'name': load.get('name'), 'description': load.get('description')})
            ssh_rows.setdefault(origin_class, []).append({
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
            for idx, ext in self.net[table].iterrows():
                if ext.get(sc['o_cl']) != 'ExternalNetworkInjection':
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
                    'controlEnabled': self._as_bool(ext.get('RegulatingControl.enabled'))})
        self._set('eq', 'ExternalNetworkInjection', eq_rows)
        self._set('ssh', 'ExternalNetworkInjection', ssh_rows)

    def _convert_synchronous_machines(self):
        # SynchronousMachine -> gen (voltage control) or sgen. Reconstruct the linked GeneratingUnit
        # and (when voltage controlled) the RegulatingControl so the importer reproduces the split.
        eq_sm, ssh_sm, eq_gu = [], [], []
        for table in ('gen', 'sgen'):
            for idx, machine in self.net[table].iterrows():
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
                               'controlEnabled': self._as_bool(machine.get('RegulatingControl.enabled'))})
        self._set('eq', 'GeneratingUnit', eq_gu)
        self._set('eq', 'SynchronousMachine', eq_sm)
        self._set('ssh', 'SynchronousMachine', ssh_sm)

    def _add_regulating_control(self, row, terminal_id) -> str | None:
        """Reconstruct a RegulatingControl from the preserved RegulatingControl.* columns. Returns
        the new RegulatingControl rdfId, or None when the element has no regulation info."""
        mode = row.get('RegulatingControl.mode')
        enabled = row.get('RegulatingControl.enabled')
        target = row.get('RegulatingControl.targetValue')
        if pd.isna(mode) and pd.isna(enabled) and pd.isna(target):
            return None
        reg_id = _new_uuid()
        self._reg_control_eq.append({'rdfId': reg_id, 'mode': mode if not pd.isna(mode) else None,
                                     'Terminal': terminal_id})
        self._reg_control_ssh.append({'rdfId': reg_id, 'enabled': self._as_bool(enabled),
                                      'targetValue': target if not pd.isna(target) else None})
        return reg_id

    def _finalize_regulating_controls(self):
        self._set('eq', 'RegulatingControl', self._reg_control_eq)
        self._set('ssh', 'RegulatingControl', self._reg_control_ssh)

    def _convert_switches(self):
        # Breaker / Disconnector / LoadBreakSwitch / Switch all map to the pandapower switch table
        # (with et == 'b'). The switch connects two buses via two terminals.
        eq_by_class: Dict[str, List[dict]] = {}
        ssh_by_class: Dict[str, List[dict]] = {}
        for idx, switch in self.net.switch.iterrows():
            if switch.get('et') != 'b':
                continue  # only bus-bus switches originate from CGMES switching devices
            origin_class = switch.get(sc['o_cl'])
            if pd.isna(origin_class):
                origin_class = 'Breaker'
            origin_id = self._id_or_new(switch.get(sc['o_id']))
            closed = bool(switch['closed'])
            in_ka = switch.get('in_ka')
            eq_by_class.setdefault(origin_class, []).append({
                'rdfId': origin_id, 'name': switch.get('name'), 'description': switch.get('description'),
                'normalOpen': not closed, 'ratedCurrent': in_ka * 1e3 if not pd.isna(in_ka) else np.nan})
            ssh_by_class.setdefault(origin_class, []).append({'rdfId': origin_id, 'open': not closed})
            # element is a bus (et == 'b')
            self._add_terminal(switch.get(sc['t_bus']), origin_id, 1, switch['bus'], closed)
            self._add_terminal(switch.get(sc['t_ele']), origin_id, 2, switch['element'], closed)
        for cls, rows in eq_by_class.items():
            self._set('eq', cls, rows)
            self._set('ssh', cls, ssh_by_class[cls])

    def _convert_shunts(self):
        # LinearShuntCompensator and StaticVarCompensator map to the shunt table. NonlinearShunt-
        # Compensators need per-section b/g points that are not preserved on the net, so they are
        # not reversed here.
        lin_eq, lin_ssh, svc_eq, svc_ssh = [], [], [], []
        for idx, shunt in self.net.shunt.iterrows():
            origin_class = shunt.get(sc['o_cl'])
            origin_id = self._id_or_new(shunt.get(sc['o_id']))
            vn_kv = shunt.get('vn_kv')
            connected = bool(shunt['in_service'])
            if origin_class == 'StaticVarCompensator':
                svc_eq.append({'rdfId': origin_id, 'name': shunt.get('name'),
                               'description': shunt.get('description'), 'voltageSetPoint': vn_kv,
                               'sVCControlMode': shunt.get('sVCControlMode')})
                svc_ssh.append({'rdfId': origin_id, 'q': float(shunt['q_mvar'])})
                self._add_terminal(shunt.get(sc['t']), origin_id, 1, shunt['bus'], connected)
            elif origin_class in (None, 'LinearShuntCompensator') or pd.isna(origin_class):
                # complex per-section admittance: s = vn^2 * conj(g + b*1j) -> g = p/vn^2, b = -q/vn^2
                g_per_section = shunt['p_mw'] / vn_kv ** 2 if vn_kv else np.nan
                b_per_section = -shunt['q_mvar'] / vn_kv ** 2 if vn_kv else np.nan
                lin_eq.append({'rdfId': origin_id, 'name': shunt.get('name'),
                               'description': shunt.get('description'), 'nomU': vn_kv,
                               'gPerSection': g_per_section, 'bPerSection': b_per_section,
                               'maximumSections': shunt.get('max_step'),
                               'normalSections': shunt.get('step')})
                lin_ssh.append({'rdfId': origin_id, 'sections': shunt.get('step')})
                self._add_terminal(shunt.get(sc['t']), origin_id, 1, shunt['bus'], connected)
            else:
                self.logger.warning("Skipping shunt %s: origin_class %s not supported yet."
                                    % (origin_id, origin_class))
        self._set('eq', 'LinearShuntCompensator', lin_eq)
        self._set('ssh', 'LinearShuntCompensator', lin_ssh)
        self._set('eq', 'StaticVarCompensator', svc_eq)
        self._set('ssh', 'StaticVarCompensator', svc_ssh)

    def _convert_impedances(self):
        # SeriesCompensator -> impedance. (EquivalentBranch, also an impedance, is handled elsewhere.)
        eq_rows = []
        for idx, imp in self.net.impedance.iterrows():
            if imp.get(sc['o_cl']) != 'SeriesCompensator':
                continue
            origin_id = self._id_or_new(imp.get(sc['o_id']))
            from_bus = imp['from_bus']
            vn_kv = self.net.bus.loc[from_bus, 'vn_kv'] if from_bus in self.net.bus.index else np.nan
            sn_mva = imp.get('sn_mva')
            z_base = (vn_kv ** 2) / sn_mva if not pd.isna(vn_kv) and sn_mva else np.nan
            eq_rows.append({'rdfId': origin_id, 'name': imp.get('name'),
                            'description': imp.get('description'),
                            'BaseVoltage': self._base_voltage_id(vn_kv),
                            'r': imp['rft_pu'] * z_base, 'x': imp['xft_pu'] * z_base,
                            'r0': imp.get('rft0_pu', np.nan) * z_base,
                            'x0': imp.get('xft0_pu', np.nan) * z_base})
            in_service = bool(imp['in_service'])
            self._add_terminal(imp.get(sc['t_from']), origin_id, 1, from_bus, in_service)
            self._add_terminal(imp.get(sc['t_to']), origin_id, 2, imp['to_bus'], in_service)
        self._set('eq', 'SeriesCompensator', eq_rows)

    def _convert_power_transformers(self):
        # 2-winding PowerTransformer only (first slice). All series impedance and the magnetizing
        # branch are placed on the HV end (endNumber 1); the LV end (endNumber 2) is left ideal.
        # This reproduces the importer's vk/vkr/i0/pfe (which sum both ends). Tap changers and
        # 3-winding transformers are not reversed yet.
        pt_rows, end_rows = [], []
        for idx, trafo in self.net.trafo.iterrows():
            if trafo.get(sc['o_cl']) not in (None, 'PowerTransformer') and not pd.isna(trafo.get(sc['o_cl'])):
                continue
            origin_id = self._id_or_new(trafo.get(sc['o_id']))
            sn = float(trafo['sn_mva'])
            vn_hv = float(trafo['vn_hv_kv'])
            vn_lv = float(trafo['vn_lv_kv'])
            in_service = bool(trafo['in_service'])
            # series impedance (positive and zero sequence) referred to the HV side
            r_hv = trafo['vkr_percent'] * vn_hv ** 2 / (sn * 100)
            z_hv = trafo['vk_percent'] * vn_hv ** 2 / (sn * 100)
            x_hv = math.sqrt(max(z_hv ** 2 - r_hv ** 2, 0.0))
            r0_hv = self._safe(trafo.get('vkr0_percent')) * vn_hv ** 2 / (sn * 100)
            z0_hv = self._safe(trafo.get('vk0_percent')) * vn_hv ** 2 / (sn * 100)
            x0_hv = math.sqrt(max(z0_hv ** 2 - r0_hv ** 2, 0.0))
            # magnetizing branch from pfe_kw / i0_percent
            g_hv = self._safe(trafo.get('pfe_kw')) / (vn_hv ** 2 * 1000) if vn_hv else 0.0
            i0_s = self._safe(trafo.get('i0_percent')) * sn / 100
            b_sq = i0_s ** 2 - (g_hv * vn_hv ** 2) ** 2
            b_hv = math.sqrt(b_sq) / vn_hv ** 2 if b_sq > 0 and vn_hv else 0.0
            clock = self._safe(trafo.get('shift_degree')) / 30 if not pd.isna(trafo.get('shift_degree')) else 0.0

            pte_hv = self._id_or_new(trafo.get(sc['pte_id_hv']))
            pte_lv = self._id_or_new(trafo.get(sc['pte_id_lv']))
            pt_rows.append({'rdfId': origin_id, 'name': trafo.get('name'),
                            'description': trafo.get('description'),
                            'isPartOfGeneratorUnit': self._as_bool(trafo.get('power_station_unit'))})
            end_rows.append({'rdfId': pte_hv, 'name': trafo.get('name'), 'PowerTransformer': origin_id,
                             'endNumber': 1, 'Terminal': trafo.get(sc['t_hv']), 'ratedS': sn, 'ratedU': vn_hv,
                             'r': r_hv, 'x': x_hv, 'b': b_hv, 'g': g_hv, 'r0': r0_hv, 'x0': x0_hv,
                             'BaseVoltage': self._base_voltage_id(vn_hv), 'phaseAngleClock': clock})
            end_rows.append({'rdfId': pte_lv, 'name': trafo.get('name'), 'PowerTransformer': origin_id,
                             'endNumber': 2, 'Terminal': trafo.get(sc['t_lv']), 'ratedS': sn, 'ratedU': vn_lv,
                             'r': 0.0, 'x': 0.0, 'b': 0.0, 'g': 0.0, 'r0': 0.0, 'x0': 0.0,
                             'BaseVoltage': self._base_voltage_id(vn_lv), 'phaseAngleClock': 0})
            self._add_terminal(trafo.get(sc['t_hv']), origin_id, 1, trafo['hv_bus'], in_service)
            self._add_terminal(trafo.get(sc['t_lv']), origin_id, 2, trafo['lv_bus'], in_service)
            self._add_tap_changer(trafo, {'hv': pte_hv, 'lv': pte_lv})
        self._set('eq', 'PowerTransformer', pt_rows)
        self._set('eq', 'PowerTransformerEnd', end_rows)

    def _add_tap_changer(self, trafo, pte_by_side: Dict[str, str]):
        # Reconstruct the tap changer on the end given by tap_side. RatioTapChanger and the phase
        # tap changers (Linear/Asymmetrical/Symmetrical) are supported; PhaseTapChangerTabular needs
        # table points that are not preserved on the net, so it is skipped (the transformer then
        # re-imports without a tap changer).
        tc_class = trafo.get(sc['tc'])
        pte_id = pte_by_side.get(trafo.get('tap_side'))
        if pte_id is None or pd.isna(tc_class):
            return
        tc_id = self._id_or_new(trafo.get(sc['tc_id']))
        common = {'rdfId': tc_id, 'TransformerEnd': pte_id, 'neutralStep': trafo.get('tap_neutral'),
                  'lowStep': trafo.get('tap_min'), 'highStep': trafo.get('tap_max')}
        if tc_class == 'RatioTapChanger':
            self._tc_eq[tc_class].append({**common, 'stepVoltageIncrement': trafo.get('tap_step_percent')})
        elif tc_class == 'PhaseTapChangerLinear':
            self._tc_eq[tc_class].append({**common, 'stepPhaseShiftIncrement': trafo.get('tap_step_degree')})
        elif tc_class == 'PhaseTapChangerAsymmetrical':
            self._tc_eq[tc_class].append({**common, 'voltageStepIncrement': trafo.get('tap_step_percent'),
                                          'windingConnectionAngle': trafo.get('tap_step_degree')})
        elif tc_class == 'PhaseTapChangerSymmetrical':
            self._tc_eq[tc_class].append({**common, 'voltageStepIncrement': trafo.get('tap_step_percent')})
        else:
            return  # PhaseTapChangerTabular and others are not supported yet
        self._tc_ssh[tc_class].append({'rdfId': tc_id, 'step': trafo.get('tap_pos')})

    def _finalize_tap_changers(self):
        for tc_class in self._tc_eq:
            self._set('eq', tc_class, self._tc_eq[tc_class])
            self._set('ssh', tc_class, self._tc_ssh[tc_class])

    def _convert_power_transformers_3w(self):
        # 3-winding PowerTransformer. The pandapower per-winding-pair short-circuit values are the
        # importer's recombination of the three CGMES end impedances (each referred to its own end's
        # voltage). We invert that pairwise system to recover the per-end r/x (positive and zero
        # sequence); the magnetizing branch is placed on the HV end.
        pt_rows, end_rows = [], []
        for idx, trafo in self.net.trafo3w.iterrows():
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
            g_hv = self._safe(trafo.get('pfe_kw')) / (u['hv'] ** 2 * 1000)
            i0_s = self._safe(trafo.get('i0_percent')) * s['hv'] / 100
            b_sq = i0_s ** 2 - (g_hv * u['hv'] ** 2) ** 2
            b_hv = math.sqrt(b_sq) / u['hv'] ** 2 if b_sq > 0 else 0.0
            clock = {'hv': 0,
                     'mv': self._safe(trafo.get('shift_mv_degree')) / 30,
                     'lv': self._safe(trafo.get('shift_lv_degree')) / 30}
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
            self._add_tap_changer(trafo, pte)
        self._set('eq', 'PowerTransformer', pt_rows)
        self._set('eq', 'PowerTransformerEnd', end_rows)

    def _x_targets(self, trafo, vk_hv_col, vk_mv_col, vk_lv_col, a_hv, a_mv, a_lv, r, k_hvmv, k_mvlv, k_lvhv):
        # given the solved per-end r values, recover the per-pair reactance sums from the vk values
        def bx(vk_col, a, pair_r):
            z_pair = self._safe(trafo.get(vk_col)) / a
            return math.sqrt(max(z_pair ** 2 - pair_r ** 2, 0.0))
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

    def _convert_equivalent_injections(self):
        # EquivalentInjection -> ward (regulationStatus False) or xward (regulationStatus True)
        eq_rows, ssh_rows = [], []
        for table, regulation_status in (('ward', False), ('xward', True)):
            df = self.net[table]
            for idx, row in df.iterrows():
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
            for idx, trafo in self.net[table].iterrows():
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
        for idx, shunt in self.net.shunt.iterrows():
            if shunt.get(sc['o_cl']) not in (None, 'LinearShuntCompensator') and not pd.isna(shunt.get(sc['o_cl'])):
                continue
            rows.append({'rdfId': _new_uuid(), 'ShuntCompensator': shunt.get(sc['o_id']),
                         'sections': shunt.get('step')})
        self._set('sv', 'SvShuntCompensatorSections', rows)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _id_or_new(value):
        # NaN is truthy, so "value or _new_uuid()" would keep a NaN id; use an explicit check.
        return _new_uuid() if value is None or pd.isna(value) else value

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

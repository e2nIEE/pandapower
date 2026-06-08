# -*- coding: utf-8 -*-

"""
Net-level round-trip tests for the CGMES exporter (pp2cim).

These tests import a CGMES grid into a pandapower net, export it back to CGMES with ``to_cim`` and
re-import the exported files. The re-imported net must reproduce the originally converted elements.

This first exporter slice covers buses (TopologicalNode / ConnectivityNode), terminals,
ACLineSegments (lines), loads (EnergyConsumer / ConformLoad / NonConformLoad / StationSupply) and
ExternalNetworkInjections. Generators (SynchronousMachine, EnergySource), transformers, switches,
shunts, impedances and wards are not part of this slice yet, so the tests assert only on the
covered element types.

Covered so far: buses, terminals, lines, loads, ExternalNetworkInjections, switches
(Breaker/Disconnector/LoadBreakSwitch/Switch), EquivalentInjections (ward/xward),
SynchronousMachines (gen/sgen), shunts (LinearShuntCompensator, StaticVarCompensator) and
impedances (SeriesCompensator), 2-winding and 3-winding PowerTransformers and RatioTapChangers.
NonlinearShuntCompensators need per-section data not preserved on the net, and phase tap changers
(Linear/Asymmetrical/Symmetrical/Tabular) are not reversed yet.
"""

import os
import tempfile

import pandas as pd
import pytest

from pandapower.test import test_path

from pandapower.converter.cim.cim2pp.from_cim import from_cim
from pandapower.converter.cim.pp2cim import to_cim
from pandapower.run import runpp

example_cim_path = os.path.join(test_path, "test_files", "example_cim")


def _import_fullgrid_bb():
    files = [os.path.join(example_cim_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BB_BE_v1.zip'),
             os.path.join(example_cim_path, 'CGMES_v2.4.15_FullGridTestConfiguration_BD_v1.zip')]
    return from_cim(file_list=files, cgmes_version='2.4.15')


def _roundtrip(net, cgmes_version='2.4.15'):
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "export.zip")
    to_cim(net, file_path=zip_path, cgmes_version=cgmes_version)
    return net, from_cim(file_list=[zip_path], cgmes_version=cgmes_version)


@pytest.fixture(scope="module")
def fullgrid_bb_roundtrip():
    return _roundtrip(_import_fullgrid_bb())


@pytest.fixture(scope="module")
def microgrid_nb_roundtrip():
    # node-breaker model: buses originate from ConnectivityNodes (not TopologicalNodes)
    files = [os.path.join(example_cim_path, 'CGMES_v2.4.15_MicroGridTestConfiguration_T4_BE_NB_Complete_v2.zip')]
    return _roundtrip(from_cim(file_list=files, cgmes_version='2.4.15'))


@pytest.fixture(scope="module")
def minigrid_roundtrip():
    # large node-breaker grid: 101 buses, 90 switches, sgens (ENI + SM), 3-winding transformers
    files = [os.path.join(example_cim_path, 'CGMES_v2.4.15_MiniGridTestConfiguration_T1_Complete_v3.zip')]
    return _roundtrip(from_cim(file_list=files, cgmes_version='2.4.15'))


# --------------------------------------------------------------------------- counts

def test_bus_count_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.bus) == len(net.bus)


def test_line_count_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.line) == len(net.line)


def test_load_count_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.load) == len(net.load)


# --------------------------------------------------------------------------- bus values

def test_bus_voltage_levels_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.set_index('origin_id') if False else net.bus.set_index('origin_id')['vn_kv'].sort_index()
    roundtrip = net_rt.bus.set_index('origin_id')['vn_kv'].sort_index()
    assert list(original.index) == list(roundtrip.index)
    assert original.round(6).tolist() == roundtrip.round(6).tolist()


# --------------------------------------------------------------------------- line values

def test_line_parameters_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.line.set_index('origin_id').sort_index()
    roundtrip = net_rt.line.set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    for col in ['length_km', 'r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km']:
        assert original[col].round(6).tolist() == pytest.approx(roundtrip[col].round(6).tolist(), abs=1e-5)


def test_line_endpoints_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    # the buses a line connects to (by their CGMES id) must be preserved
    def endpoints(n):
        bus_oid = n.bus['origin_id']
        df = n.line[['origin_id', 'from_bus', 'to_bus']].copy()
        df['from_oid'] = df['from_bus'].map(bus_oid)
        df['to_oid'] = df['to_bus'].map(bus_oid)
        return df.set_index('origin_id')[['from_oid', 'to_oid']].sort_index()
    o, r = endpoints(net), endpoints(net_rt)
    assert o.equals(r)


# --------------------------------------------------------------------------- load values

def test_load_power_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.load.set_index('origin_id').sort_index()
    roundtrip = net_rt.load.set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    assert original['p_mw'].round(6).tolist() == pytest.approx(roundtrip['p_mw'].round(6).tolist(), abs=1e-5)
    assert original['q_mvar'].round(6).tolist() == pytest.approx(roundtrip['q_mvar'].round(6).tolist(), abs=1e-5)


# --------------------------------------------------------------------------- switches

def test_switch_count_and_types_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.switch) == len(net.switch)
    original = net.switch.set_index('origin_id')['origin_class'].sort_index()
    roundtrip = net_rt.switch.set_index('origin_id')['origin_class'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


def test_switch_state_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.switch.set_index('origin_id')['closed'].sort_index()
    roundtrip = net_rt.switch.set_index('origin_id')['closed'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


# --------------------------------------------------------------------------- wards

def test_ward_count_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.ward) == len(net.ward)
    assert len(net_rt.xward) == len(net.xward)


def test_ward_power_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.ward.set_index('origin_id').sort_index()
    roundtrip = net_rt.ward.set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    assert original['ps_mw'].round(6).tolist() == pytest.approx(roundtrip['ps_mw'].round(6).tolist(), abs=1e-5)
    assert original['qs_mvar'].round(6).tolist() == pytest.approx(roundtrip['qs_mvar'].round(6).tolist(), abs=1e-5)


# --------------------------------------------------------------------------- generators

def test_gen_count_and_classification_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.gen) == len(net.gen)
    assert len(net_rt.sgen) == len(net.sgen)
    # the gen/sgen split (driven by RegulatingControl + referencePriority) must reproduce per source
    original = net.gen.set_index('origin_id')['origin_class'].sort_index()
    roundtrip = net_rt.gen.set_index('origin_id')['origin_class'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


def test_gen_setpoints_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    sm = net.gen[net.gen['origin_class'] == 'SynchronousMachine'].set_index('origin_id').sort_index()
    sm_rt = net_rt.gen[net_rt.gen['origin_class'] == 'SynchronousMachine'].set_index('origin_id').sort_index()
    assert list(sm.index) == list(sm_rt.index)
    assert sm['p_mw'].round(6).tolist() == pytest.approx(sm_rt['p_mw'].round(6).tolist(), abs=1e-5)
    assert sm['sn_mva'].round(6).tolist() == pytest.approx(sm_rt['sn_mva'].round(6).tolist(), abs=1e-5)


def test_gen_slack_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.gen.set_index('origin_id')['slack'].sort_index()
    roundtrip = net_rt.gen.set_index('origin_id')['slack'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


# --------------------------------------------------------------------------- shunts

def test_linear_and_svc_shunts_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    supported = ['LinearShuntCompensator', 'StaticVarCompensator']
    original = net.shunt[net.shunt['origin_class'].isin(supported)].set_index('origin_id').sort_index()
    roundtrip = net_rt.shunt[net_rt.shunt['origin_class'].isin(supported)].set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    # linear shunt power per section must survive the gPerSection/bPerSection round-trip
    lin = original[original['origin_class'] == 'LinearShuntCompensator']
    lin_rt = roundtrip.loc[lin.index]
    assert lin['p_mw'].round(6).tolist() == pytest.approx(lin_rt['p_mw'].round(6).tolist(), abs=1e-5)
    assert lin['q_mvar'].round(6).tolist() == pytest.approx(lin_rt['q_mvar'].round(6).tolist(), abs=1e-5)


# --------------------------------------------------------------------------- impedance

def test_series_compensator_impedance_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.impedance[net.impedance['origin_class'] == 'SeriesCompensator'].set_index('origin_id').sort_index()
    roundtrip = net_rt.impedance[
        net_rt.impedance['origin_class'] == 'SeriesCompensator'].set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    for col in ['rft_pu', 'xft_pu']:
        assert original[col].round(6).tolist() == pytest.approx(roundtrip[col].round(6).tolist(), abs=1e-5)


# --------------------------------------------------------------------------- transformers

def test_trafo2w_count_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.trafo) == len(net.trafo)


def test_trafo2w_parameters_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.trafo.set_index('origin_id').sort_index()
    roundtrip = net_rt.trafo.set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    # the series impedance and magnetizing branch are reconstructed on the HV end and must reproduce
    # the importer's per-unit transformer parameters
    for col in ['vn_hv_kv', 'vn_lv_kv', 'sn_mva', 'vk_percent', 'vkr_percent', 'i0_percent', 'pfe_kw']:
        assert original[col].astype(float).round(6).tolist() == \
               pytest.approx(roundtrip[col].astype(float).round(6).tolist(), abs=1e-4)


def test_trafo2w_endpoints_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    def endpoints(n):
        bus_oid = n.bus['origin_id']
        df = n.trafo[['origin_id', 'hv_bus', 'lv_bus']].copy()
        df['hv_oid'] = df['hv_bus'].map(bus_oid)
        df['lv_oid'] = df['lv_bus'].map(bus_oid)
        return df.set_index('origin_id')[['hv_oid', 'lv_oid']].sort_index()
    assert endpoints(net).equals(endpoints(net_rt))


def test_trafo2w_ratio_tap_changer_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    rtc = net.trafo[net.trafo['tapchanger_class'] == 'RatioTapChanger'].set_index('origin_id').sort_index()
    rtc_rt = net_rt.trafo.set_index('origin_id').loc[rtc.index]
    assert len(rtc) > 0
    for col in ['tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_pos', 'tap_step_percent']:
        assert rtc[col].astype(str).tolist() == rtc_rt[col].astype(str).tolist()


def test_phase_tap_changers_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    phase = net.trafo[net.trafo['tapchanger_class'].astype(str).str.startswith('Phase')]
    phase = phase[phase['tapchanger_class'] != 'PhaseTapChangerTabular'].set_index('origin_id').sort_index()
    assert len(phase) > 0
    phase_rt = net_rt.trafo.set_index('origin_id').loc[phase.index]
    # the tap changer class and current position must survive
    assert phase['tapchanger_class'].tolist() == phase_rt['tapchanger_class'].tolist()
    assert phase['tap_pos'].tolist() == phase_rt['tap_pos'].tolist()


def test_trafo3w_count_and_parameters_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    assert len(net_rt.trafo3w) == len(net.trafo3w)
    original = net.trafo3w.set_index('origin_id').sort_index()
    roundtrip = net_rt.trafo3w.set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    # the per-winding-pair short-circuit values must survive the pairwise-system inversion
    for col in ['vn_hv_kv', 'vn_mv_kv', 'vn_lv_kv', 'sn_hv_mva', 'sn_mv_mva', 'sn_lv_mva',
                'vk_hv_percent', 'vk_mv_percent', 'vk_lv_percent',
                'vkr_hv_percent', 'vkr_mv_percent', 'vkr_lv_percent']:
        assert original[col].astype(float).round(5).tolist() == \
               pytest.approx(roundtrip[col].astype(float).round(5).tolist(), abs=1e-4)


# --------------------------------------------------------------------------- pipeline / api

def test_to_cim_returns_structure_without_writing():
    net = _import_fullgrid_bb()
    cim = to_cim(net, cgmes_version='2.4.15')
    assert not cim['eq']['ACLineSegment'].empty
    assert not cim['eq']['Terminal'].empty
    assert not cim['tp']['TopologicalNode'].empty
    # every line has two terminals
    assert (cim['eq']['Terminal']['ConductingEquipment'].isin(cim['eq']['ACLineSegment']['rdfId']).sum()
            == 2 * len(net.line))


def test_export_zip_written_and_reimportable(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    # the fixture already re-imported the zip; a non-empty net proves the pipeline works end to end
    assert len(net_rt.bus) > 0
    assert len(net_rt.line) > 0


# --------------------------------------------------------------------------- node-breaker model

def test_microgrid_nb_bus_topology_roundtrip(microgrid_nb_roundtrip):
    net, net_rt = microgrid_nb_roundtrip
    # buses come from ConnectivityNodes in a node-breaker model
    assert (net.bus['origin_class'] == 'ConnectivityNode').all()
    assert len(net_rt.bus) == len(net.bus)
    assert (net_rt.bus['origin_class'] == 'ConnectivityNode').all()


def test_microgrid_nb_elements_roundtrip(microgrid_nb_roundtrip):
    net, net_rt = microgrid_nb_roundtrip
    for table in ['line', 'load', 'gen', 'switch', 'impedance', 'ward', 'trafo', 'trafo3w']:
        assert len(getattr(net_rt, table)) == len(getattr(net, table)), \
            "%s count changed in node-breaker round-trip" % table


def test_microgrid_nb_switch_connectivity_roundtrip(microgrid_nb_roundtrip):
    net, net_rt = microgrid_nb_roundtrip
    def switch_topology(n):
        bus_oid = n.bus['origin_id']
        df = n.switch[['origin_id', 'bus', 'element', 'closed']].copy()
        df['bus_oid'] = df['bus'].map(bus_oid)
        df['element_oid'] = df['element'].map(bus_oid)
        return df.set_index('origin_id')[['bus_oid', 'element_oid', 'closed']].sort_index()
    assert switch_topology(net).equals(switch_topology(net_rt))


def test_microgrid_nb_line_endpoints_roundtrip(microgrid_nb_roundtrip):
    net, net_rt = microgrid_nb_roundtrip
    def endpoints(n):
        bus_oid = n.bus['origin_id']
        df = n.line[['origin_id', 'from_bus', 'to_bus']].copy()
        df['from_oid'] = df['from_bus'].map(bus_oid)
        df['to_oid'] = df['to_bus'].map(bus_oid)
        return df.set_index('origin_id')[['from_oid', 'to_oid']].sort_index()
    assert endpoints(net).equals(endpoints(net_rt))


# --------------------------------------------------------------------------- full grid coverage

def test_minigrid_all_elements_roundtrip(minigrid_roundtrip):
    net, net_rt = minigrid_roundtrip
    # this grid is fully covered by the implemented converters: every element type must round-trip
    for table in ['bus', 'line', 'load', 'ext_grid', 'gen', 'sgen', 'switch', 'shunt',
                  'impedance', 'ward', 'xward', 'trafo', 'trafo3w']:
        assert len(getattr(net_rt, table)) == len(getattr(net, table)), \
            "%s count changed: %d -> %d" % (table, len(getattr(net, table)), len(getattr(net_rt, table)))


def test_minigrid_sgen_sources_roundtrip(minigrid_roundtrip):
    net, net_rt = minigrid_roundtrip
    # sgens originate from both ExternalNetworkInjection and SynchronousMachine here
    original = net.sgen.set_index('origin_id')['origin_class'].sort_index()
    roundtrip = net_rt.sgen.set_index('origin_id')['origin_class'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


# --------------------------------------------------------------------------- SV profile (results)

@pytest.fixture(scope="module")
def multivoltage_solved():
    files = [os.path.join(example_cim_path, 'example_multivoltage.zip')]
    net = from_cim(file_list=files, cgmes_version='2.4.15')
    runpp(net, calculate_voltage_angles="auto")
    return net


def _topological_node(bus_row):
    return bus_row['cim_topnode'] if bus_row['origin_class'] == 'ConnectivityNode' else bus_row['origin_id']


def test_sv_voltage_exported_from_results(multivoltage_solved):
    net = multivoltage_solved
    cim = to_cim(net, cgmes_version='2.4.15')
    sv_voltage = cim['sv']['SvVoltage']
    assert not sv_voltage.empty
    # SvVoltage is per TopologicalNode and must equal res_bus.vm_pu * vn_kv
    expected = {}
    for bus_idx, bus in net.bus.iterrows():
        tn = _topological_node(bus)
        if pd.isna(tn) or tn in expected:
            continue
        if bus_idx in net.res_bus.index and not pd.isna(net.res_bus.loc[bus_idx, 'vm_pu']):
            expected[tn] = net.res_bus.loc[bus_idx, 'vm_pu'] * bus['vn_kv']
    assert len(sv_voltage) == len(expected)
    exported = dict(zip(sv_voltage['TopologicalNode'], sv_voltage['v']))
    for tn, v in expected.items():
        assert exported[tn] == pytest.approx(v, abs=1e-6)


def test_sv_not_exported_without_results():
    # a net without power-flow results must not produce an SV profile
    net = _import_fullgrid_bb()
    cim = to_cim(net, cgmes_version='2.4.15')
    assert cim['sv']['SvVoltage'].empty
    assert 'FullModel' not in cim['sv']


def test_sv_profile_reimportable(multivoltage_solved):
    net = multivoltage_solved
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "export_sv.zip")
    to_cim(net, file_path=zip_path, cgmes_version='2.4.15')
    net_rt = from_cim(file_list=[zip_path], cgmes_version='2.4.15')
    assert len(net_rt.bus) == len(net.bus)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

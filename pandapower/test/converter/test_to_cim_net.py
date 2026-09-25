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

import pandapower as pp
from pandapower.test import test_path

from pandapower.converter.cim import cim_tools
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
    original = net.bus.set_index('origin_id')['vn_kv'].sort_index()
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


def test_line_current_limit_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.line.set_index('origin_id').sort_index()['max_i_ka']
    roundtrip = net_rt.line.set_index('origin_id').sort_index()['max_i_ka']
    assert original.notna().any()
    # the CurrentLimit / OperationalLimitSet reconstruction must preserve max_i_ka
    assert original.round(6).tolist() == pytest.approx(roundtrip.round(6).tolist(), nan_ok=True, abs=1e-6)


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


def test_energy_source_ext_grid_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    # this grid's slack is an EnergySource routed to ext_grid (it carries a voltage set point)
    assert (net.ext_grid['origin_class'] == 'EnergySource').any()
    original = net.ext_grid.set_index('origin_id')['origin_class'].sort_index()
    roundtrip = net_rt.ext_grid.set_index('origin_id')['origin_class'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


def test_gen_slack_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.gen.set_index('origin_id')['slack'].sort_index()
    roundtrip = net_rt.gen.set_index('origin_id')['slack'].sort_index()
    assert original.to_dict() == roundtrip.to_dict()


# --------------------------------------------------------------------------- shunts

def test_all_shunts_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    # every shunt subtype (Linear, Nonlinear, StaticVarCompensator) must round-trip with its class
    assert len(net_rt.shunt) == len(net.shunt)
    original = net.shunt.set_index('origin_id').sort_index()
    roundtrip = net_rt.shunt.set_index('origin_id').sort_index()
    assert list(original.index) == list(roundtrip.index)
    assert original['origin_class'].tolist() == roundtrip['origin_class'].tolist()
    # power must survive for linear and nonlinear shunts (SVC active power is defined as 0)
    pq = original[original['origin_class'] != 'StaticVarCompensator']
    pq_rt = roundtrip.loc[pq.index]
    assert pq['p_mw'].round(5).tolist() == pytest.approx(pq_rt['p_mw'].round(5).tolist(), abs=1e-4)
    assert pq['q_mvar'].round(5).tolist() == pytest.approx(pq_rt['q_mvar'].round(5).tolist(), abs=1e-4)


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


def test_trafo2w_vector_group_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    original = net.trafo.set_index('origin_id').sort_index()['vector_group']
    roundtrip = net_rt.trafo.set_index('origin_id').sort_index()['vector_group']
    assert original.notna().any()
    # connectionKind per winding is reconstructed from the vector group (e.g. 'YNyn', 'Yy')
    assert original.astype(str).tolist() == roundtrip.astype(str).tolist()


def test_trafo_current_limits_roundtrip(fullgrid_bb_roundtrip):
    net, net_rt = fullgrid_bb_roundtrip
    for table, sides in (('trafo', ('hv', 'lv')), ('trafo3w', ('hv', 'mv', 'lv'))):
        original = getattr(net, table).set_index('origin_id').sort_index()
        roundtrip = getattr(net_rt, table).set_index('origin_id').sort_index()
        for side in sides:
            value_col = 'CurrentLimit.value_%s' % side
            assert original[value_col].notna().any()
            assert original[value_col].round(4).tolist() == \
                   pytest.approx(roundtrip.loc[original.index, value_col].round(4).tolist(), nan_ok=True, abs=1e-3)
            limit_col = 'OperationalLimitType.limitType_%s' % side
            assert original[limit_col].astype(str).tolist() == \
                   roundtrip.loc[original.index, limit_col].astype(str).tolist()


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
    _, net_rt = fullgrid_bb_roundtrip
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


ALL_ELEMENT_TABLES = ['bus', 'line', 'load', 'ext_grid', 'gen', 'sgen', 'switch', 'shunt',
                      'impedance', 'ward', 'xward', 'trafo', 'trafo3w']


def _assert_all_elements_roundtrip(net, net_rt):
    for table in ALL_ELEMENT_TABLES:
        assert len(getattr(net_rt, table)) == len(getattr(net, table)), \
            "%s count changed: %d -> %d" % (table, len(getattr(net, table)), len(getattr(net_rt, table)))


def test_fullgrid_all_elements_roundtrip(fullgrid_bb_roundtrip):
    # the bus-branch reference grid must round-trip every element type completely
    _assert_all_elements_roundtrip(*fullgrid_bb_roundtrip)


def test_microgrid_nb_elements_roundtrip(microgrid_nb_roundtrip):
    _assert_all_elements_roundtrip(*microgrid_nb_roundtrip)


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


def test_node_breaker_bus_voltage_roundtrip(minigrid_roundtrip):
    net, net_rt = minigrid_roundtrip
    # node-breaker buses get their voltage via the ConnectivityNode -> VoltageLevel -> BaseVoltage
    # container chain, which must be reconstructed even without a geo/DL profile
    assert (net.bus['origin_class'] == 'ConnectivityNode').all()
    original = net.bus.set_index('origin_id')['vn_kv'].sort_index()
    roundtrip = net_rt.bus.set_index('origin_id')['vn_kv'].sort_index()
    assert original.notna().all()
    assert list(original.index) == list(roundtrip.index)
    assert original.round(6).tolist() == pytest.approx(roundtrip.round(6).tolist(), abs=1e-6)


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


# --------------------------------------------------------------------------- diagram coordinates (DL)

@pytest.fixture(scope="module")
def smallgrid_dl_roundtrip():
    files = [os.path.join(example_cim_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
             os.path.join(example_cim_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]
    net = from_cim(file_list=files, cgmes_version='2.4.15', use_GL_or_DL_profile='DL')
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "export_dl.zip")
    to_cim(net, file_path=zip_path, cgmes_version='2.4.15')
    net_rt = from_cim(file_list=[zip_path], cgmes_version='2.4.15', use_GL_or_DL_profile='DL')
    return net, net_rt


def test_bus_diagram_coordinates_roundtrip(smallgrid_dl_roundtrip):
    net, net_rt = smallgrid_dl_roundtrip
    original = net.bus.set_index('origin_id')['diagram'].dropna()
    roundtrip = net_rt.bus.set_index('origin_id')['diagram'].dropna()
    assert len(original) > 0
    assert original.to_dict() == roundtrip.loc[original.index].to_dict()


def test_line_diagram_coordinates_roundtrip(smallgrid_dl_roundtrip):
    net, net_rt = smallgrid_dl_roundtrip
    # lines carry a LineString (multiple points) which must survive in order
    original = net.line.set_index('origin_id')['diagram'].dropna()
    roundtrip = net_rt.line.set_index('origin_id')['diagram'].dropna()
    assert len(original) > 0
    assert original.to_dict() == roundtrip.loc[original.index].to_dict()


# --------------------------------------------------------------------------- geographic coordinates (GL)

@pytest.fixture(scope="module")
def smallgrid_gl_roundtrip():
    files = [os.path.join(example_cim_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_Boundary_v3.0.0.zip'),
             os.path.join(example_cim_path, 'CGMES_v2.4.15_SmallGridTestConfiguration_BaseCase_Complete_v3.0.0.zip')]
    net = from_cim(file_list=files, cgmes_version='2.4.15', use_GL_or_DL_profile='GL')
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "export_gl.zip")
    to_cim(net, file_path=zip_path, cgmes_version='2.4.15')
    net_rt = from_cim(file_list=[zip_path], cgmes_version='2.4.15', use_GL_or_DL_profile='GL')
    return net, net_rt


def test_bus_geo_coordinates_roundtrip(smallgrid_gl_roundtrip):
    net, net_rt = smallgrid_gl_roundtrip
    # bus geo is attached to the Substation; the reconstructed VoltageLevel/Substation must map it back
    original = net.bus.set_index('origin_id')['geo'].dropna()
    roundtrip = net_rt.bus.set_index('origin_id')['geo'].dropna()
    assert len(original) > 0
    assert original.to_dict() == roundtrip.loc[original.index].to_dict()


def test_line_geo_coordinates_roundtrip(smallgrid_gl_roundtrip):
    net, net_rt = smallgrid_gl_roundtrip
    original = net.line.set_index('origin_id')['geo'].dropna()
    roundtrip = net_rt.line.set_index('origin_id')['geo'].dropna()
    assert len(original) > 0
    assert original.to_dict() == roundtrip.loc[original.index].to_dict()


def test_bus_zone_restored_via_substation(smallgrid_gl_roundtrip):
    net, net_rt = smallgrid_gl_roundtrip
    # reconstructing Substation/VoltageLevel for the geo also restores the bus zone (substation name)
    assert net.bus['zone'].notna().any()
    assert net_rt.bus['zone'].notna().sum() == net.bus['zone'].notna().sum()


# --------------------------------------------------------------------------- synthetic grids

# Some CIM classes are not exercised by the sample CGMES grids (their instances are dropped on
# import or simply absent). These are covered by building a small pandapower net from scratch,
# tagging the relevant origin_class, exporting it and re-importing it.

def _build_synthetic_net():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110., name='B1')
    b2 = pp.create_bus(net, vn_kv=110., name='B2')
    pp.create_line_from_parameters(net, b1, b2, length_km=2.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
                                   c_nf_per_km=11.0, max_i_ka=0.5, name='L1')
    pp.create_load(net, b2, p_mw=5.0, q_mvar=2.0, name='LD1')
    es = pp.create_sgen(net, b2, p_mw=3.0, q_mvar=1.0, name='ES1')
    im = pp.create_impedance(net, b1, b2, rft_pu=0.01, xft_pu=0.05, sn_mva=net.sn_mva,
                             rtf_pu=0.02, xtf_pu=0.06, name='EB1')
    # add the CGMES helper columns, then tag the elements with the CIM class to export them as
    cim_tools.extend_pp_net_cim(net, override=False)
    net.sgen.loc[es, 'origin_class'] = 'EnergySource'
    net.impedance.loc[im, 'origin_class'] = 'EquivalentBranch'
    return net


@pytest.fixture(scope="module")
def synthetic_roundtrip():
    return _roundtrip(_build_synthetic_net())


def test_synthetic_from_scratch_foundation(synthetic_roundtrip):
    # a hand-built net (no origin_id) must export with generated UUIDs and re-import intact
    net, net_rt = synthetic_roundtrip
    assert len(net_rt.bus) == len(net.bus)
    assert len(net_rt.line) == len(net.line)
    assert len(net_rt.load) == len(net.load)


def test_synthetic_energy_source_roundtrip(synthetic_roundtrip):
    _, net_rt = synthetic_roundtrip
    es = net_rt.sgen[net_rt.sgen['origin_class'] == 'EnergySource']
    assert len(es) == 1
    assert es['p_mw'].iloc[0] == pytest.approx(3.0, abs=1e-6)
    assert es['q_mvar'].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_synthetic_equivalent_branch_roundtrip(synthetic_roundtrip):
    _, net_rt = synthetic_roundtrip
    eb = net_rt.impedance[net_rt.impedance['origin_class'] == 'EquivalentBranch']
    assert len(eb) == 1
    assert eb['rft_pu'].iloc[0] == pytest.approx(0.01, abs=1e-6)
    assert eb['xft_pu'].iloc[0] == pytest.approx(0.05, abs=1e-6)
    # the asymmetric (tf) values must survive too
    assert eb['rtf_pu'].iloc[0] == pytest.approx(0.02, abs=1e-6)
    assert eb['xtf_pu'].iloc[0] == pytest.approx(0.06, abs=1e-6)


@pytest.mark.xfail(reason="Known limitation, left for discussion: a 2-winding transformer's impedance "
                          "is exported entirely on the HV winding. pandapower stores only the total "
                          "impedance, so a source model that splits impedance across both windings "
                          "cannot be reproduced per-winding (the total still round-trips). See the "
                          "Limitations section of the converter docs.")
def test_2w_transformer_impedance_split_not_preserved():
    # Build a 2-winding transformer and export it. The exporter places all series impedance on the
    # HV PowerTransformerEnd (endNumber 1) and leaves the LV end (endNumber 2) ideal (r = x = 0).
    # A source model with individual impedances on *both* windings would therefore not round-trip its
    # per-winding split. This test documents that gap by asserting that the LV winding also carries
    # impedance - which currently fails by design.
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=220.)
    b2 = pp.create_bus(net, vn_kv=110.)
    pp.create_ext_grid(net, b1)
    pp.create_transformer_from_parameters(net, b1, b2, sn_mva=100., vn_hv_kv=220., vn_lv_kv=110.,
                                          vk_percent=12.0, vkr_percent=0.5, pfe_kw=0., i0_percent=0.)
    cim_tools.extend_pp_net_cim(net, override=False)
    net.trafo.loc[0, 'origin_class'] = 'PowerTransformer'

    ends = to_cim(net, cgmes_version='2.4.15')['eq']['PowerTransformerEnd']
    hv_end = ends[ends['endNumber'] == 1].iloc[0]
    lv_end = ends[ends['endNumber'] == 2].iloc[0]
    assert abs(float(hv_end['r'])) + abs(float(hv_end['x'])) > 0  # HV end carries the impedance
    # the LV end is ideal today; ideally a split source model would keep impedance on both windings
    assert abs(float(lv_end['r'])) + abs(float(lv_end['x'])) > 0, \
        "LV winding carries no impedance - the HV/LV split is not preserved (all lumped on HV)"


def test_synthetic_negative_reactance_transformer_roundtrip():
    # regression: some real transformers have a negative reactance (vk_percent < 0). vk_percent
    # carries the sign of x, which must survive the impedance reconstruction.
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=220.)
    b2 = pp.create_bus(net, vn_kv=110.)
    pp.create_ext_grid(net, b1)
    pp.create_transformer_from_parameters(net, b1, b2, sn_mva=100., vn_hv_kv=220., vn_lv_kv=110.,
                                          vk_percent=-5.0, vkr_percent=0.5, pfe_kw=0., i0_percent=0.,
                                          name='NEGX')
    cim_tools.extend_pp_net_cim(net, override=False)
    net.trafo.loc[0, 'origin_class'] = 'PowerTransformer'
    _, net_rt = _roundtrip(net)
    assert len(net_rt.trafo) == 1
    assert net_rt.trafo['vk_percent'].iloc[0] == pytest.approx(-5.0, abs=1e-6)
    assert net_rt.trafo['vkr_percent'].iloc[0] == pytest.approx(0.5, abs=1e-6)


def test_synthetic_tabular_tap_changer_roundtrip():
    # a tabular phase tap changer defines a per-step ratio table (net['trafo_characteristic_table']).
    # The exporter rebuilds the PhaseTapChangerTable/TablePoints so the per-step voltage_ratio - and
    # hence the transformer ratio at the operating tap - survives the round-trip.
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=220.)
    b2 = pp.create_bus(net, vn_kv=110.)
    pp.create_ext_grid(net, b1)
    t = pp.create_transformer_from_parameters(net, b1, b2, sn_mva=100., vn_hv_kv=220., vn_lv_kv=110.,
                                              vk_percent=12., vkr_percent=0.5, pfe_kw=0., i0_percent=0.)
    cim_tools.extend_pp_net_cim(net, override=False)
    net.trafo.loc[t, 'origin_class'] = 'PowerTransformer'
    net.trafo.loc[t, ['tapchanger_class', 'tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_pos',
                      'id_characteristic_table']] = ['PhaseTapChangerTabular', 'hv', 0, -2, 2, 1, 7]
    net['trafo_characteristic_table'] = pd.DataFrame({
        'id_characteristic': [7] * 5, 'step': [-2, -1, 0, 1, 2],
        'voltage_ratio': [0.90, 0.95, 1.00, 1.05, 1.10], 'angle_deg': [0.] * 5,
        'vk_percent': [12.] * 5, 'vkr_percent': [0.5] * 5})

    _, net_rt = _roundtrip(net)
    assert len(net_rt.trafo) == 1
    assert net_rt.trafo['tapchanger_class'].iloc[0] == 'PhaseTapChangerTabular'
    assert net_rt.trafo['tap_pos'].iloc[0] == 1
    ct = net_rt['trafo_characteristic_table']
    assert sorted(ct['step'].tolist()) == [-2, -1, 0, 1, 2]
    ratio = ct.set_index('step')['voltage_ratio']
    assert ratio.loc[-2] == pytest.approx(0.90, abs=1e-6)
    assert ratio.loc[1] == pytest.approx(1.05, abs=1e-6)  # the operating tap
    assert ratio.loc[2] == pytest.approx(1.10, abs=1e-6)


def test_synthetic_ratio_table_tap_changer_roundtrip():
    # a table-based RatioTapChanger carries a per-step ratio characteristic in addition to its linear
    # stepVoltageIncrement; the exporter rebuilds the RatioTapChangerTable so the per-step ratio
    # survives, while the linear tap_step_percent is kept as well.
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=220.)
    b2 = pp.create_bus(net, vn_kv=110.)
    pp.create_ext_grid(net, b1)
    t = pp.create_transformer_from_parameters(net, b1, b2, sn_mva=100., vn_hv_kv=220., vn_lv_kv=110.,
                                              vk_percent=12., vkr_percent=0.5, pfe_kw=0., i0_percent=0.)
    cim_tools.extend_pp_net_cim(net, override=False)
    net.trafo.loc[t, 'origin_class'] = 'PowerTransformer'
    net.trafo.loc[t, ['tapchanger_class', 'tap_side', 'tap_neutral', 'tap_min', 'tap_max', 'tap_pos',
                      'tap_step_percent', 'id_characteristic_table']] = \
        ['RatioTapChanger', 'hv', 0, -2, 2, -1, 1.25, 3]
    net['trafo_characteristic_table'] = pd.DataFrame({
        'id_characteristic': [3] * 5, 'step': [-2, -1, 0, 1, 2],
        'voltage_ratio': [0.96, 0.98, 1.00, 1.02, 1.04], 'angle_deg': [0.] * 5,
        'vk_percent': [12.] * 5, 'vkr_percent': [0.5] * 5})

    _, net_rt = _roundtrip(net)
    assert len(net_rt.trafo) == 1
    assert net_rt.trafo['tapchanger_class'].iloc[0] == 'RatioTapChanger'
    assert net_rt.trafo['tap_pos'].iloc[0] == -1
    assert net_rt.trafo['tap_step_percent'].iloc[0] == pytest.approx(1.25, abs=1e-6)  # linear part kept
    ratio = net_rt['trafo_characteristic_table'].set_index('step')['voltage_ratio']
    assert ratio.loc[-1] == pytest.approx(0.98, abs=1e-6)  # the operating tap
    assert ratio.loc[2] == pytest.approx(1.04, abs=1e-6)


def test_synthetic_ext_grid_exported_as_eni():
    # a hand-built ext_grid (no origin_class) is exported as an ExternalNetworkInjection
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=110.)
    pp.create_ext_grid(net, b)
    cim_tools.extend_pp_net_cim(net, override=False)
    cim = to_cim(net, cgmes_version='2.4.15')
    assert len(cim['eq']['ExternalNetworkInjection']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

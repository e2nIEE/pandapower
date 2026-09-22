# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Unit tests for the OpenDSS -> pandapower converter (``from_opendss``).

The fixtures are deterministic inline feeders written to a temporary ``.dss`` so
the tests need no external data. The main feeder is balanced (symmetric 3-phase
load, delta-wye transformer), so the balanced pandapower model reproduces the
OpenDSS per-bus voltage to a tight tolerance -- exactly what the positive-sequence
import promises for symmetric feeders.
"""

import faulthandler

import numpy as np
import pytest

import pandapower as pp

# pytest enables faulthandler by default, which on Windows intercepts a first-chance
# structured exception that OpenDSSDirect.py's native backend raises (and normally
# handles itself) during import -- see pandapower/converter/opendss/from_dss.py for
# the full explanation and https://github.com/dss-extensions/OpenDSSDirect.py/issues/148.
# Disable faulthandler for this pre-flight import check the same way, so it doesn't
# crash here before from_dss's own guarded import ever runs.
faulthandler_was_enabled = faulthandler.is_enabled()
faulthandler.disable()
try:
    # OpenDSSDirect.py is an optional dependency; skip the whole module without it.
    pytest.importorskip("opendssdirect")
finally:
    if faulthandler_was_enabled:
        faulthandler.enable()

from pandapower.converter.opendss import from_opendss

INLINE_FEEDER = """
clear
new circuit.test basekv=12.47 pu=1.0 phases=3 bus1=sourcebus
new linecode.lc1 nphases=3 r1=0.1 x1=0.2 c1=3.0 units=km normamps=400
new line.l1 bus1=sourcebus bus2=b1 linecode=lc1 length=1.0 units=km
new transformer.t1 phases=3 windings=2 xhl=5.0
~ wdg=1 bus=b1 conn=delta kv=12.47 kva=500 %r=0.5
~ wdg=2 bus=b2 conn=wye kv=0.48 kva=500 %r=0.5
new line.l2 bus1=b2 bus2=b3 r1=0.05 x1=0.08 c1=0 length=0.2 units=km normamps=600 phases=3
new load.load1 bus1=b3 phases=3 kv=0.48 kw=200 kvar=80 conn=wye
set voltagebases=[12.47, 0.48]
calcvoltagebases
solve
"""


# A center-tapped single-phase service transformer (OpenDSS phases=1 windings=3,
# two LV windings on the same secondary bus) -- the dominant SMART-DS element.
CENTER_TAPPED_FEEDER = """
clear
new circuit.ct basekv=12.47 pu=1.0 phases=3 bus1=src
new line.l1 bus1=src bus2=mv r1=0.1 x1=0.2 c1=0 length=0.5 units=km phases=3 normamps=400
new transformer.ct1 phases=1 windings=3 xhl=2.0 xlt=2.0 xht=1.3
~ wdg=1 bus=mv.1 conn=wye kv=7.2 kva=25 %r=0.5
~ wdg=2 bus=lv.1.0 conn=wye kv=0.12 kva=25 %r=1.0
~ wdg=3 bus=lv.0.2 conn=wye kv=0.12 kva=25 %r=1.0
new load.serv bus1=lv.1.2 phases=1 kv=0.24 kw=15 kvar=3
set voltagebases=[12.47, 0.24, 0.12]
calcvoltagebases
solve
"""


@pytest.fixture
def feeder_path(tmp_path):
    p = tmp_path / "inline.dss"
    p.write_text(INLINE_FEEDER)
    return str(p)


@pytest.fixture
def net(feeder_path):
    return from_opendss(feeder_path)


@pytest.fixture
def center_tapped_net(tmp_path):
    p = tmp_path / "ct.dss"
    p.write_text(CENTER_TAPPED_FEEDER)
    return from_opendss(str(p))


# A European-style 4-wire (3 phase + explicit neutral) matrix LineCode with no
# phase-to-neutral coupling -- the pattern found in a real, independently sourced
# Portuguese LV network (Koirala et al., "Non-Synthetic European Low Voltage Test
# System", 2020) that exposed the bug this guards: OpenDSS's own Lines.R1()/X1()
# do not Kron-reduce the neutral out of a >3-conductor matrix LineCode, so they
# return neither the raw phase diagonal nor the correct sequence value. With zero
# mutual coupling the Kron reduction is a no-op, so the correct R1/X1 is exactly
# the declared phase self-impedance.
FOUR_WIRE_ZERO_MUTUAL_FEEDER = """
clear
new circuit.zm basekv=0.4 pu=1.0 phases=3 bus1=a
new linecode.lc_zm nphases=4 baseFreq=50 units=km
~ rmatrix=[0.160263 | 0 0.160263 | 0 0 0.160263 | 0 0 0 0.230905]
~ xmatrix=[0.079 | 0 0.079 | 0 0 0.079 | 0 0 0 0.085]
new line.l1 bus1=a.1.2.3.4 bus2=b.1.2.3.4 phases=4 linecode=lc_zm length=4.34 units=m
new load.load1 bus1=b.1.2.3 phases=3 kv=0.4 kw=30 kvar=10
set voltagebases=[0.4]
calcvoltagebases
solve
"""

# Same 4-wire shape but with *asymmetric* phase-to-neutral mutual coupling, so a
# genuine Kron reduction changes the answer (unlike the zero-mutual case above,
# which is a no-op) -- this pins down that the reduction, not just a pass-
# through, is happening. Independently computed (numpy, not by hand):
#   Zpp = [[.5,.1,.1],[.1,.5,.1],[.1,.1,.5]], Zpn = [.30,.05,.15]^T, Znn = .2
#   Zred = Zpp - Zpn @ inv(Znn) @ Znp ; R1 = mean(diag(Zred)) - mean(offdiag)/2
# gives R1 = 0.3208333... (vs. 0.4 with no reduction, or 0.5 using the raw
# diagonal) -- three distinct values, so a wrong implementation can't pass by
# accident. xmatrix is R scaled by a constant 0.2 (not zero -- an all-zero
# reactance is a degenerate branch that fails pandapower's DC-initialization
# divide), which keeps X1 = 0.2 * R1 exactly and so is still hand-checkable.
FOUR_WIRE_ASYMMETRIC_FEEDER = """
clear
new circuit.am basekv=0.4 pu=1.0 phases=3 bus1=a
new linecode.lc_am nphases=4 baseFreq=50 units=km
~ rmatrix=[0.5 | 0.1 0.5 | 0.1 0.1 0.5 | 0.30 0.05 0.15 0.2]
~ xmatrix=[0.1 | 0.02 0.1 | 0.02 0.02 0.1 | 0.06 0.01 0.03 0.04]
~ cmatrix=[0 | 0 0 | 0 0 0 | 0 0 0 0]
new line.l1 bus1=a.1.2.3.4 bus2=b.1.2.3.4 phases=4 linecode=lc_am length=1000 units=m
new load.load1 bus1=b.1.2.3 phases=3 kv=0.4 kw=30 kvar=10
set voltagebases=[0.4]
calcvoltagebases
solve
"""

# The bug above turned out not to be specific to >3-conductor/neutral lines at
# all: OpenDSS's Lines.R1()/X1()/C1() are stale symmetric-component fields for
# *any* matrix-defined (rmatrix=/xmatrix=) line, regardless of conductor count
# -- confirmed against OpenDSS's own source, and empirically against a plain
# 3-conductor matrix LineCode with no neutral at all (declares 0.868/0.077
# ohm/km, R1()/X1() returned the hard-coded default 0.058/0.1206 instead).
THREE_WIRE_MATRIX_FEEDER = """
clear
new circuit.tw basekv=0.4 pu=1.0 phases=3 bus1=a
new linecode.lc_tw nphases=3 baseFreq=50 units=km
~ rmatrix=[0.868 | 0.1 0.868 | 0.1 0.1 0.868]
~ xmatrix=[0.077 | 0.01 0.077 | 0.01 0.01 0.077]
new line.l1 bus1=a.1.2.3 bus2=b.1.2.3 phases=3 linecode=lc_tw length=1 units=km
new load.load1 bus1=b.1.2.3 phases=3 kv=0.4 kw=30 kvar=10
set voltagebases=[0.4]
calcvoltagebases
solve
"""

# A single-conductor matrix-defined lateral (the common SMART-DS/IEEE-test-
# feeder pattern for single-phase service drops) -- guards against the naive
# generalization of the Kron-reduction formula (self-avg over 3, mutual-avg
# over 6) misfiring when there are fewer than 3 conductors to average over.
ONE_WIRE_MATRIX_FEEDER = """
clear
new circuit.ow basekv=0.4 pu=1.0 phases=3 bus1=src
new line.lmain bus1=src bus2=a phases=3 r1=0.1 x1=0.2 c1=0 length=0.5 units=km normamps=400
new linecode.lc_ow nphases=1 baseFreq=50 units=km
~ rmatrix=[0.868]
~ xmatrix=[0.077]
new line.l1 bus1=a.1 bus2=b.1 phases=1 linecode=lc_ow length=1 units=km
new load.load1 bus1=b.1 phases=1 kv=0.231 kw=5 kvar=2
set voltagebases=[0.4]
calcvoltagebases
solve
"""


@pytest.fixture
def four_wire_zero_mutual_net(tmp_path):
    p = tmp_path / "zm.dss"
    p.write_text(FOUR_WIRE_ZERO_MUTUAL_FEEDER)
    return from_opendss(str(p))


@pytest.fixture
def four_wire_asymmetric_net(tmp_path):
    p = tmp_path / "am.dss"
    p.write_text(FOUR_WIRE_ASYMMETRIC_FEEDER)
    return from_opendss(str(p))


@pytest.fixture
def three_wire_matrix_net(tmp_path):
    p = tmp_path / "tw.dss"
    p.write_text(THREE_WIRE_MATRIX_FEEDER)
    return from_opendss(str(p))


@pytest.fixture
def one_wire_matrix_net(tmp_path):
    p = tmp_path / "ow.dss"
    p.write_text(ONE_WIRE_MATRIX_FEEDER)
    return from_opendss(str(p))


# A series Reactor bridging sourcebus to the rest of the feeder -- some feeder
# libraries (e.g. EPRI's Ckt5/Ckt7) model the substation's Thevenin-equivalent
# source impedance this way rather than as a Transformer. A shunt reactor
# (single bus, to ground) is a different element and must be skipped, not
# misread as a second bus-to-bus branch.
REACTOR_FEEDER = """
clear
new circuit.rx basekv=12.47 pu=1.05 phases=3 bus1=sourcebus
new reactor.subrx bus1=sourcebus bus2=head phases=3 r=0.01 x=5.0
new reactor.shuntrx bus1=head phases=3 kvar=100
new line.l1 bus1=head bus2=b1 r1=0.1 x1=0.2 c1=0 length=1.0 units=km normamps=400 phases=3
new load.load1 bus1=b1 phases=3 kv=12.47 kw=500 kvar=150
set voltagebases=[12.47]
calcvoltagebases
solve
"""


@pytest.fixture
def reactor_net(tmp_path):
    p = tmp_path / "rx.dss"
    p.write_text(REACTOR_FEEDER)
    return from_opendss(str(p))


def test_element_counts(net):
    assert len(net.bus) == 4          # sourcebus, b1, b2, b3
    assert len(net.line) == 2         # l1, l2
    assert len(net.trafo) == 1        # t1
    assert len(net.load) == 1         # load1
    assert len(net.ext_grid) == 1     # Vsource
    assert net["opendss_import"]["warnings"] == []


def test_bus_voltage_bases_are_line_to_line(net):
    vn = dict(zip(net.bus["name"].str.lower(), net.bus["vn_kv"]))
    assert vn["sourcebus"] == pytest.approx(12.47, rel=1e-6)
    assert vn["b1"] == pytest.approx(12.47, rel=1e-6)
    assert vn["b2"] == pytest.approx(0.48, rel=1e-6)
    assert vn["b3"] == pytest.approx(0.48, rel=1e-6)


def test_linecode_captured_as_std_type(net):
    # l1 references LineCode lc1 -> its name is carried through as std_type;
    # l2 sets R1/X1 inline (no LineCode) -> std_type stays unset.
    std = dict(zip(net.line["name"], net.line["std_type"]))
    assert std["l1"] == "lc1"
    assert std["l2"] is None


def test_total_series_resistance_matches_opendss(net):
    # OpenDSS: l1 = 0.1 ohm/km * 1.0 km, l2 = 0.05 ohm/km * 0.2 km -> 0.11 ohm.
    total_r = float((net.line["r_ohm_per_km"] * net.line["length_km"]).sum())
    assert total_r == pytest.approx(0.11, rel=0.02)


def test_transformer_impedance(net):
    t = net.trafo.iloc[0]
    assert t["vn_hv_kv"] == pytest.approx(12.47)
    assert t["vn_lv_kv"] == pytest.approx(0.48)
    assert t["sn_mva"] == pytest.approx(0.5)
    assert t["vkr_percent"] == pytest.approx(1.0, rel=1e-6)        # 0.5 + 0.5
    assert t["vk_percent"] == pytest.approx(np.hypot(1.0, 5.0), rel=1e-6)


def test_runpp_converges(net):
    pp.runpp(net)
    assert net["converged"]


def test_center_tapped_transformer_collapsed_to_two_winding(center_tapped_net):
    # the 3-winding center-tapped service transformer must be imported (not
    # skipped) as a single 2-winding equivalent, flagged as split-phase.
    assert len(center_tapped_net.trafo) == 1
    assert center_tapped_net["opendss_import"]["n_split_phase_transformers"] == 1
    assert len(center_tapped_net.load) == 1


def test_center_tapped_feeder_converges(center_tapped_net):
    pp.runpp(center_tapped_net)
    assert center_tapped_net["converged"]


def test_roundtrip_voltage_matches_opendss(net):
    pp.runpp(net)
    odss = net["opendss_import"]["vm_pu_opendss"]
    diffs = [
        abs(odss[row["name"].lower()] - net.res_bus.vm_pu[idx])
        for idx, row in net.bus.iterrows()
        if row["name"].lower() in odss
    ]
    # balanced feeder -> agreement to well under 0.1 pp (1e-3 pu)
    assert np.mean(diffs) < 1e-3
    assert np.max(diffs) < 1e-3


def test_four_wire_zero_mutual_matches_declared_conductor(four_wire_zero_mutual_net):
    # Zero phase-to-neutral coupling -> Kron reduction is a no-op, so the correct
    # R1/X1 is exactly the declared phase self-impedance (0.160263 / 0.079 ohm/km).
    # Regression guard for the bug where OpenDSS's own Lines.R1()/X1() returned
    # 0.058/0.1206 ohm/km instead -- neither the declared value nor anything
    # physically derivable from it.
    line = four_wire_zero_mutual_net.line.iloc[0]
    assert line["r_ohm_per_km"] == pytest.approx(0.160263, rel=1e-4)
    assert line["x_ohm_per_km"] == pytest.approx(0.079, rel=1e-4)
    assert line["length_km"] == pytest.approx(4.34e-3, rel=1e-6)


def test_four_wire_asymmetric_neutral_is_kron_reduced(four_wire_asymmetric_net):
    # Asymmetric phase-to-neutral coupling: a genuine reduction changes the
    # answer, so this can't pass via a pass-through/no-op implementation.
    line = four_wire_asymmetric_net.line.iloc[0]
    assert line["r_ohm_per_km"] == pytest.approx(0.3208333, rel=1e-4)
    # must differ from both the un-reduced diagonal (0.5) and the naive
    # self-minus-mutual computed while ignoring the neutral row/column (0.4)
    assert line["r_ohm_per_km"] != pytest.approx(0.5, rel=1e-3)
    assert line["r_ohm_per_km"] != pytest.approx(0.4, rel=1e-3)
    assert line["x_ohm_per_km"] == pytest.approx(0.0641667, rel=1e-4)


def test_four_wire_feeders_converge(four_wire_zero_mutual_net, four_wire_asymmetric_net):
    pp.runpp(four_wire_zero_mutual_net)
    assert four_wire_zero_mutual_net["converged"]
    pp.runpp(four_wire_asymmetric_net)
    assert four_wire_asymmetric_net["converged"]


def test_three_wire_matrix_matches_declared_conductor(three_wire_matrix_net):
    # The bug wasn't neutral-specific: a plain 3-conductor matrix LineCode (no
    # neutral at all) also got the hard-coded 0.058/0.1206 default through
    # Lines.R1()/X1(). Expected R1/X1 = self-avg - mutual-avg = 0.868-0.1,
    # 0.077-0.01 (independently computed, not hand-picked to match a bug).
    line = three_wire_matrix_net.line.iloc[0]
    assert line["r_ohm_per_km"] == pytest.approx(0.768, rel=1e-4)
    assert line["x_ohm_per_km"] == pytest.approx(0.067, rel=1e-4)


def test_one_wire_matrix_matches_declared_conductor(one_wire_matrix_net):
    # A single-conductor matrix lateral (SMART-DS/IEEE-test-feeder style):
    # guards the Kron-reduction formula's self/mutual averaging against
    # misfiring when there are fewer than 3 conductors to average over (no
    # mutual term exists at all here, so R1/X1 must equal the single declared
    # self-impedance exactly, not that value divided by 3).
    line = one_wire_matrix_net.line.iloc[-1]  # l1, the single-phase lateral
    assert line["r_ohm_per_km"] == pytest.approx(0.868, rel=1e-4)
    assert line["x_ohm_per_km"] == pytest.approx(0.077, rel=1e-4)


def test_three_and_one_wire_matrix_feeders_converge(three_wire_matrix_net, one_wire_matrix_net):
    pp.runpp(three_wire_matrix_net)
    assert three_wire_matrix_net["converged"]
    pp.runpp(one_wire_matrix_net)
    assert one_wire_matrix_net["converged"]


def test_series_reactor_imported_as_a_branch(reactor_net):
    # subrx bridges sourcebus->head; one series reactor -> one extra line
    # (l1 is the only OTHER line), and the shunt reactor must not add a second.
    assert reactor_net["opendss_import"]["n_reactors"] == 1
    assert len(reactor_net.line) == 2  # l1 + the series reactor's branch
    reactor_line = reactor_net.line[reactor_net.line["name"] == "subrx"].iloc[0]
    assert reactor_line["r_ohm_per_km"] == pytest.approx(0.01, rel=1e-6)
    assert reactor_line["x_ohm_per_km"] == pytest.approx(5.0, rel=1e-6)


def test_shunt_reactor_is_skipped_not_misread_as_a_branch(reactor_net):
    warnings = reactor_net["opendss_import"]["warnings"]
    assert any("shuntrx" in w and "shunt" in w for w in warnings)


def test_reactor_feeder_reaches_every_bus(reactor_net):
    import pandapower.topology as top
    assert top.unsupplied_buses(reactor_net) == set()


def test_reactor_feeder_converges_with_real_voltage_drop(reactor_net):
    pp.runpp(reactor_net)
    assert reactor_net["converged"]
    # without the series reactor, every bus would stay at exactly the source's
    # 1.05 pu (no power flowing past sourcebus) -- guard against that regression.
    assert reactor_net.res_bus.vm_pu.nunique() > 1
    assert reactor_net.res_bus.vm_pu.min() < 1.05

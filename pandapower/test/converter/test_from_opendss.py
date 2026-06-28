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

import numpy as np
import pytest

import pandapower as pp

# OpenDSSDirect.py is an optional dependency; skip the whole module without it.
pytest.importorskip("opendssdirect")

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

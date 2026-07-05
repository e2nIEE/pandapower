# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Import an OpenDSS feeder into a balanced (positive-sequence) pandapower net."""

# Why balanced / positive-sequence (and not unsymmetrical 3-phase): pandapower's
# ``runpp_3ph`` is a sequence-frame solver that cannot represent truly unsymmetrical
# North-American topology (single-phase laterals, center-tapped split-phase) -- see
# issues #873 and #1442. Positive-sequence is sufficient for the symmetric (European
# / 3-phase 4-wire) feeders that pandapower targets, and OpenDSS exposes the
# positive-sequence ``R1/X1/C1`` of every line directly, so the phase->sequence
# reduction is free. Single-phase laterals are folded into the balanced model as
# full three-phase equivalents; that is the main documented fidelity loss, and it is
# reported per-import in ``net["opendss_import"]`` together with element counts, the
# OpenDSS-solved per-bus voltages and any approximations made.
#
# The circuit is read through the ``OpenDSSDirect.py`` API (lazy-imported), not a
# text parser, so every OpenDSS-supported master file is understood.

import logging
import math
from dataclasses import dataclass, field

import numpy as np

import pandapower as pp

try:
    import opendssdirect as dss

    opendssdirect_imported = True
except ImportError:
    opendssdirect_imported = False

logger = logging.getLogger(__name__)

__all__ = ["from_opendss"]

# OpenDSS LineUnits enum -> kilometres. The conversion factor cancels in the
# product r_ohm_per_km * length_km (= total ohms), so the *impedance* a pandapower
# line carries is exact even where the unit is ambiguous; only the reported
# length_km is nominal. "none" (0) is therefore safely treated as km.
_LINE_UNITS_TO_KM = {
    0: 1.0,          # none -> treat as km (impedance still exact, see above)
    1: 1.609344,     # miles
    2: 0.3048,       # kft (1000 ft)
    3: 1.0,          # km
    4: 1.0e-3,       # m
    5: 3.048e-4,     # ft
    6: 2.54e-5,      # in
    7: 1.0e-5,       # cm
}

_SQRT3 = math.sqrt(3.0)


def _kron_positive_sequence(rmat, xmat, n):
    """Positive-sequence (R1, X1) for a matrix-defined (``rmatrix``/``xmatrix``)
    line of any conductor count, computed directly from the declared matrix.

    OpenDSS's own ``Lines.R1()``/``X1()``/``C1()`` are *symmetrical-component*
    fields: they hold whatever was last assigned via ``r1=``/``x1=``/``c1=``
    (which also makes OpenDSS internally regenerate an equivalent matrix), but
    switching a line/LineCode to matrix mode (``rmatrix=``/``xmatrix=``) does
    **not** clear or recompute them -- verified against OpenDSS's own source
    (``TLineObj``/``TLineCodeObj`` construction hard-codes ``R1 := 0.0580``,
    ``X1 := 0.1206`` and the *text* property interface correctly reports
    those fields as ``'----'``/not-applicable once matrix mode is active, but
    the direct numeric getters used here do not). So for a matrix-defined
    line, ``R1()`` silently returns that stale default, not a value derived
    from the declared matrix -- confirmed both for a 4-wire (3 phase + explicit
    neutral) European LineCode and for a plain 3-conductor matrix LineCode.

    Always deriving from the raw matrix instead is safe for every case: for a
    symmetric-components-defined line OpenDSS auto-generates a consistent
    equivalent matrix (``CalcMatricesFromZ1Z0``), so re-deriving from it
    reproduces the declared R1/X1 exactly (verified). Bus tokens order phases
    before neutral by OpenDSS convention (``.1.2.3.4`` = A,B,C,N), so for
    n>3 conductors the trailing rows/columns (indices 3..n-1) are the ones to
    Kron-eliminate before averaging self/mutual over the remaining <=3 phase
    block; for n<=3 (the common case, and the only case OpenDSS's own R1/X1
    ever cover) there is nothing to eliminate.
    """
    z = np.asarray(rmat, dtype=float).reshape(n, n) + 1j * np.asarray(xmat, dtype=float).reshape(n, n)
    p = min(n, 3)
    zpp = z[:p, :p]
    if n > p:
        zpn, znp, znn = z[:p, p:], z[p:, :p], z[p:, p:]
        try:
            zpp = zpp - zpn @ np.linalg.solve(znn, znp)
        except np.linalg.LinAlgError:
            pass  # degenerate neutral block; fall back to the un-reduced phase block
    self_avg = np.trace(zpp) / p
    mutual_avg = (zpp.sum() - np.trace(zpp)) / (p * (p - 1)) if p > 1 else 0.0
    z1 = self_avg - mutual_avg
    return float(z1.real), float(z1.imag)


# Diagnostics attached to the net as ``net["opendss_import"]``.
@dataclass
class _ImportReport:
    n_buses: int = 0
    n_lines: int = 0
    n_switches: int = 0
    n_transformers: int = 0
    n_split_phase_transformers: int = 0
    n_loads: int = 0
    n_shunts: int = 0
    single_phase_lines: int = 0
    two_phase_lines: int = 0
    warnings: list = field(default_factory=list)
    vm_pu_opendss: dict = field(default_factory=dict)
    bus_phases: dict = field(default_factory=dict)

    def warn(self, message):
        """Record an approximation/skip both in the report and the logger."""
        self.warnings.append(message)
        logger.warning(message)

    def as_dict(self):
        """Return the report as a plain dict for attaching to the net."""
        return dict(self.__dict__)


def _busname(token):
    """Strip the node-connection suffix from an OpenDSS bus token.

    ``"b2.1.2.3" -> "b2"``; OpenDSS is case-insensitive and ``AllBusNames``
    returns lower-case, so we normalise here too.
    """
    return token.split(".", 1)[0].strip().lower()


def _connected_phases(token):
    """Count the phase nodes encoded in a bus token (``b2.1.2.3`` -> 3).

    A bare name (no suffix) means all phases are connected; the caller passes the
    element's declared phase count as the fallback in that case.
    """
    parts = token.split(".")[1:]
    phases = [p for p in parts if p not in ("", "0")]  # node 0 is neutral/ground
    return len(phases)


def from_opendss(path: str, solve: bool=True):
    """Build a balanced (positive-sequence) pandapower net from an OpenDSS feeder.

    The OpenDSS circuit is compiled through ``OpenDSSDirect.py`` and its elements
    are mapped to pandapower as follows:

    * Circuit / Vsource -> ``ext_grid``: slack, ``vm_pu`` from the source pu
    * Bus -> ``bus``: ``vn_kv`` = base kV (line-to-line)
    * Line + LineCode -> ``line``: r/x/c from ``R1/X1/C1``; ``LineCode`` -> std_type
    * Line (switch / 0 km) -> ``switch``: open status respected
    * Transformer (2W) -> ``trafo``: imported at the solved tap
    * Transformer (3W CT) -> ``trafo``: center-tapped split-phase mapped to a 2W equivalent
    * Load -> ``load``: kW/kvar -> p_mw/q_mvar
    * Capacitor -> ``shunt``: kvar -> -q_mvar (injection)

    Transformers are imported at their *solved* tap ratio, so on-load tap changers
    and RegControls are captured as the operating point; the controllers themselves
    are not re-implemented. Three-winding center-tapped service transformers (two
    LV windings on the same secondary bus) are collapsed to a balanced two-winding
    equivalent. Because positive-sequence modeling cannot represent 120/240 V
    split-phase operation (#873), those LV voltages carry the largest approximation
    error.

    This function requires the optional dependency ``OpenDSSDirect.py``
    (``pip install pandapower[opendss]``).

    Args:
        path (str): Path to the OpenDSS master ``.dss`` file, i.e. the file
            you would ``Redirect`` to. All transitively ``Redirect``-ed component
            files are followed.
        solve (bool): If True, solves the circuit in OpenDSS first and captures the
            per-bus voltage magnitudes (pu, phase-averaged) in the import report so
            that a round-trip can be validated without re-solving OpenDSS. Defaults
            to True.

    Returns:
        pandapowerNet: A balanced net carrying an ``opendss_import`` diagnostics
        dict with element counts, per-bus phase counts, the OpenDSS-solved
        voltages, and the list of approximations or skipped elements encountered
        during import.
    """
    if not opendssdirect_imported:
        raise NotImplementedError(
            "OpenDSSDirect.py is required to import OpenDSS circuits. Please install "
            "it, e.g. via 'pip install OpenDSSDirect.py' or 'pip install pandapower[opendss]'.")

    dss.Command("Clear")
    dss.Command(f'Redirect "{path}"')
    if solve:
        dss.Solution.Solve()

    net = pp.create_empty_network(name=dss.Circuit.Name())
    report = _ImportReport()

    bus_map = _add_buses(net, report)
    if solve and dss.Solution.Converged():
        _capture_voltages(report)

    _add_source(net, bus_map, report)
    _add_lines(net, bus_map, report)
    _add_transformers(net, bus_map, report)
    _add_loads(net, bus_map, report)
    _add_capacitors(net, bus_map, report)

    report.n_buses = len(net.bus)
    net["opendss_import"] = report.as_dict()
    return net


def _add_buses(net, report):
    bus_map = {}
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        kv_ln = dss.Bus.kVBase()  # OpenDSS bus base is line-to-neutral
        if kv_ln > 0:
            vn_kv = kv_ln * _SQRT3  # pandapower wants line-to-line
        else:
            vn_kv = 0.0
            report.warn(f"bus {name!r} has no voltage base (kVBase=0); "
                        "set 'VoltageBases' and call 'CalcVoltageBases' in the master")
        report.bus_phases[name.lower()] = len([n for n in dss.Bus.Nodes() if n != 0])
        bus_map[name.lower()] = pp.create_bus(net, vn_kv=vn_kv, name=name)
    return bus_map


def _capture_voltages(report):
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        mags = dss.Bus.puVmagAngle()[::2]  # [mag, ang, mag, ang, ...]
        mags = [m for m in mags if m > 0]
        if mags:
            report.vm_pu_opendss[name.lower()] = sum(mags) / len(mags)


def _add_source(net, bus_map, report):
    if not dss.Vsources.First():
        report.warn("no Vsource found; net has no ext_grid")
        return
    dss.Circuit.SetActiveElement("Vsource." + dss.Vsources.Name())
    bus = bus_map.get(_busname(dss.CktElement.BusNames()[0]))
    if bus is None:
        report.warn("Vsource bus not found among circuit buses")
        return
    pp.create_ext_grid(net, bus, vm_pu=dss.Vsources.PU(), va_degree=0.0,
                       name=dss.Vsources.Name())


def _add_lines(net, bus_map, report):
    i = dss.Lines.First()
    while i:
        name = dss.Lines.Name()
        f = bus_map.get(_busname(dss.Lines.Bus1()))
        t = bus_map.get(_busname(dss.Lines.Bus2()))
        if f is None or t is None:
            report.warn(f"line {name!r} references an unknown bus; skipped")
            i = dss.Lines.Next()
            continue

        phases = max(_connected_phases(dss.Lines.Bus1()), _connected_phases(dss.Lines.Bus2()))
        if phases == 0:
            phases = dss.Lines.Phases()
        if phases == 1:
            report.single_phase_lines += 1
        elif phases == 2:
            report.two_phase_lines += 1

        km = _LINE_UNITS_TO_KM.get(dss.Lines.Units(), 1.0)
        length_km = dss.Lines.Length() * km

        # A switch (or a zero-length jumper) becomes a pandapower bus-bus switch.
        if dss.Lines.IsSwitch() or length_km <= 0.0:
            closed = not dss.CktElement.IsOpen(1, 0)
            pp.create_switch(net, bus=f, element=t, et="b", closed=closed, name=name)
            report.n_switches += 1
            i = dss.Lines.Next()
            continue

        # Always derive R1/X1/C1 from the declared matrix rather than trusting
        # OpenDSS's own Lines.R1()/X1()/C1() -- those are stale symmetric-
        # component fields for any matrix-defined line, not just >3-conductor
        # ones (see `_kron_positive_sequence`). Safe for every case: it exactly
        # reproduces R1/X1 for a symmetric-components-defined line too.
        n_cond = dss.Lines.Phases()
        r1, x1 = _kron_positive_sequence(dss.Lines.RMatrix(), dss.Lines.XMatrix(), n_cond)
        c1, _ = _kron_positive_sequence(dss.Lines.CMatrix(), [0.0] * (n_cond * n_cond), n_cond)

        # The OpenDSS LineCode names the physical conductor; carry it through as
        # pandapower's std_type so the conductor identity survives the import.
        pp.create_line_from_parameters(
            net, from_bus=f, to_bus=t, length_km=length_km,
            r_ohm_per_km=r1 / km,
            x_ohm_per_km=x1 / km,
            c_nf_per_km=c1 / km,
            max_i_ka=dss.Lines.NormAmps() / 1000.0,
            std_type=dss.Lines.LineCode() or None,
            name=name,
        )
        report.n_lines += 1
        i = dss.Lines.Next()


def _add_transformers(net, bus_map, report):
    i = dss.Transformers.First()
    while i:
        _add_one_transformer(net, bus_map, report)
        i = dss.Transformers.Next()


def _add_one_transformer(net, bus_map, report):
    name = dss.Transformers.Name()
    dss.Circuit.SetActiveElement("Transformer." + name)
    wbus = [_busname(b) for b in dss.CktElement.BusNames()]
    nwdg = dss.Transformers.NumWindings()

    kva, pct_r, tap = [], [], []
    for w in range(1, nwdg + 1):
        dss.Transformers.Wdg(w)
        kva.append(dss.Transformers.kVA())
        pct_r.append(dss.Transformers.R())    # %R of this winding
        tap.append(dss.Transformers.Tap())    # solved tap ratio (captures RegControl)
    xhl = dss.Transformers.Xhl()              # HV-LV leakage reactance, %

    # Pick the HV winding and the LV winding.
    split_phase = False
    if nwdg == 2:
        hv_w, lv_w = 0, 1
    elif nwdg == 3 and len(set(wbus)) == 2:
        # Center-tapped split-phase service transformer: one HV winding + two LV
        # windings on the SAME secondary bus. Collapse to a balanced 2-winding
        # equivalent (positive-sequence cannot represent 120/240 V split phase,
        # #873, so the LV voltages here carry the larger error).
        split_phase = True
        unique = list(dict.fromkeys(wbus))
        lv_name = next(b for b in unique if wbus.count(b) == 2)
        hv_name = next(b for b in unique if wbus.count(b) == 1)
        hv_w, lv_w = wbus.index(hv_name), wbus.index(lv_name)
    else:
        report.warn(
            f"transformer {name!r} has {nwdg} windings across "
            f"{len(set(wbus))} buses; unsupported topology, skipped")
        return

    bus_hv = bus_map.get(wbus[hv_w])
    bus_lv = bus_map.get(wbus[lv_w])
    if bus_hv is None or bus_lv is None:
        report.warn(f"transformer {name!r} references an unknown bus; skipped")
        return

    # Order by the buses' own voltage base so hv_bus really is the higher side.
    if net.bus.at[bus_hv, "vn_kv"] < net.bus.at[bus_lv, "vn_kv"]:
        bus_hv, bus_lv, hv_w, lv_w = bus_lv, bus_hv, lv_w, hv_w

    # Use the connected buses' vn_kv (already kVBase*sqrt(3)) as the winding
    # ratings: OpenDSS propagates voltage bases through transformers, so the bus
    # ratio already equals the turns ratio -- this sidesteps line-to-line vs
    # line-to-neutral / sqrt(3) ambiguity. The solved tap multiplies on top.
    vn_hv = net.bus.at[bus_hv, "vn_kv"] * tap[hv_w]
    vn_lv = net.bus.at[bus_lv, "vn_kv"] * tap[lv_w]
    if vn_hv <= 0 or vn_lv <= 0:
        report.warn(f"transformer {name!r} has a zero-base winding; skipped")
        return
    if any(abs(t - 1.0) > 1e-6 for t in tap):
        report.warn(
            f"transformer {name!r} imported at solved tap {tuple(round(t, 4) for t in tap)} "
            "(RegControl baked in as a fixed tap)")

    vkr = pct_r[hv_w] + pct_r[lv_w]           # copper/short-circuit R, % (= %loadloss)
    vk = math.hypot(vkr, xhl)                  # short-circuit voltage: hypot of the R and X parts
    pp.create_transformer_from_parameters(
        net, hv_bus=bus_hv, lv_bus=bus_lv,
        sn_mva=max(kva) / 1000.0,
        vn_hv_kv=vn_hv, vn_lv_kv=vn_lv,
        vk_percent=vk, vkr_percent=vkr,
        pfe_kw=0.0, i0_percent=0.0,            # core losses dropped in v1
        shift_degree=0.0,                      # vector-group shift: no effect on balanced |V|
        name=name,
    )
    report.n_transformers += 1
    if split_phase:
        report.n_split_phase_transformers += 1


def _add_loads(net, bus_map, report):
    i = dss.Loads.First()
    while i:
        name = dss.Loads.Name()
        dss.Circuit.SetActiveElement("Load." + name)
        bus = bus_map.get(_busname(dss.CktElement.BusNames()[0]))
        if bus is None:
            report.warn(f"load {name!r} references an unknown bus; skipped")
            i = dss.Loads.Next()
            continue
        pp.create_load(net, bus, p_mw=dss.Loads.kW() / 1000.0,
                       q_mvar=dss.Loads.kvar() / 1000.0, name=name)
        report.n_loads += 1
        i = dss.Loads.Next()


def _add_capacitors(net, bus_map, report):
    i = dss.Capacitors.First()
    while i:
        name = dss.Capacitors.Name()
        dss.Circuit.SetActiveElement("Capacitor." + name)
        bus = bus_map.get(_busname(dss.CktElement.BusNames()[0]))
        if bus is None:
            report.warn(f"capacitor {name!r} references an unknown bus; skipped")
            i = dss.Capacitors.Next()
            continue
        # A shunt capacitor injects reactive power -> negative q_mvar in pandapower's
        # consumer sign convention (positive q_mvar = inductive absorption).
        pp.create_shunt(net, bus, q_mvar=-dss.Capacitors.kvar() / 1000.0, p_mw=0.0,
                        name=name)
        report.n_shunts += 1
        i = dss.Capacitors.Next()

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

import faulthandler
import logging
import math
from dataclasses import dataclass, field

import numpy as np

import pandapower as pp

try:
    # On Windows, OpenDSSDirect.py's native backend (dss_python_backend) raises an
    # internal, first-chance structured exception during its own startup -- one its
    # Free Pascal runtime normally handles and silences itself. If a Windows crash
    # handler is already installed (as Python's faulthandler does, and pytest enables
    # faulthandler by default), it intercepts that exception first and misreports it
    # as fatal, killing the whole process instead of a clean ImportError. Disable
    # faulthandler for just this import, then restore it -- it's still needed to
    # catch genuine crashes elsewhere. See
    # https://github.com/dss-extensions/OpenDSSDirect.py/issues/148
    faulthandler_was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    try:
        import opendssdirect as dss
    finally:
        if faulthandler_was_enabled:
            faulthandler.enable()

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

# Below this, a solved OpenDSS tap ratio counts as "sitting at neutral".
_TAP_RATIO_EPS = 1e-9


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
    n_reactors: int = 0
    n_switches: int = 0
    n_transformers: int = 0
    n_split_phase_transformers: int = 0
    n_reg_controls: int = 0
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


@dataclass
class _RegControlInfo:
    """One OpenDSS ``RegControl``, captured while it was the active element."""

    name: str
    transformer: str    # lower-cased transformer name
    winding: int         # 1-based: the monitored (PT/CT) winding
    tap_winding: int     # 1-based: the winding whose tap actually moves
    monitored_bus: str   # lower-cased bus name, '' if defaulted to the winding's own terminal
    vreg: float
    band: float
    ptratio: float
    forward_r: float
    forward_x: float
    is_reversible: bool
    delay: float
    tap_delay: float
    is_inverse_time: bool


def _collect_regcontrols():
    """
    Read every OpenDSS RegControl, keyed by the lower-cased controlled transformer name.

    Called once, before transformers are imported, so a transformer can pick
    its tapped winding using the RegControl that targets it (see
    `_pick_tap_fields`).
    """
    by_trafo = {}
    i = dss.RegControls.First()
    while i:
        info = _RegControlInfo(
            name=dss.RegControls.Name(),
            transformer=dss.RegControls.Transformer().lower(),
            winding=int(dss.RegControls.Winding()),
            tap_winding=int(dss.RegControls.TapWinding()),
            monitored_bus=_busname(dss.RegControls.MonitoredBus()),
            vreg=dss.RegControls.ForwardVreg(),
            band=dss.RegControls.ForwardBand(),
            ptratio=dss.RegControls.PTRatio(),
            forward_r=dss.RegControls.ForwardR(),
            forward_x=dss.RegControls.ForwardX(),
            is_reversible=bool(dss.RegControls.IsReversible()),
            delay=dss.RegControls.Delay(),
            tap_delay=dss.RegControls.TapDelay(),
            is_inverse_time=bool(dss.RegControls.IsInverseTime()),
        )
        by_trafo.setdefault(info.transformer, []).append(info)
        i = dss.RegControls.Next()
    return by_trafo


def _has_tap_grid(min_tap, max_tap, num_taps):
    """Whether a winding's declared MinTap/MaxTap/NumTaps is usable at all."""
    return round(num_taps) > 0 and max_tap > min_tap


def _is_off_ratio(ratio):
    """Whether a winding's solved tap ratio deviates from neutral (1.0)."""
    return abs(ratio - 1.0) > _TAP_RATIO_EPS


def _tap_fields_from_dss(min_tap, max_tap, num_taps, ratio):
    """
    Translate an OpenDSS winding's tap range and solved ratio into pandapower tap fields.

    Returns (tap_step_percent, tap_min, tap_max, tap_neutral, tap_pos).

    OpenDSS's tap-position axis is always centered on zero -- from
    ``-(NumTaps // 2)`` to ``-(NumTaps // 2) + NumTaps`` -- *independent* of
    whether MinTap/MaxTap happen to be symmetric about 1.0. This isn't
    documented anywhere in OpenDSS help (which just says "16 raise and 16
    lower taps about the neutral position"); it was verified empirically
    against ``RegControls.TapNumber()`` with an asymmetric MinTap/MaxTap pair.
    So tap_neutral -- the position where the ratio is exactly 1.0 -- is
    derived from where 1.0 falls inside [MinTap, MaxTap], and is *not* always
    the middle of the range.

    Returns None if the winding has no usable tap range (NumTaps <= 0 or a
    degenerate/inverted MinTap/MaxTap).
    """
    if not _has_tap_grid(min_tap, max_tap, num_taps):
        return None
    num_taps = round(num_taps)
    step_pu = (max_tap - min_tap) / num_taps
    tap_min = -(num_taps // 2)
    tap_max = tap_min + num_taps
    tap_neutral = round(tap_min + (1.0 - min_tap) / step_pu)
    tap_pos = round(tap_min + (ratio - min_tap) / step_pu)
    tap_pos = int(min(max(tap_pos, tap_min), tap_max))
    return step_pu * 100.0, tap_min, tap_max, tap_neutral, tap_pos


def _pick_regcontrol(name, report, regcontrols_by_trafo, split_phase):
    """
    Return the RegControl that governs a transformer's tap, or None if there is none to use.

    A split-phase transformer is collapsed into a 2-winding equivalent, so a
    RegControl can no longer be tied to one of its original windings.
    """
    regctrls = regcontrols_by_trafo.get(name.lower())
    if not regctrls or split_phase:
        return None
    reg = regctrls[0]
    if len(regctrls) > 1:
        report.warn(f"transformer {name!r} has {len(regctrls)} RegControls; only "
                    f"{reg.name!r} is imported")
    return reg


def _pick_tapped_winding(name, report, reg, hv_w, lv_w, hv_dev, lv_dev, min_tap, max_tap, num_taps):
    """
    Return the index of the winding whose tap becomes pandapower's tap changer.

    A RegControl names it outright; failing that it is inferred from which
    winding solved off ratio.
    """
    if reg is not None and reg.tap_winding - 1 not in (hv_w, lv_w):
        report.warn(f"transformer {name!r}: RegControl {reg.name!r} has an invalid "
                    f"tap_winding ({reg.tap_winding}); falling back to whichever winding's "
                    "solved tap deviates from neutral")
        reg = None

    if reg is not None:
        return reg.tap_winding - 1
    if hv_dev and not lv_dev:
        return hv_w
    if lv_dev and not hv_dev:
        return lv_w

    # Both windings are off ratio with no RegControl to say which one is the
    # "real" tap changer: prefer whichever has a usable OpenDSS tap grid to
    # represent, defaulting to lv only if both (or neither) do.
    hv_has_grid = _has_tap_grid(min_tap[hv_w], max_tap[hv_w], num_taps[hv_w])
    lv_has_grid = _has_tap_grid(min_tap[lv_w], max_tap[lv_w], num_taps[lv_w])
    tap_w = hv_w if hv_has_grid and not lv_has_grid else lv_w
    side = "hv" if tap_w == hv_w else "lv"
    if hv_dev and lv_dev:
        report.warn(f"transformer {name!r} has a non-unity tap on both windings; combining them "
                    f"into a single pandapower tap on the {side} side")
    return tap_w


def _pick_tap_fields(name, report, regcontrols_by_trafo, hv_w, lv_w, tap, min_tap, max_tap,
                     num_taps, split_phase):
    """
    Pick the tapped winding and translate its OpenDSS tap range into pandapower tap_* fields.

    This lets the solved tap become an explicit, movable ``tap_pos`` instead of
    being folded into ``vn_hv_kv``/``vn_lv_kv``.

    Returns ``(tap_kwargs, legacy)``:

    * ``tap_kwargs`` is a kwargs dict for ``create_transformer_from_parameters``
      (empty if neither winding shows any sign of being tapped: no RegControl
      and both windings solved at ratio 1.0, i.e. nothing beyond OpenDSS's
      meaningless tap-range defaults that every transformer carries regardless
      of use).
    * ``legacy`` is normally None, but is ``(side, factor)`` when the winding
      *is* genuinely off-ratio (RegControl or solved tap != 1.0) yet OpenDSS's
      own MinTap/MaxTap/NumTaps for it is degenerate (e.g. ``NumTaps=0``, used
      to fix a tap at a value outside the steppable grid): there is no valid
      tap_pos to represent it, so the caller must fall back to multiplying
      ``vn_<side>_kv`` directly -- the behaviour this feature replaces -- so a
      real ratio is never silently dropped.
    """
    hv_dev = _is_off_ratio(tap[hv_w])
    lv_dev = _is_off_ratio(tap[lv_w])

    reg = _pick_regcontrol(name, report, regcontrols_by_trafo, split_phase)
    if reg is None and not hv_dev and not lv_dev:
        return {}, None

    tap_w = _pick_tapped_winding(name, report, reg, hv_w, lv_w, hv_dev, lv_dev,
                                 min_tap, max_tap, num_taps)
    other_w = hv_w if tap_w == lv_w else lv_w
    tap_side = "hv" if tap_w == hv_w else "lv"
    factor = tap[tap_w] / tap[other_w]

    grid = _tap_fields_from_dss(min_tap[tap_w], max_tap[tap_w], num_taps[tap_w], factor)
    if grid is None:
        if not _is_off_ratio(factor):
            return {}, None
        report.warn(
            f"transformer {name!r} has no usable tap range on the {tap_side} winding "
            "(NumTaps<=0 or a degenerate MinTap/MaxTap); solved tap ratio "
            f"{factor:.4f} baked into vn_{tap_side}_kv instead of tap_pos")
        return {}, (tap_side, factor)
    tap_step_percent, tap_min, tap_max, tap_neutral, tap_pos = grid

    if tap_pos != tap_neutral:
        report.warn(
            f"transformer {name!r} imported with tap ratio {factor:.4f} on the {tap_side} side "
            f"-> tap_pos={tap_pos} (tap_min={tap_min}, tap_max={tap_max}, "
            f"tap_step_percent={tap_step_percent:.4f})")

    return {
        "tap_side": tap_side,
        "tap_neutral": tap_neutral,
        "tap_min": tap_min,
        "tap_max": tap_max,
        "tap_step_percent": tap_step_percent,
        "tap_pos": tap_pos,
        "tap_changer_type": "Ratio",
    }, None


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


def from_opendss(path: str, solve: bool=True, import_controllers: bool=False):
    """Build a balanced (positive-sequence) pandapower net from an OpenDSS feeder.

    The OpenDSS circuit is compiled through ``OpenDSSDirect.py`` and its elements
    are mapped to pandapower as follows:

    * Circuit / Vsource -> ``ext_grid``: slack, ``vm_pu`` from the source pu
    * Bus -> ``bus``: ``vn_kv`` = base kV (line-to-line)
    * Line + LineCode -> ``line``: r/x/c from ``R1/X1/C1``; ``LineCode`` -> std_type
    * Line (switch / 0 km) -> ``switch``: open status respected
    * Reactor (series, bus-to-bus) -> ``line``: fixed impedance from ``R``/``X``;
      a shunt reactor (single bus) is not a branch and is skipped
    * Transformer (2W) -> ``trafo``: tap changer imported live (see below)
    * Transformer (3W CT) -> ``trafo``: center-tapped split-phase mapped to a 2W equivalent
    * Load -> ``load``: kW/kvar -> p_mw/q_mvar
    * Capacitor -> ``shunt``: kvar -> -q_mvar (injection)
    * RegControl -> ``DiscreteTapControl`` (only if ``import_controllers=True``)

    A transformer's solved OpenDSS tap is translated into pandapower's
    ``tap_pos``/``tap_min``/``tap_max``/``tap_step_percent``/``tap_neutral``
    rather than being folded into ``vn_hv_kv``/``vn_lv_kv``, so the tap has
    something to actuate; this happens for every tapped winding regardless of
    ``import_controllers`` and does not change the solved voltages. With
    ``import_controllers=True``, each ``RegControl`` additionally becomes a
    ``DiscreteTapControl`` on the mapped trafo, so the tap responds to bus
    voltage during ``pandapower.control.run_control``/timeseries instead of
    being pinned at the OpenDSS operating point. Line-drop compensation,
    reverse-mode regulation and time delays are not modeled; when a RegControl
    uses them, or monitors a bus other than its own tapped winding's terminal,
    this is reported as a warning rather than guessed at (and, for an
    unsupported monitored bus, the controller is skipped entirely rather than
    silently regulating the wrong bus). Three-winding center-tapped service
    transformers (two LV windings on the same secondary bus) are collapsed to
    a balanced two-winding equivalent and are not matched to a RegControl.
    Because positive-sequence modeling cannot represent 120/240 V split-phase
    operation (#873), those LV voltages carry the largest approximation error.

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
        import_controllers (bool): If True, import each ``RegControl`` as a
            ``DiscreteTapControl`` so its tap responds during
            ``pandapower.control.run_control``. Defaults to False, so a plain
            ``pandapower.runpp`` on the returned net (which never looks at
            ``net.controller``) sees the same voltages as before this option
            existed.

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
    _add_reactors(net, bus_map, report)
    regcontrols_by_trafo = _collect_regcontrols()
    trafo_index_by_name = _add_transformers(net, bus_map, report, regcontrols_by_trafo)
    _add_reg_controls(net, report, regcontrols_by_trafo, trafo_index_by_name, import_controllers)
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


def _add_reactors(net, bus_map, report):
    """Series (bus-to-bus) Reactor elements as a fixed-impedance pandapower line.

    Some feeder libraries (e.g. EPRI's Ckt5/Ckt7 test circuits) model the
    substation's Thevenin-equivalent source impedance as a ``Reactor`` between
    ``SourceBus`` and the feeder head bus rather than a ``Transformer`` -- this
    picks that pattern up as an ordinary series impedance so the link is
    carried into the pandapower net. A shunt reactor (single bus, to ground)
    is a different physical element (not a branch) and is out of scope here.
    """
    i = dss.Reactors.First()
    while i:
        name = dss.Reactors.Name()
        dss.Circuit.SetActiveElement("Reactor." + name)
        buses = dss.CktElement.BusNames()
        if len(buses) < 2 or _busname(buses[0]) == _busname(buses[1]):
            report.warn(f"reactor {name!r} is a shunt (single bus); skipped")
            i = dss.Reactors.Next()
            continue

        f = bus_map.get(_busname(buses[0]))
        t = bus_map.get(_busname(buses[1]))
        if f is None or t is None:
            report.warn(f"reactor {name!r} references an unknown bus; skipped")
            i = dss.Reactors.Next()
            continue

        max_i_ka = dss.CktElement.NormalAmps() / 1000.0
        pp.create_line_from_parameters(
            net, from_bus=f, to_bus=t, length_km=1.0,
            r_ohm_per_km=dss.Reactors.R(), x_ohm_per_km=dss.Reactors.X(), c_nf_per_km=0.0,
            max_i_ka=max_i_ka if max_i_ka > 0 else 10.0,
            name=name,
        )
        report.n_reactors += 1
        i = dss.Reactors.Next()


def _add_transformers(net, bus_map, report, regcontrols_by_trafo):
    trafo_index_by_name = {}
    i = dss.Transformers.First()
    while i:
        _add_one_transformer(net, bus_map, report, regcontrols_by_trafo, trafo_index_by_name)
        i = dss.Transformers.Next()
    return trafo_index_by_name


def _add_one_transformer(net, bus_map, report, regcontrols_by_trafo, trafo_index_by_name):
    name = dss.Transformers.Name()
    dss.Circuit.SetActiveElement("Transformer." + name)
    wbus = [_busname(b) for b in dss.CktElement.BusNames()]
    nwdg = dss.Transformers.NumWindings()

    kva, pct_r, tap, min_tap, max_tap, num_taps = [], [], [], [], [], []
    for w in range(1, nwdg + 1):
        dss.Transformers.Wdg(w)
        kva.append(dss.Transformers.kVA())
        pct_r.append(dss.Transformers.R())    # %R of this winding
        tap.append(dss.Transformers.Tap())    # solved tap ratio (captures RegControl)
        min_tap.append(dss.Transformers.MinTap())
        max_tap.append(dss.Transformers.MaxTap())
        num_taps.append(dss.Transformers.NumTaps())
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
    # ratio already equals the nominal turns ratio -- this sidesteps line-to-line
    # vs line-to-neutral / sqrt(3) ambiguity. The solved tap is no longer folded
    # in here: it becomes an explicit tap_pos below, so a tap changer or
    # RegControl has something to actuate instead of being pinned into vn_*_kv.
    vn_hv = net.bus.at[bus_hv, "vn_kv"]
    vn_lv = net.bus.at[bus_lv, "vn_kv"]
    if vn_hv <= 0 or vn_lv <= 0:
        report.warn(f"transformer {name!r} has a zero-base winding; skipped")
        return

    tap_fields, legacy_tap = _pick_tap_fields(name, report, regcontrols_by_trafo, hv_w, lv_w, tap,
                                              min_tap, max_tap, num_taps, split_phase)
    if legacy_tap is not None:
        # No usable OpenDSS tap grid for an off-ratio winding: fall back to
        # baking the ratio into vn_*_kv (this feature's previous behaviour)
        # rather than silently dropping it.
        side, factor = legacy_tap
        if side == "hv":
            vn_hv *= factor
        else:
            vn_lv *= factor

    vkr = pct_r[hv_w] + pct_r[lv_w]           # copper/short-circuit R, % (= %loadloss)
    vk = math.hypot(vkr, xhl)                  # short-circuit voltage: hypot of the R and X parts
    tid = pp.create_transformer_from_parameters(
        net, hv_bus=bus_hv, lv_bus=bus_lv,
        sn_mva=max(kva) / 1000.0,
        vn_hv_kv=vn_hv, vn_lv_kv=vn_lv,
        vk_percent=vk, vkr_percent=vkr,
        pfe_kw=0.0, i0_percent=0.0,            # core losses dropped in v1
        shift_degree=0.0,                      # vector-group shift: no effect on balanced |V|
        name=name,
        **tap_fields,
    )
    report.n_transformers += 1
    if split_phase:
        report.n_split_phase_transformers += 1
    trafo_index_by_name[name.lower()] = tid


def _regulates_own_terminal(net, report, reg, controlled_bus):
    """
    Whether a RegControl regulates the terminal its own tapped winding sits on.

    ``DiscreteTapControl`` can only regulate that terminal, so anything else
    (an explicit remote monitored bus, or monitoring one winding while tapping
    another) is reported and skipped rather than silently regulating the wrong
    bus.
    """
    controlled_bus_name = net.bus.at[controlled_bus, "name"].lower()
    if reg.monitored_bus and reg.monitored_bus != controlled_bus_name:
        report.warn(
            f"RegControl {reg.name!r} monitors bus {reg.monitored_bus!r}, not the tapped "
            f"winding's own terminal {controlled_bus_name!r}; DiscreteTapControl can only "
            "regulate its own terminal, so it was not imported as a controller")
        return False
    if not reg.monitored_bus and reg.winding != reg.tap_winding:
        report.warn(
            f"RegControl {reg.name!r} monitors winding {reg.winding} but taps winding "
            f"{reg.tap_winding}; this configuration is not supported, so it was not imported "
            "as a controller")
        return False
    return True


def _warn_unmodelled_regcontrol_settings(report, reg):
    """Report the RegControl settings that a steady-state tap controller cannot represent."""
    notes = []
    if reg.forward_r or reg.forward_x:
        notes.append(f"line-drop compensation (R={reg.forward_r}, X={reg.forward_x}) ignored")
    if reg.is_reversible:
        notes.append("reverse-mode settings ignored")
    if reg.delay or reg.tap_delay or reg.is_inverse_time:
        notes.append("time-delay/inverse-time settings ignored")
    if notes:
        report.warn(f"RegControl {reg.name!r}: " + "; ".join(notes) +
                    " (steady-state power flow has no time/current dimension)")


def _add_one_reg_control(net, report, trafo_name, reg, tid):
    """Create the ``DiscreteTapControl`` for one RegControl; return whether it was created."""
    row = net.trafo.loc[tid]
    tap_side = row["tap_side"]
    if tap_side not in ("hv", "lv"):
        report.warn(f"RegControl {reg.name!r}: transformer {trafo_name!r} has no usable tap "
                    "range; not imported as a controller")
        return False

    controlled_bus = row["hv_bus"] if tap_side == "hv" else row["lv_bus"]
    if not _regulates_own_terminal(net, report, reg, controlled_bus):
        return False

    _warn_unmodelled_regcontrol_settings(report, reg)

    # OpenDSS regulates the PT secondary in volts (vreg +/- band/2), referred
    # to the primary by ptratio; the PT is line-to-neutral, so converting to
    # a per-unit value against the (line-to-line) bus vn_kv needs the sqrt(3)
    # -- the classic place to be off by 1.73x if skipped.
    vn_kv = net.bus.at[controlled_bus, "vn_kv"]
    vm_center_pu = reg.vreg * reg.ptratio * _SQRT3 / 1000.0 / vn_kv
    vm_half_band_pu = reg.band / 2.0 * reg.ptratio * _SQRT3 / 1000.0 / vn_kv

    pp.control.DiscreteTapControl(
        net, element_index=tid,
        vm_lower_pu=vm_center_pu - vm_half_band_pu,
        vm_upper_pu=vm_center_pu + vm_half_band_pu,
        side=tap_side,
    )
    return True


def _add_reg_controls(net, report, regcontrols_by_trafo, trafo_index_by_name, import_controllers):
    """
    Create a ``DiscreteTapControl`` for each RegControl whose transformer was imported.

    This makes the tap respond to voltage instead of staying pinned at the
    OpenDSS-solved position. Only called with effect when
    ``import_controllers`` is True; regardless of that flag, an unreachable
    transformer is still reported so the omission isn't silent.
    """
    for trafo_name, regctrls in regcontrols_by_trafo.items():
        tid = trafo_index_by_name.get(trafo_name)
        if tid is None:
            report.warn(f"RegControl on transformer {trafo_name!r} references a transformer "
                        "that was not imported; skipped")
        elif import_controllers and _add_one_reg_control(net, report, trafo_name, regctrls[0], tid):
            report.n_reg_controls += 1


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

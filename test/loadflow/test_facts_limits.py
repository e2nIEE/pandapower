# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import logging

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import pandapower as pp
import pandapower.pf.run_newton_raphson_pf as nr
from pandapower.networks import case9
from pandapower.pypower.idx_svc import SVC_CONTROLLABLE, SVC_THYRISTOR_FIRING_ANGLE


def _svc_network(devices, q_limits=None):
    net = case9()
    for bus, voltage, lower, upper in devices:
        pp.create_svc(
            net,
            bus,
            397,
            -1190,
            voltage,
            130,
            min_angle_degree=lower,
            max_angle_degree=upper,
        )
    if q_limits is not None:
        net.gen.loc[0, ["min_q_mvar", "max_q_mvar"]] = q_limits
    return net


def _tcsc_network(set_p_mw=40):
    net = pp.network.pandapowerNet(name="TCSC angle limits", sn_mva=100)
    pp.create_buses(net, 4, 110)
    pp.create_ext_grid(net, 0)
    for from_bus, to_bus in [(0, 1), (1, 3), (2, 3)]:
        pp.create_line_from_parameters(net, from_bus, to_bus, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 3, 100, 40)
    pp.create_tcsc(
        net,
        1,
        2,
        0.2,
        -20,
        set_p_mw,
        174,
        min_angle_degree=170,
        max_angle_degree=178,
    )
    return net


def _assert_bus_voltages(net, reference):
    np.testing.assert_allclose(
        net.res_bus[["vm_pu", "va_degree"]],
        reference.res_bus[["vm_pu", "va_degree"]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(net.res_ext_grid, reference.res_ext_grid, rtol=0, atol=1e-6)


def _susceptance(angle_degree, x_l, x_c):
    angle = np.deg2rad(angle_degree)
    return (2 * (np.pi - angle) + np.sin(2 * angle)) / (np.pi * x_l) + 1 / x_c


@pytest.mark.parametrize("voltage,lower,upper,limit", [(1.04, 90, 135, 135), (0.96, 140, 180, 140)])
def test_svc_angle_limit_matches_fixed_device_and_shunt(voltage, lower, upper, limit):
    net = _svc_network([(4, voltage, lower, upper)])
    fixed = copy.deepcopy(net)
    shunt = copy.deepcopy(net)
    pp.runpp(net, enforce_facts_limits=True, numba=False)
    assert net.res_svc.at[0, "thyristor_firing_angle_degree"] == pytest.approx(limit)
    assert net.res_svc.at[0, "at_limit"]
    assert net.svc.at[0, "controllable"]

    fixed.svc.loc[0, ["thyristor_firing_angle_degree", "controllable"]] = [limit, False]
    pp.runpp(fixed, numba=False)
    _assert_bus_voltages(net, fixed)
    np.testing.assert_allclose(net.res_gen, fixed.res_gen, rtol=0, atol=1e-6)

    shunt.svc = shunt.svc.iloc[:0]
    q_at_nominal_voltage = shunt.bus.at[4, "vn_kv"] ** 2 * _susceptance(limit, 397, -1190)
    pp.create_shunt(shunt, 4, q_at_nominal_voltage)
    pp.runpp(shunt, numba=False)
    _assert_bus_voltages(net, shunt)
    assert net.res_svc.at[0, "q_mvar"] == pytest.approx(shunt.res_shunt.at[0, "q_mvar"], abs=1e-6)


@pytest.mark.parametrize("set_p_mw,limit", [(-40, 170), (40, 178)])
def test_tcsc_angle_limit_matches_impedance(set_p_mw, limit):
    net = _tcsc_network(set_p_mw)
    reference = copy.deepcopy(net)
    pp.runpp(net, enforce_facts_limits=True, init="dc", numba=False)
    assert net.res_tcsc.at[0, "thyristor_firing_angle_degree"] == pytest.approx(limit)
    assert net.res_tcsc.at[0, "at_limit"]
    x_ohm = 1 / _susceptance(limit, 0.2, -20)
    base_z = 110**2 / reference.sn_mva
    reference.tcsc = reference.tcsc.iloc[:0]
    pp.create_impedance(reference, 1, 2, rft_pu=0, xft_pu=x_ohm / base_z, sn_mva=reference.sn_mva)
    pp.runpp(reference, init="dc", numba=False)
    _assert_bus_voltages(net, reference)
    cols = ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
    np.testing.assert_allclose(net.res_tcsc[cols], reference.res_impedance[cols], rtol=0, atol=1e-6)
    assert net.res_tcsc.at[0, "x_ohm"] == pytest.approx(x_ohm)


def test_svc_is_released_after_generator_q_limit(monkeypatch):
    net = _svc_network([(7, 1.06, 90, 140)], (5, 10))
    solves = []
    original = nr._run_ac_pf_without_qlims_enforced

    def observe(ppci, options):
        result = original(ppci, options)
        solves.append(ppci["svc"][:, [SVC_CONTROLLABLE, SVC_THYRISTOR_FIRING_ANGLE]].copy())
        return result

    monkeypatch.setattr(nr, "_run_ac_pf_without_qlims_enforced", observe)
    pp.runpp(net, enforce_q_lims=True, enforce_facts_limits=True, numba=False)
    assert any(not state[0, 0] for state in solves)
    assert 90 < net.res_svc.at[0, "thyristor_firing_angle_degree"] < 140
    assert not net.res_svc.at[0, "at_limit"]
    assert net.res_bus.at[7, "vm_pu"] == pytest.approx(1.06, abs=1e-7)
    assert net.res_gen.at[0, "q_mvar"] == pytest.approx(5)


def test_conflicting_svcs_do_not_depend_on_creation_order():
    devices = [(4, 1.0, 90, 135), (7, 1.06, 90, 140)]
    forward = _svc_network(devices)
    reverse = _svc_network(devices[::-1])
    for net in [forward, reverse]:
        pp.runpp(net, enforce_facts_limits=True, numba=False)
        angles = net.res_svc.thyristor_firing_angle_degree
        assert angles.between(net.svc.min_angle_degree, net.svc.max_angle_degree).all()
        assert net.res_svc.at_limit.all()
    _assert_bus_voltages(forward, reverse)
    np.testing.assert_allclose(
        forward.res_svc.thyristor_firing_angle_degree,
        reverse.res_svc.thyristor_firing_angle_degree.iloc[::-1],
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize("q_limits,q", [((5, 10), 5), ((-100, 0), 0)])
def test_svc_and_generator_limits_are_both_enforced(q_limits, q):
    net = _svc_network([(4, 1.04, 90, 135)], q_limits)
    reference = copy.deepcopy(net)
    pp.runpp(net, enforce_facts_limits=True, enforce_q_lims=True, numba=False)
    assert net.res_gen.at[0, "q_mvar"] == pytest.approx(q)
    assert net.res_svc.at[0, "thyristor_firing_angle_degree"] == pytest.approx(135)
    reference.svc.loc[0, ["thyristor_firing_angle_degree", "controllable"]] = [135, False]
    reference.gen.loc[0, ["min_q_mvar", "max_q_mvar"]] = [q, q]
    pp.runpp(reference, enforce_q_lims=True, numba=False)
    _assert_bus_voltages(net, reference)


@pytest.mark.parametrize(
    "devices",
    [
        [(3, 0.98, 120, 150), (7, 0.98, 120, 150), (6, 1.04, 90, 160)],
        [(5, 1.02, 100, 140), (7, 1.0, 120, 150), (6, 1.04, 90, 140)],
    ],
)
def test_former_cycle_cases_settle_without_spurious_warning(devices, caplog):
    net = _svc_network(devices, (0, 20))
    with caplog.at_level(logging.WARNING, logger=nr.__name__):
        pp.runpp(net, enforce_q_lims=True, enforce_facts_limits=True, numba=False)
    assert "keep switching" not in caplog.text
    assert net.converged
    assert net.res_svc.thyristor_firing_angle_degree.between(
        net.svc.min_angle_degree - 1e-6,
        net.svc.max_angle_degree + 1e-6,
    ).all()
    assert net.svc.controllable.all()


@pytest.mark.parametrize("q_limits", [False, True, 2])
def test_facts_option_off_preserves_existing_results(q_limits):
    net = _svc_network([(4, 1.04, 90, 135)])
    original = copy.deepcopy(net)
    pp.runpp(original, enforce_q_lims=q_limits, numba=False)
    pp.runpp(net, enforce_facts_limits=False, enforce_q_lims=q_limits, numba=False)
    for table in ["res_bus", "res_gen", "res_ext_grid", "res_line", "res_svc"]:
        assert_frame_equal(net[table], original[table], check_exact=True)
    assert net.res_svc.at[0, "thyristor_firing_angle_degree"] > 135
    if "at_limit" in net.res_svc:
        assert not net.res_svc.at_limit.any()


def test_q_limit_mode_two_with_only_lower_violation():
    net = case9()
    net.gen.loc[0, ["min_q_mvar", "max_q_mvar"]] = [50, 100]
    pp.runpp(net, enforce_q_lims=2, numba=False)
    assert net.res_gen.at[0, "q_mvar"] == pytest.approx(50)


@pytest.mark.parametrize("lower,upper", [(140, 130), (80, 130), (100, 190), (np.nan, 140)])
def test_invalid_facts_limits_are_rejected(lower, upper):
    net = _svc_network([(4, 1.04, lower, upper)])
    with pytest.raises(ValueError, match="90 <= min <= max <= 180"):
        pp.runpp(net, enforce_facts_limits=True, numba=False)


def test_tcsc_range_must_not_cross_resonance():
    net = _tcsc_network()
    net.tcsc.loc[0, ["min_angle_degree", "max_angle_degree"]] = [150, 170]
    with pytest.raises(ValueError, match="resonance"):
        pp.runpp(net, enforce_facts_limits=True, numba=False)


@pytest.mark.parametrize("failure", [None, FloatingPointError, np.linalg.LinAlgError])
def test_failed_release_preserves_accepted_solution(monkeypatch, failure):
    net = _svc_network([(4, 1.04, 90, 135)])
    reference = copy.deepcopy(net)
    reference.svc.loc[0, ["thyristor_firing_angle_degree", "controllable"]] = [135, False]
    pp.runpp(reference, numba=False)
    original = nr._run_ac_pf_without_qlims_enforced
    held_solved = False
    failed_trials = []

    def fail_release(case, options):
        nonlocal held_solved
        controllable = bool(case["svc"][0, SVC_CONTROLLABLE])
        result = original(case, options)
        if controllable and held_solved:
            # A failed numerical trial must not replace the last converged solution.
            case["internal"]["V"][:] = np.nan
            failed_trials.append(True)
            if failure is not None:
                raise failure("numerical release failure")
            return case, False, result[2]
        if not controllable:
            held_solved = True
        return result

    monkeypatch.setattr(nr, "_run_ac_pf_without_qlims_enforced", fail_release)
    pp.runpp(net, enforce_facts_limits=True, numba=False)
    assert failed_trials
    assert net.converged
    _assert_bus_voltages(net, reference)


@pytest.mark.parametrize(
    "devices,q",
    [
        ([(8, 1.04, 120, 140), (3, 1.06, 90, 160), (4, 0.96, 120, 135)], (5, 40)),
        ([(7, 0.96, 100, 150), (5, 1.06, 100, 140), (3, 0.96, 100, 160)], (0, 20)),
    ],
)
def test_facts_limits_recover_initially_divergent_controls(devices, q):
    net = _svc_network(devices, q)
    with pytest.raises(pp.LoadflowNotConverged):
        pp.runpp(copy.deepcopy(net), enforce_facts_limits=False, enforce_q_lims=True, numba=False)
    pp.runpp(net, enforce_facts_limits=True, enforce_q_lims=True, numba=False)
    assert net.converged
    assert net.res_svc.thyristor_firing_angle_degree.between(
        net.svc.min_angle_degree - 1e-6,
        net.svc.max_angle_degree + 1e-6,
    ).all()
    assert q[0] - 1e-6 <= net.res_gen.at[0, "q_mvar"] <= q[1] + 1e-6
    mismatch = net.res_gen.p_mw.sum() + net.res_ext_grid.p_mw.sum() - net.res_load.p_mw.sum()
    assert mismatch == pytest.approx(net.res_line.pl_mw.sum(), abs=1e-6)


def test_largest_normalized_violation_is_fixed_first(monkeypatch):
    from pandapower.pypower.idx_gen import GEN_STATUS

    net = _svc_network([(4, 1.04, 90, 135)])
    pp.runpp(net, numba=False)
    initial_q = net.res_gen.at[0, "q_mvar"]
    net.gen.loc[0, ["min_q_mvar", "max_q_mvar"]] = [initial_q + 1, initial_q + 1001]
    original = nr._run_ac_pf_without_qlims_enforced
    states = []

    def observe(case, options):
        states.append((bool(case["svc"][0, SVC_CONTROLLABLE]), bool(case["gen"][1, GEN_STATUS])))
        return original(case, options)

    monkeypatch.setattr(nr, "_run_ac_pf_without_qlims_enforced", observe)
    pp.runpp(net, enforce_facts_limits=True, enforce_q_lims=True, numba=False)
    # Both limits are violated initially; the angle's relative excess exceeds 1/1000.
    assert states[:2] == [(True, True), (False, True)]


def test_repeated_control_state_warns_and_stops(monkeypatch, caplog):
    net = _svc_network([(4, 1.04, 90, 135)])
    original = nr._run_ac_pf_without_qlims_enforced
    held_solved = False
    calls = []

    def alternate_root(case, options):
        nonlocal held_solved
        controllable = bool(case["svc"][0, SVC_CONTROLLABLE])
        initial_angle = case["svc"][0, SVC_THYRISTOR_FIRING_ANGLE]
        result = original(case, options)
        calls.append(True)
        if controllable and held_solved and np.isclose(initial_angle, np.deg2rad(135)):
            # Fault injection: a release trial selects an in-range root, but the
            # following solve returns to the original out-of-range control state.
            case["svc"][0, SVC_THYRISTOR_FIRING_ANGLE] = np.deg2rad(134)
        if not controllable:
            held_solved = True
        return result

    monkeypatch.setattr(nr, "_run_ac_pf_without_qlims_enforced", alternate_root)
    with caplog.at_level(logging.WARNING, logger=nr.__name__):
        pp.runpp(net, enforce_facts_limits=True, numba=False)
    assert "keep switching" in caplog.text
    assert len(calls) <= 5
    assert net.converged
    assert np.isfinite(net.res_bus.vm_pu).all()
    assert net.svc.controllable.all()


def test_fixed_offline_devices_and_disabling_enforcement():
    net = _svc_network([(4, 1.04, 90, 135)])
    pp.create_svc(net, 5, 397, -1190, 1.04, 130, in_service=False)
    pp.create_svc(net, 6, 397, -1190, 1.04, 140, controllable=False, min_angle_degree=90, max_angle_degree=100)
    original = copy.deepcopy(net)
    pp.runpp(net, enforce_facts_limits=True, numba=False)
    assert net.res_svc.at_limit.to_list() == [True, False, False]
    assert net.res_svc.at[2, "thyristor_firing_angle_degree"] == pytest.approx(140)
    assert_frame_equal(net.svc, original.svc, check_exact=True)
    pp.runpp(net, enforce_facts_limits=False, numba=False)
    pp.runpp(original, numba=False)
    _assert_bus_voltages(net, original)
    assert not net.res_svc.at_limit.any()


def test_zero_width_angle_range():
    net = _svc_network([(4, 1.04, 135, 135)])
    pp.runpp(net, enforce_facts_limits=True, numba=False)
    assert net.res_svc.at[0, "thyristor_firing_angle_degree"] == pytest.approx(135)
    assert net.res_svc.at[0, "at_limit"]


def test_facts_limits_with_recycled_powerflow():
    net = _svc_network([(4, 1.04, 90, 135)], (5, 10))
    pp.runpp(net, enforce_facts_limits=True, enforce_q_lims=True, numba=False)
    reference = copy.deepcopy(net)
    pp.runpp(net, recycle={"bus_pq": False, "gen": False, "trafo": False}, numba=False)
    pp.runpp(reference, enforce_facts_limits=True, enforce_q_lims=True, numba=False)
    _assert_bus_voltages(net, reference)
    np.testing.assert_allclose(net.res_gen, reference.res_gen, atol=1e-6, rtol=0)
    assert net.res_svc.at[0, "thyristor_firing_angle_degree"] == pytest.approx(135)

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Tests for the safeguarded secant update of BinarySearchControl (convergence stabilization)."""

import numpy as np
import pytest

from pandapower.auxiliary import ControllerNotConverged
from pandapower.control.controller.station_control import BinarySearchControl
from pandapower.control.run_control import run_control
from pandapower.create import (
    create_empty_network, create_bus, create_buses, create_ext_grid, create_transformer,
    create_load, create_line, create_sgen)  # noqa: F401 - create_buses used in tests
from pandapower.run import runpp


def simple_test_net():
    net = create_empty_network()
    create_bus(net, 110)
    create_buses(net, 2, 20)
    create_ext_grid(net, 0)
    create_transformer(net, 0, 1, "63 MVA 110/20 kV")
    create_load(net, 1, 3, 0.1)
    create_sgen(net, 2, p_mw=2., sn_mva=10, name="sgen1")
    create_line(net, 1, 2, length_km=0.1, std_type="NAYY 4x50 SE")
    return net


def test_near_zero_slope_no_blowup():
    """A measurement that does not respond to the output must not blow up the outputs.

    The legacy secant divided by a 1e-6 dummy slope, multiplying the residual into the
    output values. The controller cannot converge (expected), but outputs stay bounded.
    """
    net = simple_test_net()
    # second, electrically separate feeder: its line flow is unaffected by sgen 0
    create_buses(net, 2, 20)
    create_transformer(net, 0, 3, "63 MVA 110/20 kV")
    create_load(net, 4, 3, 0.5)
    create_line(net, 3, 4, length_km=0.1, std_type="NAYY 4x50 SE")
    BinarySearchControl(
        net, name="flat", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True], output_values_distribution=[1],
        input_element="res_line", input_variable=["q_to_mvar"], input_element_index=1,
        set_point=1, control_modus="Q_ctrl", tol=1e-6)
    # plain NR solver (like the CI environments): the weaker solver must never be driven
    # into voltage collapse by the controller fallback steps
    with pytest.raises(ControllerNotConverged):
        run_control(net, lightsim2grid=False)
    assert abs(net.sgen.q_mvar.at[0]) < 20, "flat-response outputs must stay bounded"


def test_pf_setpoint_not_mutated():
    """The near-zero power factor clipping must not overwrite the user-given set point."""
    net = simple_test_net()
    ctrl = BinarySearchControl(
        net, name="pf", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=0, output_element_in_service=True, output_values_distribution=1,
        input_element="res_line", input_variable="q_to_mvar", input_element_index=0,
        set_point=0.005, control_modus="PF_ctrl_ind", tol=1e-6)
    run_control(net)
    assert ctrl.set_point == 0.005
    assert ctrl.converged


def test_damping_factor_scales_first_probe():
    """damping_factor scales the residual-sized first probe of Q-type controllers."""
    results = {}
    for damping in (1.0, 0.5):
        net = simple_test_net()
        ctrl = BinarySearchControl(
            net, name="q", ctrl_in_service=True, output_element="sgen",
            output_variable="q_mvar", output_element_index=[0], output_element_in_service=[True],
            output_values_distribution=[1], input_element="res_line",
            input_variable=["q_to_mvar"], input_element_index=0, set_point=1,
            control_modus="Q_ctrl", tol=1e-6, damping_factor=damping)
        runpp(net)
        ctrl.initialize_control(net)
        assert not ctrl.is_converged(net)
        ctrl.control_step(net)
        results[damping] = net.sgen.q_mvar.at[0]
    # residual is ~1 Mvar; the probe is damping * residual
    assert results[1.0] == pytest.approx(2 * results[0.5], rel=1e-6)


def test_bracketing_monotone():
    """Once a bracket is established, all further iterates stay inside it and converge."""
    net = simple_test_net()
    ctrl = BinarySearchControl(
        net, name="unit", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True],
        output_values_distribution=[1], input_element="res_line", input_variable=["q_to_mvar"],
        input_element_index=0, set_point=0, control_modus="Q_ctrl", tol=1e-9)

    def residual(x):  # nonlinear, root at x = ln(2) / 0.3
        return 2.0 - np.exp(0.3 * x)

    root = np.log(2.0) / 0.3
    vs = {}
    x_old, x = 0.0, 10.0  # residual(0) > 0, residual(10) < 0 -> bracket on first update
    ctrl.set_point = 0
    ctrl.diff_old, ctrl.diff = residual(x_old), residual(x)
    bracket_seen = False
    for _ in range(60):
        ctrl.output_values_old = np.array([x_old])
        ctrl.output_values = np.array([x])
        x_new = ctrl._safeguarded_secant_total(vs, damping=1.0)
        if vs.get('solver', {}).get('lo') is not None:
            bracket_seen = True
            lo_x = vs['solver']['lo'][0]
            hi_x = vs['solver']['hi'][0]
            assert min(lo_x, hi_x) <= x_new <= max(lo_x, hi_x)
        x_old, x = x, x_new
        ctrl.diff_old, ctrl.diff = ctrl.diff, residual(x_new)
        if abs(ctrl.diff) < 1e-12:
            break
    assert bracket_seen
    assert x == pytest.approx(root, abs=1e-6)


def test_output_resync():
    """External modifications of the written output values are picked up before the update."""
    net = simple_test_net()
    ctrl = BinarySearchControl(
        net, name="q", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True],
        output_values_distribution=[1], input_element="res_line", input_variable=["q_to_mvar"],
        input_element_index=0, set_point=1, control_modus="Q_ctrl", tol=1e-6)
    run_control(net)
    assert ctrl.converged
    net.sgen.q_mvar.at[0] = -3.0  # external actor overwrites the setpoint
    ctrl._resync_output_values(net, ctrl._vstate)
    assert np.atleast_1d(ctrl.output_values)[0] == pytest.approx(-3.0)
    # and a full re-run still converges to the target
    run_control(net)
    assert ctrl.converged
    assert net.res_line.q_to_mvar.at[0] == pytest.approx(1.0, abs=1e-6)


def _two_station_chain_net():
    """Two V_ctrl stations along the same feeder (strong coupling through the transformer)."""
    net = simple_test_net()
    b3 = create_bus(net, 20)
    create_line(net, 2, b3, length_km=5.0, std_type="NAYY 4x50 SE")
    create_sgen(net, b3, p_mw=1., sn_mva=10, name="sgen2")
    BinarySearchControl(
        net, name="v1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True],
        output_values_distribution=[1], input_element="res_bus", input_variable="vm_pu",
        input_element_index=[1], set_point=1.02, control_modus="V_ctrl", tol=1e-6)
    BinarySearchControl(
        net, name="v2", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[1], output_element_in_service=[True],
        output_values_distribution=[1], input_element="res_bus", input_variable="vm_pu",
        input_element_index=[b3], set_point=1.03, control_modus="V_ctrl", tol=1e-6)
    return net, b3


def test_coupled_two_stations_iteration_budget():
    """Two V_ctrl stations in different substations converge within a bounded number of
    powerflows (coupling through the shared HV bus)."""
    from pandapower.test.control.test_stactrl_characterization import CountingRunpp

    net = simple_test_net()
    # second station on its own feeder
    b3, b4 = create_buses(net, 2, 20)
    create_transformer(net, 0, b3, "63 MVA 110/20 kV")
    create_load(net, b3, 3, 0.1)
    create_line(net, b3, b4, length_km=0.1, std_type="NAYY 4x50 SE")
    create_sgen(net, b4, p_mw=1., sn_mva=10, name="sgen2")
    BinarySearchControl(
        net, name="v1", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[0], output_element_in_service=[True],
        output_values_distribution=[1], input_element="res_bus", input_variable="vm_pu",
        input_element_index=[1], set_point=1.02, control_modus="V_ctrl", tol=1e-6)
    BinarySearchControl(
        net, name="v2", ctrl_in_service=True, output_element="sgen", output_variable="q_mvar",
        output_element_index=[1], output_element_in_service=[True],
        output_values_distribution=[1], input_element="res_bus", input_variable="vm_pu",
        input_element_index=[b3], set_point=1.03, control_modus="V_ctrl", tol=1e-6)
    counter = CountingRunpp()
    run_control(net, run=counter)
    assert all(net.controller.object[i].converged for i in net.controller.index)
    assert net.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-6)
    assert net.res_bus.vm_pu.at[b3] == pytest.approx(1.03, abs=1e-6)
    assert counter.count <= 12, f"coupled stations needed {counter.count} powerflows"


@pytest.mark.xfail(reason="two independently iterating V_ctrl stations behind the same "
                          "transformer have a Jacobi contraction factor near 1; the "
                          "pre-refactor implementation does not converge this case either "
                          "(verified with max_iter=100). Solved by the coupled Newton of "
                          "update_method='jacobian'.", strict=True)
def test_coupled_two_stations_same_feeder():
    net, b3 = _two_station_chain_net()
    run_control(net)
    assert net.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-6)
    assert net.res_bus.vm_pu.at[b3] == pytest.approx(1.03, abs=1e-6)


if __name__ == '__main__':
    pytest.main(['-s', __file__])

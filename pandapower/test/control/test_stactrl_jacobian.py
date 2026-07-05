# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Tests for the opt-in Jacobian sensitivity update (update_method="jacobian")."""

import copy

import numpy as np
import pytest

from pandapower.control.controller.station_control import BinarySearchControl
from pandapower.control.run_control import run_control
from pandapower.control.util.sensitivity import calc_dvm_dq
from pandapower.run import runpp
from pandapower.test.control.benchmarks.bench_station_control import (
    build_bench_net, station_dicts)
from pandapower.test.control.test_stactrl_characterization import CountingRunpp


def _bare_bench_net(n_stations):
    net = build_bench_net(n_stations, "Q_ctrl")
    net.controller.drop(net.controller.index, inplace=True)
    return net


def test_dvm_dq_matches_finite_difference():
    """Pins sign, scaling (pu/Mvar) and index mapping of calc_dvm_dq."""
    net = _bare_bench_net(2)
    runpp(net)
    q_buses = [2, 4]   # lv buses carrying the sgens
    vm_buses = [1, 3]  # mv buses
    sensitivity = calc_dvm_dq(net, q_buses, vm_buses)
    assert sensitivity.shape == (2, 2)
    assert np.all(np.isfinite(sensitivity))
    # injection raises the own feeder voltage; the feeders decouple through the slack bus
    assert np.all(np.diag(sensitivity) > 0)
    delta = 0.5  # Mvar, central difference
    for j, sgen in enumerate([0, 2]):  # sgen 0 at bus 2, sgen 2 at bus 4
        vm = {}
        for sign in (+1, -1):
            pert = copy.deepcopy(net)
            pert.sgen.at[sgen, "q_mvar"] += sign * delta
            runpp(pert)
            vm[sign] = pert.res_bus.vm_pu.values[[1, 3]]
        finite_difference = (vm[+1] - vm[-1]) / (2 * delta)
        assert np.allclose(sensitivity[:, j], finite_difference, rtol=1e-3, atol=1e-9), (
            f"column {j}: {sensitivity[:, j]} vs FD {finite_difference}")
    # non-PQ buses give NaN: the slack bus (ref)
    assert np.isnan(calc_dvm_dq(net, [0], [1])[0, 0])
    assert np.isnan(calc_dvm_dq(net, [2], [0])[0, 0])


@pytest.mark.parametrize("n", [3, 20])
def test_jacobian_vctrl_same_result_fewer_runpp(n):
    """Jacobian mode reaches the same fixed point with strictly fewer powerflows."""
    counts, nets = {}, {}
    for method in ("secant", "jacobian"):
        net = _bare_bench_net(n)
        BinarySearchControl.for_stations(net, station_dicts("V_ctrl", n), name=method,
                                         tol=1e-6, update_method=method)
        counter = CountingRunpp()
        run_control(net, run=counter)
        assert net.controller.object.at[0].converged
        counts[method] = counter.count
        nets[method] = net
    mv_buses = [1 + 2 * k for k in range(n)]
    assert np.allclose(nets["jacobian"].res_bus.vm_pu.values[mv_buses], 1.02, atol=2e-6)
    assert np.allclose(nets["jacobian"].res_sgen.q_mvar.values,
                       nets["secant"].res_sgen.q_mvar.values, atol=1e-3)
    assert counts["jacobian"] < counts["secant"], counts


def test_jacobian_coupled_same_feeder():
    """Two V_ctrl stations behind the same transformer: unsolvable for independent secant
    iterations (see test_stactrl_convergence xfail), solved by the coupled Newton."""
    from pandapower.create import create_bus, create_line, create_sgen
    from pandapower.test.control.test_stactrl_convergence import simple_test_net

    net = simple_test_net()
    b3 = create_bus(net, 20)
    create_line(net, 2, b3, length_km=5.0, std_type="NAYY 4x50 SE")
    create_sgen(net, b3, p_mw=1., sn_mva=10, name="sgen2")
    stations = [
        dict(control_modus="V_ctrl", set_point=1.02, input_element="res_bus",
             input_variable="vm_pu", input_element_index=1,
             output_element_index=[0], output_values_distribution=[1]),
        dict(control_modus="V_ctrl", set_point=1.03, input_element="res_bus",
             input_variable="vm_pu", input_element_index=b3,
             output_element_index=[1], output_values_distribution=[1]),
    ]
    BinarySearchControl.for_stations(net, stations, name="coupled", tol=1e-6,
                                     update_method="jacobian")
    counter = CountingRunpp()
    run_control(net, run=counter)
    assert net.controller.object.at[0].converged
    assert net.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-6)
    assert net.res_bus.vm_pu.at[b3] == pytest.approx(1.03, abs=1e-6)
    assert counter.count <= 8, f"coupled Newton needed {counter.count} powerflows"


def test_jacobian_fallback_no_internal():
    """Without a stored Jacobian the controller falls back to secant and still converges."""
    n = 3
    net = _bare_bench_net(n)
    BinarySearchControl.for_stations(net, station_dicts("V_ctrl", n), name="fb",
                                     tol=1e-6, update_method="jacobian")

    class JacobianStrippingRunpp(CountingRunpp):
        def __call__(self, net, **kwargs):
            super().__call__(net, **kwargs)
            net._ppc["internal"].pop("J", None)

    counter = JacobianStrippingRunpp()
    run_control(net, run=counter)
    assert net.controller.object.at[0].converged
    assert np.allclose(net.res_bus.vm_pu.values[[1, 3, 5]], 1.02, atol=2e-6)


def test_jacobian_with_qlims():
    """Stations at their Q limits leave the Newton system; limits are respected."""
    n = 2
    net = _bare_bench_net(n)
    net.sgen["min_q_mvar"] = [-0.1, -0.1, -50.0, -50.0]
    net.sgen["max_q_mvar"] = [0.1, 0.1, 50.0, 50.0]
    BinarySearchControl.for_stations(net, station_dicts("V_ctrl", n), name="lims",
                                     tol=1e-6, update_method="jacobian")
    run_control(net, enforce_q_lims=True)
    assert net.controller.object.at[0].converged
    # station 0 saturates
    assert net.sgen.q_mvar.at[0] == pytest.approx(0.1, abs=1e-6)
    assert net.sgen.q_mvar.at[1] == pytest.approx(0.1, abs=1e-6)
    assert net.res_bus.vm_pu.at[1] < 1.02
    # station 1 reaches its set point
    assert net.res_bus.vm_pu.at[3] == pytest.approx(1.02, abs=1e-6)


def test_qctrl_ignores_jacobian_flag():
    """Q_ctrl stations keep the exact secant iterates under update_method='jacobian'."""
    n = 3
    counts, results = {}, {}
    for method in ("secant", "jacobian"):
        net = _bare_bench_net(n)
        BinarySearchControl.for_stations(net, station_dicts("Q_ctrl", n), name=method,
                                         tol=1e-6, update_method=method)
        counter = CountingRunpp()
        run_control(net, run=counter)
        counts[method] = counter.count
        results[method] = net.sgen.q_mvar.values.copy()
    assert counts["jacobian"] == counts["secant"]
    assert np.allclose(results["jacobian"], results["secant"], atol=1e-12)


if __name__ == '__main__':
    pytest.main(['-s', __file__])

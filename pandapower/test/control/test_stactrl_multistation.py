# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Tests for the multi-station mode of BinarySearchControl (for_stations)."""

import os

import numpy as np
import pytest

from pandapower.control.controller.station_control import BinarySearchControl, DroopControl
from pandapower.control.run_control import run_control
from pandapower.file_io import from_json, to_json
from pandapower.run import runpp
from pandapower.test.control.benchmarks.bench_station_control import build_bench_net
from pandapower.test.control.test_stactrl_characterization import CountingRunpp


def _station_dicts(net, mode, n_stations):
    """Station dicts matching the controllers that build_bench_net(mode) would create."""
    stations = []
    for k in range(n_stations):
        sgen_a, sgen_b = 2 * k, 2 * k + 1
        mv = 1 + 2 * k
        outputs = dict(output_element_index=[sgen_a, sgen_b],
                       output_values_distribution=[0.6, 0.4])
        if mode == "Q_ctrl":
            stations.append(dict(control_modus="Q_ctrl", set_point=0.5,
                                 input_element="res_line", input_variable="q_to_mvar",
                                 input_element_index=k, **outputs))
        elif mode == "V_ctrl":
            stations.append(dict(control_modus="V_ctrl", set_point=1.02,
                                 input_element="res_bus", input_variable="vm_pu",
                                 input_element_index=mv, **outputs))
        else:  # V_ctrl_Q_droop
            stations.append(dict(control_modus="V_ctrl_Q_droop", set_point=1.02,
                                 input_element="res_trafo", input_variable="q_hv_mvar",
                                 input_element_index=k,
                                 droop=dict(q_droop_mvar=40, bus_idx=mv), **outputs))
    return stations


def _bare_bench_net(n_stations):
    """The bench net without any controllers."""
    net = build_bench_net(n_stations, "Q_ctrl")
    net.controller.drop(net.controller.index, inplace=True)
    return net


@pytest.mark.parametrize("mode", ["Q_ctrl", "V_ctrl", "V_ctrl_Q_droop"])
def test_multistation_equals_n_single_controllers(mode):
    """One for_stations instance produces the same result as n single controllers."""
    n = 5
    # reference: n single controllers (legacy path, incl. chained DroopControl)
    net_ref = build_bench_net(n, mode)
    counter_ref = CountingRunpp()
    run_control(net_ref, run=counter_ref)
    # one multi-station instance
    net = _bare_bench_net(n)
    BinarySearchControl.for_stations(net, _station_dicts(net, mode, n), name="multi", tol=1e-6)
    counter = CountingRunpp()
    run_control(net, run=counter)
    assert all(net.controller.object[i].converged for i in net.controller.index)
    assert np.allclose(net.res_sgen.q_mvar.values, net_ref.res_sgen.q_mvar.values, atol=1e-4)
    assert np.allclose(net.res_bus.vm_pu.values, net_ref.res_bus.vm_pu.values, atol=1e-6)
    assert counter.count <= counter_ref.count + 1, (
        f"multi-station needed {counter.count} powerflows, reference {counter_ref.count}")


def test_multistation_mixed_modes():
    """Stations with different control modi in one instance."""
    net = _bare_bench_net(3)
    stations = [
        dict(control_modus="V_ctrl", set_point=1.02, input_element="res_bus",
             input_variable="vm_pu", input_element_index=1,
             output_element_index=[0, 1], output_values_distribution=[0.6, 0.4]),
        dict(control_modus="Q_ctrl", set_point=0.5, input_element="res_line",
             input_variable="q_to_mvar", input_element_index=1,
             output_element_index=[2, 3], output_values_distribution=[0.5, 0.5]),
        dict(control_modus="tan_phi_ctrl", set_point=0.5, input_element="res_trafo",
             input_variable="q_lv_mvar", input_element_index=2,
             output_element_index=[4, 5], output_values_distribution=[1, 1]),
    ]
    ctrl = BinarySearchControl.for_stations(net, stations, name="mixed", tol=1e-6)
    run_control(net)
    assert ctrl.converged
    assert net.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-6)
    assert net.res_line.q_to_mvar.at[1] == pytest.approx(0.5, abs=1e-6)
    tan_phi = net.res_trafo.q_lv_mvar.at[2] / net.res_trafo.p_lv_mw.at[2]
    assert tan_phi == pytest.approx(0.5, abs=1e-5)


def test_multistation_per_station_droop():
    """One station with V droop, one without, in the same instance."""
    net = _bare_bench_net(2)
    stations = [
        dict(control_modus="V_ctrl_Q_droop", set_point=1.02, input_element="res_trafo",
             input_variable="q_hv_mvar", input_element_index=0,
             droop=dict(q_droop_mvar=40, bus_idx=1),
             output_element_index=[0, 1], output_values_distribution=[0.6, 0.4]),
        dict(control_modus="V_ctrl", set_point=1.02, input_element="res_bus",
             input_variable="vm_pu", input_element_index=3,
             output_element_index=[2, 3], output_values_distribution=[0.6, 0.4]),
    ]
    ctrl = BinarySearchControl.for_stations(net, stations, name="droopmix", tol=1e-6)
    run_control(net)
    assert ctrl.converged
    # droop station: vm = set point + q_meas / droop
    assert net.res_bus.vm_pu.at[1] == pytest.approx(
        1.02 + net.res_trafo.q_hv_mvar.at[0] / 40, abs=1e-6)
    # plain station: vm = set point
    assert net.res_bus.vm_pu.at[3] == pytest.approx(1.02, abs=1e-6)


def test_multistation_droop_equals_chained_droop():
    """The folded-in droop reproduces the chained BSC+DroopControl result."""
    net_ref = build_bench_net(1, "V_ctrl_Q_droop")  # legacy chained pair
    run_control(net_ref)
    net = _bare_bench_net(1)
    BinarySearchControl.for_stations(
        net, _station_dicts(net, "V_ctrl_Q_droop", 1), name="folded", tol=1e-6)
    run_control(net)
    assert np.allclose(net.res_bus.vm_pu.values, net_ref.res_bus.vm_pu.values, atol=1e-6)
    assert np.allclose(net.res_sgen.q_mvar.values, net_ref.res_sgen.q_mvar.values, atol=1e-4)


def test_multistation_qlims_isolated():
    """Station A hitting its Q limits must not disturb station B."""
    net = _bare_bench_net(2)
    net.sgen["min_q_mvar"] = [-0.1, -0.1, -50.0, -50.0]
    net.sgen["max_q_mvar"] = [0.1, 0.1, 50.0, 50.0]
    stations = [
        dict(control_modus="V_ctrl", set_point=1.02, input_element="res_bus",
             input_variable="vm_pu", input_element_index=1,
             output_element_index=[0, 1], output_values_distribution=[0.6, 0.4]),
        dict(control_modus="Q_ctrl", set_point=0.5, input_element="res_line",
             input_variable="q_to_mvar", input_element_index=1,
             output_element_index=[2, 3], output_values_distribution=[0.5, 0.5]),
    ]
    ctrl = BinarySearchControl.for_stations(net, stations, name="lims", tol=1e-6)
    run_control(net, enforce_q_lims=True)
    assert ctrl.converged
    # station A saturates at its limits (target 1.02 unreachable with 0.2 Mvar)
    assert net.sgen.q_mvar.at[0] == pytest.approx(0.1, abs=1e-6)
    assert net.sgen.q_mvar.at[1] == pytest.approx(0.1, abs=1e-6)
    assert net.res_bus.vm_pu.at[1] < 1.02
    # station B unaffected
    assert net.res_line.q_to_mvar.at[1] == pytest.approx(0.5, abs=1e-6)
    assert net.sgen.q_mvar.at[2] == pytest.approx(net.sgen.q_mvar.at[3], abs=1e-6)


def test_multistation_json_roundtrip(tmp_path):
    """for_stations controllers survive to_json/from_json and keep solving."""
    net = _bare_bench_net(2)
    BinarySearchControl.for_stations(net, _station_dicts(net, "V_ctrl", 2), name="rt", tol=1e-6)
    json_file = os.path.join(tmp_path, "multistation.json")
    to_json(net, json_file)
    net2 = from_json(json_file)
    ctrl2 = net2.controller.object.at[0]
    assert "_vstate" not in ctrl2.to_dict()
    assert len(ctrl2.stations) == 2
    run_control(net2)
    assert ctrl2.converged
    assert net2.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-6)
    assert net2.res_bus.vm_pu.at[3] == pytest.approx(1.02, abs=1e-6)


def test_multistation_partial_convergence():
    """Outputs of already-converged stations stay frozen while others iterate."""
    net = _bare_bench_net(2)
    # station 0 starts at its solution (set point = initial measurement), station 1 does not
    runpp(net)
    q_initial = net.res_line.q_to_mvar.at[0]
    stations = [
        dict(control_modus="Q_ctrl", set_point=float(q_initial), input_element="res_line",
             input_variable="q_to_mvar", input_element_index=0,
             output_element_index=[0, 1], output_values_distribution=[0.5, 0.5]),
        dict(control_modus="Q_ctrl", set_point=2.0, input_element="res_line",
             input_variable="q_to_mvar", input_element_index=1,
             output_element_index=[2, 3], output_values_distribution=[0.5, 0.5]),
    ]
    ctrl = BinarySearchControl.for_stations(net, stations, name="partial", tol=1e-6)
    run_control(net)
    assert ctrl.converged
    # station 0 was converged from the start and must not have been touched
    assert net.sgen.q_mvar.at[0] == 0.0
    assert net.sgen.q_mvar.at[1] == 0.0
    assert net.res_line.q_to_mvar.at[1] == pytest.approx(2.0, abs=1e-6)


def test_multistation_out_of_service_station():
    """A station whose outputs are out of service is skipped, the rest keeps working."""
    net = _bare_bench_net(2)
    net.sgen.loc[[0, 1], "in_service"] = False
    stations = _station_dicts(net, "Q_ctrl", 2)
    ctrl = BinarySearchControl.for_stations(net, stations, name="oos", tol=1e-6)
    run_control(net)
    assert ctrl.converged
    assert net.res_line.q_to_mvar.at[1] == pytest.approx(0.5, abs=1e-6)
    assert net.sgen.q_mvar.at[0] == 0.0  # untouched


def test_multistation_prerefactor_interop():
    """A frozen pre-refactor net (legacy controllers incl. chained droop) extended with a
    new for_stations controller converges as a whole."""
    from pandapower.create import create_bus, create_line, create_load, create_sgen, \
        create_transformer
    from pandapower.test.control.test_stactrl_characterization import PREREFACTOR_JSON

    net = from_json(PREREFACTOR_JSON)
    # add a new feeder controlled by a multi-station controller
    mv = create_bus(net, 20)
    lv = create_bus(net, 20)
    create_transformer(net, 0, mv, "63 MVA 110/20 kV")
    create_load(net, mv, 3, 0.1)
    new_sgen = create_sgen(net, lv, p_mw=2., sn_mva=10)
    create_line(net, mv, lv, length_km=0.1, std_type="NAYY 4x50 SE")
    BinarySearchControl.for_stations(net, [
        dict(control_modus="V_ctrl", set_point=1.015, input_element="res_bus",
             input_variable="vm_pu", input_element_index=mv,
             output_element_index=[new_sgen], output_values_distribution=[1])],
        name="new_station", tol=1e-6)
    run_control(net)
    assert all(net.controller.object[i].converged for i in net.controller.index)
    assert net.res_bus.vm_pu.at[mv] == pytest.approx(1.015, abs=1e-6)
    # legacy V_ctrl station in the frozen net still reaches its set point
    assert net.res_bus.vm_pu.at[1] == pytest.approx(1.02, abs=1e-6)


if __name__ == '__main__':
    pytest.main(['-s', __file__])

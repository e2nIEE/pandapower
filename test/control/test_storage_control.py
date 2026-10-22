# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.networks as pn
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.control import ConstControl, SelfConsumptionController, StorageController
from pandapower.control.run_control import run_control
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

DT = 0.25


def make_net(max_e=1.0, soc=50.0, min_e=0.0, scaling=1.0, p_mw=0.0, in_service=True):
    """Ext grid, one line and one storage at a second bus."""
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, 20.0)
    b1 = pp.create_bus(net, 20.0)
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(
        net, b0, b1, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, max_i_ka=1.0
    )
    pp.create_storage(
        net,
        b1,
        p_mw=p_mw,
        max_e_mwh=max_e,
        soc_percent=soc,
        min_e_mwh=min_e,
        scaling=scaling,
        in_service=in_service,
    )
    return net


def make_site_net(loads=(0.4,), sgens=(0.1,), soc=50.0, max_e=1.0):
    """Net with a storage, a list of loads and a list of static generators (all at bus 1)."""
    net = make_net(max_e=max_e, soc=soc)
    for p in loads:
        pp.create_load(net, 1, p_mw=p)
    for p in sgens:
        pp.create_sgen(net, 1, p_mw=p)
    return net


def run_step(net, ctrl, t):
    """One manual time step: initialize, control until converged, power flow, commit (as run_control does)."""
    ctrl.initialize_control(net)
    for _ in range(10):
        if ctrl.is_converged(net):
            break
        ctrl.control_step(net)
    else:
        raise AssertionError("storage controller did not converge")
    pp.runpp(net)
    ctrl.finalize_step(net, t)


# ----------------------------------------------------------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# StorageController: physics
# ----------------------------------------------------------------------------------------------------------------------


def test_charge_writes_power_and_advances_soc():
    net = make_net(soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.5)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.5)
    assert ctrl.e_mwh == pytest.approx(0.5 * DT)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(12.5)


def test_discharge_advances_soc_down():
    net = make_net(soc=50.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=-0.5)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(-0.5)
    assert ctrl.e_mwh == pytest.approx(0.5 - 0.5 * DT)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(37.5)


def test_soc_max_limit_clips_charging():
    net = make_net(soc=99.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=1.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.04)
    assert ctrl.e_mwh == pytest.approx(1.0)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(100.0)


def test_charge_at_full_is_zero():
    net = make_net(soc=100.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=1.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.0)
    assert ctrl.e_mwh == pytest.approx(1.0)


def test_soc_min_limit_clips_discharging():
    net = make_net(soc=25.0, min_e=0.2)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=-1.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(-0.2)
    assert ctrl.e_mwh == pytest.approx(0.2)


def test_discharge_at_empty_is_zero():
    net = make_net(soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=-1.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.0)
    assert ctrl.e_mwh == pytest.approx(0.0)


@pytest.mark.parametrize("eta_ch, eta_dis", [(0.9, 1.0), (1.0, 0.8), (0.9, 0.8)])
def test_efficiencies(eta_ch, eta_dis):
    net = make_net(soc=50.0)
    ctrl = StorageController(
        net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, eta_charge=eta_ch, eta_discharge=eta_dis, p_request_mw=0.5
    )
    run_step(net, ctrl, 0)
    assert ctrl.e_mwh == pytest.approx(0.5 + eta_ch * 0.5 * DT)
    ctrl.p_request_mw = -0.5
    run_step(net, ctrl, 1)
    assert ctrl.e_mwh == pytest.approx(0.5 + eta_ch * 0.5 * DT - 0.5 * DT / eta_dis)


def test_power_limits_clip_request():
    net = make_net(soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=0.3, max_discharge_mw=0.2, p_request_mw=2.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.3)
    ctrl.p_request_mw = -2.0
    ctrl.e_mwh = 0.5
    run_step(net, ctrl, 1)
    assert net.storage.at[0, "p_mw"] == pytest.approx(-0.2)


@pytest.mark.parametrize("dt, expected_p", [(0.25, 4.0), (0.5, 2.0), (1.0, 1.0)])
def test_different_delta_t(dt, expected_p):
    # energy limit: charging 1 MWh-capacity from empty needs 1 / dt MW for one interval
    net = make_net(soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=100.0, max_discharge_mw=100.0, delta_t_h=dt, p_request_mw=100.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(expected_p)
    assert ctrl.e_mwh == pytest.approx(min(1.0, expected_p * dt))


def test_zero_request_keeps_soc():
    net = make_net(soc=30.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.0)
    assert ctrl.e_mwh == pytest.approx(0.3)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(30.0)


def test_out_of_service_storage_is_untouched():
    net = make_net(soc=50.0, p_mw=0.3, in_service=False)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.5)
    assert ctrl.is_converged(net)
    ctrl.control_step(net)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.3)
    pp.runpp(net)
    ctrl.finalize_step(net, 0)
    assert ctrl.e_mwh == pytest.approx(0.5)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(50.0)


def test_scaling_applies_to_grid_side_power():
    net = make_net(soc=0.0, scaling=2.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.6)
    run_step(net, ctrl, 0)
    # grid-side power 0.6 MW is written as p_mw = 0.3 because scaling = 2
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.3)
    assert ctrl.e_mwh == pytest.approx(0.6 * DT)


def test_scaling_power_limit_is_grid_side():
    net = make_net(soc=0.0, scaling=2.0)
    ctrl = StorageController(net, 0, max_charge_mw=0.4, max_discharge_mw=1.0, p_request_mw=5.0)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.2)
    assert ctrl.e_mwh == pytest.approx(0.4 * DT)


def test_energy_conservation_without_losses():
    rng = np.random.default_rng(3)
    net = make_net(soc=40.0)
    ctrl = StorageController(net, 0, max_charge_mw=0.8, max_discharge_mw=0.8, p_request_mw=0.0)
    e0 = ctrl.e_mwh
    grid_energy = 0.0
    for t, request in enumerate(rng.uniform(-1.0, 1.0, 200)):
        ctrl.p_request_mw = float(request)
        run_step(net, ctrl, t)
        grid_energy += float(net.storage.at[0, "p_mw"]) * DT
        assert ctrl.e_mwh >= 0.0
        assert ctrl.e_mwh <= 1.0
    assert ctrl.e_mwh == pytest.approx(e0 + grid_energy, abs=1e-9)


def test_power_flow_sees_applied_power():
    net = make_net(soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.5)
    run_step(net, ctrl, 0)
    assert net.res_storage.at[0, "p_mw"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------------------------------
# SelfConsumptionController
# ----------------------------------------------------------------------------------------------------------------------


def _self_consumption(net, **kwargs):
    kwargs.setdefault("load_indices", list(net.load.index))
    kwargs.setdefault("sgen_indices", list(net.sgen.index))
    return SelfConsumptionController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, **kwargs)


def test_self_consumption_surplus_charges():
    net = make_site_net(loads=(0.2,), sgens=(0.5,), soc=0.0)
    ctrl = _self_consumption(net)
    run_step(net, ctrl, 0)
    assert ctrl.p_request_mw == pytest.approx(0.3)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.3)
    assert ctrl.e_mwh == pytest.approx(0.3 * DT)


def test_self_consumption_deficit_discharges():
    net = make_site_net(loads=(0.4,), sgens=(0.1,), soc=50.0)
    ctrl = _self_consumption(net)
    run_step(net, ctrl, 0)
    assert ctrl.p_request_mw == pytest.approx(-0.3)
    assert net.storage.at[0, "p_mw"] == pytest.approx(-0.3)
    assert ctrl.e_mwh == pytest.approx(0.5 - 0.3 * DT)


def test_self_consumption_balanced_site_is_idle():
    net = make_site_net(loads=(0.3,), sgens=(0.3,), soc=40.0)
    ctrl = _self_consumption(net)
    run_step(net, ctrl, 0)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.0)
    assert ctrl.e_mwh == pytest.approx(0.4)


def test_self_consumption_full_storage_does_not_charge():
    net = make_site_net(loads=(0.0,), sgens=(0.5,), soc=100.0)
    ctrl = _self_consumption(net)
    run_step(net, ctrl, 0)
    assert ctrl.p_request_mw == pytest.approx(0.5)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.0)
    assert ctrl.e_mwh == pytest.approx(1.0)


def test_self_consumption_empty_storage_does_not_discharge():
    net = make_site_net(loads=(0.5,), sgens=(0.0,), soc=0.0)
    ctrl = _self_consumption(net)
    run_step(net, ctrl, 0)
    assert ctrl.p_request_mw == pytest.approx(-0.5)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.0)
    assert ctrl.e_mwh == pytest.approx(0.0)


def test_self_consumption_multiple_elements_are_summed():
    net = make_site_net(loads=(0.1, 0.2, 0.3), sgens=(0.2, 0.4), soc=50.0)
    ctrl = _self_consumption(net)
    ctrl.initialize_control(net)
    ctrl.control_step(net)
    assert ctrl.p_request_mw == pytest.approx(-(0.6 - 0.6))
    net.load.at[1, "p_mw"] = 0.8  # load total becomes 1.2, site deficit 0.6 MW
    ctrl.initialize_control(net)
    ctrl.control_step(net)
    assert ctrl.p_request_mw == pytest.approx(-0.6)
    assert net.storage.at[0, "p_mw"] == pytest.approx(-0.6)


def test_self_consumption_uses_scaling():
    net = make_site_net(loads=(0.4,), sgens=(0.1,), soc=50.0)
    net.load.at[0, "scaling"] = 0.5  # load 0.2 MW
    net.sgen.at[0, "scaling"] = 2.0  # sgen 0.2 MW
    ctrl = _self_consumption(net)
    ctrl.initialize_control(net)
    ctrl.control_step(net)
    assert ctrl.p_request_mw == pytest.approx(0.0)
    net.sgen.at[0, "scaling"] = 3.0  # sgen 0.3 MW -> surplus 0.1 MW
    ctrl.initialize_control(net)
    ctrl.control_step(net)
    assert ctrl.p_request_mw == pytest.approx(0.1)


def test_self_consumption_ignores_out_of_service_elements():
    net = make_site_net(loads=(0.4, 0.5), sgens=(0.1, 0.9), soc=50.0)
    net.load.at[1, "in_service"] = False
    net.sgen.at[1, "in_service"] = False
    ctrl = _self_consumption(net)
    ctrl.initialize_control(net)
    ctrl.control_step(net)
    assert ctrl.p_request_mw == pytest.approx(-0.3)


def test_self_consumption_site_excludes_other_elements():
    net = make_site_net(loads=(0.4,), sgens=(0.1,), soc=50.0)
    pp.create_load(net, 1, p_mw=5.0)  # not part of the site
    pp.create_sgen(net, 1, p_mw=5.0)  # not part of the site
    ctrl = _self_consumption(net, load_indices=[0], sgen_indices=[0])
    ctrl.initialize_control(net)
    ctrl.control_step(net)
    assert ctrl.p_request_mw == pytest.approx(-0.3)


def test_self_consumption_follows_time_dependent_profiles():
    net = make_site_net(loads=(0.4,), sgens=(0.0,), soc=50.0)
    ctrl = _self_consumption(net)
    for t, (load, pv) in enumerate([(0.4, 0.0), (0.0, 0.6), (0.2, 0.2)]):
        net.load.at[0, "p_mw"] = load
        net.sgen.at[0, "p_mw"] = pv
        run_step(net, ctrl, t)
    # step 0: deficit 0.4 -> discharge 0.4; step 1: surplus 0.6 -> charge 0.6; step 2: balanced
    assert ctrl.e_mwh == pytest.approx(0.5 - 0.4 * DT + 0.6 * DT)


# ----------------------------------------------------------------------------------------------------------------------
# lifecycle and time series integration
# ----------------------------------------------------------------------------------------------------------------------


def test_controller_is_registered_in_net():
    net = make_net()
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0)
    assert ctrl.index in net.controller.index
    assert isinstance(net.controller.at[ctrl.index, "object"], StorageController)
    assert bool(net.controller.at[ctrl.index, "in_service"])


def test_soc_advances_once_per_time_step_with_repeated_calls():
    net = make_net(soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.5)
    for _ in range(3):
        ctrl.control_step(net)
        ctrl.control_step(net)
    pp.runpp(net)
    ctrl.finalize_step(net, 0)
    ctrl.finalize_step(net, 0)  # same time step again: no second commit
    assert ctrl.e_mwh == pytest.approx(0.5 * DT)


def test_plain_power_flows_leave_soc_unchanged():
    net = make_net(soc=50.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.5)
    run_control(net)
    run_control(net)
    pp.runpp(net)
    pp.runpp(net)
    assert ctrl.e_mwh == pytest.approx(0.5)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(50.0)


def test_run_control_outside_time_series_leaves_soc_unchanged():
    net = make_net(soc=50.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.5)
    run_control(net)
    assert net.storage.at[0, "p_mw"] == pytest.approx(0.5)  # power is applied
    assert ctrl.e_mwh == pytest.approx(0.5)  # but the energy state is not advanced
    assert net.storage.at[0, "soc_percent"] == pytest.approx(50.0)


def _failing_run(net, **kwargs):
    """Power flow that fails for steps where the total load is above 5 MW."""
    if net.load.p_mw.sum() > 5.0:
        net["converged"] = False
        raise LoadflowNotConverged("forced non-convergence")
    return pp.runpp(net, **kwargs)


def test_failed_power_flow_does_not_commit():
    net = make_site_net(loads=(0.4,), sgens=(0.0,), soc=50.0)
    ctrl = _self_consumption(net)
    time_steps = range(3)
    profiles = pd.DataFrame({"load": [0.4, 6.0, 0.4]}, index=time_steps)
    ConstControl(net, "load", "p_mw", element_index=[0], data_source=DFData(profiles), profile_name=["load"])
    run_timeseries(net, time_steps, run=_failing_run, continue_on_divergence=True, verbose=False)
    # step 0: discharge 0.4 MW; step 1 fails (no commit); step 2 discharges again from the same state
    assert ctrl.e_mwh == pytest.approx(0.5 - 2 * 0.4 * DT)
    assert ctrl.last_committed_time == 2


def test_controller_interacts_with_const_control_on_request_and_loads():
    net = make_site_net(loads=(0.0,), sgens=(0.0,), soc=0.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0)
    time_steps = range(4)
    profiles = pd.DataFrame({"req": [0.2, -0.1, 0.0, 0.3]}, index=time_steps)
    ConstControl(
        net,
        "controller",
        "object.p_request_mw",
        element_index=ctrl.index,
        data_source=DFData(profiles),
        profile_name="req",
    )
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=".json")
    ow.log_variable("storage", "soc_percent")
    ow.log_variable("storage", "p_mw")
    run_timeseries(net, time_steps, verbose=False)
    applied = ow.output["storage.p_mw"][0].values
    assert np.allclose(applied, [0.2, -0.1, 0.0, 0.3])
    assert ctrl.e_mwh == pytest.approx((0.2 - 0.1 + 0.3) * DT)


def test_output_writer_logs_start_of_interval_soc_and_power():
    net = make_site_net(loads=(0.4,), sgens=(0.1,), soc=50.0)
    ctrl = _self_consumption(net)
    time_steps = range(3)
    profiles = pd.DataFrame({"load": [0.4, 0.2, 0.0]}, index=time_steps)
    ConstControl(net, "load", "p_mw", element_index=[0], data_source=DFData(profiles), profile_name=["load"])
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=".json")
    ow.log_variable("storage", "soc_percent")
    ow.log_variable("storage", "p_mw")
    run_timeseries(net, time_steps, verbose=False)
    # p_mw is the power of each step, soc_percent the state at the start of each step
    p_expected = [-0.3, -0.1, 0.1]
    assert np.allclose(ow.output["storage.p_mw"][0].values, p_expected)
    soc_start = [50.0]
    for p in p_expected[:-1]:
        soc_start.append(soc_start[-1] + p * DT / 1.0 * 100)
    assert np.allclose(ow.output["storage.soc_percent"][0].values, soc_start)
    assert ctrl.e_mwh == pytest.approx(0.5 + sum(p_expected) * DT)


# ----------------------------------------------------------------------------------------------------------------------
# end-to-end: 96 steps of 15 minutes against an independent numpy reference
# ----------------------------------------------------------------------------------------------------------------------


def _reference_self_consumption(load, pv, e0, max_e, min_e, p_ch, p_dis, eta_ch, eta_dis, dt):
    """Independent implementation of the rule (no pandapower storage classes)."""
    e = e0
    p_grid, soc_start = [], []
    for load_t, pv_t in zip(load, pv):
        soc_start.append(e / max_e * 100)
        request = pv_t - load_t
        headroom_ch = min(p_ch, (max_e - e) / (eta_ch * dt))
        headroom_dis = min(p_dis, (e - min_e) * eta_dis / dt)
        p = min(max(request, -headroom_dis), headroom_ch)
        p_grid.append(p)
        e = e + eta_ch * max(p, 0.0) * dt - max(-p, 0.0) * dt / eta_dis
    return np.array(p_grid), np.array(soc_start), e


# Profile shapes for three summer days (15.07.-17.07.2016, 288 steps of 15 min), taken from SimBench
# 1-LV-rural1--0-sw: a household load (per-unit of its maximum) and a PV feed-in (per-unit of its peak).
# SimBench is licensed under the Open Database License (ODbL) 1.0, (c) University of Kassel and contributors.
# fmt: off
LOAD_PU = np.array([
    0.3784, 0.3784, 0.3243, 0.2811, 0.2973, 0.2432, 0.2432, 0.2324, 0.1892, 0.1784, 0.1622, 0.1730,
    0.1351, 0.1135, 0.1081, 0.1459, 0.1459, 0.1243, 0.0973, 0.0973, 0.0973, 0.0865, 0.0919, 0.1135,
    0.1351, 0.0919, 0.0757, 0.0811, 0.1027, 0.0973, 0.0919, 0.6000, 0.6054, 0.2919, 0.1838, 0.2378,
    0.1892, 0.2703, 0.2054, 0.1892, 0.3135, 0.1784, 0.1189, 0.1081, 0.1405, 0.1838, 0.1514, 0.1459,
    0.2973, 0.3838, 0.4486, 0.3568, 0.1135, 0.2595, 0.5946, 0.8487, 0.4162, 0.2324, 0.2216, 0.1784,
    0.2270, 0.1351, 0.1027, 0.1459, 0.1838, 0.1351, 0.0865, 0.0973, 0.1081, 0.1243, 0.1027, 0.0919,
    0.1027, 0.1784, 0.2162, 0.2324, 0.2487, 0.2162, 0.2541, 0.2000, 0.1892, 0.2432, 0.1838, 0.1405,
    0.1297, 0.1189, 0.1676, 0.2000, 0.2000, 0.1568, 0.2270, 0.3622, 0.4108, 0.3514, 0.2919, 0.2270,
    0.2757, 0.2541, 0.2378, 0.1946, 0.2757, 0.1676, 0.1459, 0.1351, 0.1351, 0.1568, 0.1135, 0.1081,
    0.1189, 0.1243, 0.0919, 0.0919, 0.0811, 0.1459, 0.1189, 0.0919, 0.0919, 0.1135, 0.1081, 0.0703,
    0.0595, 0.0811, 0.1189, 0.1027, 0.0865, 0.0703, 0.0973, 0.0811, 0.0865, 0.1027, 0.1027, 0.1514,
    0.2054, 0.1243, 0.2000, 0.1459, 0.0757, 0.0865, 0.0973, 0.1243, 0.1189, 0.1027, 0.1027, 0.1189,
    0.1081, 0.1297, 0.4865, 0.4541, 0.1189, 0.2108, 0.5027, 0.8595, 0.1405, 0.1027, 0.1135, 0.1189,
    0.1459, 0.1784, 0.1892, 0.1730, 0.2757, 0.1892, 0.1622, 0.0865, 0.1027, 0.1784, 0.1838, 0.1351,
    0.1568, 0.1514, 0.2000, 0.2270, 0.1622, 0.1568, 0.1622, 0.1243, 0.1243, 0.1351, 0.2432, 0.2162,
    0.1838, 0.2432, 0.2432, 0.2757, 0.2703, 0.2811, 0.3027, 0.3838, 0.3838, 0.3514, 0.4108, 0.4432,
    0.4000, 0.3297, 0.3189, 0.2865, 0.2324, 0.2162, 0.0973, 0.1081, 0.1243, 0.0973, 0.0865, 0.1027,
    0.1351, 0.1189, 0.0973, 0.0973, 0.1189, 0.0865, 0.0811, 0.0919, 0.1189, 0.1027, 0.0865, 0.0811,
    0.1027, 0.1189, 0.1081, 0.1135, 0.1459, 0.1405, 0.0865, 0.0919, 0.0919, 0.1027, 0.0649, 0.0865,
    0.1946, 0.1514, 0.1676, 0.0865, 0.0919, 0.0811, 0.1135, 0.1135, 0.1081, 0.1568, 0.4865, 0.2487,
    0.0865, 0.0649, 0.3297, 1.0000, 0.7946, 0.2487, 0.1892, 0.1892, 0.1676, 0.2324, 0.8270, 0.6541,
    0.1243, 0.1351, 0.1189, 0.1351, 0.2162, 0.1243, 0.1351, 0.1568, 0.1730, 0.0973, 0.0973, 0.1514,
    0.1622, 0.1568, 0.1730, 0.1459, 0.1784, 0.1946, 0.1351, 0.1243, 0.1784, 0.1784, 0.1892, 0.1892,
    0.2270, 0.2000, 0.1676, 0.1622, 0.1676, 0.2000, 0.1892, 0.4054, 0.4270, 0.4649, 0.4595, 0.4541,
])
PV_PU = np.array([
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0577, 0.1154, 0.1731, 0.2308, 0.2504, 0.2700, 0.2896, 0.3093, 0.3409, 0.3725, 0.4041,
    0.4358, 0.5285, 0.6212, 0.7139, 0.8067, 0.8304, 0.8542, 0.8779, 0.9017, 0.8843, 0.8668, 0.8494,
    0.8320, 0.8720, 0.9119, 0.9519, 0.9919, 0.9741, 0.9563, 0.9386, 0.9208, 0.8875, 0.8542, 0.8208,
    0.7875, 0.7038, 0.6201, 0.5364, 0.4527, 0.4066, 0.3604, 0.3143, 0.2682, 0.2291, 0.1900, 0.1509,
    0.1118, 0.0839, 0.0559, 0.0280, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0564, 0.1128, 0.1692, 0.2255, 0.2552, 0.2850, 0.3147, 0.3444, 0.3701, 0.3958, 0.4215,
    0.4472, 0.5613, 0.6754, 0.7896, 0.9037, 0.8958, 0.8879, 0.8800, 0.8721, 0.8337, 0.7953, 0.7568,
    0.7184, 0.5925, 0.4666, 0.3407, 0.2148, 0.2753, 0.3359, 0.3964, 0.4570, 0.4206, 0.3842, 0.3479,
    0.3115, 0.3183, 0.3251, 0.3319, 0.3387, 0.3075, 0.2762, 0.2450, 0.2138, 0.1957, 0.1775, 0.1593,
    0.1411, 0.1059, 0.0706, 0.0353, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0488, 0.0976, 0.1464, 0.1952, 0.2144, 0.2336, 0.2528, 0.2720, 0.2843, 0.2966, 0.3089,
    0.3211, 0.4333, 0.5454, 0.6575, 0.7696, 0.7643, 0.7591, 0.7538, 0.7486, 0.7564, 0.7643, 0.7722,
    0.7800, 0.8350, 0.8900, 0.9450, 1.0000, 0.9762, 0.9523, 0.9285, 0.9047, 0.8724, 0.8402, 0.8080,
    0.7757, 0.6884, 0.6010, 0.5136, 0.4262, 0.3847, 0.3432, 0.3017, 0.2602, 0.2230, 0.1857, 0.1484,
    0.1111, 0.0833, 0.0556, 0.0278, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
])
# fmt: on
SITE_LOAD = 0  # the site is load 0 of the Kerber grid, PV and storage are placed on its bus


def test_end_to_end_matches_numpy_reference():
    net = pn.create_kerber_vorstadtnetz_kabel_1()
    site_bus = net.load.at[SITE_LOAD, "bus"]
    profiles = pd.DataFrame({"load_pu": LOAD_PU, "pv_pu": PV_PU}, index=range(len(LOAD_PU)))
    load_peak_mw, pv_peak_mw = 0.004, 0.005
    data_source = DFData(profiles)
    ConstControl(
        net,
        "load",
        "p_mw",
        element_index=SITE_LOAD,
        data_source=data_source,
        profile_name="load_pu",
        scale_factor=load_peak_mw,
    )
    pv = pp.create_sgen(net, site_bus, p_mw=0.0)
    ConstControl(
        net, "sgen", "p_mw", element_index=pv, data_source=data_source, profile_name="pv_pu", scale_factor=pv_peak_mw
    )

    max_e, min_e, p_ch, p_dis, eta_ch, eta_dis, dt = 0.005, 0.0005, 0.002, 0.001, 0.95, 0.9, 0.25
    storage = pp.create_storage(net, site_bus, p_mw=0.0, max_e_mwh=max_e, soc_percent=50.0, min_e_mwh=min_e)
    ctrl = SelfConsumptionController(
        net,
        storage,
        max_charge_mw=p_ch,
        max_discharge_mw=p_dis,
        eta_charge=eta_ch,
        eta_discharge=eta_dis,
        delta_t_h=dt,
        load_indices=[SITE_LOAD],
        sgen_indices=[pv],
    )

    time_steps = range(len(LOAD_PU))
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=".json")
    ow.log_variable("storage", "p_mw")
    ow.log_variable("storage", "soc_percent")
    run_timeseries(net, time_steps, verbose=False)

    load = LOAD_PU * load_peak_mw
    pv_power = PV_PU * pv_peak_mw
    p_ref, soc_ref, e_end = _reference_self_consumption(
        load, pv_power, 0.5 * max_e, max_e, min_e, p_ch, p_dis, eta_ch, eta_dis, dt
    )
    assert np.allclose(ow.output["storage.p_mw"][0].values, p_ref, atol=1e-9)
    assert np.allclose(ow.output["storage.soc_percent"][0].values, soc_ref, atol=1e-7)
    assert ctrl.e_mwh == pytest.approx(e_end, abs=1e-9)
    # the day must reach the power limits and the full and empty energy limits, otherwise the test is weak
    assert np.any(np.isclose(p_ref, p_ch))
    assert np.any(np.isclose(p_ref, -p_dis))
    assert np.any(np.isclose(soc_ref, 100.0))
    assert np.any(np.isclose(soc_ref, 10.0))


def test_second_time_series_run_restarts_from_initial_soc():
    net = make_net(soc=20.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0)
    time_steps = range(4)
    profiles = pd.DataFrame({"req": [0.5, 0.5, -0.2, 0.1]}, index=time_steps)
    ConstControl(
        net,
        "controller",
        "object.p_request_mw",
        element_index=ctrl.index,
        data_source=DFData(profiles),
        profile_name="req",
    )
    run_timeseries(net, time_steps, verbose=False)
    e_first = ctrl.e_mwh
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=".json")
    ow.log_variable("storage", "soc_percent")
    run_timeseries(net, time_steps, verbose=False)
    assert ctrl.e_mwh == pytest.approx(e_first)
    assert ow.output["storage.soc_percent"][0].values[0] == pytest.approx(20.0)
    assert ctrl.last_committed_time == 3


def test_time_step_resets_state_when_time_does_not_advance():
    net = make_net(soc=10.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=1.0)
    ctrl.time_step(net, 0)
    run_step(net, ctrl, 0)
    ctrl.time_step(net, 1)
    run_step(net, ctrl, 1)
    assert ctrl.energy_not_charged_total_mwh == pytest.approx(0.0)
    assert ctrl.e_mwh > 0.1 + DT - 1e-9
    ctrl.time_step(net, 0)  # time goes back: new run
    assert ctrl.e_mwh == pytest.approx(0.1)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(10.0)
    assert ctrl.last_committed_time is None


def test_external_change_of_p_mw_warns_and_clamps(caplog):
    net = make_net(soc=99.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=0.0)
    ctrl.control_step(net)
    net.storage.at[0, "p_mw"] = 1.0  # changed after control_step, outside the controller
    pp.runpp(net)
    with caplog.at_level(logging.WARNING, logger="pandapower.control.controller.storage_control"):
        ctrl.finalize_step(net, 0)
    assert any("clamped" in rec.getMessage() for rec in caplog.records)
    assert ctrl.e_mwh == pytest.approx(1.0)
    assert net.storage.at[0, "soc_percent"] == pytest.approx(100.0)


def test_normal_clipping_does_not_warn(caplog):
    net = make_net(soc=99.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=1.0)
    with caplog.at_level(logging.WARNING, logger="pandapower.control.controller.storage_control"):
        run_step(net, ctrl, 0)
    assert not [rec for rec in caplog.records if rec.name == "pandapower.control.controller.storage_control"]


def test_clipping_indicators_track_not_charged_and_not_discharged():
    net = make_net(soc=99.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=0.2, p_request_mw=1.0)
    run_step(net, ctrl, 0)  # charging limited to 0.04 MW by the energy headroom
    assert ctrl.energy_not_charged_mwh == pytest.approx((1.0 - 0.04) * DT)
    assert ctrl.energy_not_discharged_mwh == pytest.approx(0.0)
    ctrl.p_request_mw = -1.0  # discharging limited to 0.2 MW by the power limit
    run_step(net, ctrl, 1)
    assert ctrl.energy_not_charged_mwh == pytest.approx(0.0)
    assert ctrl.energy_not_discharged_mwh == pytest.approx((1.0 - 0.2) * DT)
    assert ctrl.energy_not_charged_total_mwh == pytest.approx((1.0 - 0.04) * DT)
    assert ctrl.energy_not_discharged_total_mwh == pytest.approx((1.0 - 0.2) * DT)


def test_clipping_indicators_are_not_double_counted():
    net = make_net(soc=99.0)
    ctrl = StorageController(net, 0, max_charge_mw=1.0, max_discharge_mw=1.0, p_request_mw=1.0)
    for _ in range(3):
        ctrl.control_step(net)
    pp.runpp(net)
    ctrl.finalize_step(net, 0)
    ctrl.finalize_step(net, 0)
    assert ctrl.energy_not_charged_total_mwh == pytest.approx((1.0 - 0.04) * DT)


def test_self_consumption_site_must_be_on_storage_bus():
    net = make_site_net(loads=(0.4,), sgens=(0.1,), soc=50.0)
    pp.create_load(net, 0, p_mw=0.2)  # bus 0 is not the storage bus 1
    with pytest.raises(ValueError, match="storage bus"):
        SelfConsumptionController(net, 0, 1.0, 1.0, load_indices=[0, 1], sgen_indices=[])
    pp.create_sgen(net, 0, p_mw=0.2)
    with pytest.raises(ValueError, match="storage bus"):
        SelfConsumptionController(net, 0, 1.0, 1.0, load_indices=[], sgen_indices=[1])
    # on-bus elements are accepted
    SelfConsumptionController(net, 0, 1.0, 1.0, load_indices=[0], sgen_indices=[])


_SC_LOGGER = "pandapower.control.controller.self_consumption_control"


def test_self_consumption_warns_when_full_and_not_on_power_limit(caplog):
    net = make_site_net(loads=(0.0,), sgens=(0.5,), soc=100.0)
    ctrl = _self_consumption(net)
    with caplog.at_level(logging.WARNING, logger=_SC_LOGGER):
        run_step(net, ctrl, 0)
        ctrl.finalize_step(net, 0)  # same time step: no second warning
    warnings = [r for r in caplog.records if r.name == _SC_LOGGER and "full" in r.getMessage()]
    assert len(warnings) == 1
    assert "exchanged with the grid" in warnings[0].getMessage()


def test_self_consumption_warns_when_empty_and_not_on_power_limit(caplog):
    net = make_site_net(loads=(0.5,), sgens=(0.0,), soc=0.0)
    ctrl = _self_consumption(net)
    with caplog.at_level(logging.WARNING, logger=_SC_LOGGER):
        run_step(net, ctrl, 0)
    assert [r for r in caplog.records if r.name == _SC_LOGGER and "empty" in r.getMessage()]


def test_self_consumption_no_warning_for_power_limit_only(caplog):
    net = make_site_net(loads=(0.0,), sgens=(0.5,), soc=50.0)
    ctrl = SelfConsumptionController(
        net, 0, max_charge_mw=0.1, max_discharge_mw=1.0, load_indices=[0], sgen_indices=[0]
    )
    with caplog.at_level(logging.WARNING, logger=_SC_LOGGER):
        run_step(net, ctrl, 0)
    assert ctrl.energy_not_charged_mwh > 0
    assert not [r for r in caplog.records if r.name == _SC_LOGGER]

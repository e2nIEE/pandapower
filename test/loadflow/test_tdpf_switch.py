# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Tests of the switch thermal model integration into the TDPF.

The switch temperatures are post-processing results of the optional `switchgear-thermal-model`
library. These tests cover the input column adapter, the physical correctness of the first order
IEC 62271-306 model and the lifecycle (columns only appear when the calculation actually ran).

Unlike the transformer model, none of the thermal specifications can be derived from the switch
data, so the adapter stores them as input columns of ``net.switch``. The rated current is the one
piece of nameplate data and is taken from the existing ``in_ka`` column.
"""

import itertools
import tempfile

import numpy as np
import pandas as pd
import pytest

from pandapower import pandapowerNet
from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
    create_switch,
    create_transformer3w_from_parameters,
)
from pandapower.pf import switchgear_thermal
from pandapower.pf.switchgear_thermal import (
    SWITCH_DEFAULTS,
    TEMPERATURE_COLUMN,
    SwitchgearThermalTracker,
    calculate_switch_temperatures,
    specs_from_switch,
)
from pandapower.run import runpp

pytest.importorskip("switchgear_thermal_model")

# the thermal input columns, which have no nameplate equivalent on a switch
SWITCH_THERMAL_COLUMNS = (
    "measured_temperature_rise_kelvin",
    "temperature_rise_exponent",
    "thermal_time_constant_minutes",
    "air_temperature_degree_celsius",
)

# parameters used throughout the tests; the currents are low enough to stay below the rated current
RATED_CURRENT_KA = 0.1
AMBIENT_C = 35.0
RISE_K = 75.0
EXPONENT = 1.8
TIME_CONSTANT_MINUTES = 90.0


def _disable_line_tdpf(net):
    """Keep the line out of the thermal calculation.

    ``runpp(tdpf=True)`` validates the presence of the line TDPF columns even when no line opted
    in, so the columns have to exist. The switch thermal model does not use them.
    """
    net.line["tdpf"] = False
    net.line["r_theta_kelvin_per_mw"] = 5.0
    net.line["conductor_outer_diameter_m"] = 0.03
    net.line["mc_joule_per_m_k"] = 1000.0


def _switch_net(p_mw=10.0, switch_kwargs=None):
    """A two bus net with a line switch and a load behind it."""
    net = pandapowerNet(name="switch_thermal")
    bus1, bus2 = create_buses(net, nr_buses=2, vn_kv=[110.0, 110.0])
    create_ext_grid(net, bus1)
    line = create_line_from_parameters(
        net,
        bus1,
        bus2,
        length_km=10.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.4,
        c_nf_per_km=10.0,
        max_i_ka=1.0,
    )
    create_load(net, bus2, p_mw=p_mw)
    kwargs = {
        "closed": True,
        "in_ka": RATED_CURRENT_KA,
        "measured_temperature_rise_kelvin": RISE_K,
        "temperature_rise_exponent": EXPONENT,
        "thermal_time_constant_minutes": TIME_CONSTANT_MINUTES,
        "air_temperature_degree_celsius": AMBIENT_C,
    }
    kwargs.update(switch_kwargs or {})
    index = create_switch(net, bus1, line, et="l", **kwargs)
    # the switch thermal model does not need line TDPF, but runpp(tdpf=True) validates the line
    # columns of the lines that opted in, so keep the line out of the thermal calculation
    _disable_line_tdpf(net)
    return net, index


def _open_switch_net(**switch_kwargs):
    """A net whose only switch is open, so nothing can be calculated."""
    return _switch_net(switch_kwargs={"closed": False, **switch_kwargs})


def _steady_state_temperature(current_ka, rated_current_ka=RATED_CURRENT_KA, ambient_c=AMBIENT_C):
    """Analytic IEC 62271-306 steady state at a constant current."""
    return ambient_c + RISE_K * (abs(current_ka) / rated_current_ka) ** EXPONENT


def _run_plain(net):
    """Solve the power flow without TDPF so that the thermal post-processing can be called directly."""
    runpp(net, tdpf=False)


# --------------------------------------------------------------- input column adapter
def test_specs_from_switch():
    net, index = _switch_net()
    specs = specs_from_switch(net.switch.loc[index])

    assert specs.rated_current == pytest.approx(RATED_CURRENT_KA * 1e3)
    assert specs.measured_temperature_rise == pytest.approx(RISE_K)
    assert specs.temp_rise_exp == pytest.approx(EXPONENT)
    assert specs.thermal_time_constant == pytest.approx(TIME_CONSTANT_MINUTES)


def test_specs_from_switch_falls_back_to_defaults():
    """A direct call must not require the columns that `runpp(tdpf=True)` adds."""
    net, index = _switch_net()
    _run_plain(net)
    for column in SWITCH_THERMAL_COLUMNS:
        del net.switch[column]
    assert not any(column in net.switch.columns for column in SWITCH_THERMAL_COLUMNS)

    specs = specs_from_switch(net.switch.loc[index])

    assert specs.measured_temperature_rise == pytest.approx(SWITCH_DEFAULTS["measured_temperature_rise_kelvin"])
    assert specs.temp_rise_exp == pytest.approx(SWITCH_DEFAULTS["temperature_rise_exponent"])
    assert specs.thermal_time_constant == pytest.approx(SWITCH_DEFAULTS["thermal_time_constant_minutes"])


def test_specs_from_switch_requires_rated_current():
    net, index = _switch_net()
    net.switch.at[index, "in_ka"] = np.nan
    with pytest.raises(ValueError, match="rated current"):
        specs_from_switch(net.switch.loc[index])


# --------------------------------------------------------------- physical correctness
def test_switch_temperature_is_steady_state():
    """The default (tdpf_delay_s=None) result must be the analytic first order steady state."""
    net, index = _switch_net()
    runpp(net, tdpf=True)
    current_ka = abs(net.res_switch.at[index, "i_ka"])

    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(
        _steady_state_temperature(current_ka), rel=1e-6
    )


def test_switch_temperature_uses_ambient_temperature():
    cold_net, cold_index = _switch_net(switch_kwargs={"air_temperature_degree_celsius": 20.0})
    runpp(cold_net, tdpf=True)
    cold = cold_net.res_switch.at[cold_index, TEMPERATURE_COLUMN]

    warm_net, warm_index = _switch_net(switch_kwargs={"air_temperature_degree_celsius": 40.0})
    runpp(warm_net, tdpf=True)
    warm = warm_net.res_switch.at[warm_index, TEMPERATURE_COLUMN]

    assert warm - cold == pytest.approx(20.0, abs=1e-6)


def test_switch_temperature_rise_scales_with_current():
    """The temperature rise above ambient follows (I / I_rated) ** exponent."""
    net, index = _switch_net()
    runpp(net, tdpf=True)
    current_ka = abs(net.res_switch.at[index, "i_ka"])
    rise = net.res_switch.at[index, TEMPERATURE_COLUMN] - AMBIENT_C

    assert rise == pytest.approx(RISE_K * (current_ka / RATED_CURRENT_KA) ** EXPONENT, rel=1e-6)


def test_switch_transient_matches_hand_recursion():
    """One transient step must match the closed form first order recursion."""
    net, index = _switch_net()
    _run_plain(net)
    current_ka = abs(net.res_switch.at[index, "i_ka"])

    dt_minutes = 15.0
    calculate_switch_temperatures(net, dt_minutes=dt_minutes, initial_temperature={index: AMBIENT_C})

    asymptotic = _steady_state_temperature(current_ka)
    expected = AMBIENT_C + (asymptotic - AMBIENT_C) * (1.0 - np.exp(-dt_minutes / TIME_CONSTANT_MINUTES))
    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("delay_s", [30.0, 90.0])
def test_fractional_minute_interval_is_not_truncated(delay_s):
    """Sub-minute and non-integer minute steps must keep their true dt.

    The library truncates the profile step to whole minutes (30 s -> 0 min, 90 s -> 1 min), so the
    adapter has to preserve the dt / thermal_time_constant ratio itself.
    """
    net, index = _switch_net()
    runpp(net, tdpf=True, tdpf_delay_s=delay_s)

    current_ka = abs(net.res_switch.at[index, "i_ka"])
    dt_minutes = delay_s / 60.0
    asymptotic = _steady_state_temperature(current_ka)
    expected = AMBIENT_C + (asymptotic - AMBIENT_C) * (1.0 - np.exp(-dt_minutes / TIME_CONSTANT_MINUTES))

    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(expected, rel=1e-9)


def test_zero_delay_does_not_change_the_temperature():
    """``tdpf_delay_s=0`` means no elapsed time, as for the line TDPF."""
    net, index = _switch_net()
    runpp(net, tdpf=True, tdpf_delay_s=0.0)

    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(AMBIENT_C, rel=1e-12)


def test_switch_transient_uses_initial_temperature():
    """Starting above ambient must cool down towards the same asymptotic temperature."""
    net, index = _switch_net()
    _run_plain(net)
    current_ka = abs(net.res_switch.at[index, "i_ka"])
    asymptotic = _steady_state_temperature(current_ka)

    hot_start = asymptotic + 20.0
    calculated = calculate_switch_temperatures(
        net, dt_minutes=TIME_CONSTANT_MINUTES, initial_temperature={index: hot_start}
    )
    expected = asymptotic + (hot_start - asymptotic) * np.exp(-1.0)

    assert calculated[index] == pytest.approx(expected, rel=1e-9)
    assert calculated[index] < hot_start


def test_switch_temperature_approaches_steady_state_monotonically():
    """Repeated transient steps must heat up strictly towards the steady state."""
    net, index = _switch_net()
    _run_plain(net)
    steady_state = _steady_state_temperature(net.res_switch.at[index, "i_ka"])

    tracker = SwitchgearThermalTracker()
    profile = []
    for _ in range(5):
        temperatures = tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)
        profile.append(temperatures[index])

    assert all(later > earlier for earlier, later in itertools.pairwise(profile))
    assert profile[-1] < steady_state
    assert profile[0] > AMBIENT_C


def test_switch_steady_state_needs_two_samples():
    """The steady state is the limit of the recursion with a very large time step."""
    net, index = _switch_net()
    _run_plain(net)
    current_ka = abs(net.res_switch.at[index, "i_ka"])

    calculate_switch_temperatures(net, dt_minutes=switchgear_thermal.STEADY_STATE_DT_MINUTES)

    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(
        _steady_state_temperature(current_ka), rel=1e-9
    )


def test_switch_reverse_power_flow():
    """Heating scales with I^2, so the current magnitude must be used, not the signed value."""
    net, index = _switch_net()
    _run_plain(net)

    net.res_switch.at[index, "i_ka"] = abs(net.res_switch.at[index, "i_ka"])
    forward = calculate_switch_temperatures(net)[index]

    net.res_switch.at[index, "i_ka"] = -abs(net.res_switch.at[index, "i_ka"])
    reverse = calculate_switch_temperatures(net)[index]

    assert reverse == pytest.approx(forward, rel=1e-12)


def test_equivalence_with_direct_library_call():
    """Guards the adapter against drift from the library API."""
    from switchgear_thermal_model.schemas import InputProfile
    from switchgear_thermal_model.thermal_model import switchgear_temp

    net, index = _switch_net()
    runpp(net, tdpf=True)

    current_a = abs(net.res_switch.at[index, "i_ka"]) * 1e3
    profile = InputProfile.create(
        datetime_index=switchgear_thermal._profile_index(switchgear_thermal.STEADY_STATE_DT_MINUTES),
        load_profile=[current_a, current_a],
        ambient_temperature_profile=[AMBIENT_C, AMBIENT_C],
    )
    expected = float(switchgear_temp(profile, specs_from_switch(net.switch.loc[index])).iloc[-1])

    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------- which switches are calculated
def test_bus_bus_switch_with_z_ohm_is_calculated():
    """A bus-bus switch with an impedance creates a branch and therefore has a usable current."""
    net = pandapowerNet(name="switch_bus_bus")
    bus1, bus2, bus3 = create_buses(net, nr_buses=3, vn_kv=[110.0, 110.0, 110.0])
    create_ext_grid(net, bus1)
    create_line_from_parameters(
        net, bus1, bus2, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10.0, max_i_ka=1.0
    )
    create_load(net, bus3, p_mw=10.0)
    index = create_switch(net, bus2, bus3, et="b", closed=True, in_ka=RATED_CURRENT_KA, z_ohm=0.001)
    _disable_line_tdpf(net)

    runpp(net, tdpf=True)

    assert np.isfinite(net.res_switch.at[index, "i_ka"])
    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(
        _steady_state_temperature(net.res_switch.at[index, "i_ka"]), rel=1e-6
    )


def test_fused_switch_flow_is_reconstructed():
    """A fused bus-bus switch (z_ohm=0) has no branch, but its flow is reconstructed automatically."""
    net = pandapowerNet(name="switch_fused")
    bus1, bus2, bus3 = create_buses(net, nr_buses=3, vn_kv=[110.0, 110.0, 110.0])
    create_ext_grid(net, bus1)
    create_line_from_parameters(
        net, bus1, bus2, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10.0, max_i_ka=1.0
    )
    create_load(net, bus3, p_mw=10.0)
    index = create_switch(net, bus2, bus3, et="b", closed=True, in_ka=RATED_CURRENT_KA, z_ohm=0.0)
    _disable_line_tdpf(net)

    runpp(net, tdpf=True)

    assert np.isfinite(net.res_switch.at[index, "i_ka"])
    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(
        _steady_state_temperature(net.res_switch.at[index, "i_ka"]), rel=1e-6
    )
    # fused switches take part in the calculation, so they receive the same defaults as others
    assert net.switch.at[index, "thermal_time_constant_minutes"] == pytest.approx(
        SWITCH_DEFAULTS["thermal_time_constant_minutes"]
    )


def test_fused_switch_cycle_is_skipped_with_a_warning(caplog):
    """A fused group with a cycle has no unique flow split; TDPF must not abort."""
    net = pandapowerNet(name="switch_fused_cycle")
    bus1, bus2, bus3 = create_buses(net, nr_buses=3, vn_kv=[110.0, 110.0, 110.0])
    create_ext_grid(net, bus1)
    create_line_from_parameters(
        net, bus1, bus2, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10.0, max_i_ka=1.0
    )
    create_load(net, bus3, p_mw=10.0)
    create_switch(net, bus2, bus3, et="b", closed=True, in_ka=RATED_CURRENT_KA, z_ohm=0.0)
    create_switch(net, bus3, bus2, et="b", closed=True, in_ka=RATED_CURRENT_KA, z_ohm=0.0)
    _disable_line_tdpf(net)

    with caplog.at_level("WARNING"):
        runpp(net, tdpf=True)

    assert any("fused bus-bus switches" in record.message for record in caplog.records)
    assert np.isnan(net.res_switch[TEMPERATURE_COLUMN]).all()


def _trafo3w_switch_net(**switch_kwargs):
    """A net whose only switch is a bus-trafo3w switch, which has no solved current."""
    net = pandapowerNet(name="switch_t3")
    hv_bus, mv_bus, lv_bus = create_buses(net, nr_buses=3, vn_kv=[110.0, 20.0, 10.0])
    create_ext_grid(net, hv_bus)
    create_transformer3w_from_parameters(
        net,
        hv_bus=hv_bus,
        mv_bus=mv_bus,
        lv_bus=lv_bus,
        sn_hv_mva=100.0,
        sn_mv_mva=50.0,
        sn_lv_mva=25.0,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        vk_hv_percent=10.0,
        vk_mv_percent=8.0,
        vk_lv_percent=12.0,
        vkr_hv_percent=0.5,
        vkr_mv_percent=0.4,
        vkr_lv_percent=0.6,
        pfe_kw=20.0,
        i0_percent=0.1,
    )
    create_load(net, mv_bus, p_mw=10.0)
    index = create_switch(net, hv_bus, 0, et="t3", closed=True, in_ka=RATED_CURRENT_KA, **switch_kwargs)
    _disable_line_tdpf(net)
    return net, index


def test_t3_switch_is_skipped():
    """A bus-trafo3w switch has no solved current in res_switch and must be skipped."""
    net, index = _trafo3w_switch_net()

    runpp(net, tdpf=True)

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_open_switch_is_skipped():
    net, index = _open_switch_net()
    _run_plain(net)
    calculate_switch_temperatures(net)

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_switch_without_rated_current_is_skipped():
    """The schema default of ``in_ka`` is NaN, so such a switch cannot be scaled."""
    net, index = _switch_net(switch_kwargs={"in_ka": np.nan})
    runpp(net, tdpf=True)

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_direct_call_reconstructs_fused_switch_flows():
    """The direct entry point also reconstructs fused flows when a converged power flow exists."""
    net = pandapowerNet(name="switch_compute_flows")
    bus1, bus2, bus3 = create_buses(net, nr_buses=3, vn_kv=[110.0, 110.0, 110.0])
    create_ext_grid(net, bus1)
    create_line_from_parameters(
        net, bus1, bus2, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10.0, max_i_ka=1.0
    )
    create_load(net, bus3, p_mw=10.0)
    index = create_switch(net, bus2, bus3, et="b", closed=True, in_ka=RATED_CURRENT_KA, z_ohm=0.0)
    _disable_line_tdpf(net)

    _run_plain(net)
    assert np.isnan(net.res_switch.at[index, "i_ka"])

    calculate_switch_temperatures(net)

    assert np.isfinite(net.res_switch.at[index, "i_ka"])
    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(
        _steady_state_temperature(net.res_switch.at[index, "i_ka"]), rel=1e-6
    )


# --------------------------------------------------------------- lifecycle / regression
def test_no_columns_when_tdpf_disabled():
    net, _ = _switch_net()
    runpp(net, tdpf=False)

    assert TEMPERATURE_COLUMN not in net.res_switch.columns


def test_no_columns_when_library_missing(monkeypatch, caplog):
    """Without the optional package the line TDPF behaviour must be unchanged."""
    monkeypatch.setattr(switchgear_thermal, "SWITCHGEAR_THERMAL_AVAILABLE", False)
    net, index = _switch_net()

    with caplog.at_level("WARNING"):
        runpp(net, tdpf=True)

    assert any("switchgear-thermal-model" in record.message for record in caplog.records)
    assert TEMPERATURE_COLUMN not in net.res_switch.columns
    # the regular power flow results are still there
    assert np.isfinite(net.res_switch.at[index, "i_ka"])


def test_switch_thermal_without_switches_does_not_fail():
    net = pandapowerNet(name="no_switches")
    bus1, bus2 = create_buses(net, nr_buses=2, vn_kv=[110.0, 110.0])
    create_ext_grid(net, bus1)
    create_line_from_parameters(
        net, bus1, bus2, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10.0, max_i_ka=1.0
    )
    create_load(net, bus2, p_mw=10.0)
    _disable_line_tdpf(net)

    runpp(net, tdpf=True)

    assert len(net.switch) == 0


def test_stale_temperature_is_not_reported_when_a_switch_is_opened():
    """A switch that is skipped must not keep the temperature of the previous calculation."""
    net, index = _switch_net()
    runpp(net, tdpf=True)
    assert np.isfinite(net.res_switch.at[index, TEMPERATURE_COLUMN])

    net.switch.at[index, "closed"] = False
    calculate_switch_temperatures(net)

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_stale_temperature_is_cleared_when_nothing_is_left():
    """Even when no switch is calculated, the previous temperatures must be cleared."""
    net, index = _switch_net()
    runpp(net, tdpf=True)
    assert np.isfinite(net.res_switch.at[index, TEMPERATURE_COLUMN])

    net.switch.at[index, "in_ka"] = np.nan
    assert calculate_switch_temperatures(net) == {}
    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_switch_and_transformer_temperatures_coexist():
    """Both post-processing models run on the same net without interfering."""
    from pandapower.create import create_transformer_from_parameters

    net = pandapowerNet(name="switch_and_trafo")
    hv_bus, lv_bus = create_buses(net, nr_buses=2, vn_kv=[110.0, 10.0])
    create_ext_grid(net, hv_bus)
    create_transformer_from_parameters(
        net,
        hv_bus=hv_bus,
        lv_bus=lv_bus,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=20.0,
        i0_percent=0.1,
    )
    create_load(net, lv_bus, p_mw=10.0)
    index = create_switch(net, hv_bus, 0, et="t", closed=True, in_ka=RATED_CURRENT_KA)
    _disable_line_tdpf(net)

    runpp(net, tdpf=True)

    assert np.isfinite(net.res_switch.at[index, TEMPERATURE_COLUMN])
    assert np.isfinite(net.res_trafo.at[0, "top_oil_temperature_degree_celsius"])


# --------------------------------------------------------------- thermal state of open switches
def test_open_switch_cools_down_but_reports_nan():
    """An open switch keeps cooling internally while its result column stays NaN."""
    net, index = _switch_net()
    _run_plain(net)
    tracker = SwitchgearThermalTracker()
    for _ in range(3):
        hot = tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)[index]
    assert hot > AMBIENT_C

    net.switch.at[index, "closed"] = False
    profile = []
    for _ in range(3):
        assert tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES) == {}
        profile.append(tracker.temperature[index])
        # a switch that is not in operation has no temperature result
        assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])

    assert profile[0] < hot
    assert all(later < earlier for earlier, later in itertools.pairwise(profile))
    assert profile[-1] > AMBIENT_C


def test_out_of_service_switch_cools_down():
    """The same applies to a switch that is out of service, if the net has that column."""
    net, index = _switch_net()
    _run_plain(net)
    net.switch["in_service"] = True
    tracker = SwitchgearThermalTracker()
    for _ in range(3):
        hot = tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)[index]

    net.switch.at[index, "in_service"] = False
    assert tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES) == {}

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])
    assert tracker.temperature[index] < hot
    assert tracker.temperature[index] > AMBIENT_C


def test_reopened_switch_resumes_from_cooled_state():
    """Closing a switch must continue from its cooled state instead of restarting at ambient."""
    net, index = _switch_net()
    _run_plain(net)
    tracker = SwitchgearThermalTracker()
    for _ in range(3):
        tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)

    net.switch.at[index, "closed"] = False
    for _ in range(3):
        tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)
    cooled = tracker.temperature[index]

    net.switch.at[index, "closed"] = True
    resumed = tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)[index]
    # a cold started tracker would restart from ambient, which is colder than the cooled state
    fresh = SwitchgearThermalTracker().run(net, dt_minutes=TIME_CONSTANT_MINUTES)[index]

    assert cooled > AMBIENT_C
    assert resumed > cooled
    assert resumed > fresh


def test_closed_switch_without_current_keeps_state_unchanged():
    """A closed but uncalculable switch has no current information, so its state must not advance."""
    net, index = _switch_net()
    _run_plain(net)
    tracker = SwitchgearThermalTracker()
    tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)
    stored = tracker.temperature[index]

    net.switch.at[index, "in_ka"] = np.nan
    assert tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES) == {}

    assert tracker.temperature[index] == pytest.approx(stored)
    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_removed_switch_state_is_dropped():
    """The stored state must not outlive the switch it belongs to."""
    net, index = _switch_net()
    _run_plain(net)
    tracker = SwitchgearThermalTracker()
    tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES)
    assert index in tracker.temperature

    net.switch.drop(index=index, inplace=True)
    assert tracker.run(net, dt_minutes=TIME_CONSTANT_MINUTES) == {}

    assert tracker.temperature == {}


# --------------------------------------------------------------- validation / defaults
def test_default_columns_are_filled():
    net, _ = _switch_net()
    for column in SWITCH_THERMAL_COLUMNS:
        del net.switch[column]

    runpp(net, tdpf=True)

    for column, value in SWITCH_DEFAULTS.items():
        assert net.switch.at[0, column] == pytest.approx(value)


def test_nan_values_are_filled_with_defaults():
    net, _ = _switch_net(switch_kwargs={"measured_temperature_rise_kelvin": np.nan})
    runpp(net, tdpf=True)

    assert net.switch.at[0, "measured_temperature_rise_kelvin"] == pytest.approx(
        SWITCH_DEFAULTS["measured_temperature_rise_kelvin"]
    )


def test_non_numeric_value_raises():
    net, _ = _switch_net()
    net.switch["thermal_time_constant_minutes"] = pd.Series(["fast"], index=net.switch.index, dtype="object")
    with pytest.raises(UserWarning, match="must be numeric"):
        runpp(net, tdpf=True)


@pytest.mark.parametrize(
    "column",
    ["measured_temperature_rise_kelvin", "temperature_rise_exponent", "thermal_time_constant_minutes"],
)
def test_non_positive_value_raises(column):
    net, _ = _switch_net(switch_kwargs={column: 0.0})
    with pytest.raises(UserWarning, match=f"switch.{column} must be greater than 0"):
        runpp(net, tdpf=True)


def test_invalid_ambient_temperature_raises():
    net, _ = _switch_net(switch_kwargs={"air_temperature_degree_celsius": -300.0})
    with pytest.raises(UserWarning, match="air_temperature_degree_celsius must be greater than -273"):
        runpp(net, tdpf=True)


def test_invalid_values_of_skipped_switches_are_not_validated():
    """A switch without a rated current is skipped anyway, so it must not reject the run."""
    net, index = _switch_net(switch_kwargs={"in_ka": np.nan, "thermal_time_constant_minutes": -1.0})

    runpp(net, tdpf=True)

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_invalid_values_of_fused_switch_are_validated():
    """A fused bus-bus switch is calculated, so its thermal inputs are validated like any other."""
    net, _ = _switch_net()
    extra_bus = create_buses(net, nr_buses=1, vn_kv=[110.0])[0]
    create_switch(
        net,
        0,
        extra_bus,
        et="b",
        closed=True,
        in_ka=RATED_CURRENT_KA,
        z_ohm=0.0,
        thermal_time_constant_minutes=-1.0,
    )

    with pytest.raises(UserWarning, match="switch.thermal_time_constant_minutes must be greater than 0"):
        runpp(net, tdpf=True)


def test_non_numeric_value_of_fused_switch_raises():
    """A non-numeric thermal input on a fused switch must not silently fall back to the default."""
    net, _ = _switch_net()
    extra_bus = create_buses(net, nr_buses=1, vn_kv=[110.0])[0]
    fused = create_switch(net, 0, extra_bus, et="b", closed=True, in_ka=RATED_CURRENT_KA, z_ohm=0.0)
    net.switch["measured_temperature_rise_kelvin"] = pd.Series(
        [np.nan] * len(net.switch), index=net.switch.index, dtype="object"
    )
    net.switch.at[fused, "measured_temperature_rise_kelvin"] = "hot"

    with pytest.raises(UserWarning, match="switch.measured_temperature_rise_kelvin must be numeric"):
        runpp(net, tdpf=True)


def test_invalid_values_of_t3_switch_are_not_validated():
    """A bus-trafo3w switch has no current, so its thermal inputs must not reject the run."""
    net, index = _trafo3w_switch_net(thermal_time_constant_minutes=-1.0)

    runpp(net, tdpf=True)

    assert np.isnan(net.res_switch.at[index, TEMPERATURE_COLUMN])


def test_open_switch_values_are_not_defaulted():
    """Only the switches that can be calculated receive the thermal assumptions."""
    net, index = _open_switch_net()
    net.switch["measured_temperature_rise_kelvin"] = pd.Series([np.nan], index=net.switch.index, dtype="float64")

    from pandapower.auxiliary import _check_tdpf_switch_parameters

    _check_tdpf_switch_parameters(net)

    assert pd.isna(net.switch.at[index, "measured_temperature_rise_kelvin"])


# --------------------------------------------------------------- documentation examples
# The quick start of the library documentation
# (https://alliander-opensource.github.io/switchgear-thermal-model/examples/quick_start/) uses one
# week of 15 minute steps, a sinusoidal load between 0 A and 1000 A and these thermal parameters.
QUICKSTART_RATED_CURRENT_A = 800.0
QUICKSTART_TEMPERATURE_RISE_K = 75.0
QUICKSTART_TIME_CONSTANT_MINUTES = 60.0
QUICKSTART_EXPONENT = 1.8
QUICKSTART_AMBIENT_C = 20.0
QUICKSTART_DT_MINUTES = 15.0
QUICKSTART_STEPS = 4 * 24 * 7
# The documented load: 500 A plus a 500 A sine, one cycle per day.
QUICKSTART_LOAD_A = 500.0 + 500.0 * np.sin(np.linspace(0.0, 7 * 2 * np.pi, QUICKSTART_STEPS))
# The first transient steps follow from the documented inputs by the first order recursion. The
# library initialises the very first sample to ambient, so the first *interval* already uses the
# 500 A load and is the first temperature above ambient.
QUICKSTART_FIRST_STEPS_C = [27.119162068202108, 33.52482323884594, 39.414262924816725, 44.935918811045184]

QUICKSTART_DATETIME_INDEX = pd.date_range("2025-01-01", periods=QUICKSTART_STEPS, freq="15min")


def _quickstart_net(p_mw=1.0):
    """The quick start parameters as a pandapower net with one bus-bus switch."""
    net = pandapowerNet(name="switch_docs_quickstart")
    bus1, bus2 = create_buses(net, nr_buses=2, vn_kv=[110.0, 110.0])
    create_ext_grid(net, bus1)
    create_load(net, bus2, p_mw=p_mw)
    index = create_switch(
        net,
        bus1,
        bus2,
        et="b",
        closed=True,
        in_ka=QUICKSTART_RATED_CURRENT_A / 1e3,
        z_ohm=0.001,
        measured_temperature_rise_kelvin=QUICKSTART_TEMPERATURE_RISE_K,
        temperature_rise_exponent=QUICKSTART_EXPONENT,
        thermal_time_constant_minutes=QUICKSTART_TIME_CONSTANT_MINUTES,
        air_temperature_degree_celsius=QUICKSTART_AMBIENT_C,
    )
    # only needed to have a res_switch row; the documented load profile is imposed below
    runpp(net, tdpf=False)
    return net, index


def _run_quickstart_profile():
    """Run the documented quick start load profile through the pandapower tracker."""
    net, index = _quickstart_net()
    tracker = SwitchgearThermalTracker()
    profile = []
    for current_a in QUICKSTART_LOAD_A:
        net.res_switch.at[index, "i_ka"] = current_a / 1e3
        profile.append(tracker.run(net, dt_minutes=QUICKSTART_DT_MINUTES)[index])
    return net, index, np.array(profile)


def test_quickstart_specs_from_switch():
    net, index = _quickstart_net()
    specs = specs_from_switch(net.switch.loc[index])

    assert specs.rated_current == pytest.approx(QUICKSTART_RATED_CURRENT_A)
    assert specs.measured_temperature_rise == pytest.approx(QUICKSTART_TEMPERATURE_RISE_K)
    assert specs.temp_rise_exp == pytest.approx(QUICKSTART_EXPONENT)
    assert specs.thermal_time_constant == pytest.approx(QUICKSTART_TIME_CONSTANT_MINUTES)


def test_quickstart_first_transient_steps():
    """The first steps must follow the documented first order recursion."""
    _, _, profile = _run_quickstart_profile()

    assert profile[: len(QUICKSTART_FIRST_STEPS_C)] == pytest.approx(QUICKSTART_FIRST_STEPS_C, rel=1e-12)
    # the first interval heats up above the documented ambient temperature
    assert profile[0] > QUICKSTART_AMBIENT_C


def test_quickstart_peak_stays_below_nominal_steady_state():
    """With a 1000 A peak the switch approaches, but does not reach, the steady state of that load."""
    _, _, profile = _run_quickstart_profile()

    peak_steady_state = (
        QUICKSTART_AMBIENT_C
        + QUICKSTART_TEMPERATURE_RISE_K * (1000.0 / QUICKSTART_RATED_CURRENT_A) ** QUICKSTART_EXPONENT
    )
    assert profile.max() == pytest.approx(128.9585472127182, rel=1e-12)
    assert profile.max() < peak_steady_state
    assert profile.min() == pytest.approx(20.251177350394194, rel=1e-12)
    assert profile.min() > QUICKSTART_AMBIENT_C


def test_quickstart_matches_direct_library_call():
    """The chained single steps must reproduce one direct call of the library.

    The library initialises its first sample to ambient and only integrates the following samples,
    while the pandapower tracker integrates every step, including the first. An extra leading sample
    therefore aligns the two recursions exactly.
    """
    from switchgear_thermal_model.schemas import InputProfile
    from switchgear_thermal_model.thermal_model import switchgear_temp

    net, index, profile = _run_quickstart_profile()

    seeded_load_a = np.concatenate(([QUICKSTART_LOAD_A[0]], QUICKSTART_LOAD_A))
    profile_input = InputProfile.create(
        datetime_index=pd.date_range("2025-01-01", periods=QUICKSTART_STEPS + 1, freq="15min"),
        load_profile=seeded_load_a,
        ambient_temperature_profile=[QUICKSTART_AMBIENT_C] * (QUICKSTART_STEPS + 1),
    )
    expected = switchgear_temp(profile_input, specs_from_switch(net.switch.loc[index])).iloc[1:].to_numpy()

    assert profile == pytest.approx(expected, rel=1e-12)


def test_quickstart_steady_state_with_runpp_tdpf():
    """The quick start parameters through an ordinary ``runpp(tdpf=True)``.

    The documented peak load of 1000 A is set as the load power, so this exercises the regular
    power flow integration instead of imposing the solved current.
    """
    peak_p_mw = np.sqrt(3.0) * 110.0 * 1000.0 / 1e3
    net, index = _quickstart_net(p_mw=peak_p_mw)

    runpp(net, tdpf=True)

    current_a = abs(net.res_switch.at[index, "i_ka"]) * 1e3
    assert current_a == pytest.approx(1000.0, rel=1e-4)
    expected = (
        QUICKSTART_AMBIENT_C
        + QUICKSTART_TEMPERATURE_RISE_K * (current_a / QUICKSTART_RATED_CURRENT_A) ** QUICKSTART_EXPONENT
    )
    assert net.res_switch.at[index, TEMPERATURE_COLUMN] == pytest.approx(expected, rel=1e-6)
    assert net.res_switch.at[index, TEMPERATURE_COLUMN] > 120.0


def test_quickstart_time_series_with_runpp_tdpf():
    """The documented load profile through ``run_timeseries(tdpf=True)``.

    This is the end-to-end variant of the quick start: the load power follows the documented current
    profile, ``runpp`` solves it at every 15 minute step and the switch temperature is chained across
    the steps. The expected temperatures are computed from the *solved* currents, so the small voltage
    drop over the switch impedance does not matter.
    """
    from switchgear_thermal_model.schemas import InputProfile
    from switchgear_thermal_model.thermal_model import switchgear_temp

    from pandapower.control import ConstControl
    from pandapower.timeseries import DFData, OutputWriter, run_timeseries

    net, index = _quickstart_net()
    load_p_mw = np.sqrt(3.0) * 110.0 * QUICKSTART_LOAD_A / 1e3
    data = DFData(pd.DataFrame({"p_mw": load_p_mw}))
    ConstControl(net, "load", "p_mw", element_index=[0], data_source=data, profile_name=["p_mw"])
    output_writer = OutputWriter(
        net,
        time_steps=range(QUICKSTART_STEPS),
        output_path=None,
        log_variables=[("res_switch", TEMPERATURE_COLUMN)],
    )

    currents_a = []

    def recording_runpp(inner_net, **kwargs):
        runpp(inner_net, **kwargs)
        currents_a.append(abs(inner_net.res_switch.at[index, "i_ka"]) * 1e3)

    run_timeseries(
        net,
        time_steps=range(QUICKSTART_STEPS),
        run_control_fct=recording_runpp,
        tdpf=True,
        tdpf_delay_s=QUICKSTART_DT_MINUTES * 60.0,
        verbose=False,
    )

    profile = output_writer.output[f"res_switch.{TEMPERATURE_COLUMN}"][0].to_numpy().ravel()
    # the solved currents follow the documented profile up to the negligible voltage drop
    assert currents_a == pytest.approx(QUICKSTART_LOAD_A.tolist(), rel=1e-4)

    profile_input = InputProfile.create(
        datetime_index=pd.date_range("2025-01-01", periods=QUICKSTART_STEPS + 1, freq="15min"),
        load_profile=np.concatenate(([currents_a[0]], currents_a)),
        ambient_temperature_profile=[QUICKSTART_AMBIENT_C] * (QUICKSTART_STEPS + 1),
    )
    expected = switchgear_temp(profile_input, specs_from_switch(net.switch.loc[index])).iloc[1:].to_numpy()

    assert profile == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------- time series
def _constant_load_switch_net(n_steps, p_mw=40.0):
    """A net with a constant load, so only the thermal inertia can change the switch temperature."""
    from pandapower.control import ConstControl
    from pandapower.timeseries import DFData, OutputWriter

    net = pandapowerNet(name="switch_tdpf_chaining")
    bus1, bus2 = create_buses(net, nr_buses=2, vn_kv=[110.0, 110.0])
    create_ext_grid(net, bus1)
    create_line_from_parameters(
        net, bus1, bus2, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4, c_nf_per_km=10.0, max_i_ka=1.0
    )
    create_load(net, bus2, p_mw=p_mw)
    create_switch(
        net,
        bus1,
        0,
        et="l",
        closed=True,
        in_ka=RATED_CURRENT_KA,
        measured_temperature_rise_kelvin=RISE_K,
        temperature_rise_exponent=EXPONENT,
        thermal_time_constant_minutes=TIME_CONSTANT_MINUTES,
        air_temperature_degree_celsius=AMBIENT_C,
    )
    _disable_line_tdpf(net)

    data = DFData(pd.DataFrame({"p": [p_mw] * n_steps}))
    ConstControl(net, "load", "p_mw", element_index=[0], data_source=data, profile_name=["p"])
    OutputWriter(
        net,
        time_steps=range(n_steps),
        output_path=tempfile.gettempdir(),
        output_file_type=".json",
        log_variables=[("res_switch", TEMPERATURE_COLUMN)],
    )
    return net


def test_timeseries_switch_temperature_is_chained_across_steps():
    """The time series tracker must carry the switch temperature from one step into the next."""
    from pandapower.timeseries import run_timeseries

    n_steps = 6
    net = _constant_load_switch_net(n_steps)
    run_timeseries(net, time_steps=range(n_steps), tdpf=True, tdpf_delay_s=3600.0, verbose=False)

    profile = list(net.output_writer.iat[0, 0].output[f"res_switch.{TEMPERATURE_COLUMN}"].values.ravel())

    _run_plain(net)
    steady_state = _steady_state_temperature(net.res_switch.at[0, "i_ka"])

    assert len(profile) == n_steps
    assert all(later > earlier for earlier, later in itertools.pairwise(profile))
    assert profile[-1] < steady_state
    assert profile[-1] > profile[0]


def test_timeseries_switch_temperature_steady_state_without_delay():
    """Without thermal inertia every step is the steady state, so chaining must not change anything."""
    from pandapower.timeseries import run_timeseries

    n_steps = 4
    net = _constant_load_switch_net(n_steps)
    run_timeseries(net, time_steps=range(n_steps), tdpf=True, verbose=False)

    profile = list(net.output_writer.iat[0, 0].output[f"res_switch.{TEMPERATURE_COLUMN}"].values.ravel())

    assert np.allclose(profile, profile[0])
    assert np.isfinite(profile[0])


def test_timeseries_without_tdpf_has_no_switch_tracker():
    from pandapower.timeseries.run_time_series import init_time_series

    net = _constant_load_switch_net(2)
    ts_variables = init_time_series(net, range(2), verbose=False)

    assert ts_variables["switchgear_thermal_tracker"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

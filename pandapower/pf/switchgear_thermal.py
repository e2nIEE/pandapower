# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Switchgear temperatures for the temperature dependent power flow (TDPF).

This module wraps the optional `switchgear-thermal-model
<https://github.com/alliander-opensource/switchgear-thermal-model>`_ library, which implements the
first order thermal model of IEC 62271-306. It is used as **post-processing**: the thermal model runs
after the power flow and consumes the solved switch current as its load profile. Nothing here enters
the Newton-Raphson Jacobian and no impedance is modified.

The switch temperatures are *reported*, not fed back into the power flow. This is different from the
line TDPF, where the conductor temperature changes ``r_ohm_per_km``.

Unlike a transformer, a switch has no nameplate data from which the thermal specifications could be
derived. The specifications are therefore input columns of ``net.switch`` (see
:data:`SWITCH_DEFAULTS`), and the rated current is taken from the existing ``in_ka`` column.

All knowledge about the external API is kept in this module so that library changes are contained.
If the package is not installed, :data:`SWITCHGEAR_THERMAL_AVAILABLE` is ``False`` and
:func:`calculate_switch_temperatures` logs a warning and returns without touching the result tables.
Line-only and transformer-only TDPF therefore behave exactly as before, and no additional result
columns appear.
"""

__all__ = [
    "SWITCHGEAR_THERMAL_AVAILABLE",
    "SWITCH_DEFAULTS",
    "TEMPERATURE_COLUMN",
    "SwitchgearThermalTracker",
    "calculate_switch_temperatures",
    "dt_minutes_from_options",
    "specs_from_switch",
]

import logging
from datetime import datetime, timedelta
from importlib.util import find_spec

import numpy as np
import pandas as pd

from pandapower.pf.transformer_thermal import (
    STEADY_STATE_DT_MINUTES,
    _ambient_temperature,
    _reset_columns,
    dt_minutes_from_options,
)

SWITCHGEAR_THERMAL_AVAILABLE = find_spec("switchgear_thermal_model") is not None

if SWITCHGEAR_THERMAL_AVAILABLE:
    from switchgear_thermal_model.schemas import InputProfile
    from switchgear_thermal_model.switchgear import Switchgear
    from switchgear_thermal_model.thermal_model import switchgear_temp

logger = logging.getLogger(__name__)

# Result column written to net.res_switch. It is only created when the thermal calculation actually
# ran (see the module docstring and `calculate_switch_temperatures`). The name matches res_line and
# res_trafo, so the generic contingency temperature limit machinery works unchanged.
TEMPERATURE_COLUMN = "temperature_degree_celsius"

# New input columns of net.switch and the assumptions used when a value is missing. The three thermal
# parameters are not derivable from the switch data, so they are assumptions, not nameplate data, and
# `_check_tdpf_switch_parameters` logs every fill.
SWITCH_DEFAULTS = {
    "measured_temperature_rise_kelvin": 75.0,
    "temperature_rise_exponent": 1.8,
    "thermal_time_constant_minutes": 90.0,
    "air_temperature_degree_celsius": 35.0,
}

# arbitrary but fixed start of the synthetic profile index; only the time step matters
_PROFILE_START = datetime(2000, 1, 1)


def _float_or_default(row, column: str, default: float) -> float:
    """Value of an optional thermal input column, falling back to ``default`` if it is missing.

    This is a defensive guard for direct calls. During a normal ``runpp`` the columns were already
    defaulted by ``_check_tdpf_switch_parameters``, mirroring what ``_ambient_temperature`` does for
    the transformer air temperature.
    """
    value = row.get(column, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(value) else value


def _rated_current_a(in_ka) -> float:
    """Rated current in A from ``net.switch.in_ka``.

    ``in_ka`` is documented as the maximum current the switch can carry permanently, which is exactly
    the ``rated_current`` of the library. The library normalises the load with this value, so a zero
    or non-finite rating has to be rejected rather than producing ``inf``.
    """
    rated_current = float(in_ka) * 1e3
    if not np.isfinite(rated_current) or rated_current <= 0:
        raise ValueError(f"rated current in_ka must be positive, got {in_ka!r}")
    return rated_current


def specs_from_switch(row) -> "Switchgear":
    """Build the library specifications of a switch from its input columns.

    ====================================  ==================================================
    pandapower column                     library field
    ====================================  ==================================================
    ``in_ka``                             ``rated_current`` [A]
    ``measured_temperature_rise_kelvin``  ``measured_temperature_rise`` [K]
    ``temperature_rise_exponent``         ``temp_rise_exp`` [-]
    ``thermal_time_constant_minutes``     ``thermal_time_constant`` [min]
    ====================================  ==================================================
    """
    return Switchgear(
        rated_current=_rated_current_a(row.get("in_ka")),
        measured_temperature_rise=_float_or_default(
            row, "measured_temperature_rise_kelvin", SWITCH_DEFAULTS["measured_temperature_rise_kelvin"]
        ),
        temp_rise_exp=_float_or_default(row, "temperature_rise_exponent", SWITCH_DEFAULTS["temperature_rise_exponent"]),
        thermal_time_constant=_float_or_default(
            row, "thermal_time_constant_minutes", SWITCH_DEFAULTS["thermal_time_constant_minutes"]
        ),
    )


def _profile_index(dt_minutes: float) -> list[datetime]:
    """Synthetic two sample datetime index with a time step of ``dt_minutes``."""
    return [_PROFILE_START, _PROFILE_START + timedelta(minutes=float(dt_minutes))]


def _profile_step(dt_minutes: float) -> tuple[float, float]:
    """Whole minute profile step and its time constant scale for a real time step.

    ``switchgear_thermal_model`` 0.0.1 derives its time step from the datetime index with
    ``astype("timedelta64[m]")``, which truncates to whole minutes. A 30 second step would therefore
    become 0 and never heat or cool the switch, and a 90 second step would become 1 instead of 1.5.
    The profile index is consequently always built from a whole number of minutes, and the truncation
    is compensated by scaling ``thermal_time_constant`` so that the ratio
    ``dt / thermal_time_constant`` -- the only place the time step enters the IEC 62271-306
    recursion -- is unchanged. Returns ``(profile_step_minutes, time_constant_scale)``.
    """
    dt = float(dt_minutes)
    if np.isposinf(dt):
        # steady state, matching the line TDPF convention for tdpf_delay_s=inf
        return float(STEADY_STATE_DT_MINUTES), 1.0
    if not np.isfinite(dt) or dt <= 0:
        # no elapsed time: the model must not move (mirrors tdpf_delay_s=0 of the line TDPF). The
        # infinite time constant makes the library's 1 - exp(-step / tau) factor zero.
        return 1.0, np.inf
    step = max(1.0, float(round(dt)))
    return step, step / dt


def _candidate_switches(net, rated_current) -> np.ndarray:
    """Boolean mask of the switches that can be calculated from the current results.

    A switch needs a closed state, a finite positive rated current and a finite solved current. The
    switch schema has no ``in_service`` column by default, so it is only respected when a converter
    added it.
    """
    switch = net.switch
    closed = switch["closed"].fillna(False).to_numpy(dtype=bool)
    rated_current = pd.to_numeric(pd.Series(rated_current), errors="coerce").to_numpy(dtype=float)
    candidate = closed & np.isfinite(rated_current) & (rated_current > 0)
    if "in_service" in switch.columns:
        candidate &= switch["in_service"].fillna(False).to_numpy(dtype=bool)
    if "i_ka" not in net.res_switch.columns:
        return np.zeros(len(switch), dtype=bool)
    current = pd.to_numeric(net.res_switch["i_ka"].reindex(switch.index), errors="coerce").to_numpy(dtype=float)
    return candidate & np.isfinite(current)


def _disconnected_mask(net) -> np.ndarray:
    """Boolean mask of the switches that are explicitly open or out of service.

    A closed switch that cannot be calculated (``et='t3'``, missing rated current, or a fused group
    without a unique flow split) is *not* disconnected: its temperature is simply not advanced,
    because there is no current information to advance it with. A disconnected switch, in contrast,
    physically carries no current and must cool down.
    """
    switch = net.switch
    closed = switch["closed"].fillna(False).to_numpy(dtype=bool)
    if "in_service" in switch.columns:
        in_service = switch["in_service"].fillna(False).to_numpy(dtype=bool)
    else:
        in_service = np.ones(len(switch), dtype=bool)
    return ~closed | ~in_service


def _compute_fused_switch_flows(net) -> None:
    """Fill ``res_switch.i_ka`` for fused bus-bus switches before the thermal step.

    A closed bus-bus switch with ``z_ohm <= 0`` is fused into a single power flow node, so it has no
    branch and no current. The thermal model needs a current, which
    :func:`pandapower.toolbox.compute_switch_flows` reconstructs from the nodal balance of a converged
    power flow. Calling it here makes fused switches part of the thermal result without the user
    having to invoke it manually. A fused group that forms a cycle has no unique flow split; the
    resulting ``ValueError`` is logged. Since :func:`pandapower.toolbox.compute_switch_flows` still
    reconstructs the independent groups, only the switches of the cyclic group stay without a
    temperature instead of aborting the whole TDPF run.
    """
    table = net.switch
    if len(table) == 0 or "et" not in table.columns or not getattr(net, "converged", False):
        return
    z_ohm = (
        pd.to_numeric(table["z_ohm"], errors="coerce")
        if "z_ohm" in table.columns
        else pd.Series(0.0, index=table.index)
    )
    closed = table["closed"].fillna(False).to_numpy(dtype=bool)
    fused = (table["et"].astype(str).to_numpy() == "b") & ~(z_ohm.to_numpy(dtype=float) > 0)
    if not np.any(closed & fused):
        return
    # imported here to avoid a circular import (switchgear_thermal -> toolbox -> create -> run -> powerflow)
    from pandapower.toolbox import compute_switch_flows

    try:
        compute_switch_flows(net)
    except ValueError as exception:
        logger.warning(f"TDPF: cannot compute the flow through fused bus-bus switches: {exception}")


def calculate_switch_temperatures(net, dt_minutes: float | None = None, initial_temperature=None):
    """Calculate the temperature of all current carrying switches and write the results.

    This is post-processing and is meant to be called after the power flow results were extracted, so
    that ``net.res_switch.i_ka`` is available.

    Parameters:
        net: the pandapower network
        dt_minutes: time step of the thermal model in minutes. ``None`` means steady state. This is
            derived from the ``tdpf_delay_s`` power flow option, mirroring the line TDPF thermal
            inertia parameter.
        initial_temperature: optional mapping ``switch index -> temperature`` with the previous
            temperature of a warm start (used by the time series tracker). The library has no
            initial condition argument, so the value is injected through the first ambient sample,
            see :func:`_switch_temperature`.

    Returns:
        dict[int, float]: the resulting switch temperature per switch index. Empty if the
        calculation was skipped.

    The result column is only created when the calculation ran. Switches without a usable current or
    rated current are left as NaN, and an open switch is reported as NaN even if its stored thermal
    state (see :class:`SwitchgearThermalTracker`) keeps cooling down.
    """
    _, temperatures = _switch_temperature_step(
        net, dt_minutes=dt_minutes, initial_temperature=initial_temperature, cool_disconnected=False
    )
    return temperatures


def _switch_temperature_step(net, dt_minutes, initial_temperature, cool_disconnected: bool):
    """Advance the switch temperatures by one time step.

    Returns ``(state, temperatures)``. ``state`` is the persistent internal temperature per switch
    index, including the (not reported) temperature of open switches. ``temperatures`` only contains
    the switches whose result was written to ``net.res_switch`` in this step.

    ``cool_disconnected`` is set by the time series tracker. The standalone post-processing entry point
    does not cool open switches, because a single power flow has no time step to cool over and the
    historical convention is to leave them NaN.
    """
    if not SWITCHGEAR_THERMAL_AVAILABLE:
        logger.warning(
            "TDPF: switch temperatures need the optional 'switchgear-thermal-model' package. "
            "Install it with 'pip install pandapower[tdpf]' on Python >= 3.12 to enable them. "
            "Line and transformer TDPF results are not affected."
        )
        return {}, {}

    if len(net.switch) == 0 or "res_switch" not in net:
        return {}, {}

    # Fused z_ohm=0 bus-bus switches have no branch current until it is reconstructed from the
    # converged nodal balance. Do this here so they participate in the thermal calculation.
    _compute_fused_switch_flows(net)

    # Clear the thermal column of every switch, even if it is skipped below. Stale values from a
    # previous time step must never be reported as the current one.
    _reset_columns(net.res_switch, [TEMPERATURE_COLUMN])

    rated_current = net.switch["in_ka"] if "in_ka" in net.switch.columns else pd.Series(np.nan, index=net.switch.index)
    candidate = _candidate_switches(net, rated_current)

    skipped = int(np.count_nonzero(net.switch["closed"].fillna(False).to_numpy(dtype=bool) & ~candidate))
    if skipped:
        logger.info(
            f"TDPF: skipped {skipped} switch(es): no current in res_switch (bus-trafo3w switch et='t3', or a "
            "fused bus-bus group whose flow split is indeterminate) or no rated current in switch.in_ka."
        )

    disconnected = _disconnected_mask(net) if cool_disconnected else np.zeros(len(net.switch), dtype=bool)
    # The tracker still has to carry the stored state of closed but uncalculable switches, so it must
    # not return early here.
    if not candidate.any() and not disconnected.any() and not cool_disconnected:
        return {}, {}

    if dt_minutes is None:
        dt_minutes = STEADY_STATE_DT_MINUTES
    profile_step, time_constant_scale = _profile_step(dt_minutes)
    datetime_index = _profile_index(profile_step)

    if initial_temperature is None:
        initial_temperature = {}

    state = {}
    temperatures = {}
    for position, index in enumerate(net.switch.index):
        row = net.switch.loc[index]
        previous = initial_temperature.get(index)
        if candidate[position]:
            current_a = _current_a(net.res_switch.at[index, "i_ka"])
            try:
                temperature = _switch_temperature(row, current_a, datetime_index, previous, time_constant_scale)
            except (ValueError, TypeError) as exception:
                # invalid specifications: drop the stored state instead of carrying a stale value
                logger.warning(f"TDPF: skipping switch {index} for the thermal calculation: {exception}")
                continue
            net.res_switch.at[index, TEMPERATURE_COLUMN] = temperature
            state[index] = temperature
            temperatures[index] = temperature
        elif disconnected[position]:
            # The switch carries no current, but the IEC 62271-306 model is recursive, so it still has
            # to cool down. The result stays NaN (a switch is not "in operation"), but the tracker
            # keeps the cooled temperature so that closing it resumes from the correct state.
            try:
                temperature = _switch_temperature(row, 0.0, datetime_index, previous, time_constant_scale)
            except (ValueError, TypeError) as exception:
                logger.warning(f"TDPF: skipping switch {index} for the thermal calculation: {exception}")
                continue
            state[index] = temperature
        elif previous is not None:
            # A closed switch without usable current information: there is nothing to advance the
            # model with, so keep the state unchanged instead of guessing a current.
            state[index] = previous
    return state, temperatures


def _switch_temperature(row, current_a, datetime_index, initial_temperature, time_constant_scale=1.0) -> float:
    """Temperature of a single switch at the last sample of the profile.

    ``time_constant_scale`` compensates the whole-minute profile step for a fractional real time
    step, see :func:`_profile_step`, so that the ``dt / thermal_time_constant`` ratio is preserved.
    """
    switchgear = specs_from_switch(row)
    if time_constant_scale != 1.0:
        switchgear = switchgear.model_copy(
            update={"thermal_time_constant": switchgear.thermal_time_constant * time_constant_scale}
        )
    ambient = _ambient_temperature(row)

    # The library has no initial condition argument: the first sample is always initialised with its
    # ambient temperature. Prepending a sample whose ambient is the previous temperature therefore
    # reproduces the recursion exactly, which is how the time series warm start is injected.
    try:
        previous = float(initial_temperature)
    except (TypeError, ValueError):
        previous = ambient
    if not np.isfinite(previous):
        previous = ambient
    profile = InputProfile.create(
        datetime_index=datetime_index,
        load_profile=[current_a] * len(datetime_index),
        ambient_temperature_profile=[previous, ambient],
    )
    return float(switchgear_temp(profile, switchgear).iloc[-1])


def _current_a(i_ka) -> float:
    """Switch current in A from the solved ``i_ka`` magnitude.

    ``.abs()`` is load bearing: heating scales with I^2, so the current magnitude is the physically
    correct input for reverse power flow. The external library additionally rejects negative load
    profiles, so a signed value would raise. ``_get_branch_flows`` already returns magnitudes, but the
    ``abs()`` documents and enforces this.
    """
    return abs(float(i_ka)) * 1e3


class SwitchgearThermalTracker:
    """Chains the switch temperature across the time steps of a time series.

    The IEC 62271-306 model is recursive: the temperature of a step depends on the previous one. The
    tracker holds the last temperature per switch and uses it as the initial condition of the next
    step, which is what makes a transient time series physically meaningful.

    The state is kept for every switch that has one, including open or out-of-service switches. Those
    carry no current but still cool down, and closing them again must resume from the cooled
    temperature instead of restarting from ambient. Their result column stays NaN while they are open,
    so only the internally stored state is advanced.
    """

    def __init__(self):
        self.temperature = {}

    def reset(self):
        """Forget the stored state."""
        self.temperature = {}

    def run(self, net, dt_minutes: float | None = None):
        """Run the thermal model for one time step and remember the resulting temperatures.

        Returns the temperatures that were written to ``net.res_switch`` in this step. The persistent
        state, including the temperatures of open switches, is kept in :attr:`temperature`.
        """
        state, temperatures = _switch_temperature_step(
            net, dt_minutes=dt_minutes, initial_temperature=self.temperature, cool_disconnected=True
        )
        self.temperature = state
        return temperatures

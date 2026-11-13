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
            see :func:`_calculate_switch`.

    Returns:
        dict[int, float]: the resulting switch temperature per switch index. Empty if the
        calculation was skipped.

    The result column is only created when the calculation ran. Switches without a usable current or
    rated current are left as NaN.
    """
    if not SWITCHGEAR_THERMAL_AVAILABLE:
        logger.warning(
            "TDPF: switch temperatures need the optional 'switchgear-thermal-model' package. "
            "Install it with 'pip install pandapower[tdpf]' on Python >= 3.12 to enable them. "
            "Line and transformer TDPF results are not affected."
        )
        return {}

    if len(net.switch) == 0 or "res_switch" not in net:
        return {}

    # Clear the thermal column of every switch, even if it is skipped below. Stale values from a
    # previous time step must never be reported as the current one.
    _reset_columns(net.res_switch, [TEMPERATURE_COLUMN])

    rated_current = net.switch["in_ka"] if "in_ka" in net.switch.columns else pd.Series(np.nan, index=net.switch.index)
    candidate = _candidate_switches(net, rated_current)

    skipped = int(np.count_nonzero(net.switch["closed"].fillna(False).to_numpy(dtype=bool) & ~candidate))
    if skipped:
        logger.info(
            f"TDPF: skipped {skipped} switch(es): no current in res_switch (fused z_ohm=0 bus-bus switch or "
            "et='t3') or no rated current in switch.in_ka. Use z_ohm > 0 to model them, or compute their flows "
            "with pandapower.toolbox.compute_switch_flows(net) after a load flow and then run the TDPF calculation."
        )

    if not candidate.any():
        return {}

    if dt_minutes is None:
        dt_minutes = STEADY_STATE_DT_MINUTES
    datetime_index = _profile_index(dt_minutes)

    if initial_temperature is None:
        initial_temperature = {}

    temperatures = {}
    for index in net.switch.index[candidate]:
        try:
            temperature = _calculate_switch(net, index, datetime_index, initial_temperature.get(index))
        except (ValueError, TypeError) as exception:
            logger.warning(f"TDPF: skipping switch {index} for the thermal calculation: {exception}")
            continue
        net.res_switch.at[index, TEMPERATURE_COLUMN] = temperature
        temperatures[index] = temperature
    return temperatures


def _calculate_switch(net, index, datetime_index, initial_temperature) -> float:
    """Temperature of a single switch at the last sample of the profile."""
    row = net.switch.loc[index]
    switchgear = specs_from_switch(row)
    load_a = _current_a(net.res_switch.at[index, "i_ka"])
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
        load_profile=[load_a] * len(datetime_index),
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
    """

    def __init__(self):
        self.temperature = {}

    def reset(self):
        """Forget the stored state."""
        self.temperature = {}

    def run(self, net, dt_minutes: float | None = None):
        """Run the thermal model for one time step and remember the resulting temperatures."""
        self.temperature = calculate_switch_temperatures(
            net, dt_minutes=dt_minutes, initial_temperature=self.temperature
        )
        return self.temperature

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Transformer top-oil and hot-spot temperatures for the temperature dependent power flow (TDPF).

This module wraps the optional `transformer-thermal-model
<https://github.com/alliander-opensource/transformer-thermal-model>`_ library, which implements the
IEC 60076-7 loading guide. It is used as **post-processing**: the thermal model runs after the power
flow and consumes the solved transformer loading as its load profile. Nothing here enters the
Newton-Raphson Jacobian and the transformer resistance ``r_sc`` is never modified.

The transformer temperatures are *reported*, not fed back into the power flow. This is different from
the line TDPF, where the conductor temperature changes ``r_ohm_per_km``.

All knowledge about the external API is kept in this module so that library changes are contained.
If the package is not installed, :data:`TRANSFORMER_THERMAL_AVAILABLE` is ``False`` and
:func:`calculate_transformer_temperatures` logs a warning and returns without touching the result
tables. Line-only TDPF therefore behaves exactly as before, and no additional result columns appear.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from importlib.util import find_spec

TRANSFORMER_THERMAL_AVAILABLE = find_spec("transformer_thermal_model") is not None

if TRANSFORMER_THERMAL_AVAILABLE:
    from transformer_thermal_model.cooler import CoolerType
    from transformer_thermal_model.model import Model
    from transformer_thermal_model.schemas import (
        InputProfile,
        ThreeWindingInputProfile,
        UserThreeWindingTransformerSpecifications,
        UserTransformerSpecifications,
        WindingSpecifications,
    )
    from transformer_thermal_model.schemas.thermal_model.initial_state import (
        ColdStart,
        InitialTopOilTemp,
    )
    from transformer_thermal_model.transformer import (
        PowerTransformer,
        ThreeWindingTransformer,
    )


# Result columns written to net.res_trafo / net.res_trafo3w. They are only created when the thermal
# calculation actually ran (see the module docstring and `calculate_transformer_temperatures`).
TOP_OIL_COLUMN = "top_oil_temperature_degree_celsius"
HOT_SPOT_COLUMN = "hot_spot_temperature_degree_celsius"
# alias of the hot-spot temperature so that the existing contingency temperature limit machinery works
TEMPERATURE_COLUMN = "temperature_degree_celsius"
# per winding hot-spot columns of a three winding transformer, mapped to the column name the library
# uses in the hot-spot DataFrame it returns for three winding transformers
HOT_SPOT_3W_COLUMNS = {
    "hot_spot_temperature_degree_celsius_lv": "low_voltage_side",
    "hot_spot_temperature_degree_celsius_mv": "middle_voltage_side",
    "hot_spot_temperature_degree_celsius_hv": "high_voltage_side",
}

# Time step used to obtain the steady state solution of the recursive IEC 60076-7 model. The top-oil
# temperature follows 1 - exp(-dt / (k11 * time_const_oil)); with 1e6 minutes every exponent underflows
# to zero, so the model returns the analytic steady state in its second sample. This is also used
# instead of the library's `InitialLoad` initial condition because `InitialLoad` is broken for three
# winding transformers in transformer-thermal-model 0.6.0 (`_end_temperature_top_oil` is called with
# `load=` although the three winding signature does not accept it).
STEADY_STATE_DT_MINUTES = 1e6

# arbitrary but fixed start of the synthetic profile index; only the time step matters
_PROFILE_START = datetime(2000, 1, 1)

_COOLING_TYPES = ("ONAN", "ONAF")


def _cooling_type(cooling_type: str | None) -> "CoolerType":
    """Map a pandapower cooling type string to the library enum.

    An empty value (``None``, NaN or ``pd.NA``) falls back to ONAN, which is the same default the power
    flow applies when the ``cooling_type`` column is absent or empty. The defaulting happens here and not
    only in the caller, because ``calculate_transformer_temperatures`` is also a direct entry point that
    a partially populated column can reach without ``runpp`` normalising it first.
    """
    if pd.isna(cooling_type):
        return CoolerType.ONAN
    normalised = str(cooling_type).strip().upper()
    if normalised not in _COOLING_TYPES:
        raise ValueError(f"Unknown cooling_type {cooling_type!r}, must be one of {_COOLING_TYPES}")
    return CoolerType.ONAF if normalised == "ONAF" else CoolerType.ONAN


def specs_from_trafo(net, index) -> "UserTransformerSpecifications":
    """Build the library specifications of a two winding transformer from its nameplate data.

    Mapping of the pandapower nameplate to the IEC 60076-7 inputs:

    ====================================  ==================================================
    pandapower                            library
    ====================================  ==================================================
    ``pfe_kw``                            ``no_load_loss`` [W]
    ``vkr_percent`` / ``sn_mva``          ``load_loss`` [W]
    ``sn_mva`` / ``vn_lv_kv``             ``nom_load_sec_side`` [A]
    ====================================  ==================================================

    All optional specifications are left at ``None`` so the library applies its IEC defaults for the
    selected cooling type.
    """
    trafo = net.trafo.loc[index]
    sn_mva = float(trafo.sn_mva)
    vn_lv_kv = float(trafo.vn_lv_kv)
    return UserTransformerSpecifications(
        no_load_loss=float(trafo.pfe_kw) * 1e3,
        load_loss=float(trafo.vkr_percent) / 100.0 * sn_mva * 1e6,
        nom_load_sec_side=sn_mva * 1e3 / (np.sqrt(3.0) * vn_lv_kv),
    )


def specs_from_trafo3w(net, index) -> "UserThreeWindingTransformerSpecifications":
    """Build the library specifications of a three winding transformer from its nameplate data.

    The ``vk*_*_percent`` columns of ``net.trafo3w`` are **pairwise** (delta) values, not star
    equivalent ones -- see ``doc/elements/trafo3w.rst``. Each pair value refers to the **minimum** of
    the two rated powers involved. The library expects exactly these pairwise losses, so no
    star/delta conversion is needed.

    The column suffixes denote the *pair*, not the winding, which is the easy way to get this wrong:

    ====================  ==============  ==================
    pandapower column     physical pair   library field
    ====================  ==============  ==================
    ``vkr_hv_percent``    HV <-> MV       ``load_loss_hv_mv``
    ``vkr_mv_percent``    MV <-> LV       ``load_loss_mv_lv``
    ``vkr_lv_percent``    HV <-> LV       ``load_loss_hv_lv``
    ====================  ==============  ==================
    """
    trafo = net.trafo3w.loc[index]
    sn_hv_mva = float(trafo.sn_hv_mva)
    sn_mv_mva = float(trafo.sn_mv_mva)
    sn_lv_mva = float(trafo.sn_lv_mva)

    def _pair_loss_w(vkr_percent, sn_a_mva, sn_b_mva):
        return float(vkr_percent) / 100.0 * min(sn_a_mva, sn_b_mva) * 1e6

    def _winding(sn_mva, vn_kv):
        return WindingSpecifications(
            nom_load=sn_mva * 1e3 / (np.sqrt(3.0) * float(vn_kv)),
            nom_power=sn_mva,
        )

    return UserThreeWindingTransformerSpecifications(
        no_load_loss=float(trafo.pfe_kw) * 1e3,
        hv_winding=_winding(sn_hv_mva, trafo.vn_hv_kv),
        mv_winding=_winding(sn_mv_mva, trafo.vn_mv_kv),
        lv_winding=_winding(sn_lv_mva, trafo.vn_lv_kv),
        load_loss_hv_mv=_pair_loss_w(trafo.vkr_hv_percent, sn_hv_mva, sn_mv_mva),
        load_loss_mv_lv=_pair_loss_w(trafo.vkr_mv_percent, sn_mv_mva, sn_lv_mva),
        load_loss_hv_lv=_pair_loss_w(trafo.vkr_lv_percent, sn_hv_mva, sn_lv_mva),
    )


def _profile_index(dt_minutes: float) -> list[datetime]:
    """Synthetic two sample datetime index with a time step of ``dt_minutes``."""
    return [_PROFILE_START, _PROFILE_START + timedelta(minutes=float(dt_minutes))]


def _initial_condition(initial_top_oil_temperature):
    """Initial state of the recursive model.

    A known previous top-oil temperature is used for a warm start (time series). Otherwise a cold
    start is used: for the steady state case the accompanying time step (see
    :data:`STEADY_STATE_DT_MINUTES`) is so large that the model settles on the steady state in the
    second sample anyway.
    """
    if initial_top_oil_temperature is not None and np.isfinite(initial_top_oil_temperature):
        return InitialTopOilTemp(initial_top_oil_temp=float(initial_top_oil_temperature))
    return ColdStart()


def _ambient_temperature(trafo) -> float:
    """Ambient temperature of a transformer, defaulting to 35 degC if the column is absent or empty."""
    air_temperature = trafo.get("air_temperature_degree_celsius", np.nan)
    try:
        air_temperature = float(air_temperature)
    except (TypeError, ValueError):
        return 35.0
    return 35.0 if not np.isfinite(air_temperature) else air_temperature


def _reset_columns(res_df: pd.DataFrame, columns) -> None:
    """Create missing result columns and clear all existing values to NaN.

    The thermal columns are post-processing results that are only written for the rows whose
    calculation succeeded. Clearing them first prevents a row that is skipped in this step (e.g.
    because its nameplate data became invalid) from reporting the temperature of an earlier step.
    """
    for column in columns:
        res_df[column] = np.nan


def _current_a(i_ka, parallel=1.0) -> float:
    """Transformer current in A from the solved ``i_lv_ka`` / ``i_*_ka`` magnitude.

    ``parallel`` divides the terminal current over the parallel units of a ``net.trafo`` row.
    """
    # `.abs()` is load bearing: heating scales with I^2, so the current magnitude is the physically
    # correct input for reverse power flow (LV -> HV, e.g. a back-fed transformer). The external
    # library additionally rejects negative load profiles, so a signed value would raise.
    # `_get_branch_flows` already returns magnitudes, but the abs() documents and enforces this.
    parallel = float(parallel)
    if not np.isfinite(parallel) or parallel <= 0:
        parallel = 1.0
    return abs(float(i_ka)) * 1e3 / parallel


def calculate_transformer_temperatures(net, dt_minutes: float | None = None, initial_top_oil_temperature=None):
    """Calculate top-oil and hot-spot temperatures of all in-service transformers and write results.

    This is post-processing and is meant to be called after the power flow results were extracted, so
    that ``net.res_trafo.i_lv_ka`` and ``net.res_trafo3w.i_*_ka`` are available.

    Parameters:
        net: the pandapower network
        dt_minutes: time step of the thermal model in minutes. ``None`` means steady state. This is
            derived from the ``tdpf_delay_s`` power flow option, mirroring the line TDPF thermal
            inertia parameter.
        initial_top_oil_temperature: optional mapping ``(element, index) -> top oil temperature``
            with the previous top-oil temperature of a warm start (used by the time series tracker).

    Returns:
        dict[tuple[str, int], float]: the resulting top-oil temperature per transformer. Empty if the
        calculation was skipped.

    The result columns are only created when the calculation ran. Out-of-service transformers are
    left as NaN.
    """
    if not TRANSFORMER_THERMAL_AVAILABLE:
        logger.warning(
            "TDPF: transformer temperatures need the optional 'transformer-thermal-model' package. "
            "Install it with 'pip install pandapower[tdpf]' to enable them. Line TDPF results are "
            "not affected."
        )
        return {}

    trafo_idx = net.trafo.index[net.trafo.in_service.values] if len(net.trafo) else net.trafo.index[:0]
    trafo3w_idx = net.trafo3w.index[net.trafo3w.in_service.values] if len(net.trafo3w) else net.trafo3w.index[:0]

    # Clear the thermal columns of every transformer that exists, even if it is out of service or is
    # skipped below. Stale values from a previous time step must never be reported as the current one.
    if len(net.trafo):
        _reset_columns(net.res_trafo, [TOP_OIL_COLUMN, HOT_SPOT_COLUMN, TEMPERATURE_COLUMN])
    if len(net.trafo3w):
        _reset_columns(net.res_trafo3w, [TOP_OIL_COLUMN, *HOT_SPOT_3W_COLUMNS, TEMPERATURE_COLUMN])

    if len(trafo_idx) == 0 and len(trafo3w_idx) == 0:
        return {}

    if dt_minutes is None:
        dt_minutes = STEADY_STATE_DT_MINUTES
    datetime_index = _profile_index(dt_minutes)

    if initial_top_oil_temperature is None:
        initial_top_oil_temperature = {}

    top_oil_temperatures = {}
    if len(trafo_idx):
        top_oil_temperatures.update(_calculate_trafo(net, trafo_idx, datetime_index, initial_top_oil_temperature))
    if len(trafo3w_idx):
        top_oil_temperatures.update(_calculate_trafo3w(net, trafo3w_idx, datetime_index, initial_top_oil_temperature))
    return top_oil_temperatures


def _calculate_trafo(net, indices, datetime_index, initial_top_oil_temperature) -> dict:
    """Two winding transformers."""
    res_trafo = net.res_trafo

    top_oil_temperatures = {}
    for index in indices:
        row = net.trafo.loc[index]
        if not np.isfinite(row.sn_mva) or row.sn_mva <= 0 or not np.isfinite(row.vn_lv_kv) or row.vn_lv_kv <= 0:
            logger.warning(
                f"TDPF: skipping transformer {index} for the thermal calculation, sn_mva and vn_lv_kv must be positive"
            )
            continue
        try:
            specs = specs_from_trafo(net, index)
            transformer = PowerTransformer(
                user_specs=specs, cooling_type=_cooling_type(row.get("cooling_type", "ONAN"))
            )
            current_a = _current_a(net.res_trafo.at[index, "i_lv_ka"], row.get("parallel", 1.0))
            profile = InputProfile.create(
                datetime_index=datetime_index,
                load_profile=[current_a, current_a],
                ambient_temperature_profile=[_ambient_temperature(row)] * len(datetime_index),
            )
            initial_condition = _initial_condition(initial_top_oil_temperature.get(("trafo", index)))
            output = Model(
                temperature_profile=profile, transformer=transformer, initial_condition=initial_condition
            ).run()
        except (ValueError, TypeError) as exception:
            logger.warning(f"TDPF: skipping transformer {index} for the thermal calculation: {exception}")
            continue

        top_oil = float(output.top_oil_temp_profile.iloc[-1])
        hot_spot = float(output.hot_spot_temp_profile.iloc[-1])
        res_trafo.at[index, TOP_OIL_COLUMN] = top_oil
        res_trafo.at[index, HOT_SPOT_COLUMN] = hot_spot
        res_trafo.at[index, TEMPERATURE_COLUMN] = hot_spot
        top_oil_temperatures[("trafo", index)] = top_oil
    return top_oil_temperatures


def _calculate_trafo3w(net, indices, datetime_index, initial_top_oil_temperature) -> dict:
    """Three winding transformers."""
    res_trafo3w = net.res_trafo3w

    top_oil_temperatures = {}
    for index in indices:
        row = net.trafo3w.loc[index]
        if any(
            not np.isfinite(row[column]) or row[column] <= 0
            for column in ("sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv")
        ):
            logger.warning(
                f"TDPF: skipping three winding transformer {index} for the thermal calculation, "
                "rated powers and voltages must be positive"
            )
            continue
        try:
            specs = specs_from_trafo3w(net, index)
            transformer = ThreeWindingTransformer(
                user_specs=specs, cooling_type=_cooling_type(row.get("cooling_type", "ONAN"))
            )
            # i_hv_ka / i_mv_ka / i_lv_ka are already attributed to the individual windings by
            # _get_trafo3w_results. `.abs()` for the same reason as in `_trafo_current_a`; there is no
            # `parallel` column in net.trafo3w.
            current_hv = _current_a(net.res_trafo3w.at[index, "i_hv_ka"])
            current_mv = _current_a(net.res_trafo3w.at[index, "i_mv_ka"])
            current_lv = _current_a(net.res_trafo3w.at[index, "i_lv_ka"])
            profile = ThreeWindingInputProfile.create(
                datetime_index=datetime_index,
                ambient_temperature_profile=[_ambient_temperature(row)] * len(datetime_index),
                load_profile_high_voltage_side=[current_hv, current_hv],
                load_profile_middle_voltage_side=[current_mv, current_mv],
                load_profile_low_voltage_side=[current_lv, current_lv],
            )
            initial_condition = _initial_condition(initial_top_oil_temperature.get(("trafo3w", index)))
            output = Model(
                temperature_profile=profile, transformer=transformer, initial_condition=initial_condition
            ).run()
        except (ValueError, TypeError) as exception:
            logger.warning(f"TDPF: skipping three winding transformer {index} for the thermal calculation: {exception}")
            continue

        hot_spot = output.hot_spot_temp_profile.iloc[-1]
        for column, winding_side in HOT_SPOT_3W_COLUMNS.items():
            res_trafo3w.at[index, column] = float(getattr(hot_spot, winding_side))
        res_trafo3w.at[index, TOP_OIL_COLUMN] = float(output.top_oil_temp_profile.iloc[-1])
        res_trafo3w.at[index, TEMPERATURE_COLUMN] = float(
            max(hot_spot.low_voltage_side, hot_spot.middle_voltage_side, hot_spot.high_voltage_side)
        )
        top_oil_temperatures[("trafo3w", index)] = float(output.top_oil_temp_profile.iloc[-1])
    return top_oil_temperatures


class TransformerThermalTracker:
    """Chains the transformer top-oil temperature across the time steps of a time series.

    The IEC 60076-7 model is recursive: the top-oil temperature of a step depends on the previous one.
    The tracker holds the last top-oil temperature per transformer and uses it as the initial condition
    of the next step, which is what makes a transient time series physically meaningful.
    """

    def __init__(self):
        self.top_oil_temperature = {}

    def reset(self):
        """Forget the stored state."""
        self.top_oil_temperature = {}

    def run(self, net, dt_minutes: float | None = None):
        """Run the thermal model for one time step and remember the resulting top-oil temperatures."""
        self.top_oil_temperature = calculate_transformer_temperatures(
            net, dt_minutes=dt_minutes, initial_top_oil_temperature=self.top_oil_temperature
        )
        return self.top_oil_temperature


def dt_minutes_from_options(options) -> float | None:
    """Convert the ``tdpf_delay_s`` power flow option into a thermal model time step in minutes.

    ``None`` and ``inf`` mean steady state, matching the line TDPF convention.
    """
    tdpf_delay_s = options.get("tdpf_delay_s")
    if tdpf_delay_s is None or not np.isfinite(tdpf_delay_s):
        return None
    return float(tdpf_delay_s) / 60.0

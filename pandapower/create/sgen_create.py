# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy import nan, bool_
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import GeneratorType, Int, UnderOverExcitedType, WyeDeltaType
from pandapower.create._utils import (
    _add_to_entries_if_not_nan,
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan
)

logger = logging.getLogger(__name__)


def create_sgen(
    net: pandapowerNet,
    bus: Int,
    p_mw: float,
    q_mvar: float = 0,
    sn_mva: float = nan,
    name: Optional[str] = None,
    index: Optional[Int] = None,
    scaling: float = 1.,
    type: WyeDeltaType= 'wye',
    in_service: bool = True,
    max_p_mw: float = nan,
    min_p_mw: float = nan,
    max_q_mvar: float = nan,
    min_q_mvar: float = nan,
    controllable: bool | float = nan,
    k: float = nan,
    rx: float = nan,
    id_q_capability_characteristic: Optional[int] = None,
    reactive_capability_curve: bool = False,
    curve_style = None,
    current_source: bool = True,
    generator_type: Optional[GeneratorType] = None,
    max_ik_ka: float = nan,
    kappa: float = nan,
    lrc_pu: float = nan,
    **kwargs
) -> Int:
    """
    Adds one static generator in table net["sgen"].

    Static generators are modelled as positive and constant PQ power. This element is used to model
    generators with a constant active and reactive power feed-in. If you want to model a voltage
    controlled generator, use the generator element instead.

    gen, sgen and ext_grid in the grid are modelled in the generator system!
    If you want to model the generation of power, you have to assign a positive active power
    to the generator. Please pay attention to the correct signing of the
    reactive power as well (positive for injection and negative for consumption).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

        **p_mw** (float) - The active power of the static generator  (positive for generation!)

    OPTIONAL:
        **q_mvar** (float, 0) - The reactive power of the sgen

        **sn_mva** (float, None) - Nominal power of the sgen

        **name** (string, None) - The name for this sgen

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (float, 1.) - An optional scaling factor to be set customly.
        Multiplies with p_mw and q_mvar.

        **type** (string, None) -  Three phase Connection type of the static generator: wye/delta

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** (float, NaN) - Maximum active power injection - necessary for \
            controllable sgens in OPF

        **min_p_mw** (float, NaN) - Minimum active power injection - necessary for \
            controllable sgens in OPF

        **max_q_mvar** (float, NaN) - Maximum reactive power injection - necessary for \
            controllable sgens in OPF

        **min_q_mvar** (float, NaN) - Minimum reactive power injection - necessary for \
            controllable sgens in OPF

        **controllable** (bool, NaN) - Whether this generator is controllable by the optimal \
            powerflow; defaults to False if "controllable" column exists in DataFrame

        **k** (float, NaN) - Ratio of short circuit current to nominal current

        **rx** (float, NaN) - R/X ratio for short circuit impedance. Only relevant if type is \
            specified as motor so that sgen is treated as asynchronous motor. Relevant for \
            short-circuit calculation for all generator types

        **reactive_capability_curve** (bool, False) - True if both the id_q_capability_characteristic and the \
            curve style are present in the generator

        **id_q_capability_characteristic** (int, None) - references the index of the characteristic from the \
            net.q_capability_characteristic table (id_q_capability_curve column)

        **curve_style** (string, None) - The curve style of the generator represents the relationship \
            between active power (P) and reactive power (Q). It indicates whether the reactive power remains \
            constant as the active power changes or varies dynamically in response to it, \
            e.g. "straightLineYValues" and "constantYValue"

        **generator_type** (str, None) - can be one of "current_source" \
            (full size converter), "async" (asynchronous generator), or "async_doubly_fed"\
            (doubly fed asynchronous generator, DFIG). Represents the type of the static \
            generator in the context of the short-circuit calculations of wind power station units. \
            If None, other short-circuit-related parameters are not set

        **lrc_pu** (float, nan) - locked rotor current in relation to the rated generator \
            current. Relevant if the generator_type is "async".

        **max_ik_ka** (float, nan) - the highest instantaneous short-circuit value in case \
            of a three-phase short-circuit (provided by the manufacturer). Relevant if the \
            generator_type is "async_doubly_fed".

        **kappa** (float, nan) - the factor for the calculation of the peak short-circuit \
            current, referred to the high-voltage side (provided by the manufacturer). \
            Relevant if the generator_type is "async_doubly_fed". \
            If the superposition method is used (use_pre_fault_voltage=True), this parameter \
            is used to pass through the max. current limit of the machine in p.u.

        **current_source** (bool, True) - Model this sgen as a current source during short-\
            circuit calculations; useful in some cases, for example the simulation of full-\
            size converters per IEC 60909-0:2016.

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    EXAMPLE:
        create_sgen(net, 1, p_mw=120)

    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "sgen", index, name="static generator")

    entries = {"name": name, "bus": bus, "p_mw": p_mw, "scaling": scaling, "q_mvar": q_mvar, "sn_mva": sn_mva,
               "in_service": in_service, "type": type, "current_source": current_source, **kwargs}
    _set_entries(net, "sgen", index, True, entries=entries)

    _set_value_if_not_nan(net, index, min_p_mw, "min_p_mw", "sgen")
    _set_value_if_not_nan(net, index, max_p_mw, "max_p_mw", "sgen")
    _set_value_if_not_nan(net, index, min_q_mvar, "min_q_mvar", "sgen")
    _set_value_if_not_nan(net, index, max_q_mvar, "max_q_mvar", "sgen")
    _set_value_if_not_nan(net, index, controllable, "controllable", "sgen", dtype=bool_,
                          default_val=False)

    _set_value_if_not_nan(net, index, id_q_capability_characteristic,
                          "id_q_capability_characteristic", "sgen", dtype="Int64")

    _set_value_if_not_nan(net, index, reactive_capability_curve, "reactive_capability_curve", "sgen",
                          dtype=bool_)

    _set_value_if_not_nan(net, index, curve_style, "curve_style", "sgen", dtype=object, default_val=None)

    _set_value_if_not_nan(net, index, rx, "rx", "sgen")  # rx is always required
    if np.isfinite(kappa):
        _set_value_if_not_nan(net, index, kappa, "kappa", "sgen")
    _set_value_if_not_nan(net, index, generator_type, "generator_type", "sgen",
                          dtype="str", default_val="current_source")
    if generator_type == "current_source" or generator_type is None:
        _set_value_if_not_nan(net, index, k, "k", "sgen")
    elif generator_type == "async":
        _set_value_if_not_nan(net, index, lrc_pu, "lrc_pu", "sgen")
    elif generator_type == "async_doubly_fed":
        _set_value_if_not_nan(net, index, max_ik_ka, "max_ik_ka", "sgen")
    else:
        raise UserWarning(f"unknown sgen generator_type {generator_type}! "
                          f"Must be one of: None, 'current_source', 'async', 'async_doubly_fed'")

    return index


def create_sgens(
    net: pandapowerNet,
    buses: Sequence,
    p_mw: float | Iterable[float],
    q_mvar: float | Iterable[float] = 0,
    sn_mva: float | Iterable[float] = nan,
    name: Optional[Iterable[str]] = None,
    index: Optional[Int] | Iterable[Int] = None,
    scaling: float | Iterable[float] = 1.,
    type: WyeDeltaType = 'wye',
    in_service: bool | Iterable[bool] = True,
    max_p_mw: float | Iterable[float] = nan,
    min_p_mw: float | Iterable[float] = nan,
    max_q_mvar: float | Iterable[float] = nan,
    min_q_mvar: float | Iterable[float] = nan,
    controllable: bool | Iterable[bool] | float = nan,
    k: float | Iterable[float] = nan,
    rx: float = nan,
    id_q_capability_characteristic: Optional[Int] | Iterable[Int] = None,
    reactive_capability_curve: bool | Iterable[bool] = False,
    curve_style: Optional[str] | Optional[Iterable[str]] = None,
    current_source: bool | Iterable[bool] = True,
    generator_type: GeneratorType = "current_source",
    max_ik_ka: float = nan,
    kappa: float = nan,
    lrc_pu: float = nan,
    **kwargs
) -> npt.NDArray[Int]:
    """
    Adds a number of sgens in table net["sgen"].

    Static generators are modelled as positive and constant PQ power. This element is used to model
    generators with a constant active and reactive power feed-in. If you want to model a voltage
    controlled generator, use the generator element instead.

    INPUT:
        **net** - The net within this load should be created

        **buses** (list of int) - A list of bus ids to which the loads are connected

    OPTIONAL:

        **p_mw** (list of floats) - The active power of the sgens

             - positive value   -> generation
             - negative value  -> load

        **q_mvar** (list of floats, default 0) - The reactive power of the sgens

        **sn_mva** (list of floats, default None) - Nominal power of the sgens

        **name** (list of strings, default None) - The name for this sgen

        **scaling** (list of floats, default 1.) - An OPTIONAL scaling factor to be set customly.
        Multiplies with p_mw and q_mvar.

        **type** (string, None) -  type variable to classify the sgen

        **index** (list of int, None) - Force a specified ID if it is available. If None, the index\
             is set to a range between one higher than the highest already existing index and the \
             length of sgens that shall be created.

        **in_service** (list of boolean) - True for in_service or False for out of service

        **max_p_mw** (list of floats, default NaN) - Maximum active power sgen - necessary for \
             controllable sgens in for OPF

        **min_p_mw** (list of floats, default NaN) - Minimum active power sgen - necessary for \
             controllable sgens in for OPF

        **max_q_mvar** (list of floats, default NaN) - Maximum reactive power sgen - necessary for \
             controllable sgens in for OPF

        **min_q_mvar** (list of floats, default NaN) - Minimum reactive power sgen - necessary for \
             controllable sgens in OPF

        **controllable** (list of boolean, default NaN) - States, whether a sgen is controllable \
             or not. Only respected for OPF. Defaults to False if "controllable" column exists in DataFrame

        **k** (list of floats, None) - Ratio of nominal current to short circuit current

        **rx** (float, NaN) - R/X ratio for short circuit impedance. Only relevant if type is \
            specified as motor so that sgen is treated as asynchronous motor. Relevant for \
            short-circuit calculation for all generator types

        **reactive_capability_curve** (list of bools, False) - True if both the id_q_capability_characteristic \
            and the curve style are present in the generator.

        **id_q_capability_characteristic** (list of ints, None) - references the index of the characteristic \
            from the lookup table net.q_capability_characteristic e.g. 0, 1, 2, 3

        **curve_style** (list of strings, None) - The curve style of the generator represents the relationship \
           between active power (P) and reactive power (Q). It indicates whether the reactive power remains \
           constant as the active power changes or varies dynamically in response to it.
           e.g. "straightLineYValues" and "constantYValue"

        **generator_type** (list of strings, "current_source") - can be one of "current_source" \
            (full size converter), "async" (asynchronous generator), or "async_doubly_fed"\
            (doubly fed asynchronous generator, DFIG). Represents the type of the static \
            generator in the context of the short-circuit calculations of wind power station units

        **lrc_pu** (list of float, nan) - locked rotor current in relation to the rated generator \
            current. Relevant if the generator_type is "async".

        **max_ik_ka** (list of float, nan) - the highest instantaneous short-circuit value in case \
            of a three-phase short-circuit (provided by the manufacturer). Relevant if the \
            generator_type is "async_doubly_fed".

        **kappa** (list of float, nan) - the factor for the calculation of the peak short-circuit \
            current, referred to the high-voltage side (provided by the manufacturer). \
            Relevant if the generator_type is "async_doubly_fed".

        **current_source** (list of bool, True) - Model this sgen as a current source during short-\
            circuit calculations; useful in some cases, for example the simulation of full-\
            size converters per IEC 60909-0:2016.

    OUTPUT:
        **index** (int) - The unique IDs of the created elements

    EXAMPLE:
        create_sgens(net, buses=[0, 2], p_mw=[10., 5.], q_mvar=[2., 0.])

    """
    _check_multiple_elements(net, buses)

    index = _get_multiple_index_with_check(net, "sgen", index, len(buses))

    entries = {"bus": buses, "p_mw": p_mw, "q_mvar": q_mvar, "sn_mva": sn_mva, "scaling": scaling,
               "in_service": in_service, "name": name, "type": type, "current_source": current_source,
               "reactive_capability_curve": reactive_capability_curve, "curve_style": curve_style, **kwargs}

    _add_to_entries_if_not_nan(net, "sgen", entries, index, "min_p_mw", min_p_mw)
    _add_to_entries_if_not_nan(net, "sgen", entries, index, "max_p_mw", max_p_mw)
    _add_to_entries_if_not_nan(net, "sgen", entries, index, "min_q_mvar", min_q_mvar)
    _add_to_entries_if_not_nan(net, "sgen", entries, index, "max_q_mvar", max_q_mvar)
    _add_to_entries_if_not_nan(net, "sgen", entries, index, "controllable", controllable, dtype=bool_,
                               default_val=False)
    _add_to_entries_if_not_nan(net, "sgen", entries, index, "rx", rx)  # rx is always required
    if np.isfinite(kappa):
        _add_to_entries_if_not_nan(net, "sgen", entries, index, "kappa",
                                   kappa)  # is used for Type C also as a max. current limit
    _add_to_entries_if_not_nan(net, "sgen", entries, index, "generator_type", generator_type,
                               dtype="str", default_val="current_source")
    gen_types = ['current_source', 'async', 'async_doubly_fed']
    gen_type_match = pd.concat([entries["generator_type"] == match for match in gen_types], axis=1,
                               keys=gen_types)  # type: ignore[call-overload]

    _add_to_entries_if_not_nan(net, "sgen", entries, index, "id_q_capability_characteristic",
                               id_q_capability_characteristic, dtype="Int64")

    if gen_type_match["current_source"].any():
        _add_to_entries_if_not_nan(net, "sgen", entries, index, "k", k)
    if gen_type_match["async"].any():
        _add_to_entries_if_not_nan(net, "sgen", entries, index, "lrc_pu", lrc_pu)
    if gen_type_match["async_doubly_fed"].any():
        _add_to_entries_if_not_nan(net, "sgen", entries, index, "max_ik_ka", max_ik_ka)
    if not gen_type_match.any(axis=1).all():
        raise UserWarning(f"unknown sgen generator_type '{generator_type}'! "
                          f"Must be one of: None, 'current_source', 'async', 'async_doubly_fed'")

    defaults_to_fill = [("controllable", False), ('reactive_capability_curve', False), ("curve_style", None)]
    _set_multiple_entries(net, "sgen", index, defaults_to_fill=defaults_to_fill, entries=entries)

    return index


# =============================================================================
# Create 3ph Sgen
# =============================================================================

def create_asymmetric_sgen(
    net: pandapowerNet,
    bus: Int,
    p_a_mw: float = 0,
    p_b_mw: float = 0,
    p_c_mw: float = 0,
    q_a_mvar: float = 0,
    q_b_mvar: float = 0,
    q_c_mvar: float = 0,
    sn_mva: float = nan,
    name: Optional[str] = None,
    index: Optional[Int] = None,
    scaling: float = 1.,
    type: WyeDeltaType = 'wye',
    in_service: bool = True,
    **kwargs
) -> Int:
    """

    Adds one static generator in table net["asymmetric_sgen"].

    Static generators are modelled as negative  PQ loads. This element is used to model generators
    with a constant active and reactive power feed-in. Positive active power means generation.

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

    OPTIONAL:

        **p_a_mw** (float, default 0) - The active power of the static generator : Phase A

        **p_b_mw** (float, default 0) - The active power of the static generator : Phase B

        **p_c_mw** (float, default 0) - The active power of the static generator : Phase C

        **q_a_mvar** (float, default 0) - The reactive power of the sgen : Phase A

        **q_b_mvar** (float, default 0) - The reactive power of the sgen : Phase B

        **q_c_mvar** (float, default 0) - The reactive power of the sgen : Phase C

        **sn_mva** (float, default None) - Nominal power of the sgen

        **name** (string, default None) - The name for this sgen

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (float, 1.) - An OPTIONAL scaling factor to be set customly.
        Multiplies with p_mw and q_mvar of all phases.

        **type** (string, 'wye') -  Three phase Connection type of the static generator: wye/delta

        **in_service** (boolean) - True for in_service or False for out of service

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    EXAMPLE:
        create_asymmetric_sgen(net, 1, p_b_mw=0.12)

    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "asymmetric_sgen", index,
                                  name="3 phase asymmetric static generator")

    entries = {"name": name, "bus": bus, "p_a_mw": p_a_mw, "p_b_mw": p_b_mw, "p_c_mw": p_c_mw, "scaling": scaling,
               "q_a_mvar": q_a_mvar, "q_b_mvar": q_b_mvar, "q_c_mvar": q_c_mvar, "sn_mva": sn_mva,
               "in_service": in_service, "type": type, **kwargs}
    _set_entries(net, "asymmetric_sgen", index, True, entries=entries)

    return index


def create_sgen_from_cosphi( # no index ?
        net: pandapowerNet,
        bus: Int,
        sn_mva: float,
        cos_phi: float,
        mode: UnderOverExcitedType,
        **kwargs,
) -> Int:
    """
    Creates an sgen element from rated power and power factor cos(phi).

    INPUT:
        **net** - The net within this static generator should be created

        **bus** (int) - The bus id to which the static generator is connected

        **sn_mva** (float) - rated power of the generator

        **cos_phi** (float) - power factor cos_phi

        **mode** (str) - "underexcited" (Q absorption, decreases voltage) or "overexcited" \
                         (Q injection, increases voltage)

    OUTPUT:
        **index** (int) - The unique ID of the created sgen

    gen, sgen, and ext_grid are modelled in the generator point of view. Active power
    will therefore be positive for generation, and reactive power will be negative for
    underexcited behavior (Q absorption, decreases voltage) and
    positive for overexcited behavior (Q injection, increases voltage).
    """
    from pandapower.toolbox import pq_from_cosphi
    p_mw, q_mvar = pq_from_cosphi(sn_mva, cos_phi, qmode=mode, pmode="gen")
    return create_sgen(net, bus, sn_mva=sn_mva, p_mw=p_mw, q_mvar=q_mvar, **kwargs)

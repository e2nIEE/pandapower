# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from numpy import nan, bool_
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import (
    _add_to_entries_if_not_nan,
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
    _set_value_if_not_nan,
)

logger = logging.getLogger(__name__)


def create_storage(
    net: pandapowerNet,
    bus: Int,
    p_mw: float,
    max_e_mwh: float,
    q_mvar: float = 0,
    sn_mva: float = nan,
    soc_percent: float = nan,
    min_e_mwh: float = 0.0,
    name: str | None = None,
    index: Int | None = None,
    scaling: float = 1.,
    type: str | None = None,
    in_service: bool = True,
    max_p_mw: float = nan,
    min_p_mw: float = nan,
    max_q_mvar: float = nan,
    min_q_mvar: float = nan,
    controllable: bool | float = nan,
    **kwargs
) -> Int:
    """
    Adds a storage to the network.

    In order to simulate a storage system it is possible to use sgens or loads to model the
    discharging or charging state. The power of a storage can be positive or negative, so the use
    of either a sgen or a load is (per definition of the elements) not correct.
    To overcome this issue, a storage element can be created.

    As pandapower is not a time dependent simulation tool and there is no time domain parameter in
    default power flow calculations, the state of charge (SOC) is not updated during any power flow
    calculation.
    The implementation of energy content related parameters in the storage element allows to create
    customized, time dependent simulations by running several power flow calculations and updating
    variables manually.

    INPUT:
        **net** - The net within this storage should be created

        **bus** (int) - The bus id to which the storage is connected

        **p_mw** (float) - The momentary active power of the storage \
            (positive for charging, negative for discharging)

        **max_e_mwh** (float) - The maximum energy content of the storage \
            (maximum charge level)

    OPTIONAL:
        **q_mvar** (float, default 0) - The reactive power of the storage

        **sn_mva** (float, default NaN) - Nominal power of the storage

        **soc_percent** (float, NaN) - The state of charge of the storage

        **min_e_mwh** (float, 0) - The minimum energy content of the storage \
            (minimum charge level)

        **name** (string, default None) - The name for this storage

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (float, 1.) - An OPTIONAL scaling factor to be set customly.
        Multiplies with p_mw and q_mvar.

        **type** (string, None) -  type variable to classify the storage

        **in_service** (boolean) - True for in_service or False for out of service

        **max_p_mw** (float, NaN) - Maximum active power injection - necessary for a \
            controllable storage in OPF

        **min_p_mw** (float, NaN) - Minimum active power injection - necessary for a \
            controllable storage in OPF

        **max_q_mvar** (float, NaN) - Maximum reactive power injection - necessary for a \
            controllable storage in OPF

        **min_q_mvar** (float, NaN) - Minimum reactive power injection - necessary for a \
            controllable storage in OPF

        **controllable** (bool, NaN) - Whether this storage is controllable by the optimal \
            powerflow; defaults to False if "controllable" column exists in DataFrame

    OUTPUT:
        **index** (int) - The unique ID of the created storage

    EXAMPLE:
        create_storage(net, 1, p_mw=-30, max_e_mwh=60, soc_percent=1.0, min_e_mwh=5)

    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "storage", index)

    entries = {"name": name, "bus": bus, "p_mw": p_mw, "q_mvar": q_mvar, "sn_mva": sn_mva, "scaling": scaling,
               "soc_percent": soc_percent, "min_e_mwh": min_e_mwh, "max_e_mwh": max_e_mwh,
               "in_service": in_service, "type": type, **kwargs}
    _set_entries(net, "storage", index, True, entries=entries)

    # check for OPF parameters and add columns to network table
    _set_value_if_not_nan(net, index, min_p_mw, "min_p_mw", "storage")
    _set_value_if_not_nan(net, index, max_p_mw, "max_p_mw", "storage")
    _set_value_if_not_nan(net, index, min_q_mvar, "min_q_mvar", "storage")
    _set_value_if_not_nan(net, index, max_q_mvar, "max_q_mvar", "storage")
    _set_value_if_not_nan(net, index, controllable, "controllable", "storage",
                          dtype=bool_, default_val=False)

    return index


def create_storages(
    net: pandapowerNet,
    buses: Sequence,
    p_mw: float | Iterable[float],
    max_e_mwh: float | Iterable[float],
    q_mvar: float | Iterable[float] = 0,
    sn_mva: float | Iterable[float] = nan,
    soc_percent: float | Iterable[float] = nan,
    min_e_mwh: float | Iterable[float] = 0.0,
    name: Iterable[str] | None = None,
    index: Int | Iterable[Int] | None  = None,
    scaling: float | Iterable[float] = 1.,
    type: str | Iterable[str] | None = None,
    in_service: bool | Iterable[bool] = True,
    max_p_mw: float | Iterable[float] = nan,
    min_p_mw: float | Iterable[float] = nan,
    max_q_mvar: float | Iterable[float] = nan,
    min_q_mvar: float | Iterable[float] = nan,
    controllable: bool | Iterable[bool] | float = nan,
    **kwargs
) -> npt.NDArray[Int]:
    """
    Adds storages to the network.

    In order to simulate a storage system it is possible to use sgens or loads to model the
    discharging or charging state. The power of a storage can be positive or negative, so the use
    of either a sgen or a load is (per definition of the elements) not correct.
    To overcome this issue, a storage element can be created.

    As pandapower is not a time dependent simulation tool and there is no time domain parameter in
    default power flow calculations, the state of charge (SOC) is not updated during any power flow
    calculation.
    The implementation of energy content related parameters in the storage element allows to create
    customized, time dependent simulations by running several power flow calculations and updating
    variables manually.

    INPUT:
        **net** - The net within this storage should be created

        **buses** (list of int) - The bus ids to which the generators are connected

        **p_mw** (list of float) - The momentary active power of the storage \
            (positive for charging, negative for discharging)

        **max_e_mwh** (list of float) - The maximum energy content of the storage \
            (maximum charge level)

    OPTIONAL:
        **q_mvar** (list of float, default 0) - The reactive power of the storage

        **sn_mva** (list of float, default NaN) - Nominal power of the storage

        **soc_percent** (list of float, NaN) - The state of charge of the storage

        **min_e_mwh** (list of float, 0) - The minimum energy content of the storage \
            (minimum charge level)

        **name** (list of string, default None) - The name for this storage

        **index** (list of int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

        **scaling** (list of float, 1.) - An OPTIONAL scaling factor to be set customly.
        Multiplies with p_mw and q_mvar.

        **type** (list of string, None) -  type variable to classify the storage

        **in_service** (list of boolean, default True) - True for in_service or False for out of service

        **max_p_mw** (list of float, NaN) - Maximum active power injection - necessary for a \
            controllable storage in OPF

        **min_p_mw** (list of float, NaN) - Minimum active power injection - necessary for a \
            controllable storage in OPF

        **max_q_mvar** (list of float, NaN) - Maximum reactive power injection - necessary for a \
            controllable storage in OPF

        **min_q_mvar** (list of float, NaN) - Minimum reactive power injection - necessary for a \
            controllable storage in OPF

        **controllable** (list of bool, NaN) - Whether this storage is controllable by the optimal \
            powerflow; defaults to False if "controllable" column exists in DataFrame

    OUTPUT:
        **index** (int) - The unique ID of the created storage

    EXAMPLE:
        create_storage(net, 1, p_mw=-30, max_e_mwh=60, soc_percent=1.0, min_e_mwh=5)

    """
    _check_multiple_elements(net, buses)

    index = _get_multiple_index_with_check(net, "storage", index, len(buses))

    entries = {"name": name, "bus": buses, "p_mw": p_mw, "q_mvar": q_mvar, "sn_mva": sn_mva,
               "scaling": scaling, "soc_percent": soc_percent, "min_e_mwh": min_e_mwh,
               "max_e_mwh": max_e_mwh, "in_service": in_service, "type": type, **kwargs}

    _add_to_entries_if_not_nan(net, "storage", entries, index, "min_p_mw", min_p_mw)
    _add_to_entries_if_not_nan(net, "storage", entries, index, "max_p_mw", max_p_mw)
    _add_to_entries_if_not_nan(net, "storage", entries, index, "min_q_mvar", min_q_mvar)
    _add_to_entries_if_not_nan(net, "storage", entries, index, "max_q_mvar", max_q_mvar)
    _add_to_entries_if_not_nan(net, "storage", entries, index, "controllable", controllable, dtype=bool_,
                               default_val=False)
    defaults_to_fill = [("controllable", False)]

    _set_multiple_entries(net, "storage", index, defaults_to_fill=defaults_to_fill, entries=entries)

    return index

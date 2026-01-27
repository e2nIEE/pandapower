# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import (
    _check_element,
    _check_multiple_elements,
    _get_index_with_check,
    _get_multiple_index_with_check,
    _set_entries,
    _set_multiple_entries,
)

logger = logging.getLogger(__name__)

from typing import Optional


def create_ward(
    net: pandapowerNet,
    bus: Int,
    ps_mw: float,
    qs_mvar: float,
    pz_mw: float,
    qz_mvar: float,
    name: str | None = None,
    in_service: bool = True,
    index: Int | None = None,
    **kwargs,
) -> Int:
    """
    Creates a ward equivalent.

    A ward equivalent is a combination of an impedance load and a PQ load.

    INPUT:
        **net** (pandapowernet) - The pandapower net within the element should be created

        **bus** (int) -  bus of the ward equivalent

        **ps_mw** (float) - active power of the PQ load

        **qs_mvar** (float) - reactive power of the PQ load

        **pz_mw** (float) - active power of the impedance load in MW at 1.pu voltage

        **qz_mvar** (float) - reactive power of the impedance load in MVar at 1.pu voltage

    OUTPUT:
        ward id
    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "ward", index, "ward equivalent")

    entries = {
        "bus": bus,
        "ps_mw": ps_mw,
        "qs_mvar": qs_mvar,
        "pz_mw": pz_mw,
        "qz_mvar": qz_mvar,
        "name": name,
        "in_service": in_service,
        **kwargs,
    }
    _set_entries(net, "ward", index, entries=entries)

    return index


def create_wards(
    net: pandapowerNet,
    buses: Sequence,
    ps_mw: float | Iterable[float],
    qs_mvar: float | Iterable[float],
    pz_mw: float | Iterable[float],
    qz_mvar: float | Iterable[float],
    name: Iterable[str] | None = None,
    in_service: bool | Iterable[bool] = True,
    index: int | None = None,
    **kwargs,
) -> npt.NDArray[np.array]:
    """
    Creates ward equivalents.

    A ward equivalent is a combination of an impedance load and a PQ load.

    INPUT:
        **net** (pandapowernet) - The pandapower net within the element should be created

        **buses** (list of int) -  bus of the ward equivalent

        **ps_mw** (list of float) - active power of the PQ load

        **qs_mvar** (list of float) - reactive power of the PQ load

        **pz_mw** (list of float) - active power of the impedance load in MW at 1.pu voltage

        **qz_mvar** (list of float) - reactive power of the impedance load in MVar at 1.pu voltage

    OUTPUT:
        ward id
    """
    _check_multiple_elements(net, buses)

    index = _get_multiple_index_with_check(net, "storage", index, len(buses))

    entries = {
        "name": name,
        "bus": buses,
        "ps_mw": ps_mw,
        "qs_mvar": qs_mvar,
        "pz_mw": pz_mw,
        "qz_mvar": qz_mvar,
        "in_service": in_service,
        **kwargs,
    }

    _set_multiple_entries(net, "ward", index, entries=entries)

    return index


def create_xward(
    net: pandapowerNet,
    bus: Int,
    ps_mw: float,
    qs_mvar: float,
    pz_mw: float,
    qz_mvar: float,
    r_ohm: float,
    x_ohm: float,
    vm_pu: float,
    in_service: bool = True,
    name: str | None = None,
    index: Int | None = None,
    slack_weight: float = 0.0,
    **kwargs,
):
    """
    Creates an extended ward equivalent.

    A ward equivalent is a combination of an impedance load, a PQ load and as voltage source with
    an internal impedance.

    INPUT:
        **net** - The pandapower net within the impedance should be created

        **bus** (int) -  bus of the ward equivalent

        **ps_mw** (float) - active power of the PQ load

        **qs_mvar** (float) - reactive power of the PQ load

        **pz_mw** (float) - active power of the impedance load in MW at 1.pu voltage

        **qz_mvar** (float) - reactive power of the impedance load in MVar at 1.pu voltage

        **r_ohm** (float) - internal resistance of the voltage source

        **x_ohm** (float) - internal reactance of the voltage source

        **vm_pu** (float) - voltage magnitude at the additional PV-node

        **slack_weight** (float, default 0.0) - Contribution factor for distributed slack power
            flow calculation (active power balancing)

    OUTPUT:
        xward id
    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "xward", index, "extended ward equivalent")

    entries = {
        "bus": bus,
        "ps_mw": ps_mw,
        "qs_mvar": qs_mvar,
        "pz_mw": pz_mw,
        "qz_mvar": qz_mvar,
        "r_ohm": r_ohm,
        "x_ohm": x_ohm,
        "vm_pu": vm_pu,
        "name": name,
        "slack_weight": slack_weight,
        "in_service": in_service,
        **kwargs,
    }
    _set_entries(net, "xward", index, entries=entries)

    return index

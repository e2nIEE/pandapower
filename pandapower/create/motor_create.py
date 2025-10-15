# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging
from numpy import nan

from pandapower.auxiliary import pandapowerNet
from pandapower.pp_types import Int
from pandapower.create._utils import _check_element, _get_index_with_check, _set_entries

logger = logging.getLogger(__name__)


def create_motor(
    net: pandapowerNet,
    bus: Int,
    pn_mech_mw: float,
    cos_phi: float,
    efficiency_percent: float = 100.,
    loading_percent: float = 100.,
    name: str | None = None,
    lrc_pu: float = nan,
    scaling: float = 1.0,
    vn_kv: float = nan,
    rx: float = nan,
    index: Int | None = None,
    in_service: bool = True,
    cos_phi_n: float = nan,
    efficiency_n_percent: float = nan,
    **kwargs
) -> Int:
    """
    Adds a motor to the network.


    INPUT:
        **net** - The net within this motor should be created

        **bus** (int) - The bus id to which the motor is connected

        **pn_mech_mw** (float) - Mechanical rated power of the motor

        **cos_phi** (float, nan) - cosine phi at current operating point

    OPTIONAL:

        **name** (string, None) - The name for this motor

        **efficiency_percent** (float, 100) - Efficiency in percent at current operating point

        **loading_percent** (float, 100) - The mechanical loading in percentage of the rated \
            mechanical power

        **scaling** (float, 1.0) - scaling factor which for the active power of the motor

        **cos_phi_n** (float, nan) - cosine phi at rated power of the motor for short-circuit \
            calculation

        **efficiency_n_percent** (float, 100) - Efficiency in percent at rated power for \
            short-circuit calculation

        **lrc_pu** (float, nan) - locked rotor current in relation to the rated motor current

        **rx** (float, nan) - R/X ratio of the motor for short-circuit calculation.

        **vn_kv** (float, NaN) - Rated voltage of the motor for short-circuit calculation

        **in_service** (bool, True) - True for in_service or False for out of service

        **index** (int, None) - Force a specified ID if it is available. If None, the index one \
            higher than the highest already existing index is selected.

    OUTPUT:
        **index** (int) - The unique ID of the created motor

    EXAMPLE:
        create_motor(net, 1, pn_mech_mw=0.120, cos_ph=0.9, vn_kv=0.6, efficiency_percent=90, \
                     loading_percent=40, lrc_pu=6.0)

    """
    _check_element(net, bus)

    index = _get_index_with_check(net, "motor", index)

    entries = {"name": name, "bus": bus, "pn_mech_mw": pn_mech_mw, "cos_phi": cos_phi, "cos_phi_n": cos_phi_n,
               "vn_kv": vn_kv, "rx": rx, "efficiency_n_percent": efficiency_n_percent,
               "efficiency_percent": efficiency_percent, "loading_percent": loading_percent, "lrc_pu": lrc_pu,
               "scaling": scaling, "in_service": in_service, **kwargs}
    _set_entries(net, "motor", index, entries=entries)

    return index

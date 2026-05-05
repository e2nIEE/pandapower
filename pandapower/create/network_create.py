# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from __future__ import annotations

import logging

from pandapower.auxiliary import pandapowerNet
from pandapower.network_structure import get_structure_dict
from pandapower.results import reset_results
from pandapower.std_types import add_basic_std_types

logger = logging.getLogger(__name__)


def create_empty_network(
    name: str = "", f_hz: float = 50.0, sn_mva: float = 1, add_stdtypes: bool = True, structure: dict | None = None
) -> pandapowerNet:
    """
    This function initializes the pandapower data structure.

    Parameters:
        f_hz: power system frequency in hertz
        name: name for the network
        sn_mva: reference apparent power for per unit system
        add_stdtypes: Includes standard types to net
        structure: can contain dict from which the network structure is created, when columns that are not relevant
         for the loadflow are required

    Returns:
        net: pandapower attrdict with empty tables

    Example:
        >>> net = create_empty_network()

    """
    if structure is None:
        network_structure_dict = get_structure_dict()
    else:
        network_structure_dict = structure
    network_structure_dict["name"] = name
    network_structure_dict["f_hz"] = f_hz
    network_structure_dict["sn_mva"] = sn_mva

    net = pandapowerNet(pandapowerNet.create_dataframes(network_structure_dict))

    if add_stdtypes:
        add_basic_std_types(net)
    else:
        net.std_types = {"line": {}, "line_dc": {}, "trafo": {}, "trafo3w": {}, "fuse": {}}
    for mode in ["pf", "se", "sc", "pf_3ph"]:
        reset_results(net, mode)
    net["user_pf_options"] = {}
    return net

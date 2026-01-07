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
    name: str = "", f_hz: float = 50.0, sn_mva: float = 1, add_stdtypes: bool = True
) -> pandapowerNet:
    """
    This function initializes the pandapower datastructure.

    OPTIONAL:
        **f_hz** (float, 50.) - power system frequency in hertz

        **name** (string, None) - name for the network

        **sn_mva** (float, 1) - reference apparent power for per unit system

        **add_stdtypes** (boolean, True) - Includes standard types to net

    OUTPUT:
        **net** (attrdict) - PANDAPOWER attrdict with empty tables:

    EXAMPLE:
        net = create_empty_network()

    """
    network_structure_dict = get_structure_dict()
    network_structure_dict["name"] = name
    network_structure_dict["f_hz"] = f_hz
    network_structure_dict["sn_mva"] = sn_mva

    net = pandapowerNet(pandapowerNet.create_dataframes(network_structure_dict))

    net._empty_res_load_3ph = net._empty_res_load
    net._empty_res_sgen_3ph = net._empty_res_sgen
    net._empty_res_storage_3ph = net._empty_res_storage

    if add_stdtypes:
        add_basic_std_types(net)
    else:
        net.std_types = {"line": {}, "line_dc": {}, "trafo": {}, "trafo3w": {}, "fuse": {}}
    for mode in ["pf", "se", "sc", "pf_3ph"]:
        reset_results(net, mode)
    net["user_pf_options"] = {}
    return net

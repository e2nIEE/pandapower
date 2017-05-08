# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

import pandapower as pp
import pandapower.networks


def _get_networks_path():
    return os.path.abspath(os.path.dirname(pandapower.networks.__file__))

def mv_oberrhein(scenario="load", cosphi_load=0.98, cosphi_pv=1.0, include_substations=False):
    """
    Loads the Oberrhein network, a generic 20 kV network serviced by two 25 MVA HV/MV transformer
    stations. The network supplies 141 HV/MV substations and 6 MV loads through four MV feeders.
    The network layout is meshed, but the network is operated as a radial network with 6 open
    sectioning points.

    The network can be loaded with two different worst case scenarios for load and generation,
    which are defined by scaling factors for loads / generators as well as tap positions of the
    HV/MV transformers. These worst case scenarios are a good starting point for working with this
    network, but you are of course free to parametrize the network for your use case.

    The network also includes geographical information of lines and buses for plotting.

    OPTIONAL:
        **scenario** - (str, "load"): defines the scaling for load and generation

                - "load": high load scenario, load = 0.6 / sgen = 0, trafo taps [-2, -3]
                - "generation": high feed-in scenario: load = 0.1, generation = 0.8, trafo taps [0, 0]

        **cosphi_load** - (str, 0.98): cosine(phi) of the loads

        **cosphi_sgen** - (str, 1.0): cosine(phi) of the static generators

        **include_substations** - (bool, False): if True, the transformers of the MV/LV level are
        modelled, otherwise the loads representing the LV networks are connected directly to the
        MV node

    OUTPUT:
         **net** - pandapower network

    EXAMPLE:

    >>> import pandapower.networks
    >>> net = pandapower.networks.mv_oberrhein("generation")
    """
    if include_substations:
        net = pp.from_pickle(os.path.join(_get_networks_path(), "mv_oberrhein_substations.p"))
    else:
        net = pp.from_pickle(os.path.join(_get_networks_path(), "mv_oberrhein.p"))
    net.load.q_kvar = np.tan(np.arccos(cosphi_load)) * net.load.p_kw
    net.sgen.q_kvar = np.tan(np.arccos(cosphi_pv)) * net.sgen.p_kw

    hv_trafos = net.trafo[net.trafo.sn_kva > 1e3].index
    if scenario == "load":
        net.load.scaling = 0.6
        net.sgen.scaling = 0.0
        net.trafo.tp_pos.loc[hv_trafos] = [-2, -3]
    elif scenario == "generation":
        net.load.scaling = 0.1
        net.sgen.scaling = 0.8
        net.trafo.tp_pos.loc[hv_trafos] = [0, 0]
    else:
        raise ValueError("Unknown scenario %s - chose 'load' or 'generation'"%scenario)
    pp.runpp(net)
    return net

# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pandapower.networks
import os
import numpy as np

def _get_networks_path():
    return os.path.abspath(os.path.dirname(pandapower.networks.__file__))

def mv_oberrhein(scenario="load", cosphi_load=0.98, cosphi_pv=1.0, include_substations=False):
    """
    Loads the 20 kV Oberrhein network.
    
    OPTIONAL:
        
        **scenario** - (str, "load"): defines the scaling for load and generation
                
                - "load": high load scenario, load = 0.6 / sgen = 0, trafo taps [-2, -2]
                - "generation": high feed-in scenario: load = 0.1, generation = 0.8, trafo taps [-1, 0]
                
    RETURN:

         **net** - pandapower network

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.mv_oberrhein("generation")
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
        net.sgen.scaling = 0
        net.trafo.tp_pos.loc[hv_trafos] = [-2, -2]
    elif scenario == "generation":
        net.load.scaling = 0.1
        net.sgen.scaling = 0.8
        net.trafo.tp_pos.loc[hv_trafos] = [-1, 0]
    else:
        raise ValueError("Unknown scenario %s - chose 'load' or 'generation'"%scenario)
    pp.runpp(net)
    return net

if __name__ == "__main__":
    net = mv_oberrhein("load", include_substations=True)

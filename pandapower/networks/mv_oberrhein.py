# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np

import pandapower as pp
import pandapower.topology as top
from pandapower.__init__ import pp_dir


def mv_oberrhein(scenario="load", cosphi_load=0.98, cosphi_pv=1.0, include_substations=False,
                 separation_by_sub=False, **kwargs):
    """
    Loads the Oberrhein network, a generic 20 kV network serviced by two 25 MVA HV/MV transformer
    stations. The network supplies 141 MV/LV substations and 6 MV loads through four MV feeders.
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

        **separation_by_sub** - (bool, False): if True, the network gets separated into two
        sections, referring to both substations

    OUTPUT:
         **net** - pandapower network

         **net0, net1** - both sections of the pandapower network

    EXAMPLE:

        ``import pandapower.networks``

        ``net = pandapower.networks.mv_oberrhein("generation")``

        or with separation

        ``net0, net1 = pandapower.networks.mv_oberrhein(separation_by_sub=True)``
    """
    if include_substations:
        net = pp.from_json(os.path.join(pp_dir, "networks", "mv_oberrhein_substations.json"), **kwargs)
        # geo.convert_epsg_bus_geodata(net, epsg_out=4326, epsg_in=31467)
        # geo.convert_geodata_to_geojson(net, lonlat=False)
    else:
        net = pp.from_json(os.path.join(pp_dir, "networks", "mv_oberrhein.json"), **kwargs)
        # geo.convert_epsg_bus_geodata(net, epsg_out=4326, epsg_in=31467)
        # geo.convert_geodata_to_geojson(net, lonlat=False)
    net.load.q_mvar = np.tan(np.arccos(cosphi_load)) * net.load.p_mw
    net.sgen.q_mvar = np.tan(np.arccos(cosphi_pv)) * net.sgen.p_mw

    hv_trafos = net.trafo[net.trafo.sn_mva > 1].index
    net.trafo["tap_pos"] = net.trafo["tap_pos"].astype("float64")
    if scenario == "load":
        net.load.scaling = 0.6
        net.sgen.scaling = 0.0
        net.trafo.loc[hv_trafos, "tap_pos"] = [-2, -3]
    elif scenario == "generation":
        net.load.scaling = 0.1
        net.sgen.scaling = 0.8
        net.trafo.loc[hv_trafos, "tap_pos"] = [0, 0]
    else:
        raise ValueError("Unknown scenario %s - chose 'load' or 'generation'" % scenario)

    if separation_by_sub:
        # creating multigraph
        mg = top.create_nxgraph(net)
        # clustering connected buses
        zones = [list(area) for area in top.connected_components(mg)]
        net1 = pp.select_subnet(net, buses=zones[0], include_switch_buses=False,
                                include_results=True, keep_everything_else=True)
        net0 = pp.select_subnet(net, buses=zones[1], include_switch_buses=False,
                                include_results=True, keep_everything_else=True)

        pp.runpp(net0)
        pp.runpp(net1)
        net0.name = 'MV Oberrhein 0'
        net1.name = 'MV Oberrhein 1'
        # TODO: this should be added to the initial data not converted here.
        # geo.convert_geodata_to_geojson(net0)
        # geo.convert_geodata_to_geojson(net1)
        return net0, net1

    pp.runpp(net)
    net.name = 'MV Oberrhein'
    # geo.convert_geodata_to_geojson(net)
    return net

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pandapower as pp
import pandapower.topology as top
import pandapower.plotting.geo as geo
from pandapower.__init__ import pp_dir


def lv_schutterwald(separation_by_sub=False, include_heat_pumps=False, **kwargs):
    """
    Loads the Schutterwald network, a generic 0.4 kV network serviced by 14 MV/LV transformer
    stations of the Oberrhein network.
    The network supplies 1506 customers with the option of including 1251 heat pumps.

    The network also includes geographical information of lines and buses for plotting.

    Source: https://doi.org/10.3390/en13164052

    OPTIONAL:
        **separation_by_sub** - (bool, False): if True, the network gets separated into 14
        sections, referring to their substations

        **include_heat_pumps** - (bool, False): if True, the heat pumps from the study are
        included in the network

    OUTPUT:
         **net** - pandapower network

         **subnets** (list) - all sections of the pandapower network

    EXAMPLE:

        ``import pandapower.networks``

        ``net = pandapower.networks.lv_schutterwald()``

        or with separation

        ``net_list = pandapower.networks.lv_schutterwald(separation_by_sub=True)``
    """

    net = pp.from_json(os.path.join(pp_dir, "networks", "lv_schutterwald.json"), **kwargs)

    # modifications on the original file
    # geo.convert_crs(net, epsg_out=4326, epsg_in=31467)
    # geo.convert_geodata_to_geojson(net, lonlat=False)
    # pp.drop_inactive_elements(net)
    # net.load.replace({"type": "WP"}, "HP", inplace=True)
    # net.load.replace({"type": "HA"}, "HH", inplace=True)
    # net.load.name = net.load.name.str.replace("H", "HH", regex=False)
    # net.load.name = net.load.name.str.replace("WP", "HP", regex=False)

    if not include_heat_pumps:
        pp.drop_elements(net, "load", net.load.loc[net.load.type == "HP"].index)

    subnets = list()
    if separation_by_sub:
        # creating multigraph
        mg = top.create_nxgraph(net)
        # clustering connected buses
        zones = [list(area) for area in top.connected_components(mg)]
        for i, zone in enumerate(zones):
            net1 = pp.select_subnet(net, buses=zone, include_switch_buses=False,
                                    include_results=True, keep_everything_else=True)
            pp.runpp(net1)
            net1.name = f'LV Schutterwald {i}'
            subnets.append(net1)
        return subnets

    pp.runpp(net)
    net.name = 'LV Schutterwald'
    return net

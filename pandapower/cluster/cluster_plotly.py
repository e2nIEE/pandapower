# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Copyright (c) 2024 by Fabio S. Retorta, INESCTEC, Centre of Power and Energy Systems. All rights reserved.

import pandas as pd
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace
from pandapower.plotting.plotly.get_colors import get_plotly_color_palette
from pandapower.topology import create_nxgraph, connected_components
from draw_cluster import draw_cluster
import colorsys

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def cluster_plotly(net, clust_bus, respect_switches=True, use_line_geodata=None, colors_dict=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=2,
                  bus_size=10, auto_open=True):
    """
    Plot pandapower network clusters in plotly
    using lines/buses colors according to the voltage level they belong to.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the
    tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **clust_bus** (DataFrrame, None) - Dataframe with two columns, Bus and Cluster. 
            In the first one, the data are the index values of each bus (int).
            In the second one, the data are the number of the cluster of each bus (int).

        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        **use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata
            of the lines (True) or on net.bus_geodata of the connected buses (False)

        *colors_dict** (dict, None) - dictionary for customization of colors for each voltage level
            in the form: voltage : color

        **on_map** (bool, False) - enables using mapbox plot in plotly If provided geodata are not
            real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be
            transformed to lat-long. For each projection a string can be found at
            http://spatialreference.org/ref/epsg/

        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
            network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

        **auto_open** (bool, True) - automatically open plot in browser

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """
    # getting connected componenets without consideration of trafos
    graph = create_nxgraph(net, include_trafos=False)
    vlev_buses = connected_components(graph)
    # getting unique sets of buses for each voltage level
    vlev_bus_dict = {}
    for vl_buses in vlev_buses:
        if len(net.bus.loc[list(vl_buses), 'vn_kv'].unique().tolist()) > 1:
            logger.warning('buses from the same voltage level does not have the same vn_kv !?')
        vn_kv = net.bus.loc[list(vl_buses), 'vn_kv'].unique()[0]
        if vlev_bus_dict.get(vn_kv):
            vlev_bus_dict[vn_kv].update(vl_buses)
        else:
            vlev_bus_dict[vn_kv] = vl_buses
    # handling cluster values and buses
    Dict_clust = {}
    unique_clusters = pd.Series(clust_bus["Cluster"].unique())
    unique_clusters = unique_clusters.sort_values()
    unique_clusters = unique_clusters.reset_index(drop=True)
    ind_bus = clust_bus.index
    ind_bus = pd.Series(ind_bus)
    for i in range(len(unique_clusters)):
        lista = list()
        for j in range(len(clust_bus)):
            if unique_clusters[i] == clust_bus.iloc[j,0]:
                lista.append(ind_bus[j])
            if j == (len(clust_bus)-1):
                Dict_clust.update({unique_clusters[i]:set(lista)})

    numb_cluster = max(clust_bus['Cluster'])
    # getting number of clusters for each bus
    nvlevs = len(Dict_clust)
    colors = get_plotly_color_palette(nvlevs)
    colors_dict = colors_dict or dict(zip(Dict_clust.keys(), colors))
    # creating traces for buses and lines for each voltage level
    bus_traces = []
    for vn_kv, buses_vl in Dict_clust.items():
        vlev_color = colors_dict[vn_kv]
        bus_trace_vlev = create_bus_trace(
            net, buses=buses_vl, size=bus_size, legendgroup=str(vn_kv), color=vlev_color,
            trace_name='Cluster {0}'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev
    line_traces = []
    for vn_kv, buses_vl in vlev_bus_dict.items():
        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) &
                        net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(
        net, lines=vlev_lines, use_line_geodata=use_line_geodata,
        respect_switches=respect_switches, legendgroup=str(vn_kv), color='gray',
        width=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
                line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, color='gray', width=line_width * 2)

    return draw_cluster(line_traces + trafo_traces + bus_traces, numb_cluster, showlegend=True,
                    aspectratio=aspectratio, on_map=on_map, map_style=map_style, figsize=figsize,
                    auto_open=auto_open)


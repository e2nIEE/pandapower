# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd

from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces, create_edge_center_trace
from pandapower.plotting.plotly.mapbox_plot import *

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def get_hoverinfo(net, element):
    idx = net[element].index
    if element == "bus":
        hoverinfo = ("index = " + net.bus.index.astype(str)  + '<br>' +
                     "name = " + net.bus['name'].astype(str) + '<br>' +
                     'Vn = ' + net.bus.loc[idx, 'vn_kv'].astype(str) + ' kV' + '<br>'
                     ).tolist()
    elif element == "line":
        hoverinfo = ("index = " + net.line.index.astype(str)  + '<br>'
                     +
                     "name = " + net.line['name'].astype(str) + '<br>'
                     +
                     'length = ' + net.line.loc[idx, 'length_km'].astype(str) + ' km' + '<br>'
                     +
                     'R = ' + (net.line.loc[idx, 'length_km'] * net.line.loc[idx, 'r_ohm_per_km']).astype(
                    str) + ' Ohm' + '<br>'
                     +
                     'X = ' + (net.line.loc[idx, 'length_km'] * net.line.loc[idx, 'x_ohm_per_km']).astype(
                    str) + ' Ohm' + '<br>'
                     ).tolist()
    elif element == "trafo":
        hoverinfo = ("index = " + net.trafo.index.astype(str)  + '<br>'
                     +
                     "name = " + net.trafo['name'].astype(str) + '<br>'
                     +
                     'Vn hv = ' + net.trafo.loc[idx, 'vn_hv_kv'].astype(str) + ' kV' + '<br>'
                     +
                     'Vn lv = ' + net.trafo.loc[idx, 'vn_lv_kv'].astype(str) + ' kV' + '<br>'
                     +
                     'Tap = ' + net.trafo.loc[idx, 'tp_pos'].astype(str) + '<br>'
                     ).tolist()
    elif element == "ext_grid":
        hoverinfo = ("index = " + net.ext_grid.index.astype(str)  + '<br>'
                     +
                     "name = " + net.ext_grid['name'] + '<br>'
                     +
                     'Vm = ' + net.ext_grid.loc[idx, 'vm_pu'].astype(str) + ' p.u.' + '<br>'
                     +
                     'Va = ' + net.ext_grid.loc[idx, 'va_degree'].astype(str) + ' °' + '<br>'
                     ).tolist()
    else:
        hoverinfo = None

    return hoverinfo


def simple_plotly(net, respect_switches=True, use_line_geodata=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=1,
                  bus_size=10, ext_grid_size=20.0, bus_color="blue", line_color='grey',
                  trafo_color='green', ext_grid_color="yellow"):
    """
    Plots a pandapower network as simple as possible in plotly.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
        or on net.bus_geodata of the connected buses (False)

        **on_map** (bool, False) - enables using mapbox plot in plotly.
        If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to
        lat-long. For each projection a string can be found at http://spatialreference.org/ref/epsg/


        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the network geodata
        any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

        **ext_grid_size** (float, 20.0) - size of ext_grids to plot.

            See bus sizes for details. Note: ext_grids are plotted as rectangles

        **bus_color** (String, "blue") - Bus Color. Init as first value of color palette.

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'green') - Trafo Color. Init is green

        **ext_grid_color** (String, 'yellow') - External Grid Color. Init is yellow
    """
    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x","y"])
    if len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time...")
        create_generic_coordinates(net, respect_switches=respect_switches)
        if on_map == True:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    hoverinfo = get_hoverinfo(net, element="bus")
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, color=bus_color, infofunc=hoverinfo)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    line_trace = create_line_trace(net, net.line.index, respect_switches=respect_switches,
                                   color=line_color, width=line_width,
                                   use_line_geodata=use_line_geodata)
    line_center_trace = []
    if use_line_geodata == False:
        hoverinfo = get_hoverinfo(net, element="line")
        line_center_trace = create_edge_center_trace(line_trace, color=line_color, infofunc=hoverinfo)

    # ----- Trafos ------
    trafo_trace = create_trafo_trace(net, color=trafo_color, width=line_width * 5)
    trafo_center_trace = []
    if use_line_geodata == False:
        hoverinfo = get_hoverinfo(net, element="trafo")
        trafo_center_trace = create_edge_center_trace(trafo_trace, color=trafo_color, infofunc=hoverinfo)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'  # workaround because doesn't appear on mapbox if square
    # hoverinfo = get_hoverinfo(net, element="ext_grid")
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color=ext_grid_color, size=ext_grid_size,
                                      patch_type=marker_type, trace_name='external_grid', infofunc=hoverinfo)

    draw_traces(line_trace + trafo_trace + ext_grid_trace + bus_trace + line_center_trace + trafo_center_trace,
                aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style)



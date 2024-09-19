# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd

from pandapower.run import runpp
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.mapbox_plot import *
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def pf_res_plotly(net, cmap="Jet", use_line_geo=None, on_map=False, projection=None,
                  map_style='basic', figsize=1, aspectratio='auto', line_width=2, bus_size=10,
                  climits_volt=(0.9, 1.1), climits_load=(0, 100), cpos_volt=1.0, cpos_load=1.1,
                  filename="temp-plot.html", auto_open=True, power_unit="k", current_unit="", voltage_unit=""):
    """
        Plots a pandapower network in plotly

        using colormap for coloring lines according to line loading and buses according to voltage in p.u.
        If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

        INPUT:
            **net** - The pandapower format network.

        OPTIONAL:
            **respect_switches** (bool, False) - Respect switches when artificial geodata is created

            **cmap** (str, True) - name of the colormap

            **colors_dict** (dict, None) - by default 6 basic colors from default collor palette is used.
            Otherwise, user can define a dictionary in the form: voltage_kv : color

            **on_map** (bool, False) - enables using mapbox plot in plotly. If provided geodata are not
            real geo-coordinates in lon/lat form, on_map will be set to False.

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

            **climits_volt** (tuple, (0.9, 1.0)) - limits of the colorbar for voltage

            **climits_load** (tuple, (0, 100)) - limits of the colorbar for line_loading

            **cpos_volt** (float, 1.0) - position of the bus voltage colorbar

            **cpos_load** (float, 1.1) - position of the loading percent colorbar

            **filename** (str, "temp-plot.html") - filename / path to plot to. Should end on `*.html`

            **auto_open** (bool, True) - automatically open plot in browser

            **power_unit** (str, 'k') - default unit of displayed P, Q, S data ["", "k", "M"]

            **current_unit** (str, '') - default unit of displayed I data ["", "k"]

            **voltage_unit** (str, '') - default unit of displayed V data ["", "k"]

        OUTPUT:
            **figure** (graph_objs._figure.Figure) figure object

    """
    if 'res_bus' not in net or net.get('res_bus').shape[0] == 0:
        logger.warning(
            'There are no Power Flow results. A Newton-Raphson power flow will be executed.')
        runpp(net)

    # create geocoord if none are available
    if any(net.line.geo.isna()) and any(net.bus.geo.isna()):
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=True)
        if on_map:
            logger.warning(
                "Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # for geo_type in ["bus_geodata", "line_geodata"]:
    #     dupl_geo_idx = pd.Series(net[geo_type].index)[pd.Series(
    #             net[geo_type].index).duplicated()]
    #     if len(dupl_geo_idx):
    #         if len(dupl_geo_idx) > 20:
    #             logger.warning("In net.%s are %i duplicated " % (geo_type, len(dupl_geo_idx)) +
    #                            "indices. That can cause troubles for draw_traces()")
    #         else:
    #             logger.warning("In net.%s are the following duplicated " % geo_type +
    #                            "indices. That can cause troubles for draw_traces(): " + str(
    #                            dupl_geo_idx))

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    # hoverinfo which contains name and pf results
    precision = 3

    if voltage_unit == "":
        voltage_factor = 1e3
    else:
        voltage_factor = 1
    if power_unit == "":
        power_factor = 1e6
    elif power_unit == "k":
        power_factor = 1e3
    else:
        power_factor = 1

    hoverinfo = (
        net.bus.name.astype(str) + '<br />' +
        'V_m = ' + net.res_bus.vm_pu.round(precision).astype(str) + ' pu' + '<br />' +
        'V_m = ' + (net.res_bus.vm_pu * net.bus.vn_kv.round(2) * voltage_factor).round(precision).astype(str) + ' ' + voltage_unit + 'V' + '<br />' +
        'V_a = ' + net.res_bus.va_degree.round(precision).astype(str) + ' deg' + '<br />' +
        'P = ' + (net.res_bus.p_mw * power_factor).round(precision).astype(str) + ' ' + power_unit + 'W' + '<br />' +
        'Q = ' + (net.res_bus.q_mvar * power_factor).round(precision).astype(str) + ' ' + power_unit + 'Var').tolist()
    hoverinfo = pd.Series(index=net.bus.index, data=hoverinfo)
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, infofunc=hoverinfo, cmap=cmap,
                                 cbar_title='Bus Voltage [pu]', cmin=climits_volt[0], cmax=climits_volt[1],
                                 cpos=cpos_volt)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    # if bus geodata is available, but no line geodata
    cmap_lines = 'jet' if cmap == 'Jet' else cmap
    if use_line_geo is None:
        use_line_geo = False if any(net.line.geo.isna()) else True
    elif use_line_geo and any(net.line.geo.isna()):
        logger.warning(
            "No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geo = False
    # hoverinfo which contains name and pf results
    if current_unit == "":
        current_factor = 1e3
    else:
        current_factor = 1
    hoverinfo = (
        net.line.name.astype(str) + '<br />' +
        'I = ' + net.res_line.loading_percent.round(precision).astype(str) + ' %' + '<br />' +
        'I_from = ' + (net.res_line.i_from_ka * current_factor).round(precision).astype(str) + ' ' + current_unit + 'A' + '<br />' +
        'I_to = ' + (net.res_line.i_to_ka * current_factor).round(precision).astype(str) + ' ' + current_unit + 'A' + '<br />' +
        'P_from = ' + (net.res_line.p_from_mw * power_factor).round(precision).astype(str) + ' ' + power_unit + 'W' + '<br />' +
        'P_to = ' + (net.res_line.p_to_mw * power_factor).round(precision).astype(str) + ' ' + power_unit + 'W' + '<br />' +
        'Q_from = ' + (net.res_line.q_from_mvar * power_factor).round(precision).astype(str) + ' ' + power_unit + 'Var' + '<br />' +
        'Q_to = ' + (net.res_line.q_to_mvar * power_factor).round(precision).astype(str) + ' ' + power_unit + 'Var').tolist()

    hoverinfo = pd.Series(index=net.line.index, data=hoverinfo)
    line_traces = create_line_trace(net, use_line_geo=use_line_geo, respect_switches=True,
                                    width=line_width,
                                    infofunc=hoverinfo,
                                    cmap=cmap_lines,
                                    cmap_vals=net.res_line['loading_percent'].values,
                                    cmin=climits_load[0],
                                    cmax=climits_load[1],
                                    cbar_title='Line Loading [%]',
                                    cpos=cpos_load)

    # ----- Trafos ------
    # hoverinfo which contains name and pf results
    hoverinfo = (
        net.trafo.name.astype(str) + '<br />' +
        'I = ' + net.res_trafo.loading_percent.round(precision).astype(str) + ' %' + '<br />' +
        'I_hv = ' + (net.res_trafo.i_hv_ka * current_factor).round(precision).astype(str) + ' ' + current_unit + 'A' + '<br />' +
        'I_lv = ' + (net.res_trafo.i_lv_ka * current_factor).round(precision).astype(str) + ' ' + current_unit + 'A' + '<br />' +
        'P_hv = ' + (net.res_trafo.p_hv_mw * power_factor).round(precision).astype(str) + ' ' + power_unit + 'W' + '<br />' +
        'P_lv = ' + (net.res_trafo.p_lv_mw * power_factor).round(precision).astype(str) + ' ' + power_unit + 'W' + '<br />' +
        'Q_hv = ' + (net.res_trafo.q_hv_mvar * power_factor).round(precision).astype(str) + ' ' + power_unit + 'Var' + '<br />' +
        'Q_lv = ' + (net.res_trafo.q_lv_mvar * power_factor).round(precision).astype(str) + ' ' + power_unit + 'Var' + '<br />').tolist()
    hoverinfo = pd.Series(index=net.trafo.index, data=hoverinfo)
    trafo_traces = create_trafo_trace(net, width=line_width * 1.5, infofunc=hoverinfo,
                                      cmap=cmap_lines, cmin=0, cmax=100)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color='grey', size=bus_size * 2, trace_name='external_grid',
                                      patch_type=marker_type)

    return draw_traces(line_traces + trafo_traces + ext_grid_trace + bus_trace,
                       showlegend=False, aspectratio=aspectratio, on_map=on_map,
                       map_style=map_style, figsize=figsize, filename=filename, auto_open=auto_open)

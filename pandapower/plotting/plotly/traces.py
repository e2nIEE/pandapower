# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from itertools import compress

import numpy as np
import pandas as pd

from pandapower.plotting.plotly.get_colors import get_plotly_color, get_plotly_cmap
from pandapower.plotting.plotly.mapbox_plot import _on_map_test, _get_mapbox_token, MapboxTokenMissing

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


try:
    from plotly.graph_objs import Figure, Data, Layout, Marker, XAxis, YAxis, Line, ColorBar
except ImportError:
    logger.debug("Failed to import plotly - interactive plotting will not be available")



def _in_ipynb():
    """
    an auxiliary function which checks if plot is called from a jupyter-notebook or not
    """
    import __main__ as main
    return not hasattr(main, '__file__')



def create_bus_trace(net, buses=None, size=5, patch_type="circle", color="blue", infofunc=None,
                     trace_name='buses', legendgroup=None, cmap=None, cmap_vals=None,
                     cbar_title=None, cmin=None, cmax=None):
    """
    Creates a plotly trace of pandapower buses.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created.
        If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "square" for a rectangle
                - "diamond" for a diamond
                - much more pathc types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (list, None) - hoverinfo for each trace element

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of buses in the trace

        **cmap** (String, None) - name of a colormap which exists within plotly (Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis)
        alternatively a custom discrete colormap can be used

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

    """
    color = get_plotly_color(color)

    bus_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='text', name=trace_name,
                     marker=dict(color=color, size=size, symbol=patch_type))

    buses = net.bus.index.tolist() if buses is None else list(buses)

    buses2plot = net.bus.index.isin(buses)

    buses_with_geodata = net.bus.index.isin(net.bus_geodata.index)
    buses2plot = buses2plot & buses_with_geodata

    bus_trace['x'], bus_trace['y'] = (net.bus_geodata.loc[buses2plot, 'x'].tolist(),
                                    net.bus_geodata.loc[buses2plot, 'y'].tolist())

    bus_trace['text'] = net.bus.loc[buses2plot, 'name'] if infofunc is None else infofunc

    if legendgroup:
        bus_trace['legendgroup'] = legendgroup

    # if color map is set
    if cmap is not None:
        # TODO introduce discrete colormaps (see contour plots in plotly)
        # if cmap_vals are not given

        cmap = 'Jet' if cmap is True else cmap

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net.res_line.shape[0] == 0:
                logger.error("There are no power flow results for buses voltage magnitudes which are default for bus "
                             "colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net.res_bus.loc[buses2plot, 'vm_pu'].values

        cmap_vals = net.res_bus.loc[buses2plot, 'vm_pu'] if cmap_vals is None else cmap_vals

        cmin = cmin if cmin else cmap_vals.min()
        cmax = cmax if cmax else cmap_vals.max()

        bus_trace['marker'] = Marker(size=size,
                                     color=cmap_vals, cmin=cmin, cmax=cmax,
                                     colorscale=cmap,
                                     colorbar=ColorBar(thickness=10,
                                                       x=1.0,
                                                       titleside='right'),
                                     )

        if cbar_title:
            bus_trace['marker']['colorbar']['title'] = cbar_title

    return [bus_trace]


def _get_line_geodata_plotly(net, lines, use_line_geodata):
    xs = []
    ys = []
    if use_line_geodata:
        for line_ind, line in lines.iterrows():
            line_coords = net.line_geodata.loc[line_ind, 'coords']
            linex, liney = list(zip(*line_coords))
            xs += linex
            xs += [None]
            ys += liney
            ys += [None]
    else:
        # getting x and y values from bus_geodata for from and to side of each line

        from_bus = net.bus_geodata.loc[lines.from_bus, 'x'].tolist()
        to_bus = net.bus_geodata.loc[lines.to_bus, 'x'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        None_list = [None] * len(from_bus)
        xs = np.array([from_bus, center, to_bus, None_list]).T.flatten().tolist()

        from_bus = net.bus_geodata.loc[lines.from_bus, 'y'].tolist()
        to_bus = net.bus_geodata.loc[lines.to_bus, 'y'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        None_list = [None] * len(from_bus)
        ys = np.array([from_bus, center, to_bus, None_list]).T.flatten().tolist()

    # [:-1] is because the trace will not appear on maps if None is at the end
    return xs[:-1], ys[:-1]


def create_line_trace(net, lines=None, use_line_geodata=True, respect_switches=False, width=1.0,
                      color='grey', infofunc=None, trace_name='lines', legendgroup=None,
                      cmap=None, cbar_title=None, show_colorbar = True, cmap_vals=None, cmin=None,
                      cmax=None):
    """
    Creates a plotly trace of pandapower lines.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created.
        If None, all lines in the network are considered.

        **width** (int, 1) - line width

        **respect_switches** (bool, False) - flag for consideration of disconnected lines

        **infofunc** (list, None) - hoverinfo for each line

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "grey") - color of lines in the trace

        **legendgroup** (String, None) - defines groups of layers that will be displayed in a legend
        e.g. groups according to voltage level (as used in `vlevel_plotly`)

        **cmap** (String, None) - name of a colormap which exists within plotly if set to True default `Jet`
        colormap is used, alternative colormaps : Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis

        **cmap_vals** (list, None) - values used for coloring using colormap

        **show_colorbar** (bool, False) - flag for showing or not corresponding colorbar

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        """

    color = get_plotly_color(color)

    # defining lines to be plot
    lines = net.line.index.tolist() if lines is None else list(lines)
    if len(lines) == 0:
        return []

    nogolines = set()
    if respect_switches:
        nogolines = set(net.switch.element[(net.switch.et == "l") &
                                           (net.switch.closed == 0)])
    nogolines_mask = net.line.index.isin(nogolines)

    lines_mask = net.line.index.isin(lines)
    lines2plot_mask = ~nogolines_mask & lines_mask
    lines2plot = net.line[lines2plot_mask]

    use_line_geodata = use_line_geodata if net.line_geodata.shape[0] > 0 else False
    if use_line_geodata:
        lines_with_geodata = lines2plot.index.isin(net.line_geodata.index)
        lines2plot = lines2plot[lines_with_geodata]
    else:
        lines_with_geodata = lines2plot.from_bus.isin(net.bus_geodata.index) & \
                             lines2plot.to_bus.isin(net.bus_geodata.index)
        lines2plot = lines2plot[lines_with_geodata]


    if cmap is not None:
        # workaround: if colormap plot is used, each line need to be separate scatter object because
        # plotly still doesn't support appropriately colormap for line objects
        # TODO correct this when plotly solves existing github issue about Line colorbar

        cmap = 'jet' if cmap is True else cmap

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net.res_line.shape[0] == 0:
                logger.error("There are no power flow results for lines which are default for line colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net.res_line.loc[lines2plot.index, 'loading_percent'].values

        cmap_lines = get_plotly_cmap(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)
        cmap_lines = list(compress(cmap_lines, lines2plot_mask)) # select with mask from cmap_lines
        if infofunc is not None:
            infofunc = list(compress(infofunc, lines2plot_mask))

        line_traces = []
        col_i = 0
        for idx, line in lines2plot.iterrows():
            line_trace = dict(type='scatter', text=[], hoverinfo='text', mode='lines', name=trace_name,
                              line=Line(width=width, color=color))

            line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(net, lines2plot.loc[idx:idx], use_line_geodata)

            line_trace['line']['color'] = cmap_lines[col_i]

            line_trace['text'] = line['name'] if infofunc is None else infofunc[col_i]

            line_traces.append(line_trace)
            col_i += 1

        cmin = cmin if cmin else cmap_vals.min()
        cmax = cmax if cmax else cmap_vals.max()

        if show_colorbar:
            try:
                # TODO for custom colormaps
                cbar_cmap_name = 'Jet' if cmap is 'jet' else cmap
                # workaround to get colorbar for lines (an unvisible node is added)
                lines_cbar = dict(type='scatter',x=[net.bus_geodata.x[0]], y=[net.bus_geodata.y[0]], mode='markers',
                                  marker=Marker(size=0, cmin=cmin, cmax=cmax,
                                                color='rgb(255,255,255)',
                                                colorscale=cbar_cmap_name,
                                                colorbar=ColorBar(thickness=10,
                                                                  x=1.1,
                                                                  titleside='right'),
                                                ))
                if cbar_title:
                    lines_cbar['marker']['colorbar']['title'] = cbar_title

                line_traces.append(lines_cbar)
            except:
                pass

    else:
        line_trace = dict(type='scatter',
                          text=[], hoverinfo='text', mode='lines', name=trace_name,
                          line=Line(width=width, color=color))


        line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(net, lines2plot, use_line_geodata)

        line_trace['text'] = lines2plot['name'].tolist() if infofunc is None else infofunc

        if legendgroup:
            line_trace['legendgroup'] = legendgroup

        line_traces = [line_trace]

    if len(nogolines) > 0:
        line_trace = dict(type='scatter',
                          text=[], hoverinfo='text', mode='lines', name='disconnected lines',
                          line=Line(width=width / 2, color='grey', dash='dot'))

        lines2plot = net.line.loc[nogolines]

        line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(net, lines2plot, use_line_geodata)

        line_trace['text'] = lines2plot['name'].tolist()

        if legendgroup:
            line_trace['legendgroup'] = legendgroup

        line_traces.append(line_trace)
    return line_traces



def create_trafo_trace(net, trafos=None, color='green', width=5, infofunc=None, cmap=None,
                       trace_name='trafos', cmin=None, cmax=None, cmap_vals=None):
    """
    Creates a plotly trace of pandapower trafos.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The trafos for which the collections are created.
        If None, all trafos in the network are considered.

        **width** (int, 5) - line width

        **infofunc** (list, None) - hoverinfo for each line

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "green") - color of lines in the trace

        **cmap** (bool, False) - name of a colormap which exists within plotly (Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis)

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum


    """
    color = get_plotly_color(color)


    # defining lines to be plot
    trafos = net.trafo.index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return []

    trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) &\
                               net.trafo.lv_bus.isin(net.bus_geodata.index)

    trafos_mask = net.trafo.index.isin(trafos)
    tarfo2plot = net.trafo[trafo_buses_with_geodata & trafos_mask]


    if cmap is not None:
        cmap = 'jet' if cmap is None else cmap

        cmin = 0 if cmin is None else cmin
        cmax = 100 if cmin is None else cmax

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net.res_trafo.shape[0] == 0:
                logger.error("There are no power flow results for lines which are default for line colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net.res_trafo.loc[tarfo2plot.index,'loading_percent'].values

        cmap_colors = get_plotly_cmap(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)
        trafo_traces = []
        col_i = 0
        for trafo_ind, trafo in tarfo2plot.iterrows():
            trafo_trace = dict(type='scatter', text=[], line=Line(width=width, color=cmap_colors[col_i]),
                                  hoverinfo='text', mode='lines', name=trace_name)

            trafo_trace['text'] = trafo['name'].tolist() if infofunc is None else infofunc[col_i]

            from_bus = net.bus_geodata.loc[trafo.hv_bus, 'x']
            to_bus = net.bus_geodata.loc[trafo.lv_bus, 'x']
            trafo_trace['x'] = [from_bus, (from_bus + to_bus)/2, to_bus]

            from_bus = net.bus_geodata.loc[trafo.hv_bus, 'y']
            to_bus = net.bus_geodata.loc[trafo.lv_bus, 'y']
            trafo_trace['y'] = [from_bus, (from_bus + to_bus)/2, to_bus]

            trafo_traces.append(trafo_trace)
            col_i += 1

    else:
        trafo_trace = dict(type='scatter',
                           text=[], line=dict(width=width, color=color),
                           hoverinfo='text', mode='lines', name=trace_name)

        trafo_trace['text'] = tarfo2plot['name'].tolist() if infofunc is None else infofunc

        from_bus = net.bus_geodata.loc[tarfo2plot.hv_bus, 'x'].tolist()
        to_bus = net.bus_geodata.loc[tarfo2plot.lv_bus, 'x'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        None_list = [None] * len(from_bus)
        trafo_trace['x'] = np.array([from_bus, center, to_bus, None_list]).T.flatten().tolist()
        trafo_trace['x'] = trafo_trace['x'][:-1]

        from_bus = net.bus_geodata.loc[tarfo2plot.hv_bus, 'y'].tolist()
        to_bus = net.bus_geodata.loc[tarfo2plot.lv_bus, 'y'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        None_list = [None] * len(from_bus)
        trafo_trace['y'] = np.array([from_bus, center, to_bus, None_list]).T.flatten().tolist()
        trafo_trace['y'] = trafo_trace['y'][:-1]

        trafo_traces = [trafo_trace]

    return trafo_traces



def draw_traces(traces, on_map=False, map_style='basic', showlegend=True, figsize=1,
                aspectratio='auto'):
    """
    plots all the traces (which can be created using :func:`create_bus_trace`, :func:`create_line_trace`,
    :func:`create_trafo_trace`)
    to PLOTLY (see https://plot.ly/python/)

    INPUT:
        **traces** - list of dicts which correspond to plotly traces
        generated using: `create_bus_trace`, `create_line_trace`, `create_trafo_trace`

    OPTIONAL:
        **on_map** (bool, False) - enables using mapbox plot in plotly

        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **showlegend** (bool, 'True') - enables legend display

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the network geodata
        any custom aspectration can be given as a tuple, e.g. (1.2, 1)

    """

    if on_map:
        try:
            on_map = _on_map_test(traces[0]['x'][0], traces[0]['y'][0])
        except:
            logger.warning("Test if geo-data are in lat/long cannot be performed using geopy -> "
                           "eventual plot errors are possible.")

        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -> "
                           "plot on maps is not possible.\n"
                           "Use geo_data_to_latlong(net, projection) to transform geodata from specific projection.")

    if on_map:
        # change traces for mapbox
        # change trace_type to scattermapbox and rename x to lat and y to lon
        for trace in traces:
            trace['lat'] = trace.pop('x')
            trace['lon'] = trace.pop('y')
            trace['type'] = 'scattermapbox'

    # setting Figure object
    fig = Figure(data=Data(traces),   # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=True if aspectratio is 'auto' else False,
                     hovermode='closest',
                     margin=dict(b=5, l=5, r=5, t=5),
                     # annotations=[dict(
                     #     text="",
                     #     showarrow=False,
                     #     xref="paper", yref="paper",
                     #     x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False),
                     # legend=dict(x=0, y=1.0)
                 ),)

    # check if geodata are real geographycal lat/lon coordinates using geopy

    if on_map:
        try:
            mapbox_access_token = _get_mapbox_token()
        except Exception:
            logger.exception('mapbox token required for map plots. '
                         'Get Mapbox token by signing in to https://www.mapbox.com/.\n'
                         'After getting a token, set it to pandapower using:\n'
                         'pandapower.plotting.plotly.mapbox_plot.set_mapbox_token(\'<token>\')')
            raise MapboxTokenMissing


        fig['layout']['mapbox'] = dict(accesstoken=mapbox_access_token,
                                       bearing=0,
                                       center=dict(lat= pd.Series(traces[0]['lat']).dropna().mean(),
                                                   lon= pd.Series(traces[0]['lon']).dropna().mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=11)


    # default aspectratio: if on_map use auto, else use 'original'
    aspectratio = 'original' if not on_map and aspectratio is 'auto' else aspectratio

    if aspectratio is not 'auto':
        if aspectratio is 'original':
            # TODO improve this workaround for getting original aspectratio
            xs = []
            ys = []
            for trace in traces:
                xs += trace['x']
                ys += trace['y']
            x_dropna = pd.Series(xs).dropna()
            y_dropna = pd.Series(ys).dropna()
            xrange = x_dropna.max() - x_dropna.min()
            yrange = y_dropna.max() - y_dropna.min()
            ratio = xrange / yrange
            if ratio < 1:
                aspectratio = (ratio, 1.)
            else:
                aspectratio = (1., 1/ratio)

        aspectratio = np.array(aspectratio) / max(aspectratio)
        fig['layout']['width'], fig['layout']['height'] = ([ar * figsize * 700 for ar in aspectratio])

    # check if called from ipynb or not in order to consider appropriate plot function
    if _in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot


    plot(fig)





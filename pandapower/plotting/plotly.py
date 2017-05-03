# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


from pandapower.topology import create_nxgraph, connected_components
from pandapower import runpp
from pandapower.plotting.generic_geodata import create_generic_coordinates

import matplotlib.cm as cm
import matplotlib.colors as colors

import numpy as np
import pandas as pd

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


try:
    from plotly.graph_objs import Figure, Data, Layout, Marker, XAxis, YAxis, Line, ColorBar
except ImportError:
    logger.debug("Failed to import plotly - interactive plotting will not be available")

try:
    import seaborn
except ImportError:
    pass


#
#try:
#    import seaborn
#    import matplotlib.colors as colors_mbl
#    colors_sns = seaborn.color_palette("muted")
#    colors_sns_bright = seaborn.color_palette("bright")
#    colors = seaborn_to_plotly_palette(colors_sns)
#    colors_bright = seaborn_to_plotly_palette(colors_sns_bright)
#    colors_hex = [colors_mbl.rgb2hex(col) for col in colors_sns]
#    color_yellow = seaborn_to_plotly_color(seaborn.xkcd_palette(["amber"])[0])
#    color_yellow = colors_bright[4]
#except ImportError:
#    colors = ["blue", "green", "red", "cyan", "yellow"]
#    color_yellow = "yellow"


def in_ipynb():
    """
    an auxiliary function which checks if plot is called from a jupyter-notebook or not
    """
    import __main__ as main
    return not hasattr(main, '__file__')


def seaborn_to_plotly_palette(scl, transparence = None):
    """
    converts a seaborn color palette to a plotly colorscale
    """
    if transparence:
        return ['rgb' + str((scl[i][0] * 255, scl[i][1] * 255, scl[i][2] * 255, transparence)) for i in range(len(scl))]
    else:
        return ['rgb' + str((scl[i][0] * 255, scl[i][1] * 255, scl[i][2] * 255)) for i in range(len(scl))]


def seaborn_to_plotly_color(scl, transparence = None):
    """
    converts a seaborn color to a plotly color
    """
    if transparence:
        return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, transparence))
    else:
        if len(scl) > 3:
            return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, scl[3]))
        else:
            return 'rgb'+str((scl[0]*255, scl[1]*255, scl[2]*255))


def get_cmap_matplotlib_for_plotly(values, cmap_name='jet', cmin=None, cmax=None):
    cmap = cm.get_cmap(cmap_name)
    if cmin is None:
        cmin = values.min()
    if cmax is None:
        cmax = values.max()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    bus_fill_colors_rgba = cmap(norm(values).data)[:, 0:3] * 255.
    return ['rgb({0},{1},{2})'.format(r, g, b) for r, g, b in bus_fill_colors_rgba]


def _on_map_test(x, y):
    """
    checks if bus_geodata can be located on a map using geopy
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geolocator = Nominatim()

    except ImportError:
        # if geopy is not available there will be no geo-coordinates check
        # therefore if geo-coordinates are not real and user sets on_map=True, an empty map will be plot!
        logger.warning('Geo-coordinates check cannot be peformed because geopy package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return True
    try:
        location = geolocator.reverse("{0}, {1}".format(x, y), language='en-US')
    except GeocoderTimedOut as e:
        logger.Error("Existing net geodata cannot be geo-located: possible reason: geo-data not in lat/long ->"
                     "try geo_data_to_latlong(net, projection) to transform geodata to lat/long!")

    if location.address is None:
        return False
    else:
        return True


def geo_data_to_latlong(net, projection):
    """
    Transforms network's geodata (in `net.bus_geodata` and `net.line_geodata`) from specified projection to lat/long (WGS84).

    INPUT:
        **net** (pandapowerNet) - The pandapower network

        **projection** (String) - projection from which geodata are transformed to lat/long. some examples

                - "epsg:31467" - 3-degree Gauss-Kruger zone 3
                - "epsg:2032" - NAD27(CGQ77) / UTM zone 18N
                - "epsg:2190" - Azores Oriental 1940 / UTM zone 26N
    """
    try:
        from pyproj import Proj, transform
    except ImportError:
        logger.warning('Geo-coordinates check cannot be peformed because geopy package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return

    wgs84 = Proj(init='epsg:4326')  # lat/long

    try:
        projection = Proj(init=projection)
    except:
        logger.warning("Transformation of geodata to lat/long failed! because of:]\n"
                       "Unknown projection provided "
                         "(format 'epsg:<number>' required as available at http://spatialreference.org/ref/epsg/ )")
        return

    # transform all geodata to long/lat using set or found projection
    try:
        lon, lat = transform(projection, wgs84, net.bus_geodata.loc[:, 'x'].values, net.bus_geodata.loc[:, 'y'].values)
        net.bus_geodata.loc[:, 'x'], net.bus_geodata.loc[:, 'y'] = lat, lon

        if net.line_geodata.shape[0] > 0:
            for idx in net.line_geodata.index:
                line_coo = np.array(net.line_geodata.loc[idx, 'coords'])
                lon, lat = transform(projection, wgs84, line_coo[:, 0], line_coo[:, 1])
                net.line_geodata.loc[idx, 'coords'] = np.array([lat,lon]).T.tolist()
        return
    except:
        logger.warning('Transformation of geodata to lat/long failed!')
        return


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

        **kwargs - key word arguments are passed to the patch function

    """

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
                      cmax=None, **kwargs):
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

            **kwargs - key word arguments are passed to the patch function

        """


    # defining lines to be plot
    lines = net.line.index.tolist() if lines is None else list(lines)
    if len(lines) == 0:
        return None

    nogolines = set()
    if respect_switches:
        nogolines = set(net.switch.element[(net.switch.et == "l") &
                                           (net.switch.closed == 0)])
    nogolines_mask = net.line.index.isin(nogolines)

    lines_mask = net.line.index.isin(lines)
    lines2plot = net.line[~nogolines_mask & lines_mask]

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

        cmap_lines = get_cmap_matplotlib_for_plotly(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)

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
                       trace_name='trafos', cmin=None, cmax=None, cmap_vals=None, **kwargs):
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

            **kwargs - key word arguments are passed to the patch function

    """


    # defining lines to be plot
    trafos = net.trafo.index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return None

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

        cmap_colors = get_cmap_matplotlib_for_plotly(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)
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
    plots all the traces (which can be created using :func:`create_bus_trace`,
                                                     :func:`create_line_trace`,
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
        # TODO replace this token with a proprietary one...
        mapbox_access_token = 'pk.eyJ1Ijoiamtyc3R1bG8iLCJhIjoiY2oxcTJ3NzQxMDAwazMzcDVsdGNrdHRxeSJ9.039HhrXpcR0dD6ldBqq8oQ'
        # pk.eyJ1IjoiY2hlbHNlYXBsb3RseSIsImEiOiJjaXFqeXVzdDkwMHFrZnRtOGtlMGtwcGs4In0.SLidkdBMEap9POJGIe1eGw  token from plotly site
        # pk.eyJ1Ijoiamtyc3R1bG8iLCJhIjoiY2oxcTJ3NzQxMDAwazMzcDVsdGNrdHRxeSJ9.039HhrXpcR0dD6ldBqq8oQ    Jakov's token
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
    if in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot


    plot(fig)


def simple_plotly(net, respect_switches=True, use_line_geodata=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=1,
                  bus_size=10, ext_grid_size=20.0, bus_color="b", line_color='grey',
                  trafo_color='green', ext_grid_color="y"):
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

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to lat-long.
            For each projection a string can be found at http://spatialreference.org/ref/epsg/


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

        **bus_color** (String, colors[0]) - Bus Color. Init as first value of color palette.
        Usually colors[0] = "b".

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'g') - Trafo Color. Init is green

        **ext_grid_color** (String, 'y') - External Grid Color. Init is yellow
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
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, color=bus_color)

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

    # ----- Trafos ------
    trafo_trace = create_trafo_trace(net, color=trafo_color, width=line_width * 5)


    # ----- Ext grid ------§
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'  # doesn't appear on mapbox if square
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color=ext_grid_color, size=ext_grid_size,
                                      patch_type=marker_type, trace_name='external_grid')

    draw_traces(line_trace + trafo_trace + ext_grid_trace + bus_trace,
                aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style)




def vlevel_plotly(net, respect_switches=True, use_line_geodata=None, colors_dict=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=2,
                  bus_size=10):
    """
    Plots a pandapower network in plotly
    using lines/buses colors according to the voltage level they belong to.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
            or on net.bus_geodata of the connected buses (False)

        *colors_dict** (dict, None) - by default 6 basic colors from default collor palette is used.
            Otherwise, user can define a dictionary in the form: voltage_kv : color

        **on_map** (bool, False) - enables using mapbox plot in plotly
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to lat-long.
            For each projection a string can be found at http://spatialreference.org/ref/epsg/

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

    """

    # create geocoord if none are available
    # TODO remove this if not necessary:
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches)
        if on_map == True:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    # getting connected componenets without consideration of trafos
    graph = create_nxgraph(net, include_trafos=False)
    vlev_buses = connected_components(graph)
    # getting unique sets of buses for each voltage level
    vlev_bus_dict = {}
    for vl_buses in vlev_buses:
        if net.bus.loc[vl_buses, 'vn_kv'].unique().shape[0] > 1:
            logger.warning('buses from the same voltage level does not have the same vn_kv !?')
        vn_kv = net.bus.loc[vl_buses, 'vn_kv'].unique()[0]
        if vlev_bus_dict.get(vn_kv):
            vlev_bus_dict[vn_kv].update(vl_buses)
        else:
            vlev_bus_dict[vn_kv] = vl_buses

    # create a default colormap for voltage levels
    nvlevs = len(vlev_bus_dict)
    if colors_dict is None:
        # if number of voltage levels is greater than 6 create a new color palette with sufficient colors
        colors_vlev = colors if nvlevs <= 6 else seaborn_to_plotly_palette(seaborn.color_palette("hls", nvlevs))
        colors_dict = dict(zip(vlev_bus_dict.keys(), colors_vlev[:nvlevs]))


    # creating traces for buses and lines for each voltage level
    bus_traces = []
    line_traces = []
    for vn_kv, buses_vl in vlev_bus_dict.items():

        vlev_color = colors_dict[vn_kv]
        bus_trace_vlev = create_bus_trace(net, buses=buses_vl, size=bus_size, legendgroup=str(vn_kv),
                                          color=vlev_color, trace_name='buses {0} kV'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev

        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) & net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(net, lines=vlev_lines, use_line_geodata=use_line_geodata, on_map=on_map,
                                            respect_switches=respect_switches, legendgroup=str(vn_kv),
                                            color=vlev_color, width=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, color='gray', width=line_width * 2)

    draw_traces(line_traces + trafo_traces + bus_traces, showlegend=True,
                aspectratio=aspectratio, on_map=on_map, map_style=map_style, figsize=figsize)


def pf_res_plotly(net, cmap='Jet', use_line_geodata = None, on_map=False, projection=None,
                  map_style='basic', figsize=1, aspectratio='auto', line_width=2, bus_size=10):
    """
    Plots a pandapower network in plotly
    using colormap for coloring lines according to line loading and buses according to voltage in p.u.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches when artificial geodata is created

        *cmap** (str, True) - name of the colormap

        *colors_dict** (dict, None) - by default 6 basic colors from default collor palette is used.
            Otherwise, user can define a dictionary in the form: voltage_kv : color

        **on_map** (bool, False) - enables using mapbox plot in plotly
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to lat-long.
            For each projection a string can be found at http://spatialreference.org/ref/epsg/

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

    """


    if 'res_bus' not in net or net.get('res_bus').shape[0] == 0:
        logger.warning('There are no Power Flow results. A Newton-Raphson power flow will be executed.')
        runpp(net)

    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=True)
        if on_map == True:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
            geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    idx = net.line.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.bus['name'] + '<br>' +
                 'U = ' + net.res_bus.loc[idx, 'vm_pu'].astype(str) + ' pu' + '<br>' +
                 'U = ' + (net.res_bus.loc[idx, 'vm_pu'] * net.bus.vn_kv).astype(str) + ' kV' + '<br>' +
                 'ang = ' + net.res_bus.loc[idx, 'va_degree'].astype(str) + ' deg'
                 ).tolist()
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, infofunc=hoverinfo, cmap=cmap,
                                 cbar_title='Bus Voltage [pu]', cmin=0.9, cmax=1.1)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    # if bus geodata is available, but no line geodata
    cmap_lines = 'jet' if cmap is 'Jet' else cmap
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False
    idx = net.line.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.line['name'] + '<br>' +
                 'I = ' + net.res_line.loc[idx, 'loading_percent'].astype(str) + ' %' + '<br>' +
                 'I_from = ' + net.res_line.loc[idx, 'i_from_ka'].astype(str) + ' kA' + '<br>' +
                 'I_to = ' + net.res_line.loc[idx, 'i_to_ka'].astype(str) + ' kA' + '<br>'
                 ).tolist()
    line_traces = []
    line_traces = create_line_trace(net, use_line_geodata=use_line_geodata, respect_switches=True,
                                    width=line_width,
                                    infofunc=hoverinfo,
                                    cmap=cmap_lines,
                                    cmap_vals=net.res_line.loc[:, 'loading_percent'].values,
                                    cmin=0,
                                    cmax=100,
                                    cbar_title='Line Loading [%]')

    # ----- Trafos ------
    idx = net.trafo.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.trafo['name'] + '<br>' +
                 'I = ' + net.res_trafo.loc[idx, 'loading_percent'].astype(str) + ' %' + '<br>' +
                 'I_hv = ' + net.res_trafo.loc[idx, 'i_hv_ka'].astype(str) + ' kA' + '<br>'  +
                 'I_lv = ' + net.res_trafo.loc[idx, 'i_lv_ka'].astype(str) + ' kA' + '<br>'
                 ).tolist()
    trafo_traces = create_trafo_trace(net, width=line_width * 1.5, infofunc=hoverinfo,
                                      cmap=cmap_lines, cmin=0, cmax=100)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color='grey', size=bus_size * 2, trace_name='external_grid',
                                      patch_type=marker_type)

    draw_traces(line_traces + trafo_traces + ext_grid_trace + bus_trace,
                showlegend=False, aspectratio=aspectratio, on_map=on_map, map_style=map_style, figsize=figsize)





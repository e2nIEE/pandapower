# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


from pandapower.topology import create_nxgraph, connected_components
from pandapower import runpp
from pandapower.plotting.generic_geodata import create_generic_coordinates

import numpy as np


import pandas as pd



def in_ipynb():
    """
    an auxiliary function which checks if plot is called from a jupyter-notebook or not
    """
    import __main__ as main
    return not hasattr(main, '__file__')



from plotly.graph_objs import Figure, Data, Layout, Scatter, Marker, XAxis, YAxis, Line, ColorBar, Scattermapbox

def seaborn_to_plotly_palette(scl, transparence = None):
    ''' converts a seaborn color palette to a plotly colorscale '''
    # return [ [ float(i)/float(len(scl)-1), 'rgb'+str((scl[i][0]*255, scl[i][1]*255, scl[i][2]*255)) ] \
    #         for i in range(len(scl)) ]
    if transparence:
        return ['rgb' + str((scl[i][0] * 255, scl[i][1] * 255, scl[i][2] * 255, transparence)) for i in range(len(scl))]
    else:
        return ['rgb' + str((scl[i][0] * 255, scl[i][1] * 255, scl[i][2] * 255)) for i in range(len(scl))]

def seaborn_to_plotly_color(scl, transparence = None):
    ''' converts a seaborn color to a plotly color '''
    if transparence:
        return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, transparence))
    else:
        if len(scl) > 3:
            return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, scl[3]))
        else:
            return 'rgb'+str((scl[0]*255, scl[1]*255, scl[2]*255))


def get_cmap_matplotlib_for_plotly(values, cmap_name='jet', cmin=None, cmax=None):
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    cmap = cm.get_cmap(cmap_name)
    if cmin is None:
        cmin = values.min()
    if cmax is None:
        cmax = values.max()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    bus_fill_colors_rgba = cmap(norm(values).data)[:, 0:3] * 255.
    return ['rgb({0},{1},{2})'.format(r, g, b) for r, g, b in bus_fill_colors_rgba]


try:
    import pplog as logging
except:
    import logging

try:
    import seaborn
    import matplotlib.colors as colors_mbl
    colors_sns = seaborn.color_palette("muted")
    colors_sns_bright = seaborn.color_palette("bright")
    colors = seaborn_to_plotly_palette(colors_sns)
    colors_bright = seaborn_to_plotly_palette(colors_sns_bright)
    colors_hex = [colors_mbl.rgb2hex(col) for col in colors_sns]
    color_yellow = seaborn_to_plotly_color(seaborn.xkcd_palette(["amber"])[0])
    # color_yellow = colors_bright[4]
except:
    colors = ["b", "g", "r", "c", "y"]

logger = logging.getLogger(__name__)


# defining check whether geodata are in lat/lon using geopy

def _on_map_test_transf(net, projection=None, country=None):
    """
    checks if bus_geodata can be located on a map using geopy
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim()

    except:
        # if geopy is not available there will be no geo-coordinates check
        # therefore if geo-coordinates are not real and user sets on_map=True, an empty map will be plot!
        logger.warning('Geo-coordinates check cannot be peformed because geopy package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return True

    location = geolocator.reverse("{0}, {1}".format(net.bus_geodata.iloc[0]['x'],
                                                    net.bus_geodata.iloc[0]['y']), language='en-US')

    if location.address is None:
        if country is None and projection is None:
            logger.warning('Geo-coordinates are not in lat/lon (wgs84), '
                           'projection transformation is possible if country information is provided '
                           'as input argument country=<name_of_the_country>')
            return False

        if _transform_projection(net, projection, country) is None:
            return False
        else:
            return True

    else:
        return True


def _on_map_test(net):
    """
    checks if bus_geodata can be located on a map using geopy
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim()

    except:
        # if geopy is not available there will be no geo-coordinates check
        # therefore if geo-coordinates are not real and user sets on_map=True, an empty map will be plot!
        logger.warning('Geo-coordinates check cannot be peformed because geopy package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return True

    location = geolocator.reverse("{0}, {1}".format(net.bus_geodata.iloc[0]['x'],
                                                    net.bus_geodata.iloc[0]['y']), language='en-US')

    if location.address is None:
        return False
    else:
        return True




def _transform_projection(net, projection, country):
    try:
        from pyproj import Proj, transform
        from geopy.geocoders import Nominatim
        geolocator = Nominatim()
    except:
        logger.warning('Geo-coordinates check cannot be peformed because geopy package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return None

    wgs84 = Proj(init='epsg:4326')  # lat/long

    if projection is None:
        # searching for projection until lat/lon drops in the pre-defined country
        # dict with projections key: projection, value: name
        projections = {
            # TODO add more projections
            Proj(init='epsg:31466'): "gauss-kruger_zone2",
            Proj(init='epsg:31467'): "gauss-kruger_zone3",
            Proj(init='epsg:31468'): "gauss-kruger_zone4",
            Proj(init='epsg:31465'): "gauss-kruger_zone5",
        }

        x1, y1 = net.bus_geodata.loc[0, 'x'], net.bus_geodata.loc[0,'y']

        proj_found = False
        for projection in projections.keys():
            lon, lat = transform(projection, wgs84, x1, y1)
            location = geolocator.reverse("{0}, {1}".format(lat, lon), language='en-US')
            if location is not None and "address" in location.raw:
                if location.raw["address"]["country"] == country:
                    proj_found = True
                    break

        if not proj_found:
            logger.warning('No any corresponding projection found for the provided geo_data and specific country -> '
                           'Network cannot be plot on a map!')
            return None
        else:
            logger.warning('Projection {0} found to correspond to the provided geodata -> All network geodata '
                           'transformed from {0} to lat/lon (wgs84)'.format(projections[projection]))
    else:
        try:
            projection = Proj(init=projection)
        except:
            raise ValueError("Unknown projection provided"
                             "(format 'epsg:<number>' required as available in http://spatialreference.org/ref/epsg/ )")

    # transform all geodata to long/lat using set or found projection
    lon, lat = transform(projection, wgs84, net.bus_geodata.loc[:, 'x'].values, net.bus_geodata.loc[:, 'y'].values)
    net.bus_geodata.loc[:, 'x'], net.bus_geodata.loc[:, 'y'] = lat, lon

    if net.line_geodata.shape[0] > 0:
        for idx in net.line_geodata.index:
            line_coo = np.array(net.line_geodata.loc[idx, 'coords'])
            lon, lat = transform(projection, wgs84, line_coo[:, 0], line_coo[:, 1])
            net.line_geodata.loc[idx, 'coords'] = np.array([lat,lon]).T.tolist()

    return True
    






def create_bus_trace(net, buses=None, on_map=False, size=5, marker_type="circle", color=None, hoverinfo=None,
                     trace_name='buses', legendgroup=None,
                     cmap=False, cmap_name='Jet', cmap_vals=None, cbar_title='Bus Voltage [pu]',
                     cmin=0.9, cmax=1.1):

    # check if geodata are real geographycal lat/lon coordinates using geopy
    on_map = _on_map_test(net) if on_map else False

    # defining dict names depending if plot is on map or not
    xk = 'lat' if on_map else 'x'
    yk = 'lon' if on_map else 'y'
    trace_type = 'scattermapbox' if on_map else 'scatter'

    bus_trace = dict(type=trace_type, text=[], mode='markers', hoverinfo='text', name=trace_name,
                     marker=dict(color=color, size=size, symbol=marker_type))

    buses2plot = net.bus if buses is None else net.bus[net.bus.index.isin(buses)]

    buses_with_geodata = buses2plot.index.isin(net.bus_geodata.index)
    buses2plot = buses2plot[buses_with_geodata]

    bus_trace[xk], bus_trace[yk] = (net.bus_geodata.loc[buses2plot.index, 'x'].tolist(),
                                    net.bus_geodata.loc[buses2plot.index, 'y'].tolist())

    bus_trace['text'] = buses2plot.name.tolist() if hoverinfo is None else hoverinfo

    if legendgroup:
        bus_trace['legendgroup'] = legendgroup

    if cmap:
        # if color map is set
        cmap_vals = net.res_bus.loc[buses2plot.index, 'vm_pu'] if cmap_vals is None else cmap_vals
        bus_trace['marker'] = Marker(size=size,
                                     cmax=cmax,  # bus_volt_pu.max()
                                     cmin=cmin,  # bus_volt_pu.min(),
                                     color=cmap_vals,
                                     colorscale=cmap_name,
                                     colorbar=ColorBar(thickness=10,
                                                       title='Voltage in pu',
                                                       x=1.0,
                                                       titleside='right'),
                                     )
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

def create_line_trace(net, lines=None, use_line_geodata=True, on_map = False,
                      respect_switches=False, width=1.0, color='grey',
                      hoverinfo=None, trace_name = 'lines', legendgroup=None,
                      cmap=False, cbar_title="Line Loading [%]", cmap_name='jet', cmin=0, cmax=100, **kwargs):

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

    # check if geodata are real geographycal lat/lon coordinates using geopy
    on_map = _on_map_test(net) if on_map else False

    # defining dict names depending if plot is on map or not
    xk = 'lat' if on_map else 'x'
    yk = 'lon' if on_map else 'y'
    trace_type = 'scattermapbox' if on_map else 'scatter'

    if cmap:
        # workaround: if colormap plot is used, each line need to be separate scatter object because
        # plotly still doesn't support appropriately colormap for line objects
        # TODO correct this when plotly solves existing github issue about Line colorbar
        cmap_line_loading = get_cmap_matplotlib_for_plotly(net.res_line.loc[lines2plot.index,'loading_percent'],
                                                           cmap_name=cmap_name, cmin=cmin, cmax=cmax)

        line_traces = []
        col_i = 0
        for idx, line in lines2plot.iterrows():
            line_trace = dict(type=trace_type, text=[], hoverinfo='text', mode='lines', name=trace_name,
                              line=Line(width=width, color=color))

            line_trace[xk], line_trace[yk] = _get_line_geodata_plotly(net, lines2plot.loc[idx:idx], use_line_geodata)

            line_trace['line']['color'] = cmap_line_loading[col_i]

            line_trace['text'] = line.name.tolist() if hoverinfo is None else hoverinfo[col_i]

            line_traces.append(line_trace)
            col_i += 1

        # workaround to get colorbar for lines (an unvisible node is added)
        lines_cbar = Scatter(x=[net.bus_geodata.x[0]], y=[net.bus_geodata.y[0]], mode='markers',
                             marker=Marker(size=0, cmax=100.0, cmin=0.0,  # bus_volt_pu.min(),
                                           color='rgb(255,255,255)',
                                           colorscale='Jet',
                                           colorbar=ColorBar(thickness=10,
                                                             title=cbar_title,
                                                             x=1.1,
                                                             titleside='right'),
                                           ))
        line_traces.append(lines_cbar)

    else:
        line_trace = dict(type=trace_type,
                          text=[], hoverinfo='text', mode='lines', name=trace_name,
                          line=Line(width=width, color=color))


        line_trace[xk], line_trace[yk] = _get_line_geodata_plotly(net, lines2plot, use_line_geodata)

        line_trace['text'] = lines2plot.name.tolist() if hoverinfo is None else hoverinfo

        if legendgroup:
            line_trace['legendgroup'] = legendgroup

        line_traces = [line_trace]

    if len(nogolines) > 0:
        line_trace = dict(type=trace_type,
                          text=[], hoverinfo='text', mode='lines', name='disconnected lines',
                          line=Line(width=width / 2, color='grey', dash='dot'))

        lines2plot = net.line.loc[nogolines]

        line_trace[xk], line_trace[yk] = _get_line_geodata_plotly(net, lines2plot, use_line_geodata)

        line_trace['text'] = lines2plot.name.tolist()

        if legendgroup:
            line_trace['legendgroup'] = legendgroup

        line_traces.append(line_trace)


    return line_traces



def create_trafo_trace(net, trafos=None, on_map=False, color = 'green', width = 5,
                       hoverinfo=None, trace_name = 'trafos',
                      cmap=False, cbar_title="Line Loading [%]", cmap_name='jet', cmin=None, cmax=None, **kwargs):

    # defining lines to be plot
    trafos = net.trafo.index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return None

    trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) &\
                               net.trafo.lv_bus.isin(net.bus_geodata.index)

    trafos_mask = net.trafo.index.isin(trafos)
    tarfo2plot = net.trafo[trafo_buses_with_geodata & trafos_mask]

    # check if geodata are real geographycal lat/lon coordinates using geopy
    on_map = _on_map_test(net) if on_map else False

    # defining dict names depending if plot is on map or not
    xk = 'lat' if on_map else 'x'
    yk = 'lon' if on_map else 'y'
    trace_type = 'scattermapbox' if on_map else 'scatter'

    if cmap:
        line_traces = []
        cmin = 0 if cmin is None else cmin
        cmax = 100 if cmin is None else cmax
        cmap_name = 'jet' if cmap_name is None else cmap_name
        cmap_colors = get_cmap_matplotlib_for_plotly(net.res_trafo.loc[tarfo2plot.index,'loading_percent'],
                                                            cmap_name=cmap_name, cmin=cmin, cmax=cmax)
        trafo_traces = []
        col_i = 0
        for trafo_ind, trafo in tarfo2plot.iterrows():
            trafo_trace = dict(type=trace_type, text=[], line=Line(width=width, color=cmap_colors[col_i]),
                                  hoverinfo='text', mode='lines', name=trace_name)

            trafo_trace['text'] = trafo.name.tolist() if hoverinfo is None else hoverinfo[col_i]

            from_bus = net.bus_geodata.loc[trafo.hv_bus, 'x']
            to_bus = net.bus_geodata.loc[trafo.lv_bus, 'x']
            trafo_trace[xk] = [from_bus, (from_bus + to_bus)/2, to_bus]

            from_bus = net.bus_geodata.loc[trafo.hv_bus, 'y']
            to_bus = net.bus_geodata.loc[trafo.lv_bus, 'y']
            trafo_trace[yk] = [from_bus, (from_bus + to_bus)/2, to_bus]

            trafo_traces.append(trafo_trace)
            col_i += 1

    else:
        trafo_trace = dict(type=trace_type,
                           text=[], line=dict(width=width, color=color),
                           hoverinfo='text', mode='lines', name=trace_name)

        trafo_trace['text'] = tarfo2plot.name.tolist() if hoverinfo is None else hoverinfo

        from_bus = net.bus_geodata.loc[tarfo2plot.hv_bus, 'x'].tolist()
        to_bus = net.bus_geodata.loc[tarfo2plot.lv_bus, 'x'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        None_list = [None] * len(from_bus)
        trafo_trace[xk] = np.array([from_bus, center, to_bus, None_list]).T.flatten().tolist()
        trafo_trace[xk] = trafo_trace[xk][:-1]

        from_bus = net.bus_geodata.loc[tarfo2plot.hv_bus, 'y'].tolist()
        to_bus = net.bus_geodata.loc[tarfo2plot.lv_bus, 'y'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        None_list = [None] * len(from_bus)
        trafo_trace[yk] = np.array([from_bus, center, to_bus, None_list]).T.flatten().tolist()
        trafo_trace[yk] = trafo_trace[yk][:-1]

        trafo_traces = [trafo_trace]

    return trafo_traces



def draw_traces(net, traces, on_map = False, map_style='basic', showlegend = True, figsize=1, aspectratio = 'auto'):
    """
    plots all the traces to PLOTLY (see https://plot.ly/python/)


    INPUT:
        **net** - The pandapower format network

        **traces** - list of dicts which correspond to plotly traces
            generated using: create_bus_trace, create_line_trace, create_trafo_trace

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
        on_map = _on_map_test(net)
        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -->"
                           " plot on maps is not possible")

    if on_map:
        # TODO replace this token with a proprietary one...
        mapbox_access_token = 'pk.eyJ1IjoiY2hlbHNlYXBsb3RseSIsImEiOiJjaXFqeXVzdDkwMHFrZnRtO' \
                              'GtlMGtwcGs4In0.SLidkdBMEap9POJGIe1eGw'
        fig['layout']['mapbox'] = dict(accesstoken=mapbox_access_token,
                                       bearing=0,
                                       center=dict(lat= net.bus_geodata.x.mean(),
                                                   lon=net.bus_geodata.y.mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=11)

    # default aspectratio: if on_map use auto, else use 'original'
    aspectratio = 'original' if not on_map and aspectratio is 'auto' else aspectratio

    if aspectratio is not 'auto':
        if aspectratio is 'original':
            xrange = net.bus_geodata.x.max() - net.bus_geodata.x.min()
            yrange = net.bus_geodata.y.max() - net.bus_geodata.y.min()
            ratio = xrange / yrange
            if ratio < 1:
                aspectratio = (ratio, 1.)
            else:
                aspectratio = (1., 1/ratio)

        fig['layout']['width'], fig['layout']['height'] = ([ar * figsize * 700  for ar in aspectratio])

    # check if called from ipynb or not in order to consider appropriate plot function
    if in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    plot(fig)


def simple_plotly(net=None, respect_switches=False, use_line_geodata=None,
                  on_map=False, projection=None, country=None, map_style='basic',
                  figsize=1, aspectratio='auto',
                  line_width=1, bus_size=10, ext_grid_size=20.0,
                  bus_color=colors[0], line_color='grey', trafo_color='green', ext_grid_color=color_yellow):
    """
    Plots a pandapower network as simple as possible in plotly (https://plot.ly/python/).
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
            or on net.bus_geodata of the connected buses (False)

        **on_map** (bool, False) - enables using mapbox plot in plotly.
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

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

            See bus sizes for details. Note: ext_grids are plottet as rectangles

        **bus_color** (String, colors[0]) - Bus Color. Init as first value of color palette.
        Usually colors[0] = "b".

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'g') - Trafo Color. Init is green

        **ext_grid_color** (String, 'y') - External Grid Color. Init is yellow
    """

    if net is None:
        import pandapower.networks as nw
        logger.warning("No pandapower network provided -> Plotting mv_oberrhein")
        net = nw.mv_oberrhein()

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
    if on_map:
        on_map = _on_map_test_transf(net, projection=projection, country=country)
        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -->"
                           " plot on maps is not possible")

    # ----- Buses ------
    # initializating bus trace
    bus_trace = create_bus_trace(net, net.bus.index, on_map=on_map, size=bus_size, color=bus_color)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    line_trace = create_line_trace(net, net.line.index, on_map=on_map, respect_switches=respect_switches,
                                   color=line_color, width=line_width,
                                   use_line_geodata=use_line_geodata)

    # ----- Trafos ------
    trafo_trace = create_trafo_trace(net, on_map=on_map, color=trafo_color, width=line_width*5)


    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'  # doesn't appear on mapbox if square
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus, on_map=on_map,
                                          color=ext_grid_color, size=ext_grid_size,
                                          marker_type=marker_type, trace_name='external_grid')

    draw_traces(net, line_trace + trafo_trace + ext_grid_trace + bus_trace,
                aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style)




def vlevel_plotly(net, respect_switches=False, use_line_geodata=None,
                  colors_dict=None,
                  on_map=False, projection=None, country=None, map_style='basic',
                  figsize=1, aspectratio='auto',
                  line_width=2, bus_size=10):
    """
    Plots a pandapower network in plotly (https://plot.ly/python/)
    using lines/buses colors according to the voltage level they belong to.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
            or on net.bus_geodata of the connected buses (False)

        *colors_dict** (dict, None) - by default 6 basic colors from default collor palette is used.
            Otherwise, user can define a dictionary in the form: voltage_kv : color

        **on_map** (bool, False) - enables using mapbox plot in plotly
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

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
        create_generic_coordinates(net, respect_switches=True)
        if on_map == True:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map:
        on_map = _on_map_test_transf(net, projection=projection, country=country)
        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -->"
                           " plot on maps is not possible")

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
        bus_trace_vlev = create_bus_trace(net, buses=buses_vl, size=bus_size, on_map=on_map, legendgroup=str(vn_kv),
                                          color=vlev_color, trace_name='buses {0} kV'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev

        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) & net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(net, lines=vlev_lines, use_line_geodata=use_line_geodata, on_map=on_map,
                                            respect_switches=respect_switches, legendgroup=str(vn_kv),
                                            color=vlev_color, width=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, on_map=on_map, color='gray', width=line_width*2)

    draw_traces(net, line_traces + trafo_traces + bus_traces, showlegend=True,
                aspectratio=aspectratio, on_map=on_map, map_style=map_style, figsize=figsize)


def pf_res_plotly(net, cmap_name='jet', use_line_geodata = None,
                  on_map=False, projection=None, country=None, map_style='basic',
                  figsize=1, aspectratio='auto',
                  line_width=2, bus_size=10):
    """
    Plots a pandapower network in plotly (https://plot.ly/python/)
    using colormap for coloring lines according to line loading and buses according to voltage in p.u.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches when artificial geodata is created

        *cmap_name** (str, 'jet') - name of the colormap

        *colors_dict** (dict, None) - by default 6 basic colors from default collor palette is used.
            Otherwise, user can define a dictionary in the form: voltage_kv : color

        **on_map** (bool, False) - enables using mapbox plot in plotly
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

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
    if on_map:
        on_map = _on_map_test_transf(net, projection=projection, country=country)
        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -->"
                           " plot on maps is not possible")

    # ----- Buses ------
    # initializating bus trace
    idx = net.line.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.bus.name + '<br>' +
                 'U = ' + net.res_bus.loc[idx, 'vm_pu'].astype(str) + ' pu' + '<br>' +
                 'U = ' + (net.res_bus.loc[idx, 'vm_pu'] * net.bus.vn_kv).astype(str) + ' kV' + '<br>' +
                 'ang = ' + net.res_bus.loc[idx, 'va_degree'].astype(str) + ' deg'
                 ).tolist()
    bus_trace = create_bus_trace(net, net.bus.index, on_map=on_map, size=bus_size, hoverinfo=hoverinfo, cmap=True)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False
    idx = net.line.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.line.name + '<br>' +
                 'I = ' + net.res_line.loc[idx, 'loading_percent'].astype(str) + ' %' + '<br>' +
                 'I_from = ' + net.res_line.loc[idx, 'i_from_ka'].astype(str) + ' kA' + '<br>' +
                 'I_to = ' + net.res_line.loc[idx, 'i_to_ka'].astype(str) + ' kA' + '<br>'
                 ).tolist()
    line_traces = create_line_trace(net,use_line_geodata=use_line_geodata, on_map=on_map, respect_switches=True,
                                    width=line_width,
                                    hoverinfo=hoverinfo,
                                    cmap=True, cmap_name=cmap_name, cmin=0, cmax=100)

    # ----- Trafos ------
    idx = net.trafo.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.trafo.name + '<br>' +
                 'I = ' + net.res_trafo.loc[idx, 'loading_percent'].astype(str) + ' %' + '<br>' +
                 'I_hv = ' + net.res_trafo.loc[idx, 'i_hv_ka'].astype(str) + ' kA' + '<br>'  +
                 'I_lv = ' + net.res_trafo.loc[idx, 'i_lv_ka'].astype(str) + ' kA' + '<br>'
                 ).tolist()
    trafo_traces = create_trafo_trace(net, on_map=on_map, width=line_width*1.5, hoverinfo=hoverinfo,
                                      cmap=True, cmap_name='jet', cmin=0, cmax=100)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus, on_map=on_map,
                                      color='grey', size=bus_size*2, trace_name='external_grid',
                                      marker_type=marker_type)

    draw_traces(net, line_traces + trafo_traces + ext_grid_trace + bus_trace,
                showlegend=False, aspectratio=aspectratio, on_map=on_map, map_style=map_style, figsize=figsize)





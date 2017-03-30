

from pandapower.plotting import cmap_continous
from pandapower.topology import create_nxgraph, connected_components
from pandapower import runpp
from pandapower.plotting.generic_geodata import create_generic_coordinates

import numpy as np


import pandas as pd



def in_ipynb():
    """
    :return:
    an auxiliary function which checks if located in an jupyter-notebook or not
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


def create_bus_trace(net, buses=None, on_map=False, size=5, marker_type="circle", color=None, hoverinfo=None,
                     trace_name = 'buses',
                     cmap=False, cmap_name='Jet', cmap_vals=None, cbar_title='Bus Voltage [pu]',
                     cmin=0.9, cmax=1.1):


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
                      respect_switches=False, width=1.0, color='grey', hoverinfo=None, trace_name = 'lines',
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

        line_traces = [line_trace]

    if len(nogolines) > 0:
        line_trace = dict(type=trace_type,
                          text=[], line=Line(width=width/2, color='grey', dash='dot'),
                          hoverinfo='text', mode='lines', name='disconnected lines')

        lines2plot = net.line.loc[nogolines]

        line_trace[xk], line_trace[yk] = _get_line_geodata_plotly(net, lines2plot, use_line_geodata)

        line_trace['text'] = lines2plot.name.tolist()

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



def draw_traces(net, traces, on_map = False, map_style='basic', showlegend = True, figsize='auto', aspectratio = False):
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
        
        **figsize** (tuple, 'auto') - sets figure size as a tuple
            In "auto" mode figure size is determined automatically from plotly
            
        **aspectratio** (bool, False) - enables preserving aspect ration of the network geodata when plotting in plotly
    """

    # setting Figure object
    fig = Figure(data=Data(traces),   # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=True if figsize is 'auto' and aspectratio is False else False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text="",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    # TODO add a check if geodata are real geographic coordinates and if not place appropriate warning !
    if on_map:
        mapbox_access_token = 'pk.eyJ1IjoiY2hlbHNlYXBsb3RseSIsImEiOiJjaXFqeXVzdDkwMHFrZnRtO' \
                              'GtlMGtwcGs4In0.SLidkdBMEap9POJGIe1eGw'
        fig['layout']['mapbox'] = dict(accesstoken=mapbox_access_token,
                                       bearing=0,
                                       center=dict(lat= net.bus_geodata.x.mean(),
                                                   lon=net.bus_geodata.y.mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=13)
    if aspectratio and not on_map:
        xrange = net.bus_geodata.x.max() - net.bus_geodata.x.min()
        yrange = net.bus_geodata.y.max() - net.bus_geodata.y.min()
        aspectratio = xrange / yrange
        fig['layout']['width'] = 700 if aspectratio > 1 else 700 * aspectratio
        fig['layout']['height'] = 700 if aspectratio < 1 else 700 / aspectratio
    elif figsize is not 'auto' and not on_map:
        fig['layout']['width'], fig['layout']['height'] = figsize

    # check if called from ipynb or not in order to consider appropriate plot function
    if in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    plot(fig)


def simple_plotly(net=None, respect_switches=False, on_map=False, map_style='basic', figsize='auto',
                  line_width=1, bus_size=10, ext_grid_size=20.0,
                  bus_color=colors[0], line_color='grey', trafo_color='green', ext_grid_color=color_yellow,
                  aspectratio=False, use_line_geodata=None):

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
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches)

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
    marker_type = 'circle' if on_map else 'square'  # doesn't appear on mapbox if sqare
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus, on_map=on_map,
                                          color=ext_grid_color, size=ext_grid_size,
                                          marker_type=marker_type, trace_name='external_grid')

    draw_traces(net, line_trace + trafo_trace + ext_grid_trace + bus_trace,
                aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style)




def vlevel_plotly(net, colors_dict=None, on_map=False, map_style='basic', line_width=2, bus_size=10,
                  aspectratio = False, respect_switches=False):

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
        bus_trace_vlev = create_bus_trace(net, buses=buses_vl, size=bus_size, on_map=on_map,
                                          color=vlev_color, trace_name='buses {0} kV'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev

        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) & net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(net, lines=vlev_lines, use_line_geodata=True, on_map=on_map,
                                            respect_switches=respect_switches,
                                            color=vlev_color, width=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, on_map=on_map, color='gray', width=line_width*2)

    draw_traces(net, line_traces + trafo_traces + bus_traces, showlegend=True,
                aspectratio=aspectratio, on_map=on_map, map_style=map_style)


def pf_res_plotly(net, cmap_name='jet', on_map=False, map_style='basic', line_width=2, bus_size=10, aspectratio=False):
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
    use_line_geodata = False if net.line_geodata.shape[0] == 0 else True
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
                showlegend=False, aspectratio=aspectratio, on_map=on_map, map_style=map_style)





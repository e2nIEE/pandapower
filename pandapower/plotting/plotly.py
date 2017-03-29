

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



import plotly.plotly as pltly
from plotly.graph_objs import Figure, Data, Layout, Scatter, Marker, XAxis, YAxis, Line, ColorBar


def seaborn_to_plotly_palette( scl ):
    ''' converts a seaborn color palette to a plotly colorscale '''
    return [ [ float(i)/float(len(scl)-1), 'rgb'+str((scl[i][0]*255, scl[i][1]*255, scl[i][2]*255)) ] \
            for i in range(len(scl)) ]

def seaborn_to_plotly_color(scl, transparence = None):
    ''' converts a seaborn color to a plotly color '''
    if transparence:
        return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, transparence))
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
    colors = seaborn_to_plotly_palette(seaborn.color_palette())
    color_yellow = seaborn_to_plotly_color(seaborn.xkcd_palette(["amber"])[0],transparence=0.5)
except:
    colors = ["b", "g", "r", "c", "y"]

logger = logging.getLogger(__name__)

def create_bus_trace(net, buses=None, size=5, marker_type="circle", color=None, hoverinfo=None, trace_name = 'buses',
                     cmap=False, cmap_name='Jet', cmap_vals=None, cbar_title='Bus Voltage [pu]',
                     cmin=0.9, cmax=1.1, **kwargs):

    bus_trace = Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', name=trace_name,
                        marker=Marker(color=color, size=size, symbol=marker_type))

    # all the bus coordinates need to be positive in plotly
    # TODO use here copy() in order not to change net.bus_geodata
    bus_geodata = net.bus_geodata
    if (net.bus_geodata.x < 0).any():
        bus_geodata['x'] = bus_geodata.x + abs(bus_geodata.x.min())

    if (net.bus_geodata.y < 0).any():
        bus_geodata['y'] = bus_geodata.y + abs(bus_geodata.y.min())

    if buses is not None:
        buses2plot = net.bus[net.bus.index.isin(buses)]
    else:
        buses2plot = net.bus
    buses_with_geodata = buses2plot.index.isin(bus_geodata.index)
    buses2plot = buses2plot[buses_with_geodata]

    bus_trace['x'], bus_trace['y'] = (bus_geodata.loc[buses2plot.index, 'x'].tolist(),
                                      bus_geodata.loc[buses2plot.index, 'y'].tolist())

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



def create_line_trace(net, lines=None, use_line_geodata=True,
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


    if cmap:

        # workaround: if colormap plot is used, each line need to be separate scatter object because
        # plotly still doesn't support appropriately colormap for line objects
        # TODO correct this when plotly solves existing github issue about Line colorbar
        line_traces = []
        cmap_line_loading = get_cmap_matplotlib_for_plotly(net.res_line.loc[lines2plot.index,'loading_percent'],
                                                           cmap_name=cmap_name, cmin=cmin, cmax=cmax)

        line_traces = []
        col_i = 0
        for idx, line in lines2plot.iterrows():
            line_trace = Scatter(x=[], y=[], text=[], line=Line(width=width, color=color),
                                 hoverinfo='text', mode='lines', name=trace_name)

            if use_line_geodata:
                line_coords = net.line_geodata.loc[idx, 'coords']
                linex, liney = list(zip(*line_coords))
                line_trace['x'] += linex
                line_trace['x'] += [None]
                line_trace['y'] += liney
                line_trace['y'] += [None]
            else:
                # getting x and y values from bus_geodata for from and to side of each line
                for xy in ['x', 'y']:
                    from_bus = net.bus_geodata.loc[line.from_bus, xy].tolist()
                    to_bus = net.bus_geodata.loc[line.to_bus, xy].tolist()
                    # center point added because of the hovertool
                    center = (np.array(from_bus) + np.array(to_bus)) / 2
                    line_trace[xy] = [from_bus, center, to_bus, None]

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

        line_trace = Scatter(x=[], y=[], text=[], line=Line(width=width, color=color),
                             hoverinfo='text', mode='lines', name=trace_name)

        if use_line_geodata:
            for line_ind, line in lines2plot.iterrows():
                line_coords = net.line_geodata.loc[line_ind, 'coords']
                linex, liney = list(zip(*line_coords))
                line_trace['x'] += linex
                line_trace['x'] += [None]
                line_trace['y'] += liney
                line_trace['y'] += [None]
        else:
            # getting x and y values from bus_geodata for from and to side of each line
            for xy in ['x', 'y']:
                from_bus = net.bus_geodata.loc[lines2plot.from_bus, xy].tolist()
                to_bus = net.bus_geodata.loc[lines2plot.to_bus, xy].tolist()
                # center point added because of the hovertool
                center = (np.array(from_bus) + np.array(to_bus)) / 2
                None_list = [None] * len(from_bus)
                line_trace[xy] = np.array([from_bus, center, to_bus, None_list]).T.flatten()

        line_trace['text'] = lines2plot.name.tolist() if hoverinfo is None else hoverinfo

        line_traces = [line_trace]

    if len(nogolines) > 0:
        line_trace = Scatter(x=[], y=[], text=[], line=Line(width=width/2, color='grey', dash='dash'),
                             hoverinfo='text', mode='lines', name='disconnected lines')
        lines2plot = net.line.loc[nogolines]
        if use_line_geodata:
            for line_ind, line in lines2plot.iterrows():
                line_coords = net.line_geodata.loc[line_ind, 'coords']
                linex, liney = list(zip(*line_coords))
                line_trace['x'] += linex
                line_trace['x'] += [None]
                line_trace['y'] += liney
                line_trace['y'] += [None]
        else:
            # getting x and y values from bus_geodata for from and to side of each line
            for xy in ['x', 'y']:
                from_bus = net.bus_geodata.loc[lines2plot.from_bus, xy].tolist()
                to_bus = net.bus_geodata.loc[lines2plot.to_bus, xy].tolist()
                # center point added because of the hovertool
                center = (np.array(from_bus) + np.array(to_bus)) / 2
                None_list = [None] * len(from_bus)
                line_trace[xy] = np.array([from_bus, center, to_bus, None_list]).T.flatten()

        line_trace['text'] = lines2plot.name.tolist()

        line_traces.append(line_trace)


    return line_traces



def create_trafo_trace(net, trafos=None, color = 'green', width = 5, hoverinfo=None, trace_name = 'trafos',
                      cmap=False, cbar_title="Line Loading [%]", cmap_name='jet', cmin=None, cmax=None, **kwargs):

    # defining lines to be plot
    trafos = net.trafo.index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return None

    trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) &\
                               net.trafo.lv_bus.isin(net.bus_geodata.index)

    trafos_mask = net.trafo.index.isin(trafos)
    tarfo2plot = net.trafo[trafo_buses_with_geodata & trafos_mask]




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
            trafo_trace = Scatter(x=[], y=[], text=[], line=Line(width=width, color=cmap_colors[col_i]),
                                  hoverinfo='text', mode='lines', name=trace_name)

            trafo_trace['text'] = trafo.name.tolist() if hoverinfo is None else hoverinfo[col_i]

            for xy in ['x', 'y']:
                from_bus = net.bus_geodata.loc[trafo.hv_bus, xy]
                to_bus = net.bus_geodata.loc[trafo.lv_bus, xy]
                trafo_trace[xy] = [from_bus, (from_bus + to_bus)/2, to_bus, None]

            trafo_traces.append(trafo_trace)
            col_i += 1

    else:
        trafo_trace = Scatter(x=[], y=[], text=[], line=Line(width=width, color=color),
                             hoverinfo='text', mode='lines', name=trace_name)

        trafo_trace['text'] = tarfo2plot.name.tolist() if hoverinfo is None else hoverinfo
        for trafo_ind, trafo in tarfo2plot.iterrows():
            for xy in ['x', 'y']:
                from_bus = net.bus_geodata.loc[trafo.hv_bus, xy]
                to_bus = net.bus_geodata.loc[trafo.lv_bus, xy]
                trafo_trace[xy] += [from_bus, (from_bus + to_bus)/2, to_bus, None]

        trafo_traces = [trafo_trace]

    return trafo_traces



def create_extgrid_trace(net, color=color_yellow, size=20, symbol='square'):
    ext_grid_buses_with_geodata = net.ext_grid.bus.isin(net.bus_geodata.index)

    ext_grid2plot = net.ext_grid[ext_grid_buses_with_geodata]

    ext_grid_trace = Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text', name='external grid',
                             marker=Marker(color=color, symbol=symbol, size=size))

    ext_grid_trace['x'] = net.bus_geodata.loc[ext_grid2plot.bus, 'x'].tolist()
    ext_grid_trace['y'] = net.bus_geodata.loc[ext_grid2plot.bus, 'y'].tolist()
    ext_grid_trace['text'] = ext_grid2plot.name

    return [ext_grid_trace]



def draw_traces(net, traces, showlegend = True, aspectratio = False, with_geo = False):

    # setting Figure object
    fig = Figure(data=Data(traces),   # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=False if aspectratio else True,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     annotations=[dict(
                         text="",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    if with_geo:
        fig['layout']['geo'] = dict(resolution=50, scope='europe',showframe=False, showland=True)

    if aspectratio:
        xrange = net.bus_geodata.x.max() - net.bus_geodata.x.min()
        yrange = net.bus_geodata.y.max() - net.bus_geodata.y.min()
        aspectratio = xrange / yrange
        fig['layout']['width'] = 700 if aspectratio > 1 else 700 * aspectratio
        fig['layout']['height'] = 700 if aspectratio < 1 else 700 / aspectratio


    # check if called from ipynb or not in order to consider appropriate plot function
    if in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    plot(fig)


def simple_plotly(net=None, respect_switches=False, line_width=1.0, bus_size=10, ext_grid_size=20.0,
                bus_color=colors[0][1], line_color='grey', trafo_color='green', ext_grid_color=color_yellow,
                  aspectratio = False):

    if net is None:
        import pandapower.networks as nw
        logger.warning("No pandapower network provided -> Plotting mv_oberrhein")
        net = nw.mv_oberrhein()

    # create geocoord if none are available
    # TODO remove this if not necessary:
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x","y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches)



    # ----- Buses ------
    # initializating bus trace
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, color=bus_color)


    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    use_line_geodata = False if len(net.line_geodata) == 0 else True

    line_trace = create_line_trace(net, net.line.index, respect_switches=respect_switches,
                                   color=line_color, linewidth=line_width,
                                   use_line_geodata=use_line_geodata)


    # ----- Trafos ------
    trafo_trace = create_trafo_trace(net, color=trafo_color, width=line_width*5)


    # ----- Ext grid ------
    # get external grid from create_bus_trace

    ext_grid_trace = create_extgrid_trace(net, color=ext_grid_color, symbol='sqare', size=ext_grid_size)


    draw_traces(net, line_trace + bus_trace + trafo_trace + ext_grid_trace, aspectratio=aspectratio)


def vlevel_plotly(net, cmap_list=None, line_width=3.0, bus_size=10, aspectratio = False, respect_switches=False):
    if 'res_bus' not in net or net.get('res_bus').shape[0] == 0:
        logger.warning('There are no Power Flow results. A Newton-Raphson power flow will be executed.')
        runpp(net)

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


    cmap_list_def = [(400,'red'),(220,'green'),(110,'blue'),(20,'green'),(10,'blue'),(0.4,'red')]
    cmap_list = cmap_list_def if cmap_list is None else cmap_list
    cmap_dict = dict(cmap_list)
    cmap,norm = cmap_continous(cmap_list)

    # getting connected componenets without consideration of trafos
    graph = create_nxgraph(net, include_trafos=False)
    vlev_buses = connected_components(graph)
    # getting unique sets of buses for each voltage level
    vlev_dict = {}
    for vl_buses in vlev_buses:
        if net.bus.loc[vl_buses, 'vn_kv'].unique().shape[0] > 1:
            logger.warning('buses from the same voltage level does not have the same vn_kv !?')
        vn_kv = net.bus.loc[vl_buses, 'vn_kv'].unique()[0]
        if vlev_dict.get(vn_kv):
            vlev_dict[vn_kv].update(vl_buses)
        else:
            vlev_dict[vn_kv] = vl_buses

    # creating traces for buses and lines for each voltage level
    bus_traces = []
    line_traces = []
    for vn_kv, buses_vl in vlev_dict.items():

        vlev_color = cmap_dict[vn_kv]
        bus_trace_vlev = create_bus_trace(net, buses=buses_vl, size=bus_size,
                                          color=vlev_color, trace_name='buses {0} kV'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev

        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) & net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(net, lines=vlev_lines, use_line_geodata=True,
                                            respect_switches=respect_switches,
                                            color=vlev_color, linewidth=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, color='gray', width=5)

    draw_traces(net, line_traces + trafo_traces + bus_traces, showlegend=True, aspectratio=aspectratio)


def pf_res_plotly(net, cmap_name='jet', line_width=3.0, bus_size=10, aspectratio=False):
    if 'res_bus' not in net or net.get('res_bus').shape[0] == 0:
        logger.warning('There are no Power Flow results. A Newton-Raphson power flow will be executed.')
        pp.runpp(net)

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

    # ----- Buses ------
    # initializating bus trace
    idx = net.line.index
    # hoverinfo which contains name and pf results
    hoverinfo = (net.bus.name + '<br>' +
                 'U = ' + net.res_bus.loc[idx, 'vm_pu'].astype(str) + ' pu' + '<br>' +
                 'U = ' + (net.res_bus.loc[idx, 'vm_pu'] * net.bus.vn_kv).astype(str) + ' kV' + '<br>' +
                 'ang = ' + net.res_bus.loc[idx, 'va_degree'].astype(str) + ' deg'
                 ).tolist()
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, hoverinfo=hoverinfo, cmap=True)


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
    line_traces = create_line_trace(net,use_line_geodata=use_line_geodata, respect_switches=True,
                                    linewidth=line_width,
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
    trafo_traces = create_trafo_trace(net, width=line_width, hoverinfo=hoverinfo,
                                      cmap=True, cmap_name='jet', cmin=0, cmax=100)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    ext_grid_trace = create_extgrid_trace(net)

    draw_traces(net, ext_grid_trace + line_traces + trafo_traces + bus_trace,
                showlegend=False, aspectratio=aspectratio)





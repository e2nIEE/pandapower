import logging
import pandas as pd
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace
from pandapower.plotting.plotly.get_colors import get_plotly_color_palette
from pandapower.topology import create_nxgraph, connected_components
from pandapower.plotting.plotly.draw_layers import draw_layers

logger = logging.getLogger(__name__)


def layers_plotly(net, bus_groups, respect_switches=True, use_line_geo=None, colors_dict=None, on_map=False,
                  map_style='basic', figsize=1, aspectratio='auto', line_width=2,
                  bus_size=10, auto_open=True):
    """
    Plot pandapower network buses grouped into layers in plotly
    If no geodata is available, artificial geodata is generated. For advanced plotting see the
    tutorial

    INPUT:
        **net** - The pandapower format network.

    OPTIONAL:
        **bus_groups** (DataFrame, None) - Dataframe with two columns, `bus` and `group`.
            In the first one, the data is the index value of each bus (int).
            In the second one, the data is the number of the group the bus belongs to (int).

        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        **use_line_geo** (bool, True) - defines if lines patches are based on net.line.geo
        of the lines (True) or on net.bus.geo of the connected buses (False)

        *colors_dict** (dict, None) - dictionary for customization of colors for each voltage level
        in the form: voltage : color

        **on_map** (bool, False) - enables using mapLibre plot in plotly If provided geodata are not
        real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be
        transformed to lat-long. For each projection a string can be found at
        http://spatialreference.org/ref/epsg/

        **map_style** (str, 'basic') - enables using mapLibre plot in plotly

            - 'basic'
            - 'carto-darkmatter'
            - 'carto-darkmatter-nolabels'
            - 'carto-positron'
            - 'carto-positron-nolabels'
            - 'carto-voyager'
            - 'carto-voyager-nolabels'
            - 'dark'
            - 'light'
            - 'open-street-map'
            - 'outdoors'
            - 'satellite''
            - 'satellite-streets'
            - 'streets'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
        network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

        **auto_open** (bool, True) - automatically open plot in browser

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """
    # getting connected components without consideration of trafos
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
    # handling groups
    layers = {}
    unique_layers = pd.Series(bus_groups["group"].unique())
    unique_layers = unique_layers.sort_values()
    unique_layers = unique_layers.reset_index(drop=True)
    ind_bus = bus_groups.index
    ind_bus = pd.Series(ind_bus)
    for i in range(len(unique_layers)):
        set_a = set()
        for j in range(len(bus_groups)):
            if unique_layers[i] == bus_groups.iloc[j, 0]:
                set_a.add(ind_bus[j])
        layers[unique_layers[i]] = set_a

    num_layer = len(bus_groups['group'].unique())
    # getting number of groups for each bus
    nvlevs = len(layers)
    colors = get_plotly_color_palette(nvlevs)
    colors_dict = colors_dict or dict(zip(layers.keys(), colors))
    # creating traces for buses and lines for each voltage level
    bus_traces = []
    for vn_kv, buses_vl in layers.items():
        vlev_color = colors_dict[vn_kv]
        bus_trace_vlev = create_bus_trace(
            net, buses=buses_vl, size=bus_size, legendgroup=str(vn_kv), color=vlev_color,
            trace_name='Group {0}'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev
    line_traces = []
    for vn_kv, buses_vl in vlev_bus_dict.items():
        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) &
                        net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(
            net=net,
            lines=vlev_lines,
            use_line_geo=use_line_geo,
            respect_switches=respect_switches,
            legendgroup=str(vn_kv), color='gray',
            width=line_width,
            trace_name='lines {0} kV'.format(vn_kv)
        )
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, color='gray', width=line_width * 2)

    return draw_layers(
        line_traces + trafo_traces + bus_traces,
        num_layer,
        showlegend=True,
        aspectratio=aspectratio,
        on_map=on_map,
        map_style=map_style,
        figsize=figsize,
        auto_open=auto_open
    )

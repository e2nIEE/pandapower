# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import sys
import math

import numpy as np
import pandas as pd
from packaging import version
from collections.abc import Iterable

from pandapower.auxiliary import soft_dependency_error
from pandapower.plotting.plotly.get_colors import get_plotly_color, get_plotly_cmap
from pandapower.plotting.plotly.mapbox_plot import _on_map_test, _get_mapbox_token, \
    MapboxTokenMissing

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

try:
    from plotly import __version__ as plotly_version
    from plotly.graph_objs.scatter.marker import ColorBar
    from plotly.graph_objs import Figure, Layout
    from plotly.graph_objs.layout import XAxis, YAxis
    from plotly.graph_objs.scatter import Line, Marker
    from plotly.graph_objs.scattermapbox import Line as scmLine
    from plotly.graph_objs.scattermapbox import Marker as scmMarker
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


def version_check():
    if "plotly_version" not in locals() and "plotly_version" not in globals():
        raise UserWarning("You are trying to use plotly, which is not installed.\r\n"
                          "Please upgrade your python-plotly installation, "
                          "e.g., via pip install --upgrade plotly")
    if version.parse(plotly_version) < version.parse("3.1.1"):
        raise UserWarning(f"Your plotly version {plotly_version} is no longer supported.\r\n"
                          "Please upgrade your python-plotly installation, "
                          "e.g., via pip install --upgrade plotly")


def _in_ipynb():
    """
    An auxiliary function which checks if plot is called from a jupyter-notebook or not
    """
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False


def sum_line_length(pts):
    pt_diff = lambda p: (p[0][0] - p[1][0], p[0][1] - p[1][1])
    diffs = map(pt_diff, zip(pts[:-1], pts[1:]))
    line_length = sum(math.hypot(d1, d2) for d1, d2 in diffs)
    return line_length


def get_line_neutral(coord):
    if len(coord) == 1:
        return coord[0]
    half_length = sum_line_length(coord) / 2.0
    length = 0.0
    ind = 0
    while length < half_length:
        ind += 1
        length = sum_line_length(coord[:ind])

    start_coord = coord[ind - 2]
    end_coord = coord[ind - 1]
    mid = [(a1 + a2) / 2.0 for a1, a2 in zip(start_coord, end_coord)]

    return mid


def create_edge_center_trace(line_trace, size=1, patch_type="circle", color="white", infofunc=None,
                             trace_name='edge_center', use_line_geodata=False, showlegend=False,
                             legendgroup=None, hoverlabel=None):
    """
    Creates a plotly trace of pandapower buses.

    INPUT:
        **line traces** (from pandapowerNet) - The already generated line traces with center geodata

    OPTIONAL:

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "square" for a rectangle
                - "diamond" for a diamond
                - much more patch types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (pd.Series, None) - hoverinfo for each trace element. Indices should correspond
        to the pandapower element indices

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of buses in the trace

    """
    # color = get_plotly_color(color)

    center_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='text', name=trace_name,
                        marker=dict(color=color, size=size, symbol=patch_type),
                        showlegend=showlegend, legendgroup=legendgroup)
    if hoverlabel is not None:
        center_trace.update({'hoverlabel': hoverlabel})

    if not use_line_geodata:
        center_trace['x'], center_trace['y'] = (line_trace[0]["x"][1::4], line_trace[0]["y"][1::4])
    else:
        x, y = [], []
        for trace in line_trace:
            coord = list(zip(trace["x"], trace["y"]))
            mid_coord = get_line_neutral(coord)
            x.append(mid_coord[0])
            y.append(mid_coord[1])

        center_trace['x'], center_trace['y'] = (x, y)

    center_trace['text'] = infofunc

    return center_trace


def create_bus_trace(net, buses=None, size=5, patch_type="circle", color="blue", infofunc=None,
                     trace_name='buses', legendgroup=None, cmap=None, cmap_vals=None,
                     cbar_title=None, cmin=None, cmax=None, cpos=1.0, colormap_column="vm_pu"):
    """
    Creates a plotly trace of pandapower buses. It is a wrapper function for the more generic
    _create_node_trace function.

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
                - much more patch types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (pd.Series, None) - hoverinfo for bus elements. Indices should correspond to
        the pandapower element indices

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of buses in the trace

        **cmap** (String, None) - name of a colormap which exists within plotly
        (Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow,
        Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis) alternatively a custom
        discrete colormap can be used. Append "_r" for inversion.

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        **colormap_column** (str, "vm_pu") - set color of bus according to this variable

    """
    node_element = 'bus'
    branch_element = 'line'
    return _create_node_trace(net,
                              nodes=buses,
                              size=size,
                              patch_type=patch_type,
                              color=color,
                              infofunc=infofunc,
                              trace_name=trace_name,
                              legendgroup=legendgroup,
                              cmap=cmap,
                              cmap_vals=cmap_vals,
                              cbar_title=cbar_title,
                              cmin=cmin,
                              cmax=cmax,
                              cpos=cpos,
                              colormap_column=colormap_column,
                              node_element=node_element,
                              branch_element=branch_element)


def _create_node_trace(net, nodes=None, size=5, patch_type='circle', color='blue', infofunc=None,
                       trace_name='nodes', legendgroup=None, cmap=None, cmap_vals=None,
                       cbar_title=None, cmin=None, cmax=None, cpos=1.0, colormap_column='vm_pu',
                       node_element='bus', branch_element='line'):
    """
    Creates a plotly trace of node elements. In pandapower, it should be called by
    create_bus_traces. The rather generic, non-power net specific names were introduced to make it
    usable in other packages, e.g. for pipe networks.

    INPUT:
        **net** (pandapowerNet) - The network

    OPTIONAL:
        **nodes** (list, None) - The nodes for which the collections are created.
                                 If None, all nodes in the network are considered.

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "square" for a rectangle
                - "diamond" for a diamond
                - much more patch types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (pd.Series, None) - hoverinfo for node elements. Indices should correspond to
                                         the node element indices

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of nodes in the trace

        **cmap** (String, None) - name of a colormap which exists within plotly
            (Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow,
            Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis) alternatively a custom
            discrete colormap can be used

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        **colormap_column** (str, "vm_pu") - set color of bus according to this variable

        **node_element** (str, "bus") - name of the node element in the net. In a pandapower net,
                                        this is alwas "bus"

        **branch_element** (str, "line") - name of the branch element in the net. In a pandapower
                                           net, this is alwas "line"

    """
    if not PLOTLY_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "plotly")
    color = get_plotly_color(color)
    node_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='text', name=trace_name,
                     marker=dict(color=color, size=size, symbol=patch_type))
    nodes = net[node_element].index.tolist() if nodes is None else list(nodes)
    node_geodata = node_element + "_geodata"
    node_plot_index = [b for b in nodes if b in list(set(nodes) & set(net[node_geodata].index))]
    node_trace['x'], node_trace['y'] = \
        (net[node_geodata].loc[node_plot_index, 'x'].tolist(),
         net[node_geodata].loc[node_plot_index, 'y'].tolist())
    if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
            len(infofunc) == len(nodes):
        infofunc = pd.Series(index=nodes, data=infofunc)
    node_trace['text'] = net[node_element].loc[node_plot_index, 'name'] if infofunc is None else \
        infofunc.loc[nodes]
    if legendgroup:
        node_trace['legendgroup'] = legendgroup
    # if color map is set
    if cmap is not None:
        # TODO introduce discrete colormaps (see contour plots in plotly)
        # if cmap_vals are not given

        cmap = 'Jet' if cmap is True else cmap

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net["res_"+branch_element].shape[0] == 0:
                if branch_element == "line":
                    logger.error(
                        "There are no power flow results for buses voltage magnitudes which are"
                        "default for bus colormap coloring..."
                        "set cmap_vals input argument if you want colormap according to some "
                        "specific values...")
                else:
                    logger.error(
                     "There are no simulation results which are default for %s colormap coloring..."
                     "set cmap_vals input argument if you want colormap according to some "
                     "specific values..." %node_element)
            cmap_vals = net["res_"+node_element].loc[node_plot_index, colormap_column].values

        cmap_vals = net["res_"+node_element].loc[
            node_plot_index, colormap_column] if cmap_vals is None else cmap_vals

        cmin = cmap_vals.min() if cmin is None else cmin
        cmax = cmap_vals.max() if cmax is None else cmax

        node_trace['marker'] = Marker(size=size,
                                     color=cmap_vals, cmin=cmin, cmax=cmax,
                                     colorscale=cmap,
                                     colorbar=ColorBar(thickness=10,
                                                       x=cpos),
                                     symbol=patch_type
                                     )

        if cbar_title:
            node_trace['marker']['colorbar']['title'] = cbar_title

        node_trace['marker']['colorbar']['title']['side'] = 'right'
    return [node_trace]


def _get_branch_geodata_plotly(net, branches, use_branch_geodata, branch_element='line',
                               node_element='bus'):
    xs = []
    ys = []
    if use_branch_geodata:
        for line_ind, _ in branches.iterrows():
            line_coords = net[branch_element+'_geodata'].at[line_ind, 'coords']
            linex, liney = list(zip(*line_coords))
            xs += linex
            xs += [None]
            ys += liney
            ys += [None]
    else:
        # getting x and y values from bus_geodata for from and to side of each line
        n = node_element
        n_geodata = n + "_geodata"
        from_n = 'from_'+n
        to_n = 'to_'+n
        from_node = net[n_geodata].loc[branches[from_n], 'x'].tolist()
        to_node = net[n_geodata].loc[branches[to_n], 'x'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_node) + np.array(to_node)) / 2
        none_list = [None] * len(from_node)
        xs = np.array([from_node, center, to_node, none_list]).T.flatten().tolist()

        from_node = net[n_geodata].loc[branches[from_n], 'y'].tolist()
        to_node = net[n_geodata].loc[branches[to_n], 'y'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_node) + np.array(to_node)) / 2
        none_list = [None] * len(from_node)
        ys = np.array([from_node, center, to_node, none_list]).T.flatten().tolist()

    # [:-1] is because the trace will not appear on maps if None is at the end
    return xs[:-1], ys[:-1]


def create_line_trace(net, lines=None, use_line_geodata=True, respect_switches=False, width=1.0,
                      color='grey', infofunc=None, trace_name='lines', legendgroup='lines',
                      cmap=None, cbar_title=None, show_colorbar=True, cmap_vals=None, cmin=None,
                      cmax=None, cpos=1.1, cmap_vals_category='loading_percent', hoverlabel=None):
    """
    Creates a plotly trace of pandapower lines. It is a power net specific wrapper function for the
    more generic _create_line_trace function.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created.
        If None, all lines in the network are considered.

        **width** (int, 1) - line width


        **infofunc** (pd.Series, None) - hoverinfo for line elements. Indices should correspond to
        the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "grey") - color of lines in the trace

        **legendgroup** (String, None) - defines groups of layers that will be displayed in a legend
        e.g. groups according to voltage level (as used in `vlevel_plotly`)

        **cmap** (String, None) - name of a colormap which exists within plotly if set to True default `Jet`
        colormap is used, alternative colormaps : Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis
        To revert the color map, append "_r" to the name.

        **cmap_vals** (list, None) - values used for coloring using colormap

        **show_colorbar** (bool, False) - flag for showing or not corresponding colorbar

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        """

    branch_element = "line"
    node_element = "bus"
    separator_element = "switch"

    return _create_branch_trace(net=net,
                                branches=lines,
                                use_branch_geodata=use_line_geodata,
                                respect_separators=respect_switches,
                                width=width,
                                color=color,
                                infofunc=infofunc,
                                trace_name=trace_name,
                                legendgroup=legendgroup,
                                cmap=cmap,
                                cbar_title=cbar_title,
                                show_colorbar=show_colorbar,
                                cmap_vals=cmap_vals,
                                cmin=cmin,
                                cmax=cmax,
                                cpos=cpos,
                                branch_element=branch_element,
                                separator_element=separator_element,
                                node_element=node_element,
                                cmap_vals_category=cmap_vals_category,
                                hoverlabel=hoverlabel)


def _create_branch_trace(net, branches=None, use_branch_geodata=True, respect_separators=False,
                         width=1.0, color='grey', infofunc=None, trace_name='lines',
                         legendgroup=None, cmap=None, cbar_title=None, show_colorbar=True,
                         cmap_vals=None, cmin=None, cmax=None, cpos=1.1, branch_element='line',
                         separator_element='switch', node_element='bus',
                         cmap_vals_category='loading_percent', hoverlabel=None):
    """
    Creates a plotly trace of branch elements. The rather generic, non-power net specific names
    were introduced to make it usable in other packages, e.g. for pipe networks.

   INPUT:
       **net** (pandapowerNet) - The network

   OPTIONAL:
       **branches** (list, None) - The branches for which the collections are created.
                                   If None, all branches in the network are considered.

       **use_branch_geodata** (bool, True) - whether the geodata of the branch tables should be used

       **respect_separators** (bool, True) - whether separating elements like switches should be
                                             considered

       **width** (int, 1) - branch width

       **color** (String, "grey") - color of lines in the trace

       **infofunc** (pd.Series, None) - hoverinfo for line elements. Indices should correspond to
           the pandapower element indices

       **trace_name** (String, "lines") - name of the trace which will appear in the legend

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

       **cpos** (float, 1.1) - position of the colorbar

       **branch_element** (str, "line") - name of the branch element in the net. In a pandapower
                                          net, this is alwas "line"

       **separator_element** (str, "switch") - name of the separator element in the net. In a
                                               pandapower net, this is alwas "switch"

      **node_element** (str, "bus") - name of the node element in the net. In a pandapower net,
                                      this is alwas "bus" (net.bus)

       """
    if not PLOTLY_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "plotly")

    color = get_plotly_color(color)

    # defining branches (lines) to be plot
    branches = net[branch_element].index.tolist() if branches is None else list(branches)
    if len(branches) == 0:
        return []

    if infofunc is not None:
        if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
                len(infofunc) == len(branches):
            infofunc = pd.Series(index=branches, data=infofunc)
        if len(infofunc) != len(branches) and len(infofunc) != len(net[branch_element]):
            raise UserWarning("Different amount of hover info than {}s to "
                              "plot".format(branch_element))
        assert isinstance(infofunc, pd.Series), \
            "infofunc should be a pandas series with the net.{}.index to the infofunc " \
            "contents".format(branch_element)
    no_go_branches = set()
    if respect_separators:
        if separator_element == "switch":
            no_go_branches = set(branches) & \
                             set(net[separator_element].element[(net[separator_element].et == "l") &
                                                             (net[separator_element].closed == 0)])
        elif separator_element == "valve":
            no_go_branches = set(branches) & \
                             set(net[separator_element][(~net[separator_element].in_service) |
                                                        (net[separator_element].opened)])
        else:
            raise NotImplementedError("respect separtors is only implements for switches, "
                                      "not for {}s.".format(separator_element))
    branches_to_plot = net[branch_element].loc[list(set(net[branch_element].index) &
                                                    (set(branches) - no_go_branches))]
    no_go_branches_to_plot = None
    branch_geodata = branch_element + "_geodata"
    node_geodata = node_element + "_geodata"
    use_branch_geodata = use_branch_geodata if net[branch_geodata].shape[0] > 0 else False
    if use_branch_geodata:
        branches_to_plot = branches_to_plot.loc[np.intersect1d(branches_to_plot.index,
                                                               net[branch_geodata].index)]
    else:
        branches_with_geodata = branches_to_plot['from_'+node_element].isin(
                                                    net[node_geodata].index) & \
                                branches_to_plot['to_'+node_element].isin(net[node_geodata].index)
        branches_to_plot = branches_to_plot.loc[branches_with_geodata]
    cmap_branches = None
    if cmap is not None:
        # workaround: if colormap plot is used, each line need to be separate scatter object because
        # plotly still doesn't support appropriately colormap for line objects
        # TODO correct this when plotly solves existing github issue about Line colorbar

        cmap = 'jet' if cmap is True else cmap

        if cmap_vals is not None:
            if not isinstance(cmap_vals, np.ndarray):
                cmap_vals = np.asarray(cmap_vals)
        else:
            if net['res_'+branch_element].shape[0] == 0:
                logger.error(
                    "There are no simulation results for branches which are default for {}"
                    "colormap coloring..."
                    "set cmap_vals input argument if you want colormap according to some specific "
                    "values...".format(branch_element))
            cmap_vals = net['res_'+branch_element].loc[branches_to_plot.index,
                                                       cmap_vals_category].values

        cmap_branches = get_plotly_cmap(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)
        if len(cmap_branches) == len(net[branch_element]):
            # some branches are not plotted although cmap_value were provided for all branches
            branch_idx_map = dict(zip(net[branch_element].loc[branches].index.tolist(), range(len(branches))))
            cmap_branches = [cmap_branches[branch_idx_map[idx]] for idx in branches_to_plot.index]
        else:
            assert len(cmap_branches) == len(branches_to_plot), \
                "Different amounts of cmap values and branches to plot were supplied"
    branch_traces = []
    for col_i, (idx, branch) in enumerate(branches_to_plot.iterrows()):
        line_color = color
        line_info = branch['name'] if infofunc is None else infofunc.at[idx]
        if cmap is not None:
            try:
                line_color = cmap_branches[col_i]
            except IndexError:
                logger.warning("No color and info for {} {:d} (name: {}) available".format(
                    branch_element, idx, branch['name']))

        line_trace = dict(type='scatter', text=[], hoverinfo='text', mode='lines', name=trace_name,
                          line=Line(width=width, color=color), showlegend=False,
                          legendgroup=legendgroup)

        line_trace['x'], line_trace['y'] = _get_branch_geodata_plotly(net,
                                                                      branches_to_plot.loc[idx:idx],
                                                                      use_branch_geodata,
                                                                      branch_element, node_element)

        line_trace['line']['color'] = line_color

        line_trace['text'] = line_info

        branch_traces.append(line_trace)

    # enable legend for one element to make the legend group show up in the legend:
    branch_traces[0]["showlegend"] = True

    if show_colorbar and cmap is not None:

        cmin = cmap_vals.min() if cmin is None else cmin
        cmax = cmap_vals.max() if cmax is None else cmax
        try:
            # TODO for custom colormaps
            cbar_cmap_name = 'Jet' if cmap == 'jet' else cmap
            # workaround to get colorbar for branches (an unvisible node is added)
            # get x and y of first line.from_bus:
            x = [net[node_geodata].x[net[branch_element]["from_"+node_element][net[branch_element].index[0]]]]
            y = [net[node_geodata].y[net[branch_element]["from_"+node_element][net[branch_element].index[0]]]]
            branches_cbar = dict(type='scatter', x=x, y=y, mode='markers',
                              marker=Marker(size=0, cmin=cmin, cmax=cmax,
                                            color='rgb(255,255,255)',
                                            opacity=0,
                                            colorscale=cbar_cmap_name,
                                            colorbar=ColorBar(thickness=10,
                                                              x=cpos),
                                            ))
            if cbar_title:
                branches_cbar['marker']['colorbar']['title'] = cbar_title

            branches_cbar['marker']['colorbar']['title']['side'] = 'right'

            branch_traces.append(branches_cbar)
        except:
            pass
    if len(no_go_branches) > 0:
        ng_traces = []
        no_go_branches_to_plot = net[branch_element].loc[list(no_go_branches)]
        for idx, branch in no_go_branches_to_plot.iterrows():
            line_color = color
            line_trace = dict(type='scatter',
                              text=[], hoverinfo='text', mode='lines', name='disconnected branches',
                              line=Line(width=width / 2, color='grey', dash='dot'),
                              legendgroup="disconnected " + legendgroup, showlegend=False)

            line_trace['x'], line_trace['y'] = _get_branch_geodata_plotly(net,
                                                                          no_go_branches_to_plot.loc[
                                                                          idx:idx],
                                                                          use_branch_geodata,
                                                                          branch_element, node_element)

            line_trace['line']['color'] = line_color
            try:
                line_trace['text'] = infofunc.loc[idx]
            except (KeyError, IndexError, AttributeError):
                line_trace["text"] = branch['name']

            ng_traces.append(line_trace)
        # enable legend for one element to make the legend group show up in the legend:
        ng_traces[0]["showlegend"] = True
        branch_traces += ng_traces

    # sort infofunc so that it is the correct order lines_to_plot + no_go_lines_to_plot
    if infofunc is not None:
        if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
                len(infofunc) == len(net[branch_element]):
            infofunc = pd.Series(index=net[branch_element].index, data=infofunc)
        assert isinstance(infofunc, pd.Series), \
            "infofunc should be a pandas series with the net.{}.index to the infofunc contents" \
            .format(branch_element)
        sorted_idx = branches_to_plot.index.tolist()
        if no_go_branches_to_plot is not None:
            sorted_idx += no_go_branches_to_plot.index.tolist()
        infofunc = infofunc.loc[sorted_idx]
    center_trace = create_edge_center_trace(branch_traces, color=color, infofunc=infofunc,
                                            use_line_geodata=use_branch_geodata,
                                            showlegend=False, legendgroup=legendgroup,
                                            hoverlabel=hoverlabel)
    branch_traces.append(center_trace)

    return branch_traces


def create_trafo_trace(net, trafos=None, color='green', trafotype='2W', width=5, infofunc=None, cmap=None,
                       trace_name='2W transformers', cmin=None, cmax=None, cmap_vals=None, matching_params=None,
                       use_line_geodata=None):

    """
    Creates a plotly trace of pandapower trafos.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The trafos for which the collections are created.
        If None, all trafos in the network are considered.

        **trafotype** (String, "2W") - trafotype can be 2W or 3W and can define which transformer table is taken (trafo or trafo3w)

        **width** (int, 5) - line width

        **infofunc** (pd.Series, None) - hoverinfo for trafo elements. Indices should correspond
        to the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "green") - color of lines in the trace

        **cmap** (bool, False) - name of a colormap which exists within plotly (Greys, YlGnBu,
        Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
        Blackbody, Earth, Electric, Viridis)

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum


    """
    if not PLOTLY_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "plotly")

    color = get_plotly_color(color)

    if trafotype == '2W':
        trafotable='trafo'

        trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) & \
                               net.trafo.lv_bus.isin(net.bus_geodata.index)
        connections = [['hv_bus', 'lv_bus']]
    elif trafotype=='3W':
        trafotable = 'trafo3w'
        trafo_buses_with_geodata = net.trafo3w.hv_bus.isin(net.bus_geodata.index) & \
                               net.trafo3w.mv_bus.isin(net.bus_geodata.index) & \
                               net.trafo3w.lv_bus.isin(net.bus_geodata.index)
        connections = [['hv_bus', 'lv_bus'], ['hv_bus', 'mv_bus'], ['mv_bus', 'lv_bus']]

    else:
        raise UserWarning('trafo_type should be 2W or 3W')
    # defining lines to be plot
    trafos = net[trafotable].index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return []

    trafos_mask = net[trafotable].index.isin(trafos)
    trafos_to_plot = net[trafotable].loc[trafo_buses_with_geodata & trafos_mask]

    if infofunc is not None:
        if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
                len(infofunc) == len(trafos):
            infofunc = pd.Series(index=trafos, data=infofunc)
        assert isinstance(infofunc, pd.Series), \
            "infofunc should be a pandas series with the net.trafo.index to the infofunc contents"
        infofunc = infofunc.loc[trafos_to_plot.index]

    cmap_colors = []
    if cmap is not None:
        cmap = 'jet' if cmap is None else cmap

        cmin = 0 if cmin is None else cmin
        cmax = 100 if cmin is None else cmax

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net['res_'+trafotable].shape[0] == 0:
                logger.error("There are no power flow results for lines which are default for line colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net['res_'+trafotable].loc[trafos_to_plot.index, 'loading_percent'].values

        cmap_colors = get_plotly_cmap(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)

    trafo_traces = []

    for col_i, (idx, trafo) in enumerate(trafos_to_plot.iterrows()):
        if cmap is not None:
            color = cmap_colors[col_i]
        for from_bus1, to_bus1 in connections:

            trafo_trace = dict(type='scatter', text=[], line=Line(width=width, color=color),
                                 hoverinfo='text', mode='lines', name=trace_name,
                               legendgroup=trace_name, showlegend=False)

            trafo_trace['text'] = trafo['name'] if infofunc is None else infofunc.loc[idx]

            for k in ('x', 'y'):
                from_bus = net.bus_geodata.loc[trafo[from_bus1], k]
                to_bus = net.bus_geodata.loc[trafo[to_bus1], k]
                trafo_trace[k] = [from_bus, (from_bus + to_bus) / 2, to_bus]

            trafo_traces.append(trafo_trace)
    trafo_traces[0]["showlegend"] = True

    center_trace = create_edge_center_trace(trafo_traces, color=color, infofunc=infofunc,
                                            use_line_geodata=use_line_geodata,
                                            showlegend=False, legendgroup=trace_name)
    trafo_traces.append(center_trace)
    return trafo_traces


def create_weighted_marker_trace(net, elm_type="load", elm_ids=None, column_to_plot="p_mw",
                                 sizemode="area", color="red", patch_type="circle",
                                 marker_scaling=1., trace_name="", infofunc=None,
                                 node_element="bus", show_scale_legend=True,
                                 scale_marker_size=None, scale_marker_color=None,
                                 scale_legend_unit=None):
    """Create a single-color plotly trace markers/patches (e.g., bubbles) of value-dependent size.

    Can be used with pandapipes.plotting.plotly.simple_plotly (pass as "additional_trace").
    If present in the respective pandapower-net table, the "in_service" and "scaling" column will be
    taken into account as factors to calculate the markers' weights.
    Negative values might lead to unexpected results, especially when pos. and neg. values are
    mixed in one column! All values are treated as absolute values.
    If value = 0, no marker will be created.

    INPUT:
        **net** (pandapowerNet) - the pandapower net of the plot

    OPTIONAL:
        **elm_type** (str, default "load") - the element table in the net that holds the values
        to plot

        **elm_ids** (list, default None) - the element IDs of the elm_type table for which markers
        will be created. If None, all IDs in net[elm_type] that are not 0 or NaN in the
        `column_to_plot` will be used.

        **column_to_plot** (str, default "p_mw") - the column in the net[elm_type] table that
        will be plotted. The suffix (everything behind the last '_' in the column name) will be
        used as the unit in the infofunction.

        **sizemode** (str, default "area") - whether the markers' "area" or "diameter"  will be
        proportional to the represented value

        **color** (str, default "red") - color for the markers

        **patch_type** (str, default "circle") - plotly marker style that will be used (other
        options are triangle-up, triangle-down and many more; for non-filled markers, append "-open"
        cf. https://plotly.com/python/marker-style/)

        **marker_scaling** (float, default 1.) - factor to scale the size of all markers

        **trace_name** (str, default ""): trace name for the legend. If empty, elm_type will be used.

        **infofunc** (pd.Series, default None): hover-infofuction to overwrite the internal infofunction

        **node_element** (str, default "bus") - the name of node elements in the net. "bus" for
        pandapower networks, "junction" for pandapipes networks

        **show_scale_legend** (bool, default True): display a marker legend at the top right of the
         plot

        **scale_marker_size** (float / list of floats, default None): adjust the size of the scale,
        gets multiplied with `marker_scaling`. Default size is the average size of the respective
        weighted markers rounded to 5. A list of values will result in several scale marker legends.

        **scale_marker_color** (str, default None): specify the color of the scale legend marker.
        If None, `color` will be used.

        **scale_legend_unit** (str, default None): specifies the unit shown in the scale legend
        marker's string. It does not trigger any unit conversions! If None, the last part of
        `column_to_plot` will be used (all upper case).

    OUTPUT:
        **marker_trace** (dict): dict for the plotly trace

    """
    # filter for relevant elements:
    elm_ids = net[elm_type].loc[~net[elm_type][column_to_plot].isna()].index.tolist() \
        if elm_ids is None else list(elm_ids)

    # apply the scaling and in_service column, if available
    elms_df = net[elm_type].loc[elm_ids]
    scaling = elms_df.scaling if "scaling" in elms_df.columns else 1
    in_service = elms_df.in_service if "in_service" in elms_df.columns else 1
    elms_df["values_scaled"] = elms_df[column_to_plot] * in_service * scaling

    # get sum per bus, if not a bus table already:
    if elm_type not in [node_element, "res_"+node_element]:
        values_by_bus = elms_df.groupby("bus").sum().values_scaled
    else:
        values_by_bus = elms_df.values_scaled
    values_by_bus = values_by_bus.loc[values_by_bus != 0]

    if any(values_by_bus < 0):
        logger.warning("A marker trace cannot be created for negative values!\n"
                       "They will be considered as absolute values. Items with negative values:\n"
                       + str(values_by_bus.loc[values_by_bus < 0]))

    # add geodata:
    node_geodata = node_element + "_geodata"
    x_list = net[node_geodata].loc[values_by_bus.index, 'x'].tolist()
    y_list = net[node_geodata].loc[values_by_bus.index, 'y'].tolist()

    # set up hover info:
    unit = column_to_plot.split("_")[-1]
    unit = unit.upper()
    trace_name = trace_name if len(trace_name) else elm_type
    if infofunc is None:
        infofunc = values_by_bus.apply(lambda x: f"{trace_name} {x:.2f} {unit}")
    if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
            len(infofunc) == len(values_by_bus.index):
        infofunc = pd.Series(index=values_by_bus.index, data=infofunc)

    # set up the marker trace:
    marker_trace = dict(type='scatter',
                        text=infofunc,
                        mode='markers',
                        hoverinfo='text',
                        name=trace_name,
                        x=x_list,
                        y=y_list)
    marker_trace["marker"] = dict(color=color,
                                  size=values_by_bus.abs() * marker_scaling,
                                  symbol=patch_type,
                                  sizemode=sizemode)

    # additional info for the create_scale_trace function:

    if not isinstance(scale_marker_size, Iterable):
        scale_marker_size = [scale_marker_size]

    marker_trace["meta"] = dict(marker_scaling=marker_scaling,
                                column_to_plot=column_to_plot,
                                scale_legend_unit=scale_legend_unit or
                                                  column_to_plot.split("_")[1].upper(),
                                show_scale_legend=show_scale_legend,
                                scale_marker_size=scale_marker_size,
                                scale_marker_color=scale_marker_color or color)

    return marker_trace


def create_scale_trace(net, weighted_trace, down_shift=0):
    """Create a scale (marker size legend) for a weighted_marker_trace.

    Will be used with pandapipes.plotting.plotly.simple_plotly, when "additional_trace" contains
    a trace created by :func:`create_weighted_marker_trace` with :code:`show_scale_legend=True`.
    The default reference marker is of average size of all weighted markers, rounded to the next 5,
    and comes with a string with the respective reference value and unit.

    INPUT:
        **net** (pandapowerNet) - the pandapower net of the plot

        **weighted_trace** (dict) - weighted plotly trace

        **down_shift** (int) - shift to align different scales below each other (prop. to y-offset)

    """
    marker = weighted_trace["marker"]
    scale_info = weighted_trace["meta"]
    unit = scale_info["scale_legend_unit"]  # p_mw...q_mvar ..

    # scale trace
    x_max, y_max = net.bus_geodata.x.max(), net.bus_geodata.y.max()
    x_min, y_min = net.bus_geodata.x.min(), net.bus_geodata.y.min()
    x_pos = x_max + (x_max - x_min) * 0.2
    x_pos2 = x_max + ((x_max - x_min) * 0.2 * 2)

    scale_traces = []
    for idx, sze in enumerate(scale_info["scale_marker_size"]):
        y_pos = y_max - ((y_max - y_min) * (0.2 * (down_shift + idx)))

        # default is the average rounded to 5
        if not sze:
            scale_size = math.ceil(marker["size"].mean() / 5) * 5
        else:
            scale_size = sze * scale_info['marker_scaling']

        # second (dummy) position is needed for correct marker sizing
        scale_trace = dict(type="scatter",
                           x=[x_pos, x_pos2],
                           y=[y_pos, y_pos],
                           mode="markers+text",
                           hoverinfo="skip",
                           marker=dict(size=[scale_size, 0],
                                       color=scale_info.get("scale_marker_color"),
                                       symbol=marker["symbol"],
                                       sizemode=marker["sizemode"]),
                           text=[f"{scale_size / scale_info['marker_scaling']} {unit}", ""],
                           textposition="top center",
                           showlegend=False,
                           textfont=dict(family="Helvetica",
                                         size=14,
                                         color="DarkSlateGrey"))
        scale_traces.append(scale_trace)

    return scale_traces


def draw_traces(traces, on_map=False, map_style='basic', showlegend=True, figsize=1,
                aspectratio='auto', filename='temp-plot.html', auto_open=True, **kwargs):
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

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
        network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **filename** (str, "temp-plot.html") - plots to a html file called filename

        **auto_open** (bool, 'True') - automatically open plot in browser

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """
    if not PLOTLY_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "plotly")
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
            if 'x' in trace.keys():
                trace['lon'] = trace.pop('x')
            if 'y' in trace.keys():
                trace['lat'] = trace.pop('y')
            trace['type'] = 'scattermapbox'
            if "line" in trace and isinstance(trace["line"], Line):
                # scattermapboxplot lines do not support dash for some reason, make it a red line instead
                if "dash" in trace["line"]._props:
                    _prps = dict(trace["line"]._props)
                    _prps.pop("dash", None)
                    _prps["color"] = "red"
                    trace["line"] = scmLine(_prps)
                else:
                    trace["line"] = scmLine(dict(trace["line"]._props))
            elif "marker" in trace and isinstance(trace["marker"], Marker):
                trace["marker"] = scmMarker(trace["marker"]._props)

    # setting Figure object
    fig = Figure(data=traces,  # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=(aspectratio == 'auto'),
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
                     legend={'itemsizing': 'constant'},
                     ),
                )
    a = kwargs.get('annotation')
    if a:
        fig.add_annotation(a)

    # check if geodata are real geographical lat/lon coordinates using geopy
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
                                       center=dict(lat=pd.Series(traces[0]['lat']).dropna().mean(),
                                                   lon=pd.Series(traces[0]['lon']).dropna().mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=kwargs.pop('zoomlevel', 11))

    # default aspectratio: if on_map use auto, else use 'original'
    aspectratio = 'original' if not on_map and aspectratio == 'auto' else aspectratio

    if aspectratio != 'auto':
        if aspectratio == 'original':
            # TODO improve this workaround for getting original aspectratio
            xs = []
            ys = []
            for trace in traces:
                xs += trace.get('x') or trace['lon']
                ys += trace.get('y') or trace['lat']
            x_dropna = pd.Series(xs).dropna()
            y_dropna = pd.Series(ys).dropna()
            xrange = x_dropna.max() - x_dropna.min()
            yrange = y_dropna.max() - y_dropna.min()
            ratio = xrange / yrange
            if ratio < 1:
                aspectratio = (ratio, 1.)
            else:
                aspectratio = (1., 1 / ratio)

        aspectratio = np.array(aspectratio) / max(aspectratio)
        fig['layout']['width'], fig['layout']['height'] = ([ar * figsize * 700 for ar in aspectratio])

    # check if called from ipynb or not in order to consider appropriate plot function
    if _in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
        plot(fig, filename=filename)
    else:
        from plotly.offline import plot as plot
        plot(fig, filename=filename, auto_open=auto_open)

    return fig

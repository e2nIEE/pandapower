# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import math

import numpy as np
import pandas as pd
from packaging import version

from pandapower.plotting.plotly.mapbox_plot import _on_map_test
from pandapower.plotting.plotly.traces import _in_ipynb

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
except ImportError:
    logger.info("Failed to import plotly - interactive plotting will not be available")


def version_check():
    if "plotly_version" not in locals() and "plotly_version" not in globals():
        raise UserWarning("You are trying to use plotly, which is not installed.\r\n"
                          "Please upgrade your python-plotly installation, "
                          "e.g., via pip install --upgrade plotly")
    if version.parse(plotly_version) < version.parse("3.1.1"):
        raise UserWarning(f"Your plotly version {plotly_version} is no longer supported.\r\n"
                          "Please upgrade your python-plotly installation, "
                          "e.g., via pip install --upgrade plotly")


def draw_layers(traces, num_layers=0, on_map=False, map_style='basic', showlegend=True, figsize=1,
                aspectratio='auto', filename=None, auto_open=True, **kwargs):
    """
    plots all the traces (which can be created using :func:`create_bus_trace`, :func:`create_line_trace`,
    :func:`create_trafo_trace`)
    to PLOTLY (see https://plot.ly/python/)

    INPUT:
        **traces** - list of dicts which correspond to plotly traces
        generated using: `create_bus_trace`, `create_line_trace`, `create_trafo_trace`

    OPTIONAL:
        **num_layers** (int, 0) - Used for nameing the file if filename is not set.
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

        **filename** (str, None) - plots to a html file called filename if not None.

        **auto_open** (bool, 'True') - automatically open plot in browser

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

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
                     title_font={"size": 16},
                     showlegend=showlegend,
                     autosize=(aspectratio == 'auto'),
                     hovermode='closest',
                     margin={"b": 5, "l": 5, "r": 5, "t": 5},
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False),
                 ),
                 )
    a = kwargs.get('annotation')
    if a:
        fig.add_annotation(a)

    # check if geodata are real geographical lat/lon coordinates using geopy
    if on_map:
        fig['layout']['mapbox'] = {
            "bearing": 0,
            "center": {
                "lat": pd.Series(traces[0]['lat']).dropna().mean(),
                "lon": pd.Series(traces[0]['lon']).dropna().mean()
            },
            "style": map_style,
            "pitch": 0,
            "zoom": kwargs.pop('zoomlevel', 11)
        }

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
        if filename is not None:
            plot(fig, filename=filename)
        else:
            plot(fig, filename=f'Layers_{num_layers}.html')
    else:
        from plotly.offline import plot as plot
        if filename is not None:
            plot(fig, filename=filename, auto_open=auto_open)
        else:
            plot(fig, filename=f'Layers_{num_layers}.html', auto_open=auto_open)

    return fig

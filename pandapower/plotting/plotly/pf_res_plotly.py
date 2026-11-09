# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = ["pf_res_plotly"]

import logging
from typing import Literal

import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.geo import convert_crs
from pandapower.plotting.plotly.traces import (
    create_bus_trace,
    create_line_trace,
    create_trafo_trace,
    draw_traces,
)
from pandapower.run import runpp

logger = logging.getLogger(__name__)


def pf_res_plotly(
    net: pandapowerNet,
    cmap: str = "Jet",
    use_line_geo: bool | None = None,
    on_map: bool = False,
    projection: int | None = None,
    map_style: str = "basic",
    figsize: int = 1,
    aspectratio: str = "auto",
    line_width: int = 2,
    bus_size: int = 10,
    climits_volt: tuple[float, float] = (0.9, 1.1),
    climits_load: tuple[float, float] = (0, 100),
    cpos_volt: float = 1.0,
    cpos_load: float = 1.1,
    filename: str = "temp-plot.html",
    auto_open: bool = True,
    power_unit: Literal["", "k", "M"] = "M",
    current_unit: Literal["", "k"] = "k",
    voltage_unit: Literal["", "k"] = "k",
    zoomlevel: int = 11,
    plot_out_of_service: bool = True,
):
    """
    Plots a pandapower network in plotly

    using colormap for coloring lines according to line loading and buses according to voltage in p.u.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    Parameters:
        net: The pandapower format network.
        cmap (str, True): name of the colormap
        use_line_geo: Should line geodata be used for plotting lines. If None: it will automatically decide.
        on_map (bool, False): enables using mapLibre plot in plotly. If provided geodata are not real
            geo-coordinates in lon/lat form, on_map will be set to False.
        projection (String, None): defines a projection from which network geo-data will be transformed to lat-long.
            For each projection a string can be found at https://spatialreference.org/ref/epsg/
        map_style (str, 'basic'): enables using mapLibre plot in plotly

            - 'basic'
            - 'carto-darkmatter'
            - 'carto-darkmatter-nolabels'
            - 'carto-positron'
            - 'carto-positron-nolabels'
            - 'carto-voyager'
            - 'carto-voyager-nolabels'
            - 'dark'
            - 'satellite'

        figsize: aspectratio is multiplied by it in order to get final image size
        aspectratio: when 'auto' it preserves original aspect ratio of the network geodata
            any custom aspectration can be given as a tuple, e.g. (1.2, 1)
        line_width: width of lines
        bus_size:  size of buses to plot.
        climits_volt: limits of the colorbar for voltage
        climits_load: limits of the colorbar for line_loading
        cpos_volt: position of the bus voltage colorbar
        cpos_load: position of the loading percent colorbar
        filename: filename / path to plot to. Should end on `*.html`
        auto_open: automatically open plot in browser
        power_unit: default unit of displayed P, Q, S data ["", "k", "M"]
        current_unit: default unit of displayed I data ["", "k"]
        voltage_unit: default unit of displayed V data ["", "k"]
        zoomlevel: initial zoomlevel of map plot (only if on_map=True)
        plot_out_of_service: should out of service elements be plotted?

    Returns:
        figure (graph_objs._figure.Figure): figure object
    """
    if "res_bus" not in net or net.res_bus.shape[0] == 0:
        logger.warning("There are no Power Flow results. A Newton-Raphson power flow will be executed.")
        runpp(net)

    # create geocoord if none are available
    if any(net.line.geo.isna()) and any(net.bus.geo.isna()):
        logger.warning(
            "No or insufficient geodata available --> Creating artificial coordinates." + " This may take some time"
        )
        create_generic_coordinates(net, respect_switches=True)
        if on_map:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographical lat/lon coordinates using geopy
    if on_map and projection is not None:
        convert_crs(net, epsg_in=projection, epsg_out=4326)

    # ----- Buses ------
    # initializing bus trace
    # hoverinfo which contains name and pf results
    precision = 3

    voltage_factor: float
    if voltage_unit == "k":
        voltage_factor = 1.0
    else:
        voltage_factor = 1e3

    power_factor: float
    if power_unit == "M":
        power_factor = 1.0
    elif power_unit == "k":
        power_factor = 1e3
    else:
        power_factor = 1e6

    bus_ids = net.bus.index
    if not plot_out_of_service:
        bus_ids = bus_ids[net.bus.in_service is True]
    hoverinfo = (
        net.bus.name.astype(str)
        + "<br />V_m = "
        + net.res_bus.vm_pu.round(precision).astype(str)
        + " pu<br />V_m = "
        + (net.res_bus.vm_pu * net.bus.vn_kv.round(2) * voltage_factor).round(precision).astype(str)
        + f" {voltage_unit}V<br />V_a = "
        + net.res_bus.va_degree.round(precision).astype(str)
        + " deg<br />P = "
        + (net.res_bus.p_mw * power_factor).round(precision).astype(str)
        + f" {power_unit}W<br />Q = "
        + (net.res_bus.q_mvar * power_factor).round(precision).astype(str)
        + f" {power_unit}Var"
    ).tolist()
    hoverinfo = pd.Series(index=net.bus.index, data=hoverinfo)
    bus_trace = create_bus_trace(
        net,
        bus_ids,
        size=bus_size,
        infofunc=hoverinfo[bus_ids],
        cmap=cmap,
        cbar_title="Bus Voltage [pu]",
        cmin=climits_volt[0],
        cmax=climits_volt[1],
        cpos=cpos_volt,
    )

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    # if bus geodata is available, but no line geodata
    cmap_lines = "jet" if cmap == "Jet" else cmap
    if use_line_geo is None:
        use_line_geo = not any(net.line.geo.isna())
    elif use_line_geo and any(net.line.geo.isna()):
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geo = False

    # hoverinfo that contains name and pf results
    current_factor: float
    if current_unit == "k":
        current_factor = 1.0
    else:
        current_factor = 1e3

    line_ids = net.line.index
    if not plot_out_of_service:
        line_ids = line_ids[net.line.in_service is True]
    hoverinfo = (
        net.line.name.astype(str)
        + "<br />I = "
        + net.res_line.loading_percent.round(precision).astype(str)
        + " %<br />I_from = "
        + (net.res_line.i_from_ka * current_factor).round(precision).astype(str)
        + f" {current_unit}A<br />I_to = "
        + (net.res_line.i_to_ka * current_factor).round(precision).astype(str)
        + f" {current_unit}A<br />P_from = "
        + (net.res_line.p_from_mw * power_factor).round(precision).astype(str)
        + f" {power_unit}W<br />P_to = "
        + (net.res_line.p_to_mw * power_factor).round(precision).astype(str)
        + f" {power_unit}W<br />Q_from = "
        + (net.res_line.q_from_mvar * power_factor).round(precision).astype(str)
        + f" {power_unit}Var<br />Q_to = "
        + (net.res_line.q_to_mvar * power_factor).round(precision).astype(str)
        + f" {power_unit}Var"
    ).tolist()
    hoverinfo = pd.Series(index=net.line.index, data=hoverinfo)
    line_traces = create_line_trace(
        net,
        line_ids,
        use_line_geo=use_line_geo,
        respect_switches=True,
        width=line_width,
        infofunc=hoverinfo[line_ids],
        cmap=cmap_lines,
        cmap_vals=net.res_line["loading_percent"].values,
        cmin=climits_load[0],
        cmax=climits_load[1],
        cbar_title="Line Loading [%]",
        cpos=cpos_load,
    )

    # ----- Trafos ------
    trafo_ids = net.trafo.index
    if not plot_out_of_service:
        trafo_ids = trafo_ids[net.trafo.in_service is True]
    # hoverinfo which contains name and pf results
    hoverinfo = (
        net.trafo.name.astype(str)
        + "<br />I = "
        + net.res_trafo.loading_percent.round(precision).astype(str)
        + f" %<br />I_hv = "
        + (net.res_trafo.i_hv_ka * current_factor).round(precision).astype(str)
        + f" {current_unit}A<br />I_lv = "
        + (net.res_trafo.i_lv_ka * current_factor).round(precision).astype(str)
        + f" {current_unit}A<br />P_hv = "
        + (net.res_trafo.p_hv_mw * power_factor).round(precision).astype(str)
        + f" {power_unit}W<br />P_lv = "
        + (net.res_trafo.p_lv_mw * power_factor).round(precision).astype(str)
        + f" {power_unit}W<br />Q_hv = "
        + (net.res_trafo.q_hv_mvar * power_factor).round(precision).astype(str)
        + f" {power_unit}Var<br />Q_lv = "
        + (net.res_trafo.q_lv_mvar * power_factor).round(precision).astype(str)
        + f" {power_unit}Var<br />"
    ).tolist()
    hoverinfo = pd.Series(index=net.trafo.index, data=hoverinfo)
    trafo_traces = create_trafo_trace(
        net, trafo_ids, width=line_width * 1.5, infofunc=hoverinfo[trafo_ids], cmap=cmap_lines, cmin=0, cmax=100
    )

    # ----- Ext grid ------
    ext_grid_trace = []
    # get external grid from create_bus_trace
    if "ext_grid" in net and len(net.ext_grid):
        ext_grid_ids = net.ext_grid.index
        if not plot_out_of_service:
            ext_grid_ids = ext_grid_ids[net.ext_grid.in_service is True]
        marker_type = "circle" if on_map else "square"
        ext_grid_trace = create_bus_trace(
            net,
            buses=net.ext_grid[ext_grid_ids].bus,
            color="grey",
            size=bus_size * 2,
            trace_name="external_grid",
            patch_type=marker_type,
        )

    return draw_traces(
        line_traces + trafo_traces + ext_grid_trace + bus_trace,
        showlegend=False,
        aspectratio=aspectratio,
        on_map=on_map,
        map_style=map_style,
        figsize=figsize,
        filename=filename,
        auto_open=auto_open,
        zoomlevel=zoomlevel,
    )

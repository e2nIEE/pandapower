# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import sys
import math
import logging
from collections import defaultdict

import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

from pandapower.auxiliary import soft_dependency_error, pandapowerNet
from pandapower.plotting.plotting_toolbox import get_collection_sizes
from pandapower.plotting.collections import (
    create_bus_collection,
    create_line_collection,
    create_trafo_collection,
    create_trafo3w_collection,
    create_line_switch_collection,
    draw_collections,
    create_bus_bus_switch_collection,
    create_ext_grid_collection,
    create_sgen_collection,
    create_gen_collection,
    create_load_collection,
    create_dcline_collection,
    create_vsc_collection,
)
from pandapower.plotting.generic_geodata import create_generic_coordinates

logger = logging.getLogger(__name__)


def bus_info(bus):
    return ("bus", bus)


def line_info(line):
    return ("line", line)


def trafo_info(idx):
    return ("trafo", idx)


def trafo3w_info(idx):
    return ("trafo3w", idx)


def hover(event, ax, net, hover_text):
    """
    Update the hover text in an interactive pandapower plot based on the mouse position.

    Expects collections to have an `info` attribute containing a list of
    (element, index) tuples, e.g. ("bus", 3) or ("line", 5).

    Parameters
    ----------
    event : matplotlib.backend_bases.MouseEvent
        Mouse-move event from Matplotlib.
    ax : matplotlib.axes.Axes
        Axes object containing the collections.
    net : pp.pandapowerNet
        pandapower network with DataFrames (bus, line, trafo, trafo3w, ...).
    hover_text : matplotlib.text.Text
        Text artist whose content, position and visibility are updated.
    """
    fig = ax.figure
    visible = hover_text.get_visible()

    if event.inaxes is not ax:
        if visible:
            hover_text.set_visible(False)
            fig.canvas.draw_idle()
        return

    for collection in ax.collections:
        info = getattr(collection, "info", None)
        if not info:
            continue

        contains, props = collection.contains(event)
        if not contains or "ind" not in props or len(props["ind"]) == 0:
            continue

        coll_idx = props["ind"][0]
        element_info = info[coll_idx]

        if isinstance(element_info, tuple) and len(element_info) == 2:
            element, idx = element_info
        else:
            element, idx = str(element_info), None

        df = getattr(net, element, None)

        if df is not None and idx is not None and idx in df.index and "name" in df.columns:
            name = df.at[idx, "name"]
            hover_info = f"{element}: {name} | Index: {idx}"
        elif idx is not None:
            hover_info = f"{element} | Index: {idx}"
        else:
            hover_info = str(element_info)

        # text and position
        hover_text.set_text(hover_info)
        hover_text.set_position((event.xdata, event.ydata))
        hover_text.set_visible(True)
        fig.canvas.draw_idle()
        return

    if visible:
        hover_text.set_visible(False)
        fig.canvas.draw_idle()


def simple_plot(
        net: pandapowerNet,
        respect_switches: bool = False,
        line_width: float = 2.0,
        bus_size: float = 1.0,
        ext_grid_size: float = 1.0,
        trafo_size: float = 1.0,
        plot_loads: bool = False,
        plot_gens: bool = False,
        plot_sgens: bool = False,
        orientation=None,
        load_size: float = 1.0,
        gen_size: float = 1.0,
        sgen_size: float = 1.0,
        switch_size: float = 2.0,
        switch_distance: float = 1.0,
        plot_line_switches: bool = False,
        scale_size: bool = True,
        bus_color="#1c3f52",
        line_color="grey",
        dcline_color="c",
        trafo_color="k",
        ext_grid_color="#179c7d",
        switch_color="k",
        library="igraph",
        show_plot: bool = True,
        ax=None,
        draw_by_type: bool=True,
        bus_dc_size: float = 1.0,
        bus_dc_color="m",
        line_dc_color="c",
        vsc_size: float = 4.0,
        vsc_color="orange",
        highlight_buses=None,
        highlight_lines=None,
        enable_hover=True,
        highlight_bus_size_factor=1.5,
        highlight_line_width_factor=2.0,
        highlight_color="#f58220"
):
    """
        Plots a pandapower network as simple as possible. If no geodata is available, artificial
        geodata is generated. For advanced plotting see the tutorial

        Parameters:
            net: The pandapower format network.
            respect_switches (bool, False) Respect switches if artificial geodata is created. This Flag is ignored if
                plot_line_switches is True
            line_width (float, 1.0): width of lines
            bus_size (float, 1.0): Relative size of buses to plot. The value bus_size is multiplied with
                mean_distance_between_buses, which equals the distance between the max geocoord and the min divided by
                200. mean_distance_between_buses = sum((net.bus.geo.max() - net.bus.geo.min()) / 200)
            ext_grid_size (float, 1.0): Relative size of ext_grids to plot. See bus sizes for details. Note: ext_grids
                are plotted as rectangles
            trafo_size (float, 1.0): Relative size of trafos to plot.
            plot_loads (bool, False): Flag to decide whether load symbols should be drawn.
            plot_gens (bool, False): Flag to decide whether gen symbols should be drawn.
            plot_sgens (bool, False): Flag to decide whether sgen symbols should be drawn.
            load_size (float, 1.0): Relative size of loads to plot.
            sgen_size (float, 1.0): Relative size of sgens to plot.
            switch_size (float, 2.0): Relative size of switches to plot. See bus size for details
            switch_distance (float, 1.0): Relative distance of the switch to its corresponding bus. See bus size for
                details
            plot_line_switches (bool, False): Flag if line switches are plotted
            scale_size (bool, True): Flag if bus_size, ext_grid_size, bus_size- and distance will be scaled with respect
                to grid mean distances
            bus_color (String, colors[0]): Bus Color. Init as first value of color palette. Usually colors[0] = "b".
            line_color (String, 'grey'): Line Color. Init is grey
            dcline_color (String, 'c'): Line Color. Init is cyan
            trafo_color (String, 'k'): Trafo Color. Init is black
            ext_grid_color (String, 'y'): External Grid Color. Init is yellow
            switch_color (String, 'k'): Switch Color. Init is black
            library (String, "igraph"): library name to create generic coordinates (case of missing geodata). "igraph"
                to use igraph package or "networkx" to use networkx package.
            show_plot (bool, True): Shows plot at the end of plotting
            ax (object, None): matplotlib axis to plot to
            highlight_buses (iterable, None): buses, to highlight
            highlight_lines (iterable, None): lines to highlight
            enable_hover (bool, True): enable hovering functionality
            highlight_bus_size_factor (float, 1.5): bus_size for highlighted buses
            highlight_line_width_factor (float, 2.0): line_width for highlighted lines
            highlight_color (str, "r"): color for highlighted elements
        Returns:
            axes of figure
    """
    try:
        if hasattr(net, "bus_geodata") or hasattr(net, "line_geodata"):
            raise UserWarning(
                """The supplied network uses an outdated geodata format. Please update your geodata by
                   \rrunning `pandapower.plotting.geo.convert_geodata_to_geojson(net)`"""
            )
    except UserWarning as e:
        logger.warning(e)

    # don't hide lines if switches are plotted
    if plot_line_switches:
        respect_switches = False

    # create geocoord if none are available
    if (len(net.line.geo) == 0 and len(net.bus.geo) == 0) or (
            net.line.geo.isna().any() and net.bus.geo.isna().any()):
        logger.warning(
            "No or insufficient geodata available --> Creating artificial coordinates." +
            " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches, library=library)

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = get_collection_sizes(
            net,
            bus_size,
            ext_grid_size,
            trafo_size,
            load_size,
            sgen_size,
            switch_size,
            switch_distance,
            gen_size,
        )
        bus_size = sizes["bus"]
        ext_grid_size = sizes["ext_grid"]
        trafo_size = sizes["trafo"]
        sgen_size = sizes["sgen"]
        load_size = sizes["load"]
        switch_size = sizes["switch"]
        switch_distance = sizes["switch_distance"]
        gen_size = sizes["gen"]

    bc = create_bus_collection(net, net.bus.index, size=bus_size, color=bus_color,
                               zorder=10, infofunc=bus_info)
    collections = [bc]

    if highlight_buses is not None:
        hl_buses_idx = list(set(highlight_buses) & set(net.bus.index))
        if len(hl_buses_idx):
            hbc = create_bus_collection(
                net, hl_buses_idx,
                size=bus_size * highlight_bus_size_factor,
                color=highlight_color,
                zorder=bc.zorder + 1 if hasattr(bc, "zorder") else 11,
                infofunc=bus_info)
            collections.append(hbc)

    # if bus geodata is available, but no line geodata
    use_bus_geodata = len(net.line.geo.dropna()) == 0
    in_service_lines = net.line[net.line.in_service].index
    nogolines = (
        set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)])
        if respect_switches
        else set()
    )
    plot_lines = in_service_lines.difference(nogolines)
    plot_dclines = net.dcline.in_service
    plot_lines_dc = net.line_dc.loc[net.line_dc.in_service].index

    lc = create_line_collection(
        net,
        plot_lines,
        color=line_color,
        linewidths=line_width,
        use_bus_geodata=use_bus_geodata,
        infofunc=line_info)
    collections.append(lc)

    if highlight_lines is not None:
        hl_lines_idx = list(set(highlight_lines) & set(plot_lines))
        if len(hl_lines_idx):
            hlc = create_line_collection(
                net,
                hl_lines_idx,
                color=highlight_color,
                linewidths=line_width * highlight_line_width_factor,
                use_bus_geodata=use_bus_geodata,
                infofunc=line_info)
            collections.append(hlc)

    # create dcline collections
    if len(net.dcline) > 0:
        dclc = create_dcline_collection(
            net,
            plot_dclines,
            color=dcline_color,
            linewidths=line_width
        )
        collections.append(dclc)

    # create bus dc collection
    if len(net.bus_dc) > 0:
        bc_dc = create_bus_collection(
            net,
            net.bus_dc.index,
            size=bus_dc_size,
            color=bus_dc_color,
            zorder=10,
            bus_table="bus_dc",
        )
        collections.append(bc_dc)

    # create VSC collection
    if len(net.vsc) > 0:
        vsc_ac = create_vsc_collection(
            net, net.vsc.index, size=vsc_size, color=vsc_color, zorder=12
        )
        collections.append(vsc_ac)

    # create line_dc collections
    if len(net.line_dc) > 0:
        lc_dc = create_line_collection(
            net,
            plot_lines_dc,
            color=line_dc_color,
            linewidths=line_width,
            use_bus_geodata=use_bus_geodata,
            line_table="line_dc",
        )
        collections.append(lc_dc)

    # create ext_grid collections
    if len(net.ext_grid) > 0:
        sc = create_ext_grid_collection(
            net,
            size=ext_grid_size,
            orientation=0,
            ext_grids=net.ext_grid.index,
            patch_edgecolor=ext_grid_color,
            zorder=11,
        )
        collections.append(sc)

    # create trafo collection if trafo is available
    trafo_buses_with_geo_coordinates = [
        t
        for t, trafo in net.trafo.iterrows()
        if trafo.hv_bus in net.bus.geo.index and trafo.lv_bus in net.bus.geo.index
    ]
    if len(trafo_buses_with_geo_coordinates) > 0:
        tc = create_trafo_collection(
            net,
            trafo_buses_with_geo_coordinates,
            color=trafo_color,
            size=trafo_size,
            infofunc=trafo_info
        )
        collections.append(tc)

    # create trafo3w collection if trafo3w is available
    trafo3w_buses_with_geo_coordinates = [
        t
        for t, trafo3w in net.trafo3w.iterrows()
        if trafo3w.hv_bus in net.bus.geo.index and
           trafo3w.mv_bus in net.bus.geo.index and
           trafo3w.lv_bus in net.bus.geo.index
    ]
    if len(trafo3w_buses_with_geo_coordinates) > 0:
        tc = create_trafo3w_collection(net, trafo3w_buses_with_geo_coordinates,
                                       color=trafo_color, infofunc=trafo3w_info)
        collections.append(tc)

    if plot_line_switches and len(net.switch):
        sc = create_line_switch_collection(
            net,
            size=switch_size,
            distance_to_bus=switch_distance,
            use_line_geodata=not use_bus_geodata,
            zorder=12,
            color=switch_color,
        )
        collections.append(sc)

    angles = calculate_unique_angles(net) if draw_by_type else None

    if plot_sgens and len(net.sgen):
        sgc = create_sgen_collection(
            net,
            size=sgen_size,
            orientation=orientation,
            unique_angles=angles,
            draw_by_type=draw_by_type,
        )
        collections.append(sgc)

    if plot_gens and len(net.gen):
        gc = create_gen_collection(
            net,
            size=gen_size,
            orientation=orientation,
            unique_angles=angles,
            draw_by_type=draw_by_type
        )
        collections.append(gc)

    if plot_loads and len(net.load):
        lc = create_load_collection(
            net, size=load_size, orientation=orientation, unique_angles=angles
        )
        collections.append(lc)

    if len(net.switch):
        bsc = create_bus_bus_switch_collection(net, size=switch_size)
        collections.append(bsc)

    ax = draw_collections(collections, ax=ax)

    if enable_hover:
        fig = ax.figure
        hover_text = ax.text(0, 0, "", fontsize=12, fontweight="bold", color='white',
                             ha='center', va='center', zorder=99,
                             bbox={"boxstyle": "round",
                                   "facecolor": '#179c7d',
                                   "alpha": 1,
                                   "edgecolor": 'white'})
        hover_text.set_visible(False)
        fig.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: hover(event, ax, net, hover_text)
        )

    if show_plot:
        if not MATPLOTLIB_INSTALLED:
            soft_dependency_error(
                str(sys._getframe().f_code.co_name) + "()", "matplotlib"
            )
        plt.show()
    return ax


def calculate_unique_angles(net: pandapowerNet) -> dict[int, dict[str, dict[str, float] | float]]:
    """
    Calculate the angles for each patch at each bus. (currently only respects sgen, gen and load)
    Only a single patch for all loads is currently supported.

    :param pandapowerNet net: the network to calculate angles for
    :returns: a dictionary containing layout information for each patch at bus, load has only one patch at bottom.
    :rtype: dict[int, dict[str, Union[dict[str, float], float]]]
    """
    sgen_counts = net.sgen.groupby(['bus', 'type'], dropna=False).size().unstack(fill_value=0)
    gen_counts = net.gen.groupby(['bus', 'type'], dropna=False).size().unstack(fill_value=0)
    loads = pd.Series(1, index=net.load.bus.unique(), name='load')

    patch_counts = pd.concat([sgen_counts, gen_counts, loads], axis=1).fillna(0)
    patches_per_bus = patch_counts.ne(0).sum(axis=1)

    patches: dict[int, dict[str, dict[str, float] | float]] = defaultdict(dict)
    counts: dict[int, int] = defaultdict(int)
    for df, df_name in [(sgen_counts, "sgen"), (gen_counts, "gen")]:
        index: int
        for index, row in df.iterrows():
            patch_angle = float(2 * math.pi / patches_per_bus[index])
            c: str | float
            for c, v in row.items():
                _type: str
                if v > 0:
                    if isinstance(c, float) and math.isnan(c):
                        _type = "none"
                    else:
                        _type = str(c)
                    if df_name not in patches[index]:
                        patches[index][df_name] = {}
                    patches[index][df_name][_type] = patch_angle * counts[index]
                    counts[index] += 1
    for index, _ in loads.items():
        patch_angle = float(2 * math.pi / patches_per_bus[index])
        patches[index]['load'] = patch_angle * counts[index]
        counts[index] += 1
    return patches

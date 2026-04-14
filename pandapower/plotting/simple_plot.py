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


_LINE_PALETTE: list[str] = ["#b2d235", "#fdb913", "#f58220", "#bb0056"]
_BUS_PALETTE:  list[str] = ["#1c3f52", "#179c7d", "#179c7d", "#fdb913", "#bb0056"]

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


# ── module-level palette / colormap / toggle helpers ─────────────────────────
def _pick_n_colors(n: int, palette: list[str]) -> list[str]:
    """
    Sample *n* colors evenly from *palette* using floor-based index mapping.
    First and last palette entries are always included.
    """
    if n <= 0:
        return []
    if n == 1:
        return [palette[0]]
    if n >= len(palette):
        return (list(palette) + [palette[-1]] * (n - len(palette)))[:n]
    step = (len(palette) - 1) / (n - 1)
    return [palette[math.floor(i * step)] for i in range(n)]


def _build_cmap_from_limits(
    limits: tuple | list,
    colormap_type: str,
    kind: str = "line",
) -> list:
    """
    Build a pandapower cmap_list from sorted numeric breakpoints.

    Parameters
    ----------
    limits :
        Sorted breakpoints, e.g. ``(0, 50, 100)`` for lines or
        ``(0.9, 0.95, 1.0, 1.05, 1.1)`` for buses.
    colormap_type : str
        ``"discrete"`` or ``"continuous"``.
    kind : str
        ``"line"`` uses a green→red palette;
        ``"bus"``  uses a blue→green→red palette.
    """
    palette = _LINE_PALETTE if kind == "line" else _BUS_PALETTE
    n = len(limits)
    if colormap_type == "discrete":
        colors = _pick_n_colors(n - 1, palette)
        return [((limits[i], limits[i + 1]), colors[i]) for i in range(n - 1)]
    colors = _pick_n_colors(n, palette)
    return [(limits[i], colors[i]) for i in range(n)]


def _toggle_colormap_cb(event, state, ax, normal_colls, cmap_colls, colorbars, button):
    """
    Swap between Normal (flat-color) and Colormap view.

    Why two collections + set_visible (not in-place color patching)
    ────────────────────────────────────────────────────────────────
    ScalarMappable.update_scalarmappable() is called on every canvas.draw()
    and rewrites _facecolors from _A + _cmap + _norm for any collection
    whose scalar array is set.  In-place patching via set_facecolor/set_color
    is therefore silently discarded on the very next draw cycle.  Toggling
    visibility between two separately created and fully initialised
    collections is the only approach that survives repeated draw() calls.

    Why copy_collections=False is required in draw_collections
    ──────────────────────────────────────────────────────────
    draw_collections shallow-copies every collection into ax.collections by
    default.  set_visible() must target the exact Python objects that live in
    ax.collections; copy_collections=False guarantees identity.

    Why figure resize + colorbar repositioning is needed
    ─────────────────────────────────────────────────────
    plt.colorbar(ax=ax) shrinks the main ax to make room for colorbars within
    the same figure width.  After hiding the colorbars in Normal mode the ax
    stays at the shrunken position, leaving an empty strip on the right.
    The fix is two-fold:
      1. Restore ax to full-width bounds (normal_ax_pos) in Normal mode.
      2. Expand figure width in Colormap mode so the network keeps the same

         absolute pixel size; colorbar axes positions are recomputed to
         maintain the same absolute gap from the ax right edge.

    state dict keys
    ───────────────
    active              bool   True = Colormap mode currently visible
    normal_ax_pos       list   ax bounds [x0, y0, w, h] – Normal mode
    cmap_ax_pos         list   ax bounds [x0, y0, w, h] – Colormap mode
    normal_figsize      tuple  figure (width, height) in inches – Normal mode
    cmap_figsize        tuple  figure (width, height) in inches – Colormap mode
    cmap_cbar_positions list   precomputed colorbar bounds for the wider figure
    """
    state["active"] = not state["active"]
    active = state["active"]
    fig = ax.figure

    for c in normal_colls:
        c.set_visible(not active)
    for c in cmap_colls:
        c.set_visible(active)

    if active:  # → Colormap mode: widen figure, restore shrunken ax, show colorbars
        fig.set_size_inches(state["cmap_figsize"], forward=True)
        ax.set_position(state["cmap_ax_pos"])
        for cbar, pos in zip(colorbars, state["cmap_cbar_positions"]):
            cbar.ax.set_visible(True)
            cbar.ax.set_position(pos)
    else:  # → Normal mode: hide colorbars, restore full-width ax, shrink figure
        for cbar in colorbars:
            cbar.ax.set_visible(False)
        ax.set_position(state["normal_ax_pos"])
        fig.set_size_inches(state["normal_figsize"], forward=True)

    # label = what the NEXT click will switch TO
    button.label.set_text("Normal" if active else "Colormap")
    fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────


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
        draw_by_type: bool = True,
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
        highlight_color="#f58220",
        # ── colormap parameters ──────────────────────────────────────────────
        colormap_type: str = "continuous",
        line_limits: tuple = (0, 25, 50, 75, 100),
        bus_limits: tuple = (0.9, 0.95, 1.0, 1.05, 1.1),
        cmap_lines: list = None,
        cmap_buses: list = None,
        plot_colorbars: bool = True,
):
    """
    Plots a pandapower network as simple as possible. If no geodata is available,
    artificial geodata is generated. For advanced plotting see the tutorial.

    A "Colormap / Normal" toggle button is always added to the figure when
    load-flow results are available (or can be computed).  The plot starts in
    Normal (flat-color) mode.  Clicking the button switches to a colormap view
    that colors buses by vm_pu and lines by loading_percent; clicking again
    reverts to Normal mode.  The figure window resizes automatically:

    * Normal mode  – compact figure, ax fills the full width (no reserved

      colorbar space).
    * Colormap mode – figure is widened so the network keeps the same absolute

      pixel size; the extra width accommodates the colorbars.

    If result tables are empty, pp.runpp(net) is called automatically.  If the
    load flow fails or the colormaps module cannot be imported, the toggle
    button is omitted and only Normal mode is available.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network to plot.
    respect_switches : bool, default False
        Respect switches when creating artificial geodata.
        Ignored when ``plot_line_switches=True``.
    line_width : float, default 2.0
        Width of line segments.
    bus_size : float, default 1.0
        Relative bus marker size (scaled by mean bus distance).
    ext_grid_size : float, default 1.0
        Relative ext_grid symbol size.
    trafo_size : float, default 1.0
        Relative trafo symbol size.
    plot_loads : bool, default False
        Draw load symbols.
    plot_gens : bool, default False
        Draw generator symbols.
    plot_sgens : bool, default False
        Draw static generator symbols.
    load_size : float, default 1.0
        Relative load symbol size.
    gen_size : float, default 1.0
        Relative gen symbol size.
    sgen_size : float, default 1.0
        Relative sgen symbol size.
    switch_size : float, default 2.0
        Relative switch symbol size.
    switch_distance : float, default 1.0
        Relative switch distance from its bus.
    plot_line_switches : bool, default False
        Draw line switch symbols.
    scale_size : bool, default True
        Scale all symbol sizes relative to mean bus geodistance.
    bus_color : str, default "#1c3f52"
        Flat bus marker color used in Normal mode.
    line_color : str, default "grey"
        Flat line color used in Normal mode.
    dcline_color : str, default "c"
        DC line color.
    trafo_color : str, default "k"
        Transformer symbol color.
    ext_grid_color : str, default "#179c7d"
        External grid symbol color.
    switch_color : str, default "k"
        Switch symbol color.
    library : str, default "igraph"
        Layout library for generic coordinates (``"igraph"`` or ``"networkx"``).
    show_plot : bool, default True
        Call ``plt.show()`` at the end.
    ax : matplotlib.axes.Axes or None
        Existing axes to draw into.
    draw_by_type : bool, default True
        Group sgen/gen symbols by element type.
    bus_dc_size : float, default 1.0
        Relative DC bus marker size.
    bus_dc_color : str, default "m"
        DC bus marker color.
    line_dc_color : str, default "c"
        DC line color.
    vsc_size : float, default 4.0
        Relative VSC symbol size.
    vsc_color : str, default "orange"
        VSC symbol color.
    highlight_buses : iterable or None
        Bus indices to highlight.
    highlight_lines : iterable or None
        Line indices to highlight.
    enable_hover : bool, default True
        Enable interactive hover labels.
    highlight_bus_size_factor : float, default 1.5
        Size multiplier for highlighted buses.
    highlight_line_width_factor : float, default 2.0
        Line-width multiplier for highlighted lines.
    highlight_color : str, default "#f58220"
        Color for highlighted elements.
    colormap_type : str, default "continuous"
        ``"discrete"`` – flat color bands; ``"continuous"`` – smooth gradient.
    line_limits : tuple, default (0, 50, 100)
        Breakpoints for line loading in **%**.  Colors are auto-generated from
        a green→red palette.  Overridden entirely by ``cmap_lines`` if given.
    bus_limits : tuple, default (0.9, 0.95, 1.0, 1.05, 1.1)
        Breakpoints for bus voltage in **p.u.**.  Colors are auto-generated from
        a blue→green→red palette.  Overridden entirely by ``cmap_buses`` if given.
    cmap_lines : list or None
        Full custom colormap definition for lines; overrides ``line_limits``.
        Discrete:   ``[((min, max), color), …]``
        Continuous: ``[(value, color), …]``
    cmap_buses : list or None
        Full custom colormap definition for buses; overrides ``bus_limits``.
        Same format as ``cmap_lines``.
    plot_colorbars : bool, default True
        Show colorbars in the Colormap view.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        if hasattr(net, "bus_geodata") or hasattr(net, "line_geodata"):
            raise UserWarning(
                """The supplied network uses an outdated geodata format. Please update your geodata by
                   \rrunning `pandapower.plotting.geo.convert_geodata_to_geojson(net)`"""
            )
    except UserWarning as e:
        logger.warning(e)

    # line switches being plotted requires all lines to be visible
    if plot_line_switches:
        respect_switches = False

    # create generic coordinates if no geodata is available
    if (len(net.line.geo) == 0 and len(net.bus.geo) == 0) or (
            net.line.geo.isna().any() and net.bus.geo.isna().any()):
        logger.warning(
            "No or insufficient geodata available --> Creating artificial coordinates."
            " This may take some time"
        )
        create_generic_coordinates(net, respect_switches=respect_switches, library=library)

    if scale_size:
        # scale all symbol sizes relative to the mean distance between buses
        sizes = get_collection_sizes(
            net, bus_size, ext_grid_size, trafo_size,
            load_size, sgen_size, switch_size, switch_distance, gen_size,
        )
        bus_size        = sizes["bus"]
        ext_grid_size   = sizes["ext_grid"]
        trafo_size      = sizes["trafo"]
        sgen_size       = sizes["sgen"]
        load_size       = sizes["load"]
        switch_size     = sizes["switch"]
        switch_distance = sizes["switch_distance"]
        gen_size        = sizes["gen"]

    # ── colormap setup ────────────────────────────────────────────────────────
    # Always attempt colormap preparation so the toggle button can be offered.
    cmap_l = norm_l = cmap_b = norm_b = None
    colormap_ready = False

    try:
        if net.res_bus.empty or net.res_line.empty:
            logger.info("Result tables empty – running pp.runpp(net) automatically.")
            import pandapower as pp
            pp.runpp(net)

        from pandapower.plotting.colormaps import cmap_discrete, cmap_continuous

        if colormap_type == "discrete":
            _cl = cmap_lines or _build_cmap_from_limits(line_limits, "discrete", "line")
            _cb = cmap_buses or _build_cmap_from_limits(bus_limits,  "discrete", "bus")
            cmap_l, norm_l = cmap_discrete(_cl)
            cmap_b, norm_b = cmap_discrete(_cb)
            colormap_ready = True

        elif colormap_type == "continuous":
            _cl = cmap_lines or _build_cmap_from_limits(line_limits, "continuous", "line")
            _cb = cmap_buses or _build_cmap_from_limits(bus_limits,  "continuous", "bus")
            cmap_l, norm_l = cmap_continuous(_cl)
            cmap_b, norm_b = cmap_continuous(_cb)
            colormap_ready = True

        else:
            logger.error(
                f"Unknown colormap_type='{colormap_type}'. "
                "Allowed values: 'discrete', 'continuous'. Toggle disabled."
            )

    except Exception as exc:
        logger.warning(
            f"Colormap setup failed ({exc!r}). "
            "Toggle button will not be shown; plotting in Normal mode only."
        )
    # ─────────────────────────────────────────────────────────────────────────

    # ── bus collections: one flat-color (Normal), one colormap (Colormap) ────
    normal_bc = create_bus_collection(
        net, net.bus.index, size=bus_size,
        color=bus_color, zorder=10, infofunc=bus_info,
    )
    cmap_bc = None
    if colormap_ready:
        cmap_bc = create_bus_collection(
            net, net.bus.index, size=bus_size,
            cmap=cmap_b, norm=norm_b, zorder=10, infofunc=bus_info,
        )

    collections = [normal_bc]
    if cmap_bc is not None:
        collections.append(cmap_bc)
    # ─────────────────────────────────────────────────────────────────────────

    if highlight_buses is not None:
        hl_buses_idx = list(set(highlight_buses) & set(net.bus.index))
        if len(hl_buses_idx):
            hbc = create_bus_collection(
                net, hl_buses_idx,
                size=bus_size * highlight_bus_size_factor,
                color=highlight_color, zorder=11, infofunc=bus_info,
            )
            collections.append(hbc)

    # fall back to bus geodata when no line geodata is present
    use_bus_geodata = len(net.line.geo.dropna()) == 0
    in_service_lines = net.line[net.line.in_service].index
    nogolines = (
        set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)])
        if respect_switches
        else set()
    )
    plot_lines    = in_service_lines.difference(nogolines)
    plot_dclines  = net.dcline.in_service
    plot_lines_dc = net.line_dc.loc[net.line_dc.in_service].index

    # ── line collections: one flat-color (Normal), one colormap (Colormap) ───
    # named normal_lc / cmap_lc to prevent rebinding by create_load_collection
    normal_lc = create_line_collection(
        net, plot_lines,
        color=line_color, linewidths=line_width,
        use_bus_geodata=use_bus_geodata, infofunc=line_info,
    )
    cmap_lc = None
    if colormap_ready:
        cmap_lc = create_line_collection(
            net, plot_lines,
            cmap=cmap_l, norm=norm_l, linewidths=line_width,
            use_bus_geodata=use_bus_geodata, infofunc=line_info,
        )

    collections.append(normal_lc)
    if cmap_lc is not None:
        collections.append(cmap_lc)
    # ─────────────────────────────────────────────────────────────────────────

    if highlight_lines is not None:
        hl_lines_idx = list(set(highlight_lines) & set(plot_lines))
        if len(hl_lines_idx):
            hlc = create_line_collection(
                net, hl_lines_idx,
                color=highlight_color,
                linewidths=line_width * highlight_line_width_factor,
                use_bus_geodata=use_bus_geodata, infofunc=line_info,
            )
            collections.append(hlc)

    if len(net.dcline) > 0:
        dclc = create_dcline_collection(
            net, plot_dclines, color=dcline_color, linewidths=line_width
        )
        collections.append(dclc)

    if len(net.bus_dc) > 0:
        bc_dc = create_bus_collection(
            net, net.bus_dc.index, size=bus_dc_size, color=bus_dc_color,
            zorder=10, bus_table="bus_dc",
        )
        collections.append(bc_dc)

    if len(net.vsc) > 0:
        vsc_ac = create_vsc_collection(
            net, net.vsc.index, size=vsc_size, color=vsc_color, zorder=12
        )
        collections.append(vsc_ac)

    if len(net.line_dc) > 0:
        lc_dc = create_line_collection(
            net, plot_lines_dc, color=line_dc_color, linewidths=line_width,
            use_bus_geodata=use_bus_geodata, line_table="line_dc",
        )
        collections.append(lc_dc)

    if len(net.ext_grid) > 0:
        sc = create_ext_grid_collection(
            net, size=ext_grid_size, orientation=0,
            ext_grids=net.ext_grid.index,
            patch_edgecolor=ext_grid_color, zorder=11,
        )
        collections.append(sc)

    trafo_buses_with_geo_coordinates = [
        t for t, trafo in net.trafo.iterrows()
        if trafo.hv_bus in net.bus.geo.index and trafo.lv_bus in net.bus.geo.index
    ]
    if len(trafo_buses_with_geo_coordinates) > 0:
        tc = create_trafo_collection(
            net, trafo_buses_with_geo_coordinates,
            color=trafo_color, size=trafo_size, infofunc=trafo_info,
        )
        collections.append(tc)

    trafo3w_buses_with_geo_coordinates = [
        t for t, trafo3w in net.trafo3w.iterrows()
        if trafo3w.hv_bus in net.bus.geo.index
        and trafo3w.mv_bus in net.bus.geo.index
        and trafo3w.lv_bus in net.bus.geo.index
    ]
    if len(trafo3w_buses_with_geo_coordinates) > 0:
        tc = create_trafo3w_collection(
            net, trafo3w_buses_with_geo_coordinates,
            color=trafo_color, infofunc=trafo3w_info,
        )
        collections.append(tc)

    if plot_line_switches and len(net.switch):
        sc = create_line_switch_collection(
            net, size=switch_size, distance_to_bus=switch_distance,
            use_line_geodata=not use_bus_geodata, zorder=12, color=switch_color,
        )
        collections.append(sc)

    angles = calculate_unique_angles(net) if draw_by_type else None

    if plot_sgens and len(net.sgen):
        sgc = create_sgen_collection(
            net, size=sgen_size, orientation=orientation,
            unique_angles=angles, draw_by_type=draw_by_type,
        )
        collections.append(sgc)

    if plot_gens and len(net.gen):
        gc = create_gen_collection(
            net, size=gen_size, orientation=orientation,
            unique_angles=angles, draw_by_type=draw_by_type,
        )
        collections.append(gc)

    if plot_loads and len(net.load):
        # separate variable name to prevent rebinding normal_lc / cmap_lc
        load_coll = create_load_collection(
            net, size=load_size, orientation=orientation, unique_angles=angles
        )
        collections.append(load_coll)

    if len(net.switch):
        bsc = create_bus_bus_switch_collection(net, size=switch_size)
        collections.append(bsc)

    # copy_collections=False: axes hold the exact Python objects from our list,
    # so set_visible() calls in the toggle callback take effect immediately.
    # Colorbars are created manually below; suppress the auto-colorbar logic.
    ax = draw_collections(collections, ax=ax, plot_colorbars=False,
                          copy_collections=False)

    # ── initial visibility: Normal mode ───────────────────────────────────────
    # Must be set AFTER draw_collections so the collections are in the axes,
    # but BEFORE plt.show() fires the first canvas.draw().
    if colormap_ready:
        cmap_bc.set_visible(False)
        cmap_lc.set_visible(False)
    # ─────────────────────────────────────────────────────────────────────────

    fig = ax.figure

    # ── record Normal-mode layout ─────────────────────────────────────────────
    # ax at this point fills the full figure (no colorbars created yet).
    normal_figsize = tuple(fig.get_size_inches())
    normal_ax_pos  = list(ax.get_position().bounds)  # [x0, y0, w, h]
    # ─────────────────────────────────────────────────────────────────────────

    # ── colorbars + Colormap-mode layout ──────────────────────────────────────
    _colorbars          : list = []
    cmap_ax_pos         : list = list(normal_ax_pos)
    cmap_figsize        : tuple = normal_figsize
    cmap_cbar_positions : list = []

    if colormap_ready and plot_colorbars:
        # plt.colorbar(ax=ax) shrinks the main ax to make room for the colorbar
        cbar_l = plt.colorbar(cmap_lc, ax=ax, label="Line loading [%]")
        cbar_b = plt.colorbar(cmap_bc, ax=ax, label="Bus voltage [p.u.]")
        _colorbars = [cbar_l, cbar_b]

        # ax is now at the shrunken Colormap-mode position
        cmap_ax_pos = list(ax.get_position().bounds)
        ax_x1       = cmap_ax_pos[0] + cmap_ax_pos[2]   # right edge of shrunken ax

        # Expand figure width so the network has the same absolute pixel width
        # in Colormap mode as in Normal mode:
        #   normal_abs_w = normal_figsize[0] * normal_ax_pos[2]
        #   cmap_abs_w   = cmap_figsize[0]   * cmap_ax_pos[2]   ← want equal
        expand = normal_ax_pos[2] / cmap_ax_pos[2] if cmap_ax_pos[2] > 0 else 1.0
        cmap_figsize = (normal_figsize[0] * expand, normal_figsize[1])

        # Precompute colorbar positions for the wider figure.
        # Goal: maintain the same absolute gap (in inches) between the ax right
        # edge and each colorbar that matplotlib chose for normal_figsize.
        #
        # gap_abs = (cbar.x0 – ax_x1) * normal_figsize[0]          [inches]
        # In cmap_figsize the ax right edge (same fraction) is at:
        #   ax_x1 * cmap_figsize[0]                                 [inches]
        # So new colorbar x0 fraction:
        #   new_x0 = ax_x1 + gap_abs / cmap_figsize[0]
        # Colorbar width stays the same in inches → new fraction:
        #   new_w  = old_w * normal_figsize[0] / cmap_figsize[0]
        for cbar in _colorbars:
            cb = list(cbar.ax.get_position().bounds)   # [x0, y0, w, h] in normal_figsize
            gap_abs = (cb[0] - ax_x1) * normal_figsize[0]
            new_x0  = ax_x1 + gap_abs / cmap_figsize[0]
            new_w   = cb[2] * normal_figsize[0] / cmap_figsize[0]
            cmap_cbar_positions.append([new_x0, cb[1], new_w, cb[3]])

        # Restore Normal mode as the initial display:
        # • expand ax back to full width
        # • hide colorbars
        # • keep figure at normal_figsize
        ax.set_position(normal_ax_pos)
        for cbar in _colorbars:
            cbar.ax.set_visible(False)
    # ─────────────────────────────────────────────────────────────────────────

    # ── toggle button ─────────────────────────────────────────────────────────
    if colormap_ready:
        from matplotlib.widgets import Button

        btn_ax     = fig.add_axes([0.02, 0.01, 0.13, 0.04])
        # starts in Normal mode → first click switches TO Colormap → label "Colormap"
        toggle_btn = Button(btn_ax, "Colormap", color="#1c3f52", hovercolor="#179c7d")
        toggle_btn.label.set_color("white")
        toggle_btn.label.set_fontsize(9)
        toggle_btn.label.set_fontweight("bold")

        _state = {
            "active"             : False,          # False = Normal mode active
            "normal_ax_pos"      : normal_ax_pos,
            "cmap_ax_pos"        : cmap_ax_pos,
            "normal_figsize"     : normal_figsize,
            "cmap_figsize"       : cmap_figsize,
            "cmap_cbar_positions": cmap_cbar_positions,
        }
        _normal_colls = [normal_bc, normal_lc]
        _cmap_colls   = [cmap_bc,   cmap_lc]

        toggle_btn.on_clicked(
            lambda event: _toggle_colormap_cb(
                event, _state, ax,
                _normal_colls, _cmap_colls,
                _colorbars, toggle_btn,
            )
        )

        # strong references on ax prevent garbage collection of the button widget
        ax._simple_plot_refs = {
            "toggle_btn"         : toggle_btn,
            "btn_ax"             : btn_ax,
            "state"              : _state,
            "normal_colls"       : _normal_colls,
            "cmap_colls"         : _cmap_colls,
            "colorbars"          : _colorbars,
        }
    # ─────────────────────────────────────────────────────────────────────────

    if enable_hover:
        hover_text = ax.text(
            0, 0, "", fontsize=12, fontweight="bold", color="white",
            ha="center", va="center", zorder=99,
            bbox={"boxstyle": "round", "facecolor": "#179c7d",
                  "alpha": 1, "edgecolor": "white"},
        )
        hover_text.set_visible(False)
        fig.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: hover(event, ax, net, hover_text),
        )

    if show_plot:
        if not MATPLOTLIB_INSTALLED:
            soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
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

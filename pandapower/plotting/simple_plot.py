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

# Color palettes
_LINE_PALETTE: list[str] = ["#b2d235", "#fdb913", "#f58220", "#C7105C"]
_BUS_PALETTE: list[str] = ["#005b7f", "#179c7d", "#179c7d", "#fdb913", "#C7105C"]
# Button colors: active (current mode) = magenta, inactive = dark blue
_BTN_ACTIVE_COLOR = "#C7105C"
_BTN_INACTIVE_COLOR = "#1c3f52"


def bus_info(bus):
    """Return an info tuple identifying a bus element.

    Args:
        bus (int): Bus index.

    Returns:
        tuple: ``("bus", bus)``
    """
    return ("bus", bus)


def line_info(line):
    """Return an info tuple identifying a line element.

    Args:
        line (int): Line index.

    Returns:
        tuple: ``("line", line)``
    """
    return ("line", line)


def trafo_info(idx):
    """Return an info tuple identifying a two-winding transformer element.

    Args:
        idx (int): Transformer index.

    Returns:
        tuple: ``("trafo", idx)``
    """
    return ("trafo", idx)


def trafo3w_info(idx):
    """Return an info tuple identifying a three-winding transformer element.

    Args:
        idx (int): Three-winding transformer index.

    Returns:
        tuple: ``("trafo3w", idx)``
    """
    return ("trafo3w", idx)


def hover(event, ax, net, hover_text):
    """Update the hover text in an interactive pandapower plot based on mouse position.

    Expects collections to have an ``info`` attribute containing a list of
    ``(element, index)`` tuples, e.g. ``("bus", 3)`` or ``("line", 5)``.

    When the hovered element is a bus or line and the corresponding result
    table (``net.res_bus`` or ``net.res_line``) is non-empty, the hover label
    is extended with the following load-flow results:

    * Bus:  ``vm_pu`` and ``va_degree``
    * Line: ``loading_percent`` and ``i_ka``

    Args:
        event (matplotlib.backend_bases.MouseEvent): Mouse-move event from
            Matplotlib.
        ax (matplotlib.axes.Axes): Axes object containing the collections.
        net (pandapowerNet): pandapower network with DataFrames
            (bus, line, trafo, trafo3w, ...).
        hover_text (matplotlib.text.Text): Text artist whose content,
            position, and visibility are updated.
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

        if (
            df is not None
            and idx is not None
            and idx in df.index
            and "name" in df.columns
        ):
            name = df.at[idx, "name"]
            hover_info = f"Element: {element}\nName: {name}\nIndex: {idx}"
        elif idx is not None:
            hover_info = f"{element} | Index: {idx}"
        else:
            hover_info = str(element_info)

        # Append loadflow results when available
        if element == "bus" and idx is not None:
            res_bus = getattr(net, "res_bus", None)
            if res_bus is not None and not res_bus.empty and idx in res_bus.index:
                if "vm_pu" in res_bus.columns and "va_degree" in res_bus.columns:
                    hover_info += (
                        f"\nV_m = {res_bus.vm_pu.at[idx]:.2f} p.u."
                                   f"\nV_m = {(res_bus.vm_pu.at[idx] * net.bus.vn_kv.at[idx]):.2f} kV"
                                   f"\nV_a = {res_bus.va_degree.at[idx]:.2f} deg")
        elif element == "line" and idx is not None:
            res_line = getattr(net, "res_line", None)
            if res_line is not None and not res_line.empty and idx in res_line.index:
                if "loading_percent" in res_line.columns and "i_ka" in res_line.columns:
                    hover_info += (
                        f"\nLoading: {res_line.at[idx, 'loading_percent']:.2f} %"
                        f"\nI_kA: {res_line.at[idx, 'i_ka']:.2f} kA")

        hover_text.set_text(hover_info)
        hover_text.set_ha("left")
        hover_text.set_va("bottom")
        hover_text.set_multialignment("left")
        hover_text.set_position((event.xdata, event.ydata))
        hover_text.set_visible(True)
        fig.canvas.draw_idle()
        return

    if visible:
        hover_text.set_visible(False)
        fig.canvas.draw_idle()


# -- Colormap helpers ---------------------------------------------------------
def _pick_n_colors(n: int, palette: list[str]) -> list[str]:
    """Sample *n* colors evenly from *palette* using floor-based index mapping.

    The first and last palette entries are always included.

    Args:
        n (int): Number of colors to sample.  Returns an empty list when
            ``n <= 0``.
        palette (list of str): Source color palette to sample from.

    Returns:
        list of str: List of *n* hex color strings sampled from *palette*.
            If ``n >= len(palette)``, the last palette color is repeated as
            needed.
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
    """Build a pandapower ``cmap_list`` from sorted numeric breakpoints.

    Args:
        limits (tuple or list): Sorted breakpoints, e.g.
            ``(0, 25, 50, 75, 100)`` for lines or
            ``(0.9, 0.95, 1.0, 1.05, 1.1)`` for buses.
        colormap_type (str): ``"discrete"`` for flat color bands or
            ``"continuous"`` for a smooth gradient.
        kind (str, optional): ``"line"`` uses the line palette; ``"bus"``
            uses the bus palette.  Default is ``"line"``.

    Returns:
        list: Discrete: ``[((lo, hi), color), ...]``
            Continuous: ``[(value, color), ...]``
    """
    palette = _LINE_PALETTE if kind == "line" else _BUS_PALETTE
    n = len(limits)
    if colormap_type == "discrete":
        colors = _pick_n_colors(n - 1, palette)
        return [((limits[i], limits[i + 1]), colors[i]) for i in range(n - 1)]
    colors = _pick_n_colors(n, palette)
    return [(limits[i], colors[i]) for i in range(n)]


def _extract_cbar_ticks(cmap_list: list, colormap_type: str) -> list:
    """Extract tick positions from a ``cmap_list`` at the user-defined breakpoints.

    Ensures that discrete and continuous colorbars show identical tick marks
    regardless of colormap type.

    Args:
        cmap_list (list): Discrete: ``[((lo, hi), color), ...]``
            Continuous: ``[(value, color), ...]``
        colormap_type (str): ``"discrete"`` or ``"continuous"``.

    Returns:
        list: Sorted unique tick values derived from the breakpoints in
            *cmap_list*.
    """
    if colormap_type == "discrete":
        ticks = []
        for (lo, hi), _ in cmap_list:
            if lo not in ticks:
                ticks.append(lo)
            if hi not in ticks:
                ticks.append(hi)
        return sorted(ticks)
    # Continuous: one value per entry
    return [v for v, _ in cmap_list]


def _set_colormap_mode(
    mode: str,
    state: dict,
    ax,
    normal_colls: list,
    cmap_colls: list,
    colorbars: list,
    btn_normal,
    btn_colormap,
):
    """Switch the figure between ``"normal"`` and ``"colormap"`` display mode.

    Calling this function with the mode that is already active is a no-op.

    Args:
        mode (str): Target display mode: ``"normal"`` or ``"colormap"``.
        state (dict): Mutable layout and mode state built in ``simple_plot``.
        ax (matplotlib.axes.Axes): Main network axes.
        normal_colls (list of Collection): Flat-color bus and line collections
            used in Normal mode.
        cmap_colls (list of Collection): Colormap bus and line collections
            used in Colormap mode.
        colorbars (list of Colorbar): Colorbars belonging to the Colormap
            view.
        btn_normal (matplotlib.widgets.Button): Button that activates Normal
            mode.
        btn_colormap (matplotlib.widgets.Button): Button that activates
            Colormap mode.

    Note:
        **Button highlighting**
        The button representing the currently active mode is always rendered
        in ``_BTN_ACTIVE_COLOR`` (magenta); the other button uses
        ``_BTN_INACTIVE_COLOR`` (dark blue), giving the user clear visual
        feedback about which mode is currently displayed.

        **Why two collections + set_visible (not in-place color patching)**
        ``ScalarMappable.update_scalarmappable()`` is called on every
        ``canvas.draw()`` and rewrites ``_facecolors`` from
        ``_A + _cmap + _norm``.  Any in-place color patch via
        ``set_facecolor`` / ``set_color`` is silently overwritten on the next
        draw cycle.  Swapping visibility between two fully initialized
        collections is the only approach that survives repeated ``draw()``
        calls.

        **Why ``copy_collections=False`` is required in draw_collections**
        ``draw_collections`` shallow-copies every collection by default.
        ``set_visible()`` must target the exact Python objects held in
        ``ax.collections``; ``copy_collections=False`` guarantees identity.

        **Why figure resize and ax repositioning is needed**
        ``plt.colorbar(ax=ax)`` permanently shrinks the main axes.  Restoring
        the saved ``normal_ax_pos`` and reverting ``fig.set_size_inches`` in
        Normal mode eliminates the empty strip on the right side of the
        figure.
    """
    if state["active"] == mode:
        return  # already in the requested mode – nothing to do

    state["active"] = mode
    is_cmap = (mode == "colormap")
    fig = ax.figure

    # Swap collection visibility
    for c in normal_colls:
        c.set_visible(not is_cmap)
    for c in cmap_colls:
        c.set_visible(is_cmap)

    # Resize figure and reposition axes + colorbars
    if is_cmap:
        fig.set_size_inches(state["cmap_figsize"], forward=True)
        ax.set_position(state["cmap_ax_pos"])
        for cbar, pos in zip(colorbars, state["cmap_cbar_positions"]):
            cbar.ax.set_visible(True)
            cbar.ax.set_position(pos)
    else:
        for cbar in colorbars:
            cbar.ax.set_visible(False)
        ax.set_position(state["normal_ax_pos"])
        fig.set_size_inches(state["normal_figsize"], forward=True)

    # Button highlighting: active mode = magenta, inactive mode = dark blue
    btn_normal.ax.set_facecolor(
        _BTN_ACTIVE_COLOR if not is_cmap else _BTN_INACTIVE_COLOR
    )
    btn_colormap.ax.set_facecolor(
        _BTN_ACTIVE_COLOR if is_cmap else _BTN_INACTIVE_COLOR
    )

    fig.canvas.draw_idle()


def simple_plot(
        net: pandapowerNet,
        respect_switches: bool = False,
        line_width: float = 3.0,
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
        plot_bus_switches: bool = False,
        scale_size: bool = True,
        bus_color="#1c3f52",
        line_color="grey",
        dcline_color="c",
        trafo_color="k",
        ext_grid_color="#1c3f52",
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
        highlight_bus_size_factor=2.0,
        highlight_line_width_factor=2.5,
        highlight_color="red",
        colormap_type: str = "continuous",
        line_limits: tuple = (0, 25, 50, 75, 100),
        bus_limits: tuple = (0.9, 0.95, 1.0, 1.05, 1.1),
        cmap_lines: list = None,
        cmap_buses: list = None,
        plot_colorbars: bool = True,
):
    """Plot a pandapower network as simply as possible.

    If no geodata is available, artificial geodata is generated automatically.
    For advanced plotting options see the pandapower plotting tutorial.

    Two mode buttons ("Normal" and "Colormap") are added to the figure whenever
    load-flow results are available or can be computed automatically.  The plot
    starts in **Normal** (flat-color) mode:

    * The **Normal** button is highlighted in magenta (= currently active).
    * The **Colormap** button is shown in dark blue (= inactive).

    Clicking a button switches to that mode and updates the highlighting
    accordingly.  Clicking the already-active button is a no-op.

    In Colormap mode buses are colored by ``vm_pu`` and lines by
    ``loading_percent``.  Colorbars always show ticks only at the
    user-defined breakpoints regardless of ``colormap_type``.

    The figure window resizes automatically:

    * Normal mode   – compact figure; axes fill the full width.
    * Colormap mode – figure is widened so the network keeps the same absolute
      pixel size; the extra width accommodates the colorbars.

    Args:
        net (pandapowerNet): The pandapower network to plot.
        respect_switches (bool, optional): Respect open switches when creating
            artificial geodata.  Ignored when ``plot_line_switches=True``.
            Default is ``False``.
        line_width (float, optional): Width of line segments.
            Default is ``2.0``.
        bus_size (float, optional): Relative bus marker size (scaled by mean
            bus geodistance when ``scale_size=True``).  Default is ``1.0``.
        ext_grid_size (float, optional): Relative external grid symbol size.
            Default is ``1.0``.
        trafo_size (float, optional): Relative transformer symbol size.
            Default is ``1.0``.
        plot_loads (bool, optional): Draw load symbols.  Default is ``False``.
        plot_gens (bool, optional): Draw generator symbols.
            Default is ``False``.
        plot_sgens (bool, optional): Draw static generator symbols.
            Default is ``False``.
        orientation (float or None, optional): Base orientation angle in
            radians for sgen, gen, and load symbols.  ``None`` uses the
            element-specific default.  Default is ``None``.
        load_size (float, optional): Relative load symbol size.
            Default is ``1.0``.
        gen_size (float, optional): Relative gen symbol size.
            Default is ``1.0``.
        sgen_size (float, optional): Relative sgen symbol size.
            Default is ``1.0``.
        switch_size (float, optional): Relative switch symbol size.
            Default is ``2.0``.
        switch_distance (float, optional): Relative switch distance from its
            bus.  Default is ``1.0``.
        plot_line_switches (bool, optional): Draw line switch symbols.
            Default is ``False``.
        plot_line_switches (bool, optional): Draw bus switch symbols.
            Default is ``False``.
        scale_size (bool, optional): Scale all symbol sizes relative to the
            mean bus geodistance.  Default is ``True``.
        bus_color (str, optional): Flat bus marker color used in Normal mode.
            Default is ``"#1c3f52"``.
        line_color (str, optional): Flat line color used in Normal mode.
            Default is ``"grey"``.
        dcline_color (str, optional): DC line color.  Default is ``"c"``.
        trafo_color (str, optional): Transformer symbol color.
            Default is ``"k"``.
        ext_grid_color (str, optional): External grid symbol color.
            Default is ``"#C7105C"``.
        switch_color (str, optional): Switch symbol color.
            Default is ``"k"``.
        library (str, optional): Layout library used for generic coordinate
            generation.  ``"igraph"`` or ``"networkx"``.
            Default is ``"igraph"``.
        show_plot (bool, optional): Call ``plt.show()`` at the end.
            Default is ``True``.
        ax (matplotlib.axes.Axes or None, optional): Existing axes to draw
            into.  A new figure and axes are created when ``None``.
            Default is ``None``.
        draw_by_type (bool, optional): Group sgen and gen symbols by element
            type.  Default is ``True``.
        bus_dc_size (float, optional): Relative DC bus marker size.
            Default is ``1.0``.
        bus_dc_color (str, optional): DC bus marker color.
            Default is ``"m"``.
        line_dc_color (str, optional): DC line color.  Default is ``"c"``.
        vsc_size (float, optional): Relative VSC symbol size.
            Default is ``4.0``.
        vsc_color (str, optional): VSC symbol color.
            Default is ``"orange"``.
        highlight_buses (iterable or None, optional): Bus indices to
            highlight.  Default is ``None``.
        highlight_lines (iterable or None, optional): Line indices to
            highlight.  Default is ``None``.
        enable_hover (bool, optional): Enable interactive hover labels.
            Default is ``True``.
        highlight_bus_size_factor (float, optional): Size multiplier applied
            to highlighted bus markers.  Default is ``2.0``.
        highlight_line_width_factor (float, optional): Line-width multiplier
            applied to highlighted lines.  Default is ``2.5``.
        highlight_color (str, optional): Color used for highlighted elements.
            Default is ``"#C7105C"``.
        colormap_type (str, optional): ``"discrete"`` for flat color bands or
            ``"continuous"`` for a smooth gradient.  Colorbar ticks are placed
            only at the breakpoints in both cases.
            Default is ``"continuous"``.
        line_limits (tuple, optional): Breakpoints for line loading in **%**.
            Colors are auto-generated from the line palette.  Entirely
            overridden by ``cmap_lines`` when provided.
            Default is ``(0, 25, 50, 75, 100)``.
        bus_limits (tuple, optional): Breakpoints for bus voltage in **p.u.**.
            Colors are auto-generated from the bus palette.  Entirely
            overridden by ``cmap_buses`` when provided.
            Default is ``(0.9, 0.95, 1.0, 1.05, 1.1)``.
        cmap_lines (list or None, optional): Full custom colormap definition
            for lines; overrides ``line_limits``.
            Discrete: ``[((min, max), color), ...]``
            Continuous: ``[(value, color), ...]``
            Default is ``None``.
        cmap_buses (list or None, optional): Full custom colormap definition
            for buses; overrides ``bus_limits``.  Same format as
            ``cmap_lines``.  Default is ``None``.
        plot_colorbars (bool, optional): Show colorbars in the Colormap view.
            Default is ``True``.

    Returns:
        matplotlib.axes.Axes: The axes object containing the network plot.
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
        bus_size = sizes["bus"]
        ext_grid_size = sizes["ext_grid"]
        trafo_size = sizes["trafo"]
        sgen_size = sizes["sgen"]
        load_size = sizes["load"]
        switch_size = sizes["switch"]
        switch_distance = sizes["switch_distance"]
        gen_size = sizes["gen"]

    # ── colormap setup ────────────────────────────────────────────────────────
    # Always attempt colormap preparation so the mode buttons can be offered.
    cmap_l = norm_l = cmap_b = norm_b = None
    cmap_buses_ready = cmap_lines_ready = colormap_ready = False
    _cl = _cb = None

    has_buses = len(net.bus) > 0
    has_lines = len(net.line) > 0

    try:
        needs_runpp = (
            (has_buses and net.res_bus.empty)
            or (has_lines and net.res_line.empty)
        )
        if needs_runpp:
            logger.info("Result tables empty – running pp.runpp(net) automatically.")
            from pandapower.run import runpp
            runpp(net)

        from pandapower.plotting.colormaps import cmap_discrete, cmap_continuous

        if colormap_type == "discrete":
            if has_lines:
                _cl = cmap_lines or _build_cmap_from_limits(line_limits, "discrete", "line")
                cmap_l, norm_l = cmap_discrete(_cl)
                cmap_lines_ready = True
            if has_buses:
                _cb = cmap_buses or _build_cmap_from_limits(bus_limits, "discrete", "bus")
                cmap_b, norm_b = cmap_discrete(_cb)
                cmap_buses_ready = True

        elif colormap_type == "continuous":
            if has_lines:
                _cl = cmap_lines or _build_cmap_from_limits(line_limits, "continuous", "line")
                cmap_l, norm_l = cmap_continuous(_cl)
                cmap_lines_ready = True
            if has_buses:
                _cb = cmap_buses or _build_cmap_from_limits(bus_limits, "continuous", "bus")
                cmap_b, norm_b = cmap_continuous(_cb)
                cmap_buses_ready = True

        else:
            logger.error(
                f"Unknown colormap_type='{colormap_type}'. "
                "Allowed values: 'discrete', 'continuous'. Mode buttons disabled."
            )

        colormap_ready = cmap_buses_ready or cmap_lines_ready

    except Exception as exc:
        logger.warning(
            f"Colormap setup failed ({exc!r}). "
            "Mode buttons will not be shown; plotting in Normal mode only."
        )

    normal_bc = cmap_bc = None
    if has_buses:
        normal_bc = create_bus_collection(
            net, net.bus.index, size=bus_size,
            color=bus_color, zorder=8, infofunc=bus_info,
        )
        if cmap_buses_ready:
            cmap_bc = create_bus_collection(
                net, net.bus.index, size=bus_size,
                cmap=cmap_b, norm=norm_b, zorder=9, infofunc=bus_info,
            )

    collections = []
    if normal_bc is not None:
        collections.append(normal_bc)
    if cmap_bc is not None:
        collections.append(cmap_bc)

    # fall back to bus geodata when no line geodata is present
    use_bus_geodata = not has_lines or len(net.line.geo.dropna()) == 0
    in_service_lines = (
        net.line[net.line.in_service].index if has_lines else pd.Index([])
    )
    nogolines = (
        set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)])
        if respect_switches
        else set()
    )
    plot_lines = in_service_lines.difference(nogolines)
    plot_dclines = net.dcline.in_service
    plot_lines_dc = net.line_dc.loc[net.line_dc.in_service].index

    # ── line collections: flat-color (Normal) + colormap (Colormap) ──────────
    # named normal_lc / cmap_lc to prevent rebinding by create_load_collection
    normal_lc = cmap_lc = None
    if has_lines:
        normal_lc = create_line_collection(
            net, plot_lines,
            color=line_color, linewidths=line_width,
            use_bus_geodata=use_bus_geodata, zorder=7, infofunc=line_info,
        )
        if cmap_lines_ready:
            cmap_lc = create_line_collection(
                net, plot_lines,
                cmap=cmap_l, norm=norm_l, linewidths=line_width,
                use_bus_geodata=use_bus_geodata, zorder=8, infofunc=line_info,
            )

    if normal_lc is not None:
        collections.append(normal_lc)
    if cmap_lc is not None:
        collections.append(cmap_lc)

    # ── highlighting ----------------------------------------------------------
    if highlight_buses is not None and has_buses:
        hl_buses_idx = list(set(highlight_buses) & set(net.bus.index))
        if hl_buses_idx:
            hbc = create_bus_collection(
                net, hl_buses_idx,
                size=bus_size * highlight_bus_size_factor,
                color=highlight_color, zorder=98, infofunc=bus_info,
            )
            collections.append(hbc)

    if highlight_lines is not None and has_lines:
        hl_lines_idx = list(set(highlight_lines) & set(plot_lines))
        if hl_lines_idx:
            hlc = create_line_collection(
                net, hl_lines_idx,
                color=highlight_color,
                linewidths=line_width * highlight_line_width_factor,
                use_bus_geodata=use_bus_geodata, zorder=98, infofunc=line_info,
            )
            collections.append(hlc)

    # ── other collections -----------------------------------------------------
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
            patch_edgecolor=ext_grid_color, zorder=12,
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
            use_line_geodata=not use_bus_geodata, zorder=10, color=switch_color,
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

    if plot_bus_switches and len(net.switch):
        bsc = create_bus_bus_switch_collection(net, size=switch_size, zorder=10)
        collections.append(bsc)

    # copy_collections=False: axes hold the exact Python objects so that
    # set_visible() in _set_colormap_mode takes effect immediately.
    # Colorbars are managed manually; suppress draw_collections auto-colorbars.
    ax = draw_collections(collections, ax=ax, plot_colorbars=False,
                          copy_collections=False)

    # ── initial visibility: Normal mode ───────────────────────────────────────
    # Must be set AFTER draw_collections so the collections are in the axes,
    # but BEFORE plt.show() fires the first canvas.draw().
    if cmap_bc is not None:
        cmap_bc.set_visible(False)
    if cmap_lc is not None:
        cmap_lc.set_visible(False)

    fig = ax.figure
    normal_figsize = tuple(fig.get_size_inches())
    normal_ax_pos = list(ax.get_position().bounds)

    _colorbars: list = []
    cmap_ax_pos: list = list(normal_ax_pos)
    cmap_figsize: tuple = normal_figsize
    cmap_cbar_positions: list = []

    if colormap_ready and plot_colorbars:
        if cmap_lc is not None:
            cbar_l = plt.colorbar(cmap_lc, ax=ax, label="Line loading [%]")
            cbar_l.set_ticks(_extract_cbar_ticks(_cl, colormap_type))
            _colorbars.append(cbar_l)
        if cmap_bc is not None:
            cbar_b = plt.colorbar(cmap_bc, ax=ax, label="Bus voltage [p.u.]")
            cbar_b.set_ticks(_extract_cbar_ticks(_cb, colormap_type))
            _colorbars.append(cbar_b)

        # Proceed only when at least one colorbar was created.
        if _colorbars:
            cmap_ax_pos = list(ax.get_position().bounds)
            ax_x1 = cmap_ax_pos[0] + cmap_ax_pos[2]
            expand = (
                normal_ax_pos[2] / cmap_ax_pos[2] if cmap_ax_pos[2] > 0 else 1.0
            )
            cmap_figsize = (normal_figsize[0] * expand, normal_figsize[1])

            for cbar in _colorbars:
                cb = list(cbar.ax.get_position().bounds)
                gap_abs = (cb[0] - ax_x1) * normal_figsize[0]
                new_x0 = ax_x1 + gap_abs / cmap_figsize[0]
                new_w = cb[2] * normal_figsize[0] / cmap_figsize[0]
                cmap_cbar_positions.append([new_x0, cb[1], new_w, cb[3]])

            ax.set_position(normal_ax_pos)
            for cbar in _colorbars:
                cbar.ax.set_visible(False)

    if colormap_ready:
        from matplotlib.widgets import Button

        btn_normal_ax = fig.add_axes([0.02, 0.01, 0.10, 0.04])
        btn_colormap_ax = fig.add_axes([0.135, 0.01, 0.10, 0.04])

        btn_normal = Button(btn_normal_ax, "Normal",
                            color=_BTN_INACTIVE_COLOR, hovercolor=_BTN_ACTIVE_COLOR)
        btn_colormap = Button(btn_colormap_ax, "Colormap",
                              color=_BTN_INACTIVE_COLOR, hovercolor=_BTN_ACTIVE_COLOR)

        for btn in (btn_normal, btn_colormap):
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold")

        _state = {
            "active": "normal",
            "normal_ax_pos": normal_ax_pos,
            "cmap_ax_pos": cmap_ax_pos,
            "normal_figsize": normal_figsize,
            "cmap_figsize": cmap_figsize,
            "cmap_cbar_positions": cmap_cbar_positions,
        }
        # Filter out None entries when an element type is absent.
        _normal_colls = [c for c in [normal_bc, normal_lc] if c is not None]
        _cmap_colls = [c for c in [cmap_bc, cmap_lc] if c is not None]

        btn_normal.on_clicked(
            lambda e: _set_colormap_mode(
                "normal", _state, ax,
                _normal_colls, _cmap_colls,
                _colorbars, btn_normal, btn_colormap,
            )
        )
        btn_colormap.on_clicked(
            lambda e: _set_colormap_mode(
                "colormap", _state, ax,
                _normal_colls, _cmap_colls,
                _colorbars, btn_normal, btn_colormap,
            )
        )

        ax._simple_plot_refs = {
            "btn_normal": btn_normal,
            "btn_colormap": btn_colormap,
            "btn_normal_ax": btn_normal_ax,
            "btn_colormap_ax": btn_colormap_ax,
            "state": _state,
            "normal_colls": _normal_colls,
            "cmap_colls": _cmap_colls,
            "colorbars": _colorbars,
        }

    if enable_hover:
        hover_text = ax.text(
            0, 0, "", fontsize=12, fontweight="bold", color="white",
            ha="center", va="center", zorder=99,
            bbox={"boxstyle": "round", "facecolor": "#C7105C",
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


def calculate_unique_angles(
    net: pandapowerNet,
) -> dict[int, dict[str, dict[str, float] | float]]:
    """Calculate patch placement angles for sgen, gen, and load symbols at each bus.

    Only a single patch for all loads at a given bus is currently supported.

    Args:
        net (pandapowerNet): The pandapower network to calculate patch angles
            for.

    Returns:
        dict: Nested mapping of the form
            ``{bus_index: {element_type: {sub_type: angle_rad}}}``.
            Angular offsets are in radians.  For loads the inner value is a
            plain ``float`` instead of a nested dict, because loads are not
            grouped by sub-type.
    """
    sgen_counts = (
        net.sgen.groupby(["bus", "type"], dropna=False).size().unstack(fill_value=0)
    )
    gen_counts = (
        net.gen.groupby(["bus", "type"], dropna=False).size().unstack(fill_value=0)
    )
    loads = pd.Series(1, index=net.load.bus.unique(), name="load")

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
        patches[index]["load"] = patch_angle * counts[index]
        counts[index] += 1

    return patches

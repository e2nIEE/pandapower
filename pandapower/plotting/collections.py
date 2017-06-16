# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle, RegularPolygon


def create_bus_collection(net, buses=None, size=5, marker="o", patch_type="circle", colors=None,
                          z = None, cmap=None, norm=None, infofunc=None, picker=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower buses.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created.
        If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **marker** (str, "o") - patch marker

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "rect" for a rectangle
                - "poly<n>" for a polygon with n edges

        **infofunc** (function, None) - infofunction for the patch element

        **colors** (list, None) - list of colors for every element

        **cmap** - colormap for the patch colors

        **picker** - picker argument passed to the patch collection

        **kwargs - key word arguments are passed to the patch function

    """
    buses = net.bus.index.tolist() if buses is None else list(buses)
    if len(buses) == 0:
        return None
    infos = []

    def figmaker(x, y, i):
        if patch_type=="circle":
            if colors:
                fig = Circle((x, y), size, color=colors[i], **kwargs)
            else:
                fig = Circle((x, y), size, **kwargs)
        elif patch_type=="rect":
            if colors:
                fig = Rectangle([x - size, y - size], 2*size, 2*size, color=colors[i], **kwargs)
            else:
                fig = Rectangle([x - size, y - size], 2*size, 2*size, **kwargs)
        elif patch_type.startswith("poly"):
            edges = int(patch_type[4:])
            if colors:
                fig = RegularPolygon([x, y], numVertices=edges, radius=size, color=colors[i],
                                     **kwargs)
            else:
                fig = RegularPolygon([x, y], numVertices=edges, radius=size, **kwargs)
        if infofunc:
            infos.append(infofunc(buses[i]))
        return fig
    patches = [figmaker(x, y, i)
               for i, (x, y) in enumerate(zip(net.bus_geodata.loc[buses].x.values,
                                              net.bus_geodata.loc[buses].y.values))
               if x != np.nan]
    pc = PatchCollection(patches, match_original=True, picker=picker)
    pc.bus_indices = np.array(buses)
    if cmap:
        pc.set_cmap(cmap)
        pc.set_norm(norm)
        if z is None:
            z = net.res_bus.vm_pu.loc[buses]
        pc.set_array(np.array(z))
        pc.has_colormap = True
        pc.cbar_title = "Bus Voltage [pu]"

    pc.patch_type = patch_type
    pc.size = size
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    pc.info = infos
    return pc


def create_line_collection(net, lines=None, use_line_geodata=True, 
                           infofunc=None, cmap=None,
                           norm=None, picker=False, z=None,
                           cbar_title="Line Loading [%]", **kwargs):
    """
    Creates a matplotlib line collection of pandapower lines.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created. If None, all lines in the network are considered.

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True) or on net.bus_geodata of the connected buses (False)

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    """
    lines = net.line_geodata.index.tolist() if lines is None and use_line_geodata else \
        net.line.index.tolist() if lines is None and not use_line_geodata else list(lines)
    if len(lines) == 0:
        return None
    if use_line_geodata:
        data = [(net.line_geodata.coords.loc[line],
                 infofunc(line) if infofunc else [])
                 for line in lines if line in net.line_geodata.index]
    else:
        data = [([(net.bus_geodata.x.at[a], net.bus_geodata.y.at[a]),
                  (net.bus_geodata.x.at[b], net.bus_geodata.y.at[b])],
                 infofunc(line) if infofunc else [])
                for line, (a, b) in net.line[["from_bus", "to_bus"]].iterrows()
                if line in lines and a in net.bus_geodata.index and b in net.bus_geodata.index]
    data, info = list(zip(*data))

    # This would be done anyways by matplotlib - doing it explicitly makes it a) clear and
    # b) prevents unexpected behavior when observing colors being "none"
    lc = LineCollection(data, picker=picker, **kwargs)
    lc.line_indices = np.array(lines)
    if cmap:
        if z is None:
            z = net.res_line.loading_percent.loc[lines]
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        lc.set_array(np.array(z))
        lc.has_colormap = True
        lc.cbar_title = "Line Loading [%]"
    lc.info = info
    return lc


def create_trafo_collection(net, trafos=None, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The transformers for which the collections are created.
        If None, all transformers in the network are considered.

        **kwargs - key word arguments are passed to the patch function

    """
    trafos = net.trafo if trafos is None else net.trafo.loc[trafos]

    hv_geo = list(zip(net.bus_geodata.loc[trafos["hv_bus"], "x"].values,
                 net.bus_geodata.loc[trafos["hv_bus"], "y"].values))
    lv_geo = list(zip(net.bus_geodata.loc[trafos["lv_bus"], "x"].values,
                 net.bus_geodata.loc[trafos["lv_bus"], "y"].values))

    tg = list(zip(hv_geo, lv_geo))

    return LineCollection([(tgd[0], tgd[1]) for tgd in tg], **kwargs)

def create_trafo_symbol_collection(net, trafos=None, picker=False, size=1.,
                                   infofunc=None, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The transformers for which the collections are created.
        If None, all transformers in the network are considered.

        **kwargs - key word arguments are passed to the patch function

    """
    trafo_table = net.trafo if trafos is None else net.trafo.loc[trafos]
    lines = []
    circles = []
    infos = []
    for i, trafo in trafo_table.iterrows():
        p1 = net.bus_geodata[["x", "y"]].loc[trafo.hv_bus].values
        p2 = net.bus_geodata[["x", "y"]].loc[trafo.lv_bus].values
        if np.all(p1 == p2):
            continue
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        off = size*0.35
        circ1 = (0.5 - off/d) * (p1 - p2) + p2
        circ2 = (0.5 + off/d) * (p1 - p2) + p2
        circles.append(Circle(circ1, size, fc=(1,0,0,0), ec=(0,0,0,1)))
        circles.append(Circle(circ2, size, fc=(1,0,0,0), ec=(0,0,0,1)))

        lp1 = (0.5 - off/d - size/d) * (p2 - p1) + p1
        lp2 = (0.5 - off/d - size/d) * (p1 - p2) + p2
        lines.append([p1, lp1])
        lines.append([p2, lp2])
        if not infofunc is None:
            infos.append(infofunc(i))
            infos.append(infofunc(i))
    if kwargs.__contains__("color"):
        lc = LineCollection((lines), picker=picker, **kwargs)
    else:
        lc = LineCollection((lines), color="k", picker=picker, **kwargs)
    lc.info = infos
    pc = PatchCollection(circles, match_original=True, picker=picker, **kwargs )
    pc.info = infos
    return lc, pc

def create_load_symbol_collection(net, size=1., infofunc=None, **kwargs):
    lines = []
    polys = []
    infos = []
    off = 1.7
    for i, load in net.load.iterrows():
        p1 = net.bus_geodata[["x", "y"]].loc[load.bus]
        p2 = p1 - np.array([0, size*off])
        polys.append(RegularPolygon(p2, numVertices=3, radius=size, orientation=np.pi))
        lines.append((p1, p2 + np.array([0, size/2])))
        if infofunc is not None:
            infos.append(infofunc(i))
    load1 = PatchCollection(polys, facecolor="w", edgecolor="k", **kwargs)
    load2 = LineCollection(lines, color="k", **kwargs)
    load1.info = infos
    load2.info = infos
    return load1, load2

def create_ext_grid_symbol_collection(net, size=1., infofunc=None, picker=False,
                                      **kwargs):
    lines = []
    polys = []
    infos = []
    for i, ext_grid in net.ext_grid.iterrows():
        p1 = net.bus_geodata[["x", "y"]].loc[ext_grid.bus]
        p2 = p1 + np.array([0, size])
        polys.append(Rectangle([p2[0] - size/2, p2[1] - size/2], size, size))
        lines.append((p1, p2 - np.array([0, size/2])))
        if infofunc is not None:
            infos.append(infofunc(i))
    ext_grid1 = PatchCollection(polys, facecolor=(1,0,0,0), edgecolor=(0,0,0,1),
                                hatch="XX", picker=picker, **kwargs)
    ext_grid2 = LineCollection(lines, color="k", picker=picker, **kwargs)
    ext_grid1.info = infos
    ext_grid2.info = infos
    return ext_grid1, ext_grid2

def draw_collections(collections, figsize=(10, 8), ax=None, plot_colorbars=True):
    """
    Draws matplotlib collections which can be created with the create collection functions.

    Input:
        **collections** (list) - iterable of collection objects

    OPTIONAL:
        **figsize** (tuple, (10,8)) - figsize of the matplotlib figure

        **ax** (axis, None) - matplotlib axis object to plot into, new axis is created if None
    """

    if not ax:
        plt.figure(facecolor="white", figsize=figsize)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                            wspace=0.02, hspace=0.04)
    ax = ax or plt.gca()

    for c in collections:
        if c:
            cc = copy.copy(c)
            ax.add_collection(cc)
            if plot_colorbars and hasattr(c, "has_colormap"):
                extend = c.extend if hasattr(c, "extend") else "neither"
                cbar_load = plt.colorbar(c, extend=extend, ax=ax)
                if hasattr(c, "cbar_title"):
                    cbar_load.ax.set_ylabel(c.cbar_title)
    try:
        ax.set_facecolor("white")
    except:
        ax.set_axis_bgcolor("white")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal', 'datalim')
    ax.autoscale_view(True, True, True)
    ax.margins(.02)
    plt.tight_layout()

if __name__ == "__main__":
    import pandapower as pp
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10, geodata=(5,10))
    b2 = pp.create_bus(net, 0.4, geodata=(5,15))
    b3 = pp.create_bus(net, 0.4, geodata=(0,22))
    b4 = pp.create_bus(net, 0.4, geodata=(8, 20))
    pp.create_load(net, b1, p_kw=100)
    pp.create_load(net, b3, p_kw=100)
    pp.create_ext_grid(net, b4)

    pp.create_line(net, b2, b3, 2.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, b2, b4, 2.0, std_type="NAYY 4x50 SE")
    pp.create_transformer(net, b1, b2, std_type="0.63 MVA 10/0.4 kV")
    pp.create_transformer(net, b3, b4, std_type="0.63 MVA 10/0.4 kV")

    bc = create_bus_collection(net, size=0.2, color="k")
    lc = create_line_collection(net, use_line_geodata=False, color="k", linewidth=3.)
    lt, bt = create_trafo_symbol_collection(net, size=1.0, linewidth=3.)
    load1, load2 = create_load_symbol_collection(net, linewidth=2.,
                                            infofunc=lambda x: ("load", x))
    eg1, eg2 = create_ext_grid_symbol_collection(net, size=2.,
                                            infofunc=lambda x: ("ext_grid", x))


    draw_collections([bc, lc, load1, load2, lt, bt, eg1, eg2])

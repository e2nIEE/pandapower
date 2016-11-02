# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle, RegularPolygon
import matplotlib.pyplot as plt
import copy

def create_bus_collection(net, buses=None, size=5, marker="o", patch_type="circle", colors=None,
                          cmap=None, norm=None, infofunc=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower buses.
    
    Input:

        **net** (PandapowerNet) - The pandapower network
               
    Optional:
    
        **buses** (list, None) - The buses for which the collections are created. If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **marker** (str, "o") - patch marker

        **patch_type** (str, "circle") - patch type, can be
        
                - "circle" for a circle
                - "rect" for a rectanlge
                - "poly<n>" for a polygon with n edges 
        
        **infofunc** (function, None) - infofunction for the patch element
        
        **colors** (list, None) - list of colors for every element
        
        **cmap** - colormap for the patch colors
        
        **picker** - picker argument passed to the patch collection
        
        **kwargs - key word arguments are passed to the patch function
        
    """
    buses = net.bus.index.tolist() if buses is None else list(buses)
    patches = []
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
               if x != -1 and x != np.nan]
    pc = PatchCollection(patches, match_original=True)
    if cmap:
        pc.set_cmap(cmap)
        pc.set_norm(norm)
        pc.set_array(net.res_bus.vm_pu.loc[buses])
        pc.has_colormap = True
        pc.cbar_title = "Bus Voltage [pu]"

    pc.patch_type = patch_type
    pc.size = size
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    pc.info = infos
    return pc


def create_line_collection(net, lines=None, use_line_geodata=True, infofunc=None, cmap=None,
                           norm=None, **kwargs):
    """
    Creates a matplotlib line collection of pandapower lines.
    
    Input:

        **net** (PandapowerNet) - The pandapower network
               
    Optional:

        **lines** (list, None) - The lines for which the collections are created. If None, all lines in the network are considered.

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True) or on net.bus_geodata of the connected buses (False)
        
         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function
        
    """
    if lines is None:
        lines = net.line.index
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
    lc = LineCollection(data, **kwargs)
    if cmap:
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        lc.set_array(net.res_line.loading_percent.loc[lines])
        lc.has_colormap = True
        lc.cbar_title = "Line Loading [%]"
    lc.info = info
    return lc

def create_trafo_collection(net, trafos=None, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.
    
    Input:

        **net** (PandapowerNet) - The pandapower network
               
    Optional:

        **trafos** (list, None) - The transformers for which the collections are created. If None, all transformers in the network are considered.

        **kwargs - key word arguments are passed to the patch function
        
    """
    trafos = net.trafo if trafos is None else net.trafo.loc[trafos]

    hv_geo = list(zip(net.bus_geodata.loc[trafos["hv_bus"], "x"].values,
                 net.bus_geodata.loc[trafos["hv_bus"], "y"].values))
    lv_geo = list(zip(net.bus_geodata.loc[trafos["lv_bus"], "x"].values,
                 net.bus_geodata.loc[trafos["lv_bus"], "y"].values))

    tg = list(zip(hv_geo, lv_geo))

    return LineCollection([(tgd[0], tgd[1]) for tgd in tg], **kwargs)

def draw_collections(collections, figsize=(10, 8), ax=None, plot_colorbars=True):
    """
    Draws matplotlib collections which can be created with the create collection functions.

    Input:

        **collections** (list) - iterable of collection objects
               
    Optional:

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
                cbar_load = plt.colorbar(c, extend=c.extend if hasattr(c, "extend") else "neither")                
                if hasattr(c, "cbar_title"):
                    cbar_load.ax.set_ylabel(c.cbar_title)
    ax.set_axis_bgcolor("white")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal', 'datalim')
    ax.autoscale_view(True, True, True)
    ax.margins(.02)
    plt.tight_layout()

# -*- coding: utf-8 -*-
from __future__ import division
from builtins import zip
__author__ = "Alexander Scheidler"

import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle, RegularPolygon
import copy
from .gis import find_gis_components, direct_connections
import matplotlib as mpl

def create_bus_collection(net, busses, size=5, marker="o", patch_type="circle", colors=None,
                          infofunc=None, cmap=None, picker=False, **kwargs):
    """
    Creates a matplotlib patch collection

    Busses are the nodal points of the network that all other elements connect to.

    Input:

        **net** (PandapowerNet) - The pandapower network
        
        **busses** (list) - The busses for which the collections are created
        
    Optional:
    
        **size** (int, 5) - patch size

        **marker** (str, "o") - patch 

        **patch_type** (str, "circle") - patch 
        
        *colors** (list, None) - 
        
    """
    if len(busses) == 0:
        return PatchCollection([])
    busses = list(busses)
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
            infos.append(infofunc(busses[i]))
        return fig
    patches = [figmaker(x, y, i)
               for i, (x, y) in enumerate(zip(net.bus_geodata.loc[busses].x.values,
                                              net.bus_geodata.loc[busses].y.values))
               if x != -1 and x != np.nan]
    pc = PatchCollection(patches, match_original=True, cmap=cmap, picker=picker)
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    pc.info = infos
    return pc

def create_line_collection(net, lines, use_line_geodata=True, infofunc=None, **kwargs):
    """
    
    """

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
    lc.info = info
    return lc


def create_connection_collection(net, connections, lwd, c, z=5):
    lines = [[(net.bus_geodata.x.at[a], net.bus_geodata.y.at[a]),
              (net.bus_geodata.x.at[b], net.bus_geodata.y.at[b])]
              for a, b in connections
              if a in net.bus_geodata.index and b in net.bus_geodata.index]
    return LineCollection(lines, alpha=1, color=c, linewidth=lwd, zorder = z)

def create_direct_line_collection(net, lines=None, gis_components=None, linewidth=1., alpha=.75,
                                  zorder=1, color=(.4,.4,.4), linestyles="solid"):
    """
    """
    ll = copy.deepcopy(gis_components) or list(find_gis_components(net))
    if lines is not None:
        ll = [l for l in ll if len(set(l[2]) & set(lines)) > 0]
    return LineCollection(direct_connections(net, ll), color=color, linewidth=linewidth,
                          alpha=alpha, zorder=zorder, linestyles=linestyles)


def create_trennstellen_collection(net, switches=None, scale=10, **kwargs):
    """
    """   
    lc = []
    if switches is None:
        switches = net.switch.query("closed==0 and et=='l'").index
    for bus, element in net.switch.loc[switches][["bus", "element"]].values:
        direction = 1 if net.line.from_bus.at[element] == bus else -1
        if not element in net.line_geodata.index:
            continue
        b1, b2 = np.array(net.line_geodata.coords.loc[element][::direction][:2])
        d = b1 - (b1 - b2) * 0.5
        b = (b1 - b2) / np.linalg.norm(b1 - b2)
        dv = np.array([-b[1], b[0]])
        lc.append([d + dv * scale, d - dv * scale])
    return LineCollection(lc, **kwargs)

def create_trafo_collection(net, tid, **kwargs):
    """
    Returns a Collection of lines from the
    lv-bus to the hv-bus. Note: its recommended to
    make them somewhat thicker using linewidths,
    in order to make them visible
    """
    trafos = net.trafo.loc[tid]

    hv_geo = list(zip(net.bus_geodata.loc[trafos["hv_bus"], "x"].values,
                 net.bus_geodata.loc[trafos["hv_bus"], "y"].values))
    lv_geo = list(zip(net.bus_geodata.loc[trafos["lv_bus"], "x"].values,
                 net.bus_geodata.loc[trafos["lv_bus"], "y"].values))

    tg = list(zip(hv_geo, lv_geo))

    return LineCollection([(tgd[0], tgd[1]) for tgd in tg], **kwargs)

                            

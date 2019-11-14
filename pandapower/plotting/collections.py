# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import inspect
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection, Collection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from pandas import isnull
from pandapower.plotting.patch_makers import load_patches, _rotate_dim2, node_patches, gen_patches,\
    sgen_patches, ext_grid_patches

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CustomTextPath(TextPath):
    """
    Create a path from the text. This class provides functionality for deepcopy, which is not
    implemented for TextPath.
    """

    def __init__(self, xy, s, size=None, prop=None, _interpolation_steps=1, usetex=False):
        """
        Create a path from the text. No support for TeX yet. Note that
        it simply is a path, not an artist. You need to use the
        PathPatch (or other artists) to draw this path onto the
        canvas.

        xy : position of the text.
        s : text
        size : font size
        prop : font property
        """
        if prop is None:
            prop = FontProperties()
        TextPath.__init__(self, xy, s, size=size, prop=prop,
                          _interpolation_steps=_interpolation_steps, usetex=usetex)
        self.s = s
        self.usetex = usetex
        self.prop = prop

    def __deepcopy__(self, memo=None):
        """
        Returns a deepcopy of the `CustomTextPath`, which is not implemented for TextPath
        """
        return self.__class__(copy.deepcopy(self._xy), copy.deepcopy(self.s), size=self.get_size(),
                              prop=copy.deepcopy(self.prop),
                              _interpolation_steps=self._interpolation_steps, usetex=self.usetex)


def create_annotation_collection(texts, coords, size, prop=None, **kwargs):
    """
    Creates PatchCollection of Texts shown at the given coordinates

    Input:
        **texts** (iterable of strings) - The texts to be
        **coords** (iterable of tuples) - The pandapower network
        **size** (int) - The pandapower network

    OPTIONAL:
        **prop** - FontProperties being passed to the TextPatches
        **kwargs** - Any other keyword-arguments will be passed to the PatchCollection.
    """
    tp = []
    # we convert TextPaths to PathPatches to create a PatchCollection
    if hasattr(size, "__iter__"):
        for i, t in enumerate(texts):
            tp.append(PathPatch(CustomTextPath(coords[i], t, size=size[i], prop=prop)))
    else:
        for t, c in zip(texts, coords):
            tp.append(PathPatch(CustomTextPath(c, t, size=size, prop=prop)))

    return PatchCollection(tp, **kwargs)


def add_cmap_to_collection(collection, cmap, norm, z, cbar_title, clim=None):
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    collection.set_array(np.ma.masked_invalid(z))
    collection.has_colormap = True
    collection.cbar_title = cbar_title
    if clim is not None:
        collection.set_clim(clim)
    return collection


def create_node_collection(nodes, coords, size=5, patch_type="circle", color=None, picker=False,
                           infos=None, **kwargs):
    if len(coords) == 0:
        return None

    infos = list(infos) if infos is not None else []
    patches = node_patches(coords, size, patch_type, color, **kwargs)
    pc = PatchCollection(patches, match_original=True, picker=picker)
    pc.node_indices = np.array(nodes)

    pc.patch_type = patch_type
    pc.size = size
    if 'orientation' in kwargs:
        pc.orientation = kwargs['orientation']
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    pc.info = infos
    pc.patch_type = patch_type
    pc.size = size
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    if 'orientation' in kwargs:
        pc.orientation = kwargs['orientation']

    return pc


def create_line2d_collection(coords, indices, infos=None, picker=None, **kwargs):
    # This would be done anyways by matplotlib - doing it explicitly makes it a) clear and
    # b) prevents unexpected behavior when observing colors being "none"
    lc = LineCollection(coords, picker=picker, **kwargs)
    lc.indices = np.array(indices)
    lc.info = infos if infos is not None else list()
    return lc


def create_node_element_collection(node_coords, patch_maker, size=1., infos=None, orientation=np.pi,
                                   picker=False, patch_facecolor="w", patch_edgecolor="k",
                                   line_color="k", **kwargs):
    """
    Creates matplotlib collections of node elements. All node element collections usually consist of
    one patch collection representing the element itself and a small line collection that connects
    the element to the respective node.

    Input:
        **node_coords** (iterable) - the coordinates (x, y) of the nodes with shape (N, 2)

        **patch_maker** (function) - a function to generate the patches of which the collections \
            consist (cf. the patch_maker module)

    OPTIONAL:
        **size** (float, 1) - patch size

        **infos** (iterable, None) - additional infos to add for each element

        **orientation** (float, np.pi) - orientation of load collection. pi is directed downwards,
            increasing values lead to clockwise direction changes.

        **patch_facecolor** (matplotlib color, "w") - color of the patch face (content)

        **patch_edgecolor** (matplotlib color, "k") - color of the patch edges

        **line_color** (matplotlib color, "k") - color of the connecting lines

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **patch_coll** - patch collection representing the element

        **line_coll** - connecting line collection
    """
    angles = orientation if hasattr(orientation, '__iter__') else [orientation] * len(node_coords)
    assert len(node_coords) == len(angles), \
        "The length of coordinates does not match the length of the orientation angles!"
    infos = [] if infos is None else infos
    lines, polys = patch_maker(node_coords, size, orientation, **kwargs)
    patch_coll = PatchCollection(polys, facecolor=patch_facecolor, edgecolor=patch_edgecolor,
                                 picker=picker, **kwargs)
    line_coll = LineCollection(lines, color=line_color, picker=picker, **kwargs)
    patch_coll.info = infos
    line_coll.info = infos
    return patch_coll, line_coll


def create_bus_collection(net, buses=None, size=5, patch_type="circle", colors=None, z=None,
                          cmap=None, norm=None, infofunc=None, picker=False, bus_geodata=None,
                          cbar_title="Bus Voltage [pu]", **kwargs):
    """
    Creates a matplotlib patch collection of pandapower buses.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created.
            If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "rect" for a rectangle
                - "poly<n>" for a polygon with n edges

        **infofunc** (function, None) - infofunction for the patch element

        **colors** (list, None) - list of colors for every element

        **z** (array, None) - array of bus voltage magnitudes for colormap. Used in case of given
            cmap. If None net.res_bus.vm_pu is used.

        **cmap** (ListedColormap, None) - colormap for the patch colors

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **picker** (bool, False) - picker argument passed to the patch collection

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

        **cbar_title** (str, "Bus Voltage [pu]") - colormap bar title in case of given cmap

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **pc** - patch collection
    """
    buses = net.bus.index.tolist() if buses is None else list(buses)
    if len(buses) == 0:
        return None
    if bus_geodata is None:
        bus_geodata = net["bus_geodata"]

    coords = list(zip(bus_geodata.loc[buses, "x"].values, bus_geodata.loc[buses, "y"].values))

    infos = [infofunc(buses[i]) for i in range(len(buses))]

    pc = create_node_collection(buses, coords, size, patch_type, colors, picker, infos, **kwargs)

    if cmap is not None:
        add_cmap_to_collection(pc, cmap, norm, z, cbar_title)

    return pc


def create_line_collection(net, lines=None, line_geodata=None, bus_geodata=None,
                           use_bus_geodata=False, infofunc=None,
                           cmap=None, norm=None, picker=False, z=None,
                           cbar_title="Line Loading [%]", clim=None,
                           plot_colormap=True, **kwargs):
    """
    Creates a matplotlib line collection of pandapower lines.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created. If None, all lines
            in the network are considered.

        **line_geodata** (DataFrame, None) - coordinates to use for plotting. If None,
            net["line_geodata"] is used

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

        **use_bus_geodata** (bool, False) - Defines whether bus or line geodata are used.

         **infofunc** (function, None) - infofunction for the patch element

        **cmap** - colormap for the patch colors

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **picker** (bool, False) - picker argument passed to the patch collection

        **z** (array, None) - array of line loading magnitudes for colormap. Used in case of given
            cmap. If None net.res_line.loading_percent is used.

        **cbar_title** (str, "Line Loading [%]") - colormap bar title in case of given cmap

        **clim** (tuple of floats, None) - setting the norm limits for image scaling

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection
    """
    global linetab
    if use_bus_geodata is False and net.line_geodata.empty:
        # if bus geodata is available, but no line geodata
        logger.warning("use_bus_geodata is automatically set to True, since net.line_geodata is "
                       "empty.")
        use_bus_geodata = True

    if use_bus_geodata:
        linetab = net.line if lines is None else net.line.loc[lines]
    lines = net.line.index.tolist() if lines is None else list(lines)
    if len(lines) == 0:
        return None
    if line_geodata is None:
        line_geodata = net["line_geodata"]

    if len(lines) == 0:
        return None

    lines_with_geo = []
    if use_bus_geodata:
        if bus_geodata is None:
            bus_geodata = net["bus_geodata"]
        data = []
        buses_with_geodata = bus_geodata.index.values
        bg_dict = bus_geodata.to_dict()  # transforming to dict to make access faster
        for line, fb, tb in zip(linetab.index, linetab.from_bus.values, linetab.to_bus.values):
            if fb in buses_with_geodata and tb in buses_with_geodata:
                lines_with_geo.append(line)
                data.append(([(bg_dict["x"][fb], bg_dict["y"][fb]),
                              (bg_dict["x"][tb], bg_dict["y"][tb])],
                             infofunc(line) if infofunc else []))
        lines_without_geo = set(lines) - set(lines_with_geo)
        if lines_without_geo:
            logger.warning("Could not plot lines %s. Bus geodata is missing for those lines!"
                           % lines_without_geo)
    else:
        data = []
        coords_dict = line_geodata.coords.to_dict()  # transforming to dict to make access faster
        for line in lines:
            if line in line_geodata.index:
                lines_with_geo.append(line)
                data.append((coords_dict[line], infofunc(line) if infofunc else []))

        lines_without_geo = set(lines) - set(lines_with_geo)
        if len(lines_without_geo) > 0:
            logger.warning("Could not plot lines %s. Line geodata is missing for those lines!"
                           % lines_without_geo)

    if len(data) == 0:
        return None

    data, info = list(zip(*data))

    # This would be done anyways by matplotlib - doing it explicitly makes it a) clear and
    # b) prevents unexpected behavior when observing colors being "none"
    lc = LineCollection(data, picker=picker, **kwargs)
    lc.line_indices = np.array(lines_with_geo)
    if cmap is not None:
        if z is None:
            z = net.res_line.loading_percent.loc[lines_with_geo]
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        if clim is not None:
            lc.set_clim(clim)

        lc.set_array(np.ma.masked_invalid(z))
        lc.has_colormap = plot_colormap
        lc.cbar_title = cbar_title
    lc.info = info

    return lc


def create_trafo_connection_collection(net, trafos=None, bus_geodata=None, infofunc=None,
                                       cmap=None, clim=None, norm=None, z=None,
                                       cbar_title="Transformer Loading", **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The transformers for which the collections are created.
            If None, all transformers in the network are considered.

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection
    """
    trafos = net.trafo if trafos is None else net.trafo.loc[trafos]

    if bus_geodata is None:
        bus_geodata = net["bus_geodata"]

    hv_geo = list(zip(bus_geodata.loc[trafos["hv_bus"], "x"].values,
                      bus_geodata.loc[trafos["hv_bus"], "y"].values))
    lv_geo = list(zip(bus_geodata.loc[trafos["lv_bus"], "x"].values,
                      bus_geodata.loc[trafos["lv_bus"], "y"].values))

    tg = list(zip(hv_geo, lv_geo))

    info = [infofunc(tr) if infofunc is not None else [] for tr in trafos.index.values]

    lc = LineCollection([(tgd[0], tgd[1]) for tgd in tg], **kwargs)
    lc.info = info
    if cmap is not None:
        if z is None:
            z = net.res_trafo.loading_percent.loc[trafos.index]
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        if clim is not None:
            lc.set_clim(clim)

        lc.set_array(np.ma.masked_invalid(z))
        lc.has_colormap = True
        lc.cbar_title = cbar_title
    return lc


def create_trafo3w_connection_collection(net, trafos=None, bus_geodata=None, infofunc=None,
                                         **kwargs):
    """
    Creates a matplotlib line collection of pandapower 3W-transformers.
    This function can be used to create line collections for voltage fall diagrams.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The 3W-transformers for which the collections are created.
            If None, all 3W-transformers in the network are considered.

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection
    """
    trafos = net.trafo3w if trafos is None else net.trafo3w.loc[trafos]

    if bus_geodata is None:
        bus_geodata = net["bus_geodata"]

    hv_geo, mv_geo, lv_geo = (list(zip(*(bus_geodata.loc[trafos[column], var].values
                                         for var in ['x', 'y'])))
                              for column in ['hv_bus', 'mv_bus', 'lv_bus'])

    # create 3 connection lines, each of 2 points, for every trafo3w
    tg = [x for c in [list(combinations(y, 2))
                      for y in zip(hv_geo, mv_geo, lv_geo)]
          for x in c]

    # 3 times infofunc for every trafo
    info = [infofunc(x) if infofunc is not None else []
            for tr in [(t, t, t) for t in trafos.index.values]
            for x in tr]

    lc = LineCollection(tg, **kwargs)
    lc.info = info

    return lc


def create_trafo_collection(net, trafos=None, picker=False, size=None,
                            infofunc=None, cmap=None, norm=None, z=None, clim=None,
                            cbar_title="Transformer Loading",
                            plot_colormap=True, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The transformers for which the collections are created.
            If None, all transformers in the network are considered.

        **picker** (bool, False) - picker argument passed to the patch collection

        **size** (int, None) - size of transformer symbol circles. Should be >0 and
            < 0.35*bus_distance

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection

        **pc** - patch collection
    """
    trafo_table = net.trafo if trafos is None else net.trafo.loc[trafos]
    lines = []
    circles = []
    infos = []
    color = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    if cmap is not None and z is None:
        z = net.res_trafo.loading_percent

    for i, idx in enumerate(trafo_table.index):
        p1 = net.bus_geodata[["x", "y"]].loc[net.trafo.at[idx, "hv_bus"]].values
        p2 = net.bus_geodata[["x", "y"]].loc[net.trafo.at[idx, "lv_bus"]].values
        if np.all(p1 == p2):
            continue
        d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        if size is None:
            size_this = np.sqrt(d) / 5
        else:
            size_this = size
        off = size_this * 0.35
        circ1 = (0.5 - off / d) * (p1 - p2) + p2
        circ2 = (0.5 + off / d) * (p1 - p2) + p2
        ec = color if cmap is None else cmap(norm(z.at[idx]))
        circles.append(Circle(circ1, size_this, fc=(1, 0, 0, 0), ec=ec))
        circles.append(Circle(circ2, size_this, fc=(1, 0, 0, 0), ec=ec))

        lp1 = (0.5 - off / d - size_this / d) * (p2 - p1) + p1
        lp2 = (0.5 - off / d - size_this / d) * (p1 - p2) + p2
        lines.append([p1, lp1])
        lines.append([p2, lp2])
        if infofunc is not None:
            infos.append(infofunc(i))
            infos.append(infofunc(i))
    if len(circles) == 0:
        return None, None
    lc = LineCollection(lines, color=color, picker=picker, linewidths=linewidths, **kwargs)
    lc.info = infos
    pc = PatchCollection(circles, match_original=True, picker=picker, linewidth=linewidths,
                         **kwargs)
    pc.info = infos
    if cmap is not None:
        z_duplicated = np.repeat(z.values, 2)
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        if clim is not None:
            lc.set_clim(clim)
        lc.set_array(np.ma.masked_invalid(z_duplicated))
        lc.has_colormap = plot_colormap
        lc.cbar_title = cbar_title
    return lc, pc


# noinspection PyArgumentList
def create_trafo3w_collection(net, trafo3ws=None, picker=False, infofunc=None,
                              cmap=None, norm=None, z=None, clim=None,
                              cbar_title="3W-Transformer Loading",
                              plot_colormap=True,
                              **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafo3ws** (list, None) - The three winding transformers for which the collections are
            created. If None, all three winding transformers in the network are considered.

        **picker** (bool, False) - picker argument passed to the patch collection

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection

        **pc** - patch collection
    """
    trafo3w_table = net.trafo3w if trafo3ws is None else net.trafo3w.loc[trafo3ws]
    lines = []
    circles = []
    infos = []
    color = kwargs.pop("color", "k")
    linewidth = kwargs.pop("linewidths", 2.)
    if cmap is not None and z is None:
        z = net.res_trafo3w.loading_percent
    for i, idx in enumerate(trafo3w_table.index):
        # get bus geodata
        p1 = net.bus_geodata[["x", "y"]].loc[net.trafo3w.at[idx, "hv_bus"]].values
        p2 = net.bus_geodata[["x", "y"]].loc[net.trafo3w.at[idx, "mv_bus"]].values
        p3 = net.bus_geodata[["x", "y"]].loc[net.trafo3w.at[idx, "lv_bus"]].values
        if np.all(p1 == p2) and np.all(p1 == p3):
            continue
        p = np.array([p1, p2, p3])
        # determine center of buses and minimum distance center-buses
        center = sum(p) / 3
        d = np.linalg.norm(p - center, axis=1)
        r = d.min() / 3
        # determine closest bus to center and vector from center to circle midpoint in closest
        # direction
        closest = d.argmin()
        to_closest = (p[closest] - center) / d[closest] * 2 * r / 3
        # determine vectors from center to circle midpoint
        order = list(range(closest, 3)) + list(range(closest))
        cm = np.empty((3, 2))
        cm[order.pop(0)] = to_closest
        ang = 2 * np.pi / 3  # 120 degree
        cm[order.pop(0)] = _rotate_dim2(to_closest, ang)
        cm[order.pop(0)] = _rotate_dim2(to_closest, -ang)
        # determine midpoints of circles
        m = center + cm
        # determine endpoints of circles
        e = (center - p) * (1 - 5 * r / 3 / d).reshape(3, 1) + p
        # save circle and line collection data
        ec = color if cmap is None else cmap(norm(z.at[idx]))
        for j in range(3):
            circles.append(Circle(m[j], r, fc=(1, 0, 0, 0), ec=ec))
            lines.append([p[j], e[j]])

        if infofunc is not None:
            infos.append(infofunc(i))
            infos.append(infofunc(i))
    if len(circles) == 0:
        return None, None
    lc = LineCollection(lines, color=color, picker=picker, linewidths=linewidth, **kwargs)
    lc.info = infos
    pc = PatchCollection(circles, match_original=True, picker=picker, linewidth=linewidth, **kwargs)
    pc.info = infos
    if cmap is not None:
        z_duplicated = np.repeat(z.values, 3)
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        if clim is not None:
            lc.set_clim(clim)
        lc.set_array(np.ma.masked_invalid(z_duplicated))
        lc.has_colormap = plot_colormap
        lc.cbar_title = cbar_title
    return lc, pc


def create_busbar_collection(net, buses=None, infofunc=None, cmap=None, norm=None, picker=False,
                             z=None, cbar_title="Bus Voltage [p.u.]", clim=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower buses plotted as busbars

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created. If None, all buses
            which have the entry coords in bus_geodata are considered.

        **line_geodata** (DataFrame, None) - coordinates to use for plotting. If None,
            net["line_geodata"] is used

        **infofunc** (function, None) - infofunction for the line element

        **cmap** - colormap for the line colors

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **picker** (bool, False) - picker argument passed to the patch collection

        **z** (array, None) - array of line loading magnitudes for colormap. Used in case of given
            cmap. If None net.res_line.loading_percent is used.

        **cbar_title** (str, "Line Loading [%]") - colormap bar title in case of given cmap

        **clim** (tuple of floats, None) - setting the norm limits for image scaling

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **bbc** - busbar collection
    """

    if buses is None:
        buses = net.bus_geodata.loc[~isnull(net.bus_geodata.coords)].index

    if cmap is not None:
        # determine color of busbar by vm_pu
        if z is None and net is not None:
            z = net.res_bus.vm_pu.loc[buses]
        else:
            logger.warning("z is None and no net is provided")

    # the busbar is just a line collection with coords from net.bus_geodata
    return create_line_collection(net, lines=buses, line_geodata=net.bus_geodata, bus_geodata=None,
                                  norm=norm, cmap=cmap, infofunc=infofunc, picker=picker, z=z,
                                  cbar_title=cbar_title, clim=clim, **kwargs)


def create_load_collection(net, loads=None, size=1., infofunc=None, orientation=np.pi, picker=False,
                           **kwargs):
    """
    Creates a matplotlib patch collection of pandapower loads.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **loads** (list of ints, None) - the loads to include in the collection

        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch element

        **orientation** (float, np.pi) - orientation of load collection. pi is directed downwards,
            increasing values lead to clockwise direction changes.

        **picker** (bool, False) - picker argument passed to the patch collectionent

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **load_pc** - patch collection

        **load_lc** - line collection
    """
    if loads is None:
        loads = net.load.index
    infos = [infofunc(i) for i in range(len(loads))]
    node_coords = net.bus_geodata.loc[:, ["x", "y"]].values[net.load.loc[loads, "bus"].values]
    load_pc, load_lc = create_node_element_collection(
        node_coords, load_patches, size=size, infos=infos, orientation=orientation,
        offset=2. * size, picker=picker, **kwargs)
    return load_pc, load_lc


def create_gen_collection(net, gens=None, size=1., infofunc=None, orientation=np.pi, picker=False,
                          **kwargs):
    """
    Creates a matplotlib patch collection of pandapower gens.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **gens** (list of ints, None) - the generators to include in the collection

        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch element

        **orientation** (float or list of floats, np.pi) - orientation of gen collection. pi is\
            directed downwards, increasing values lead to clockwise direction changes.

        **picker** (bool, False) - picker argument passed to the patch collectionent

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **gen_pc** - patch collection

        **gen_lc** - line collection
    """
    if gens is None:
        gens = net.gen.index
    infos = [infofunc(i) for i in range(len(gens))]
    node_coords = net.bus_geodata.loc[:, ["x", "y"]].values[net.gen.loc[gens, "bus"].values]
    gen_pc, gen_lc = create_node_element_collection(
        node_coords, gen_patches, size=size, infos=infos, orientation=orientation,
        offset=1.7 * size, picker=picker, **kwargs)
    return gen_pc, gen_lc


def create_sgen_collection(net, sgens=None, size=1., infofunc=None, orientation=np.pi, picker=False,
                           **kwargs):
    """
    Creates a matplotlib patch collection of pandapower sgen.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **sgens** (list of ints, None) - the static generators to include in the collection

        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch elem

        **picker** (bool, False) - picker argument passed to the patch collectionent

        **orientation** (float, np.pi) - orientation of static generator collection. pi is directed\
            downwards, increasing values lead to clockwise direction changes.

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **sgen_pc** - patch collection

        **sgen_lc** - line collection
    """
    if sgens is None:
        sgens = net.sgen.index
    infos = [infofunc(i) for i in range(len(sgens))]
    node_coords = net.bus_geodata.loc[:, ["x", "y"]].values[net.sgen.loc[sgens, "bus"].values]
    sgen_pc, sgen_lc = create_node_element_collection(
        node_coords, sgen_patches, size=size, infos=infos, orientation=orientation,
        offset=1.7 * size, r_triangle=size * 0.4, picker=picker, **kwargs)
    return sgen_pc, sgen_lc


def create_ext_grid_collection(net, size=1., infofunc=None, orientation=0, picker=False,
                               ext_grids=None, ext_grid_buses=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower ext_grid. Parameters
    ext_grids, ext_grid_buses can be used to specify, which ext_grids the collection should be
    created for.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch element

        **orientation** (float, 0) - orientation of load collection. 0 is directed upwards,
            increasing values lead to clockwise direction changes.

        **picker** (bool, False) - picker argument passed to the patch collection

        **ext_grid_buses** (np.ndarray, None) - buses to be used as ext_grid locations

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **ext_grid1** - patch collection

        **ext_grid2** - patch collection
    """
    if ext_grids is None:
        ext_grids = net.ext_grid.index.values
    if ext_grid_buses is None:
        ext_grid_buses = net.ext_grid.bus.loc[ext_grids].values
    else:
        assert len(ext_grids) == len(ext_grid_buses), \
            "Length mismatch between chosen ext_grids and ext_grid_buses."
    infos = [infofunc(ext_grid_idx) for ext_grid_idx in ext_grids]

    node_coords = net.bus_geodata.loc[:, ["x", "y"]].values[ext_grid_buses]
    ext_grid_pc, ext_grid_lc = create_node_element_collection(
        node_coords, ext_grid_patches, size=size, infos=infos, orientation=orientation,
        offset=1.7 * size, r_triangle=size * 0.4, picker=picker, hatch='XXX', **kwargs)
    return ext_grid_pc, ext_grid_lc


def onBusbar(net, bus, line_coords):
    """
    Checks if the first or the last coordinates of a line are on a bus

    Input:
        **net** (pandapowerNet) - The pandapower network

        **bus** (int) - ID of the target bus on one end of the line

        **line_coords (array, shape= (,2L)) -
        The linegeodata of the line. The first row should be the coordinates
        of bus a and the last should be the coordinates of bus b. The points
        in the middle represent the bending points of the line

    OUTPUT:
        **intersection** (tupel, shape= (2L,))- Intersection point of the line with the given bus. \
            Can be used for switch position

    """
    # If the line has no Intersection line will be returned and the bus coordinates can be used to
    # calculate the switch position
    intersection = None
    bus_coords = net.bus_geodata.loc[bus, "coords"]
    # Checking if bus has "coords" - if it is a busbar
    if bus_coords is not None and bus_coords is not np.NaN and line_coords is not None:
        for i in range(len(bus_coords) - 1):
            try:
                # Calculating slope of busbar-line. If the busbar-line is vertical ZeroDivisionError
                # occurs
                m = (bus_coords[i + 1][1] - bus_coords[i][1]) / \
                    (bus_coords[i + 1][0] - bus_coords[i][0])
                # Clculating the off-set of the busbar-line
                b = bus_coords[i][1] - bus_coords[i][0] * m
                # Checking if the first end of the line is on the busbar-line
                if 0 == m * line_coords[0][0] + b - line_coords[0][1]:
                    # Checking if the end of the line is in the Range of the busbar-line
                    if bus_coords[i + 1][0] <= line_coords[0][0] <= bus_coords[i][0]\
                            or bus_coords[i][0] <= line_coords[0][0] <= bus_coords[i + 1][0]:
                        # Intersection found. Breaking for-loop
                        intersection = line_coords[0]
                        break
                # Checking if the second end of the line is on the busbar-line
                elif 0 == m * line_coords[-1][0] + b - line_coords[-1][1]:
                    if bus_coords[i][0] >= line_coords[-1][0] >= bus_coords[i + 1][0] \
                            or bus_coords[i][0] <= line_coords[-1][0] <= bus_coords[i + 1][0]:
                        # Intersection found. Breaking for-loop
                        intersection = line_coords[-1]
                        break
            # If the busbar-line is a vertical line and the slope is infinitely
            except ZeroDivisionError:
                # Checking if the first line-end is at the same position
                if bus_coords[i][0] == line_coords[0][0]:
                    # Checking if the first line-end is in the Range of the busbar-line
                    if bus_coords[i][1] >= line_coords[0][1] >= bus_coords[i + 1][1] \
                            or bus_coords[i][1] <= line_coords[0][1] <= bus_coords[i + 1][1]:
                        # Intersection found. Breaking for-loop
                        intersection = line_coords[0]
                        break
                # Checking if the second line-end is at the same position
                elif bus_coords[i][0] == line_coords[-1][0]:
                    if bus_coords[i][1] >= line_coords[-1][1] >= bus_coords[i + 1][1] \
                            or bus_coords[i][1] <= line_coords[-1][1] <= bus_coords[i + 1][1]:
                        # Intersection found. Breaking for-loop
                        intersection = line_coords[-1]
                        break
    # If the bus has no "coords" it mus be a normal bus
    elif bus_coords is np.NaN:
        bus_geo = (net["bus_geodata"].loc[bus, "x"], net["bus_geodata"].loc[bus, "y"])
        # Checking if the first end of the line is on the bus
        if bus_geo == line_coords[0]:
            intersection = line_coords[0]
        # Checking if the second end of the line is on the bus
        elif bus_geo == line_coords[-1]:
            intersection = line_coords[-1]

    return intersection


def create_line_switch_collection(net, size=1, distance_to_bus=3, use_line_geodata=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower line-bus switches.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:

        **size** (float, 1) - Size of the switch patches

        **distance_to_bus** (float, 3) - Distance of the switch patch from the bus patch

        **use_line_geodata** (bool, False) - If True, line coordinates are used to identify the
                                             switch position

        **kwargs - Key word arguments are passed to the patch function

    OUTPUT:
        **switches** - patch collection
    """
    lbs_switches = net.switch.index[net.switch.et == "l"]

    color = kwargs.pop("color", "k")

    switch_patches = []
    for switch in lbs_switches:
        sb = net.switch.bus.loc[switch]
        line = net.line.loc[net.switch.element.loc[switch]]
        fb = line.from_bus
        tb = line.to_bus

        line_buses = {fb, tb}
        target_bus = list(line_buses - {sb})[0]

        if sb not in net.bus_geodata.index or target_bus not in net.bus_geodata.index:
            logger.warning("Bus coordinates for switch %s not found, skipped switch!" % switch)
            continue

        # switch bus and target coordinates
        pos_sb = net.bus_geodata.loc[sb, ["x", "y"]].values
        pos_tb = np.zeros(2)

        use_bus_geodata = False

        if use_line_geodata:
            if line.name in net.line_geodata.index:
                line_coords = net.line_geodata.coords.loc[line.name]
                # check, which end of the line is nearer to the switch bus
                intersection = onBusbar(net, target_bus, line_coords=line_coords)
                if intersection is not None:
                    pos_sb = intersection
                if len(line_coords) >= 2:
                    if abs(line_coords[0][0] - pos_sb[0]) < 0.01 and \
                            abs(line_coords[0][1] - pos_sb[1]) < 0.01:
                        pos_tb = np.array([line_coords[1][0], line_coords[1][1]])
                    else:
                        pos_tb = np.array([line_coords[-2][0], line_coords[-2][1]])
                else:
                    use_bus_geodata = True
            else:
                use_bus_geodata = True

        if not use_line_geodata or use_bus_geodata:
            pos_tb = net.bus_geodata.loc[target_bus, ["x", "y"]]

        # position of switch symbol
        vec = pos_tb - pos_sb
        mag = np.linalg.norm(vec)
        pos_sw = pos_sb + vec / mag * distance_to_bus

        # rotation of switch symbol
        angle = np.arctan2(vec[1], vec[0])
        rotation = Affine2D().rotate_around(pos_sw[0], pos_sw[1], angle)

        # color switch by state
        col = color if net.switch.closed.loc[switch] else "white"

        # create switch patch (switch size is respected to center the switch on the line)
        patch = Rectangle((pos_sw[0] - size / 2, pos_sw[1] - size / 2), size, size, facecolor=col,
                          edgecolor=color)
        # apply rotation
        patch.set_transform(rotation)

        switch_patches.append(patch)

    switches = PatchCollection(switch_patches, match_original=True, **kwargs)
    return switches


def create_bus_bus_switch_collection(net, size=1., helper_line_style=':', helper_line_size=1.,
                                     helper_line_color="gray", **kwargs):
    """
    Creates a matplotlib patch collection of pandapower bus-bus switches. Switches are plotted in
    the center between two buses with a "helper" line (dashed and thin) being drawn between the
    buses as well.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:

        **size** (float, 1.0) - Size of the switch patches

        **helper_line_style** (string, ':') - Line style of the "helper" line being plotted between\
            two buses connected by a bus-bus switch

        **helper_line_size** (float, 1.0) - Line width of the "helper" line being plotted between \
            two buses connected by a bus-bus switch

        **helper_line_color** (string, "gray") - Line color of the "helper" line being plotted \
            between two buses connected by a bus-bus switch

        **kwargs - Key word arguments are passed to the patch function

    OUTPUT:
        **switches**, **helper_lines** - tuple of patch collections
    """
    lbs_switches = net.switch.index[net.switch.et == "b"]
    color = kwargs.pop("color", "k")
    switch_patches = []
    line_patches = []
    for switch in lbs_switches:
        switch_bus = net.switch.bus.loc[switch]
        target_bus = net.switch.element.loc[switch]
        if switch_bus not in net.bus_geodata.index or target_bus not in net.bus_geodata.index:
            logger.warning("Bus coordinates for switch %s not found, skipped switch!" % switch)
            continue
        # switch bus and target coordinates
        pos_sb = net.bus_geodata.loc[switch_bus, ["x", "y"]].values.astype(np.float64)
        pos_tb = net.bus_geodata.loc[target_bus, ["x", "y"]].values.astype(np.float64)
        # position of switch symbol
        vec = pos_tb - pos_sb
        pos_sw = pos_sb + vec * 0.5 if not np.allclose(pos_sb, pos_tb) else pos_tb
        # rotation of switch symbol
        angle = np.arctan2(vec[1], vec[0])
        rotation = Affine2D().rotate_around(pos_sw[0], pos_sw[1], angle)
        # color switch by state
        col = color if net.switch.closed.loc[switch] else "white"
        # create switch patch (switch size is respected to center the switch on the line)
        patch = Rectangle((pos_sw[0] - size / 2, pos_sw[1] - size / 2), size, size, facecolor=col,
                          edgecolor=color)
        # apply rotation
        patch.set_transform(rotation)
        # add to collection lists
        switch_patches.append(patch)
        line_patches.append([pos_sb.tolist(), pos_tb.tolist()])
    # create collections and return
    switches = PatchCollection(switch_patches, match_original=True, **kwargs)
    helper_lines = LineCollection(line_patches, linestyles=helper_line_style,
                                  linewidths=helper_line_size, colors=helper_line_color)
    return switches, helper_lines


def draw_collections(collections, figsize=(10, 8), ax=None, plot_colorbars=True, set_aspect=True,
                     axes_visible=(False, False), copy_collections=True, draw=True):
    """
    Draws matplotlib collections which can be created with the create collection functions.

    Input:
        **collections** (list) - iterable of collection objects, may include tuples of collections

    OPTIONAL:
        **figsize** (tuple, (10,8)) - figsize of the matplotlib figure

        **ax** (axis, None) - matplotlib axis object to plot into, new axis is created if None

        **plot_colorbars** (bool, True) - defines whether colorbars should be plotted

        **set_aspect** (bool, True) - defines whether 'equal' and 'datalim' aspects of axis scaling
            should be set.

        **axes_visible** (tuple, (False, False)) - defines visibility of (xaxis, yaxis)

    OUTPUT:
        **ax** - matplotlib axes
    """

    if ax is None:
        plt.figure(facecolor="white", figsize=figsize)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                            wspace=0.02, hspace=0.04)
    ax = ax or plt.gca()

    add_collections_to_axes(ax, collections, plot_colorbars=plot_colorbars,
                            copy_collections=copy_collections)

    try:
        ax.set_facecolor("white")
    except:
        ax.set_axis_bgcolor("white")
    ax.xaxis.set_visible(axes_visible[0])
    ax.yaxis.set_visible(axes_visible[1])
    if not any(axes_visible):
        # removes bounding box of the plot also
        ax.axis("off")
    if set_aspect:
        ax.set_aspect('equal', 'datalim')
    ax.autoscale_view(True, True, True)
    ax.margins(.02)
    if draw:
        plt.draw()
    return ax


def add_single_collection(c, ax, plot_colorbars, copy_collections):
    if copy_collections:
        c = copy.copy(c)
    ax.add_collection(c)
    if plot_colorbars and hasattr(c, "has_colormap") and c.has_colormap:
        extend = c.extend if hasattr(c, "extend") else "neither"
        cbar_load = plt.colorbar(c, extend=extend, ax=ax)
        if hasattr(c, "cbar_title"):
            cbar_load.ax.set_ylabel(c.cbar_title)


def add_collections_to_axes(ax, collections, plot_colorbars=True, copy_collections=True):
    for c in collections:
        if Collection in inspect.getmro(c.__class__):
            # if Collection is in one of the base classes of c
            add_single_collection(c, ax, plot_colorbars, copy_collections)
        elif isinstance(c, tuple) or isinstance(c, list):
            # if c is a tuple or a list of collections
            add_collections_to_axes(ax, c, plot_colorbars, copy_collections)
        else:
            logger.warning("{} in collections is of unknown type. Skipping".format(c))


if __name__ == "__main__":
    # if 0:
    #     import pandapower as pp
    #
    #     ntw = pp.create_empty_network()
    #     b1 = pp.create_bus(ntw, 10, geodata=(5, 10))
    #     b2 = pp.create_bus(ntw, 0.4, geodata=(5, 15))
    #     b3 = pp.create_bus(ntw, 0.4, geodata=(0, 22))
    #     b4 = pp.create_bus(ntw, 0.4, geodata=(8, 20))
    #     pp.create_gen(ntw, b1, p_mw=0.1)
    #     pp.create_load(ntw, b3, p_mw=0.1)
    #     pp.create_ext_grid(ntw, b4)
    #
    #     pp.create_line(ntw, b2, b3, 2.0, std_type="NAYY 4x50 SE")
    #     pp.create_line(ntw, b2, b4, 2.0, std_type="NAYY 4x50 SE")
    #     pp.create_transformer(ntw, b1, b2, std_type="0.63 MVA 10/0.4 kV")
    #     pp.create_transformer(ntw, b3, b4, std_type="0.63 MVA 10/0.4 kV")
    #
    #     bus_col = create_bus_collection(ntw, size=0.2, color="k")
    #     line_col = create_line_collection(ntw, use_line_geodata=False, color="k", linewidth=3.)
    #     lt, bt = create_trafo_collection(ntw, size=2, linewidth=3.)
    #     load_col1, load_col2 = create_load_collection(ntw, linewidth=2.,
    #                                                   infofunc=lambda x: ("load", x))
    #     gen1, gen2 = create_gen_collection(ntw, linewidth=2.,
    #                                        infofunc=lambda x: ("gen", x))
    #     eg1, eg2 = create_ext_grid_collection(ntw, size=2.,
    #                                           infofunc=lambda x: ("ext_grid", x))
    #
    #     draw_collections([bus_col, line_col, load_col1, load_col2, gen1, gen2, lt, bt, eg1, eg2])
    # else:
    #     pass
    pass

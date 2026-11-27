# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = [
    "ellipse_patches",
    "ext_grid_patches",
    "gen_patches",
    "load_patches",
    "node_patches",
    "polygon_patches",
    "pv_patch",
    "rectangle_patches",
    "sgen_patches",
    "storage_patches",
    "trafo_patches",
    "vsc_patches",
    "ward_patches",
    "wp_patch",
    "wye_patch",
    "xward_patches",
]

import logging
import math
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING

import geojson
import geojson.utils
import numpy as np
from geojson import Point

from pandapower.auxiliary import soft_dependency_error
from pandapower.plotting.plotting_toolbox import (
    _rotate_dim2,
    get_angle_list,
    get_color_list,
    get_linewidth_list,
    get_list,
    safe_geojson_loads,
)

if find_spec("matplotlib"):
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Arc, Circle, Ellipse, Patch, PathPatch, Polygon, Rectangle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

if TYPE_CHECKING and not MATPLOTLIB_INSTALLED:
    raise UserWarning("matplotlib required for type checking")

logger = logging.getLogger(__name__)


def wye_patch(node_geo, offset, size, r_triangle, angle, facecolor, edgecolor) -> tuple[list["Patch"], list]:
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    polys: list[Patch] = []
    lines = []
    mid_circ = node_geo + _rotate_dim2(np.array([0, offset + size]), angle)
    circ_edge = node_geo + _rotate_dim2(np.array([0, offset]), angle)
    mid_tri1 = mid_circ + _rotate_dim2(np.array([r_triangle, -r_triangle / 4]), angle)
    mid_tri2 = mid_circ + _rotate_dim2(np.array([-r_triangle, r_triangle / 4]), angle)
    # dropped perpendicular foot of triangle1
    perp_foot1 = mid_tri1 + _rotate_dim2(np.array([0, -r_triangle / 2]), angle)
    line_end1 = perp_foot1 + +_rotate_dim2(np.array([-2.5 * r_triangle, 0]), angle)
    perp_foot2 = mid_tri2 + _rotate_dim2(np.array([0, r_triangle / 2]), angle)
    line_end2 = perp_foot2 + +_rotate_dim2(np.array([2.5 * r_triangle, 0]), angle)

    polys.append(Circle(mid_circ, size, fc=facecolor, ec=edgecolor))
    polys.append(
        RegularPolygon(mid_tri1, numVertices=3, radius=r_triangle, orientation=-angle, fc=facecolor, ec=edgecolor)
    )
    polys.append(
        RegularPolygon(
            mid_tri2, numVertices=3, radius=r_triangle, orientation=math.pi - angle, fc=facecolor, ec=edgecolor
        )
    )
    lines.append((node_geo, circ_edge))
    lines.append((perp_foot1, line_end1))
    lines.append((perp_foot2, line_end2))
    return polys, lines


def wp_patch(
    node_geo: Point,
    offset: float,
    size: float,
    angle: float,
    facecolor: str,
    edgecolor: str,
    blade_coord1: float,
    blade_coord2: float,
    hub_size: float,
    path: "type[Path]",
) -> tuple[list["Patch"], list]:
    """
    Generate Patch for wind power plant.

    Parameters:
        node_geo: Location at which the patch will be generated.
        offset:
        size:
        angle:
        facecolor:
        edgecolor:
        blade_coord1:
        blade_coord2:
        hub_size:
        path:

    Returns:
        - a list of Patch objects
        - a list of coords for lines
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    polys: list[Patch] = []
    lines = []
    mid_circ = node_geo + _rotate_dim2(np.array([0, offset + size]), angle)
    circ_edge = node_geo + _rotate_dim2(np.array([0, offset]), angle)
    circ_topedge = circ_edge + _rotate_dim2(np.array([0, 2 * size]), angle)
    mid_midcirc = node_geo + _rotate_dim2(np.array([0, offset + size]), angle)

    codes, verts = zip(
        *[
            (path.MOVETO, mid_midcirc),
            (
                path.LINETO,
                circ_topedge,
            ),
            (
                path.LINETO,
                (circ_topedge + _rotate_dim2(np.array([-blade_coord1, -blade_coord1]), angle)),
            ),
            (
                path.LINETO,
                (mid_midcirc + _rotate_dim2(np.array([-blade_coord2, +blade_coord2]), angle)),
            ),
            # Blade1
            (path.CLOSEPOLY, mid_midcirc),
        ]
    )
    polys.append(PathPatch(path(verts, codes), fc="k", ec="none", lw=200000000))
    polys.append(Circle(mid_circ, size, fc=facecolor, ec=edgecolor))
    lines.append((node_geo, circ_edge))

    for j in range(3):
        angle = j * 2 * math.pi / 3
        rotated_verts = []
        for vertex in verts:
            # Calculate the vector from the center of the hub to the vertex
            x_diff = vertex[0] - mid_midcirc[0]
            y_diff = vertex[1] - mid_midcirc[1]

            # Rotate the vector by the desired angle
            rotated_x = x_diff * math.cos(angle) - y_diff * math.sin(angle)
            rotated_y = x_diff * math.sin(angle) + y_diff * math.cos(angle)

            # Calculate the coordinates of the rotated vertex
            rotated_vertex = [mid_midcirc[0] + rotated_x, mid_midcirc[1] + rotated_y]

            rotated_verts.append(rotated_vertex)

        rotated_patch = PathPatch(path(rotated_verts, codes), fc="k", ec="none")

        polys.append(rotated_patch)

    polys.append(Circle(mid_midcirc, hub_size, fc="k", ec=edgecolor))  # Center hub
    return polys, lines


def pv_patch(node_geo, offset, size, angle, pv_rect_size, pv_tri_size, facecolor, edgecolor) -> tuple[list, list]:
    """
    Generate PV patch Items

    Parameters:
        node_geo:
        offset:
        size:
        angle:
        pv_rect_size:
        pv_tri_size:
        facecolor:
        edgecolor:

    Returns:
        - a list of Patch objects
        - a list of Line objects
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    polys: list[Patch] = []
    lines = []
    mid_rect = node_geo + _rotate_dim2(np.array([0, 2 * size]), angle)
    rect_lbottom = (mid_rect[0] - (pv_rect_size / 4), mid_rect[1])
    _pv_patch = Rectangle(
        rect_lbottom,
        pv_rect_size / 2,
        pv_rect_size,
        angle=-angle * (180 / math.pi),
        rotation_point=(mid_rect[0], mid_rect[1]),
        ec=edgecolor,
        fc=facecolor,
    )
    polys.append(_pv_patch)
    lines.append((node_geo, mid_rect))
    triangle_base = pv_tri_size
    triangle_height = pv_tri_size * 0.75
    triangle_points = [
        (mid_rect + _rotate_dim2(np.array([-triangle_base / 2, +2 * size]), angle)),
        (mid_rect + _rotate_dim2(np.array([+triangle_base / 2, +2 * size]), angle)),
        (mid_rect + _rotate_dim2(np.array([0, +1.5 * triangle_height]), angle)),
    ]
    triangle_patch = Polygon(triangle_points, ec="k", fc="none")
    polys.append(triangle_patch)
    return polys, lines


def node_patches(node_coords, size, patch_type, colors=None, **kwargs):
    """
    Creates node patches from coordinates translating the patch type into patches.

    Parameters:
        node_coords: coordinates of the nodes to draw
        size: size of the patch (can be interpreted differently, depending on the patch type)
        patch_type: type of patches to create  - can be one of

            - "circle" or "ellipse" for an ellipse (circles are just ellipses with the same width and height)
            - "rect" or "rectangle" for a rectangle
            - "poly<n>" for a polygon with n edges

        colors: colors or color of the patches

    Keyword Arguments:
        These are passed to the patch initialization
        (might contain "width", "height", "angle" depending on the patch type)

    Returns:
        patches - list of rectangle patches for the nodes
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    if patch_type.lower() == "ellipse" or patch_type.lower() == "circle":
        # circles are just ellipses
        if patch_type.lower() == "circle" and len(set(kwargs.keys()) & {"width", "height"}) == 1:
            wh = kwargs["width"] if "width" in kwargs else kwargs["height"]
            width = wh
            height = wh
        else:
            width = kwargs.pop("width", 2 * size)
            height = kwargs.pop("height", 2 * size)
        angle = kwargs.pop("angle", 0)
        return ellipse_patches(node_coords, width, height, angle, color=colors, **kwargs)
    elif patch_type.lower() == "rect" or patch_type.lower() == "rectangle":
        width = kwargs.pop("width", 2 * size)
        height = kwargs.pop("height", 2 * size)
        return rectangle_patches(node_coords, width, height, color=colors, **kwargs)
    elif patch_type.lower().startswith("poly"):
        edges = int(patch_type[4:])
        return polygon_patches(node_coords, size, edges, color=colors, **kwargs)
    else:
        logger.error("Wrong patchtype. Please choose a correct patch type.")
        raise ValueError("Wrong patchtype")


def ellipse_patches(node_coords, width: float, height: float, angle: float = 0.0, color=None, **kwargs):
    """
    Function to create a list of ellipse patches from node coordinates.

    Parameters:
        node_coords (iterable): coordinates of the nodes to draw
        width: width of the ellipse (described by an exterior rectangle)
        height: height of the ellipse (described by an exterior rectangle)
        angle: angle by which to rotate the ellipse
        color (iterable, float): color or colors of the patches

    Keyword Args:
        additional keyword arguments to pass to the Ellipse initialization

    Returns:
        list of ellipse patches for the nodes
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    patches = []
    angles = get_angle_list(angle, len(node_coords))
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col, ang in zip(node_coords, colors, angles):
            patches.append(Ellipse((x, y), width, height, angle=ang, color=col, **kwargs))
    else:
        for (x, y), ang in zip(node_coords, angles):
            patches.append(Ellipse((x, y), width, height, angle=ang, **kwargs))
    return patches


def rectangle_patches(node_coords, width, height, color=None, **kwargs):
    """
    Function to create a list of rectangle patches from node coordinates.

    Parameters:
        node_coords (iterable): coordinates of the nodes to draw
        width (float): width of the rectangle
        height (float): height of the rectangle
        color (iterable, float): color or colors of the patches
        kwargs (dict): additional keyword arguments to pass to the Rectangle initialization

    Returns:
        list of rectangle patches for the nodes
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    patches = []
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col in zip(node_coords, colors):
            patches.append(Rectangle((x - width / 2, y - height / 2), width, height, color=color, **kwargs))
    else:
        for x, y in node_coords:
            patches.append(Rectangle((x - width / 2, y - height / 2), width, height, **kwargs))
    return patches


def polygon_patches(node_coords, radius: float, num_edges: int, color=None, **kwargs) -> "list[RegularPolygon]":
    """
    Function to create a list of polygon patches from node coordinates. The number of edges for the polygon can be
    defined.

    Parameters:
        node_coords (iterable): coordinates of the nodes to draw
        radius: radius for the polygon (from centroid to edges)
        num_edges: number of edges of the polygon
        color (iterable, float): color or colors of the patches

    Keyword Args:
        additional keyword arguments to pass to the Polygon initialization

    Returns:
        list of rectangle patches for the nodes
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    patches: list[RegularPolygon] = []
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col in zip(node_coords, colors):
            patches.append(RegularPolygon((x, y), numVertices=num_edges, radius=radius, color=col, **kwargs))
    else:
        for x, y in node_coords:
            patches.append(RegularPolygon((x, y), numVertices=num_edges, radius=radius, **kwargs))
    return patches


def load_patches(node_coords, size: float, angles, **kwargs) -> "tuple[list, list[RegularPolygon], set[str]]":
    """
    Creation function of patches for loads.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the loads belong to.
        size: size of the patch
        angles (iterable(float), float): angles by which to rotate the patches (in radians)

    Keyword Args:
         offset: offset for the patch (rotation)
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to load patches
        - list containing the load patches
        - set of keywords removed from kwargs

    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(f"{sys._getframe().f_code.co_name}()", "matplotlib")
    offset = kwargs.get("offset", 1.2 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    polys, lines = [], []
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset + size]), all_angles[i])
        p3 = node_geo + _rotate_dim2(np.array([0, offset + size / 2]), all_angles[i])
        polys.append(
            RegularPolygon(
                p2, numVertices=3, radius=size, orientation=-all_angles[i], fc=facecolors[i], ec=edgecolors[i]
            )
        )
        lines.append((node_geo, p3))
    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def gen_patches(
    node_coords, size: float, angles, patch_type, draw_by_type: bool = True, **kwargs
) -> "tuple[list, list[Polygon | Circle | Arc], set[str]]":
    """
    Creation function of patches for generators.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the generators belong to.
        size: size of the patch
        angles (iterable(float) or float): angles by which to rotate the patches (in radians)
        patch_type (iterable(str) or str): types to use for patches

    Keyword Args:
         offset: offset for the patch (rotation)
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to generator patches
        - list containing the generator patches
        - set of keywords removed from kwargs

    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    polys: list[Polygon | Circle | Arc] = []
    lines: list = []
    offset = kwargs.get("offset", 2.0 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    all_patches = get_list(patch_type, len(node_coords), "patches", "patch types")
    edgecolor = kwargs.get("patch_edgecolor", "k")
    facecolor = kwargs.get("patch_facecolor", (1, 0, 0, 0))
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    for i, node_geo in enumerate(node_coords):
        _angle = all_angles[i]
        _patch_type = all_patches
        if draw_by_type:
            pass  # TODO: add support for other patch types
        p2 = node_geo + _rotate_dim2(np.array([0, size + offset]), _angle)
        circ_edge = node_geo + _rotate_dim2(np.array([0, offset]), _angle)
        polys.append(Circle(p2, size, fc=facecolors[i], ec=edgecolors[i]))
        polys.append(
            Arc(p2 + np.array([-size / 6.2, -size / 2.6]), size / 2, size, theta1=65, theta2=120, ec=edgecolors[i])
        )
        polys.append(
            Arc(p2 + np.array([size / 6.2, size / 2.6]), size / 2, size, theta1=245, theta2=300, ec=edgecolors[i])
        )
        lines.append((node_geo, circ_edge))

    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def sgen_patches(node_coords, size: float, angles, patch_type, draw_by_type: bool = True, **kwargs):
    """
    Creation function of patches for static generators.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the static generators belong to.
        size: size of the patch
        angles (iterable(float) or float): angles by which to rotate the patches (in radians)
        patch_type (iterable(str) or str): type of patches
        draw_by_type: should static generators be drawn by type or grouped as one

    Keyword Args:
         offset: offset for the patch (rotation)
         r_triangle:
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to static generator patches
        - list containing the static generator patches
        - set of keywords removed from kwargs

    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    lines = []
    polys: list[Patch] = []
    offset = kwargs.get("offset", 2 * size)
    r_triangle = kwargs.get("r_triangles", size * 0.4)
    hub_size = size * 0.15  # Hub size
    blade_coord1 = size * 0.08  # Blade angle coord
    blade_coord2 = size * 0.22
    pv_rect_size = size * 2
    pv_tri_size = size
    all_angles = get_angle_list(angles, len(node_coords))
    all_patch_types = get_list(patch_type, len(node_coords), "patches", "patch types")
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))

    for i, node_geo in enumerate(node_coords):
        _angle = all_angles[i]
        _patch_type = all_patch_types[i]
        if draw_by_type:
            if _patch_type == "WP":
                _polys, _lines = wp_patch(
                    node_geo,
                    offset,
                    size,
                    _angle,
                    facecolors[i],
                    edgecolors[i],
                    blade_coord1,
                    blade_coord2,
                    hub_size,
                    Path,
                )
            elif _patch_type == "PV":
                _polys, _lines = pv_patch(
                    node_geo, offset, size, _angle, pv_rect_size, pv_tri_size, facecolors[i], edgecolors[i]
                )
            else:
                _polys, _lines = wye_patch(node_geo, offset, size, r_triangle, _angle, facecolors[i], edgecolors[i])
        else:
            _polys, _lines = wye_patch(node_geo, offset, size, r_triangle, _angle, facecolors[i], edgecolors[i])
        polys.extend(_polys)
        lines.extend(_lines)

    return lines, polys, {"offset", "r_triangle", "patch_edgecolor", "patch_facecolor"}


def storage_patches(node_coords, size: float, angles, **kwargs):
    """
    Creation function of patches for storage systems.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the storage system belong to.
        size: size of the patch
        angles (iterable(float), float): angles by which to rotate the patches (in radians)

    Keyword Args:
         offset: offset for the patch (rotation)
         r_triangle:
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to storage patches
        - list containing the storage patches
        - set of keywords removed from kwargs

    """
    polys: list = []
    lines: list[tuple] = []
    offset = kwargs.get("offset", 1 * size)
    r_triangle = kwargs.get("r_triangles", size * 0.4)
    for i, node_geo in enumerate(node_coords):
        mid_circ = node_geo + _rotate_dim2(np.array([0, offset + r_triangle * 2.0]), angles[i])
        circ_edge = node_geo + _rotate_dim2(np.array([0, offset]), angles[i])
        mid_tri1 = mid_circ + _rotate_dim2(np.array([-r_triangle, -r_triangle]), angles[i])

        # dropped perpendicular foot of triangle1
        perp_foot1 = mid_tri1 + _rotate_dim2(np.array([r_triangle * 0.5, -r_triangle / 4]), angles[i])
        line_end1 = perp_foot1 + _rotate_dim2(np.array([1 * r_triangle, 0]), angles[i])

        perp_foot2 = mid_tri1 + _rotate_dim2(np.array([0, -r_triangle]), angles[i])
        line_end2 = perp_foot2 + _rotate_dim2(np.array([2.0 * r_triangle, 0]), angles[i])

        lines.append((node_geo, circ_edge))
        lines.append((perp_foot1, line_end1))
        lines.append((perp_foot2, line_end2))
    return lines, polys, {"offset", "r_triangle", "patch_edgecolor", "patch_facecolor"}


def ext_grid_patches(node_coords, size: float, angles, **kwargs):
    """
    Creation function of patches for external grids.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the external grids belong to.
        size: size of the patch
        angles (iterable(float), float): angles by which to rotate the patches (in radians)

    Keyword Args:
         offset: offset for the patch (rotation)
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to external grid patches
        - list containing the external grid patches
        - set of keywords removed from kwargs

    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    offset = kwargs.get("offset", 2 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    polys, lines = [], []
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset]), all_angles[i])
        p_ll = p2 + _rotate_dim2(np.array([-size, 0]), all_angles[i])
        polys.append(
            Rectangle(
                p_ll,
                2 * size,
                2 * size,
                angle=(-all_angles[i] / math.pi * 180),
                fc=facecolors[i],
                ec=edgecolors[i],
                hatch="XXX",
            )
        )
        lines.append((node_geo, p2))
    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def trafo_patches(coords, size: float, **kwargs):
    """
    Creates a list of patches and line coordinates representing transformers each connecting two nodes.

    Parameters:
        coords ((N, (2, 2)) shaped iterable): list of connecting node coordinates (usually should be `[((x11, y11),
            (x12, y12)), ((x21, y21), (x22, y22)), ...]`)
        size: size of the trafo patches

    Keyword Args:
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:
        - list of coordinates for lines connecting nodes and transformer patches
        - list containing the transformer patches (rings)
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", (1, 0, 0, 0))
    edgecolors = get_color_list(edgecolor, len(coords))
    facecolors = get_color_list(facecolor, len(coords))
    linewidths = kwargs.get("linewidths", 2.0)
    linewidths = get_linewidth_list(linewidths, len(coords), name_entries="trafos")
    circles, lines = [], []

    # load and extract the coords from the geojson object
    coords = [list(geojson.utils.coords(safe_geojson_loads(x))) for x in coords]

    for i, (p1, p2) in enumerate(coords):
        p1 = np.array(p1)
        p2 = np.array(p2)
        if np.all(p1 == p2):
            continue
        d = np.sqrt(np.sum((p1 - p2) ** 2))
        if size is None:
            size_this = np.sqrt(d) / 5
        else:
            size_this = size
        off = size_this * 0.35
        circ1 = (0.5 - off / d) * (p1 - p2) + p2
        circ2 = (0.5 + off / d) * (p1 - p2) + p2
        circles.append(Circle(circ1, size_this, fc=facecolors[i], ec=edgecolors[i], lw=linewidths[i]))
        circles.append(Circle(circ2, size_this, fc=facecolors[i], ec=edgecolors[i], lw=linewidths[i]))

        lp1 = (0.5 - off / d - size_this / d) * (p2 - p1) + p1
        lp2 = (0.5 - off / d - size_this / d) * (p1 - p2) + p2
        lines.append([p1, lp1])
        lines.append([p2, lp2])
    return lines, circles, {"patch_edgecolor", "patch_facecolor"}


def ward_patches(node_coords, size: float, angles, **kwargs):
    """
    Creation function of patches for wards.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the wards belong to.
        size: size of the patch
        angles (iterable(float), float): angles by which to rotate the patches (in radians)

    Keyword Args:
         offset: offset for the patch (rotation)
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to ward patches
        - list containing the ward patches
        - set of keywords removed from kwargs

    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    offset = kwargs.get("offset", 2 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    polys: list[PathPatch | Rectangle] = []
    lines = []
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset]), all_angles[i])
        p_ll = p2 + _rotate_dim2(np.array([-size, 0]), all_angles[i])
        polys.append(
            Rectangle(
                p_ll, 2 * size, 2 * size, angle=(-all_angles[i] / math.pi * 180), fc=facecolors[i], ec=edgecolors[i]
            )
        )
        lines.append((node_geo, p2))

        text_foot: list[float] = (
            p2 + np.array([-1.425 * size, -0.34 * size]) + _rotate_dim2(np.array([size, size]), all_angles[i])
        )

        text_path = TextPath((0, 0), "W", size=0.9 * size, prop=FontProperties(weight="light"))
        transform = Affine2D().translate(text_foot[0], text_foot[1]).rotate(all_angles[i])
        text_patch = PathPatch(transform.transform_path(text_path), edgecolor="black", facecolor="black", lw=0.5)
        polys.append(text_patch)

    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def xward_patches(node_coords, size: float, angles, **kwargs) -> "tuple[list, list[PathPatch | Rectangle], set[str]]":
    """
    Creation function of patches for xwards.

    Parameters:
        node_coords (iterable): coordinates of the nodes that the xwards belong to.
        size: size of the patch
        angles (iterable(float), float): angles by which to rotate the patches (in radians)

    Keyword Args:
         offset: offset for the patch (rotation)
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines leading to elements of xward patches
        - list containing the xward patches
        - set of keywords removed from kwargs

    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    offset = kwargs.get("offset", 2 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "k")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    polys: list[PathPatch | Rectangle] = []
    lines = []
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset]), all_angles[i])
        p_ll = p2 + _rotate_dim2(np.array([-size, 0]), all_angles[i])
        polys.append(
            Rectangle(
                p_ll, 2 * size, 2 * size, angle=(-all_angles[i] / math.pi * 180), fc=facecolors[i], ec=edgecolors[i]
            )
        )
        lines.append((node_geo, p2))

        text_foot = p2 + np.array([-1.755 * size, -0.34 * size]) + _rotate_dim2(np.array([size, size]), all_angles[i])

        text_path = TextPath((0, 0), "XW", size=0.9 * size, prop=FontProperties(weight="light"))
        transform = Affine2D().translate(text_foot[0], text_foot[1]).rotate(all_angles[i])
        text_patch = PathPatch(transform.transform_path(text_path), edgecolor="black", facecolor="black", lw=0.5)
        polys.append(text_patch)

    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def vsc_patches(coords, size: float, **kwargs):
    """
    Creates a list of patches and line coordinates representing VSCs each connecting an AC and a DC node.

    Parameters:
        coords ((N, (2, 2)) shaped iterable): list of connecting node coordinates (usually should be `[((x11, y11),
            (x12, y12)), ((x21, y21), (x22, y22)), ...]`)
        size: size of the VSC patches

    Keyword Args:
         patch_edgecolor: color for the patch outline
         patch_facecolor: color for the patch icon

    Returns:

        - list of coordinates for lines connecting nodes and VSC patches
        - list containing the VSC patches (squares)
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(coords))
    facecolors = get_color_list(facecolor, len(coords))
    linewidths = kwargs.get("linewidths", 2.0)
    linewidths = get_linewidth_list(linewidths, len(coords), name_entries="vscs")
    squares, lines = [], []

    # load and extract the coords from the geojson object
    coords = [list(geojson.utils.coords(safe_geojson_loads(x))) for x in coords]

    for i, (p1, p2) in enumerate(coords):
        p1 = np.array(p1)
        p2 = np.array(p2)
        if np.all(p1 == p2):
            continue
        d = np.sqrt(np.sum((p1 - p2) ** 2))  # distance
        if size is None:
            size_this = np.sqrt(d) / 3
        else:
            size_this = size
        xy1 = p1 - np.array([size_this, size_this])
        xy2 = p2 - np.array([size_this, size_this])
        squares.append(
            Rectangle(
                xy1, size_this * 2, size_this * 2, fc=facecolors[i], ec=edgecolors[i], lw=linewidths[i], hatch="+++"
            )
        )
        squares.append(
            Rectangle(
                xy2, size_this * 2, size_this * 2, fc=facecolors[i], ec=edgecolors[i], lw=linewidths[i], hatch="---"
            )
        )
        lines.append([p1, p2])
    return lines, squares, {"patch_edgecolor", "patch_facecolor"}

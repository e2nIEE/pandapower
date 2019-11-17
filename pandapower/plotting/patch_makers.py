from matplotlib.patches import RegularPolygon, Arc, Circle, Rectangle, Ellipse
import numpy as np
try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _rotate_dim2(arr, ang):
    """
    :param arr: array with 2 dimensions
    :param ang: angle [rad]
    """
    return np.dot(np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]), arr)


def load_patches(node_coords, size, angles, **kwargs):
    offset = kwargs.pop("offset", 2. * size)
    polys, lines = list(), list()
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, size * offset]), angles[i])
        p3 = node_geo + _rotate_dim2(np.array([0, size * (offset - 0.5)]), angles[i])
        polys.append(RegularPolygon(p2, numVertices=3, radius=size, orientation=-angles[i]))
        lines.append((node_geo, p3))
    return lines, polys


def gen_patches(node_coords, size, angles, **kwargs):
    polys, lines = list(), list()
    offset = kwargs.pop("offset", 2. * size)
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, size * offset]), angles[i])
        polys.append(Circle(p2, size))
        polys.append(
            Arc(p2 + np.array([-size / 6.2, -size / 2.6]), size / 2, size, theta1=45, theta2=135))
        polys.append(
            Arc(p2 + np.array([size / 6.2, size / 2.6]), size / 2, size, theta1=225, theta2=315))
        lines.append((node_geo, p2 + np.array([0, size])))
    return lines, polys


def sgen_patches(node_coords, size, angles, **kwargs):
    polys, lines = list(), list()
    offset = kwargs.pop("offset", 2. * size)
    r_triangle = kwargs.pop("r_triangles", size * 0.4)
    for i, node_geo in enumerate(node_coords):
        mid_circ = node_geo + _rotate_dim2(np.array([0, size * offset]), angles[i])
        circ_edge = node_geo + _rotate_dim2(np.array([0, size * (offset - 1)]), angles[i])
        mid_tri1 = mid_circ + _rotate_dim2(np.array([r_triangle, -r_triangle / 4]), angles[i])
        mid_tri2 = mid_circ + _rotate_dim2(np.array([-r_triangle, r_triangle / 4]), angles[i])
        # dropped perpendicular foot of triangle1
        perp_foot1 = mid_tri1 + _rotate_dim2(np.array([0, -r_triangle / 2]), angles[i])
        line_end1 = perp_foot1 + + _rotate_dim2(np.array([-2.5 * r_triangle, 0]), angles[i])
        perp_foot2 = mid_tri2 + _rotate_dim2(np.array([0, r_triangle / 2]), angles[i])
        line_end2 = perp_foot2 + + _rotate_dim2(np.array([2.5 * r_triangle, 0]), angles[i])
        polys.append(Circle(mid_circ, size))
        polys.append(RegularPolygon(mid_tri1, numVertices=3, radius=r_triangle,
                                    orientation=-angles[i]))
        polys.append(RegularPolygon(mid_tri2, numVertices=3, radius=r_triangle,
                                    orientation=np.pi - angles[i]))
        lines.append((node_geo, circ_edge))
        lines.append((perp_foot1, line_end1))
        lines.append((perp_foot2, line_end2))
    return lines, polys


def ext_grid_patches(node_coords, size, angles, **kwargs):
    polys, lines = list(), list()
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, size]), angles[i])
        polys.append(Rectangle((p2[0] - size / 2, p2[1] - size / 2), size, size))
        lines.append((node_geo, p2 - _rotate_dim2(np.array([0, size / 2]), angles[i])))
    return lines, polys


def get_list(individuals, number_entries, name_ind, name_ent):
    if hasattr(individuals, "__iter__"):
        if number_entries == len(individuals):
            return individuals
        elif number_entries > len(individuals):
            logger.warning("The number of given %s (%d) is smaller than the number of %s (%d) to"
                           " draw! The %s will be repeated to fit."
                           % (name_ind, len(individuals), name_ent, number_entries, name_ind))
            return (individuals * (int(number_entries / len(individuals)) + 1))[:number_entries]
        else:
            logger.warning("The number of given %s (%d) is larger than the number of %s (%d) to"
                           " draw! The %s will be capped to fit."
                           % (name_ind, len(individuals), name_ent, number_entries, name_ind))
            return individuals[:number_entries]
    return [individuals] * number_entries


def get_color_list(color, number_entries, name_entries="nodes"):
    return get_list(color, number_entries, "colors", name_entries)


def get_angle_list(angle, number_entries,  name_entries="nodes"):
    return get_list(angle, number_entries, "angles", name_entries)


def ellipse_patches(node_coords, width, height, angle=0, color=None, **kwargs):
    patches = list()
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
    patches = list()
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col in zip(node_coords, colors):
            patches.append(Rectangle((x - width / 2, y - height / 2), color=color, **kwargs))
    else:
        for x, y in node_coords:
            patches.append(Rectangle((x - width / 2, y - height / 2), **kwargs))
    return patches


def polygon_patches(node_coords, radius, num_edges, color=None, **kwargs):
    patches = list()
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col in zip(node_coords, colors):
            patches.append(RegularPolygon([x, y], numVertices=num_edges, radius=radius, color=color,
                                          **kwargs))
    else:
        for x, y in node_coords:
            patches.append(RegularPolygon([x, y], numVertices=num_edges, radius=radius, **kwargs))
    return patches


def node_patches(node_coords, size, patch_type, colors=None, **kwargs):
    if patch_type == 'ellipse' or patch_type == 'circle':  # circles are just ellipses
        width = kwargs.pop("width", 2 * size)
        height = kwargs.pop("height", 2 * size)
        angle = kwargs.pop('angle', 0)
        return ellipse_patches(node_coords, width, height, angle, color=colors, **kwargs)
    elif patch_type == "rect":
        width = kwargs.pop("width", 2 * size)
        height = kwargs.pop("height", 2 * size)
        return rectangle_patches(node_coords, width, height, color=colors, **kwargs)
    elif patch_type.startswith("poly"):
        edges = int(patch_type[4:])
        return polygon_patches(node_coords, size, edges, color=colors, **kwargs)
    else:
        logger.error("Wrong patchtype. Please choose a correct patch type.")
        raise ValueError("Wrong patchtype")


def trafo_patches(coords, size, color):
    colors = get_color_list(color, len(coords))
    circles, lines = list(), list()
    for (p1, p2), col in zip(coords, colors):
        p1 = np.array(p1)
        p2 = np.array(p2)
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
        circles.append(Circle(circ1, size_this, fc=(1, 0, 0, 0), ec=col))
        circles.append(Circle(circ2, size_this, fc=(1, 0, 0, 0), ec=col))

        lp1 = (0.5 - off / d - size_this / d) * (p2 - p1) + p1
        lp2 = (0.5 - off / d - size_this / d) * (p1 - p2) + p2
        lines.append([p1, lp1])
        lines.append([p2, lp2])
    return circles, lines

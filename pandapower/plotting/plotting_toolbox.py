import ast
import re

import numpy as np
import pandas as pd
import geojson

from typing_extensions import deprecated

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _rotate_dim2(arr, ang):
    """
    Rotate the input vector with the given angle.

    :param arr: array with 2 dimensions
    :type arr: np.array
    :param ang: angle [rad]
    :type ang: float
    """
    return np.dot(np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]), arr)


def _get_coords_from_geojson(gj_str):
    if pd.isnull(gj_str):
        return gj_str
    pattern = r'"coordinates"\s*:\s*((?:\[(?:\[[^]]+],?\s*)+\])|\[[^]]+\])'
    matches = re.findall(pattern, gj_str)

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("More than one match found in GeoJSON string")
    for m in matches:
        return ast.literal_eval(m)
    return None

def get_collection_sizes(net, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, load_size=1.0,
                         sgen_size=1.0, switch_size=2.0, switch_distance=1.0, gen_size=1.0):
    """
    Calculates the size for most collection types according to the distance between min and max
    geocoord so that the collections fit the plot nicely

    .. note: This is implemented because if you would choose a fixed values (e.g. bus_size = 0.2),\
        the size could be to small for large networks and vice versa

    :param net: pandapower network for which to create plot
    :type net: pandapowerNet
    :param bus_size: relative bus size
    :type bus_size: float, default 1.
    :param ext_grid_size: relative external grid size
    :type ext_grid_size: float, default 1.
    :param trafo_size: relative trafo size
    :type trafo_size: float, default 1.
    :param load_size: relative load size
    :type load_size: float, default 1.
    :param sgen_size: relative static generator size
    :type sgen_size: float, default 1.
    :param switch_size: relative switch size
    :type switch_size: float, default 2.
    :param switch_distance: relative distance between switches
    :type switch_distance: float, default 1.
    :return: sizes (dict) - dictionary containing all scaled sizes
    """

    lst = net.bus.geo.apply(geojson.loads).apply(geojson.utils.coords).apply(next).values
    mean_distance_between_buses = sum(map(lambda a, b: (a-b)/200, *(map(max, zip(*lst)), map(min, zip(*lst)))))

    sizes = {
        "bus": bus_size * mean_distance_between_buses,
        "ext_grid": ext_grid_size * mean_distance_between_buses * 1.5,
        "switch": switch_size * mean_distance_between_buses * 1,
        "switch_distance": switch_distance * mean_distance_between_buses * 2,
        "load": load_size * mean_distance_between_buses,
        "sgen": sgen_size * mean_distance_between_buses,
        "trafo": trafo_size * mean_distance_between_buses,
        "gen": gen_size * mean_distance_between_buses
    }
    return sizes


def get_list(individuals, number_entries, name_ind, name_ent):
    """
    Auxiliary function to create a list of specified length from an input value that could be either
    an iterable or a single value. Strings are treated as non-iterables. In case of iterables and
    the length not matching the specified length, the input values are repeated or capped to match
    the specified length.

    :param individuals: list or other iterable to adapt to the given length
    :type individuals: iterable
    :param number_entries: length to which individuals shall be entended / capped
    :type number_entries: int
    :param name_ind: Name of the individuals (only for logging pupose).
    :type name_ind: str
    :param name_ent: Name of the entries to which the length belongs (only for logging pupose).
    :type name_ent: str
    :return: new_individuals (list) - a list of length __number_entries__ containing the \
        (extended / capped) individuals
    """
    if hasattr(individuals, "__iter__") and not isinstance(individuals, str):
        if number_entries == len(individuals):
            return list(individuals)
        elif number_entries > len(individuals):
            logger.warning("The number of given %s (%d) is smaller than the number of %s (%d) to"
                           " draw! The %s will be repeated to fit."
                           % (name_ind, len(individuals), name_ent, number_entries, name_ind))
            return (list(individuals) * (int(number_entries / len(individuals)) + 1))[
                   :number_entries]
        else:
            logger.warning("The number of given %s (%d) is larger than the number of %s (%d) to"
                           " draw! The %s will be capped to fit."
                           % (name_ind, len(individuals), name_ent, number_entries, name_ind))
            return list(individuals)[:number_entries]
    return [individuals] * number_entries


def get_color_list(color, number_entries, name_entries="nodes"):
    if (len(color) == 3 or len(color) == 4) and all(isinstance(c, float) for c in color):
        logger.info("Interpreting color %s as rgb or rgba!" % str(color))
        return get_list([color], number_entries, "colors", name_entries)
    return get_list(color, number_entries, "colors", name_entries)


def get_angle_list(angle, number_entries, name_entries="nodes"):
    return get_list(angle, number_entries, "angles", name_entries)


def get_linewidth_list(linewidth, number_entries, name_entries="lines"):
    return get_list(linewidth, number_entries, "linewidths", name_entries)


def get_index_array(indices, net_table_indices):
    if indices is None:
        return np.copy(net_table_indices.values)
    elif isinstance(indices, set):
        return np.array(list(indices))
    return np.array(indices)


def coords_from_node_geodata(element_indices, from_nodes, to_nodes, node_geodata, table_name,
                             node_name="Bus", ignore_no_geo_diff=True, node_geodata_to=None):
    """
    Auxiliary function to get the node coordinates for a number of branches with respective from
    and to nodes. The branch elements for which there is no geodata available are not included in
    the final list of coordinates.

    :param element_indices: Indices of the branch elements for which to find node geodata
    :type element_indices: iterable
    :param from_nodes: Indices of the starting nodes
    :type from_nodes: iterable
    :param to_nodes: Indices of the ending nodes
    :type to_nodes: iterable
    :param node_geodata: Dataframe containing x and y coordinates of the nodes
    :type node_geodata: pd.DataFrame
    :param table_name: Name of the table that the branches belong to (only for logging)
    :type table_name: str
    :param node_name: Name of the node type (only for logging)
    :type node_name: str, default "Bus"
    :param ignore_no_geo_diff: States if branches should be left out, if their length is zero, i.e. \
        from_node_coords = to_node_coords
    :type ignore_no_geo_diff: bool, default True
    :param node_geodata_to: Dataframe containing x and y coordinates of the "to" nodes (optional, default node_geodata)
    :type node_geodata_to: pd.DataFrame
    :return: Return values are:\
        - coords (list) - list of branch coordinates as geojson valid strings
        - elements_with_geo (set) - the indices of branch elements for which coordinates wer found \
            in the node geodata table
    """
    if len(element_indices) == 0:
        return np.array([], dtype=object), np.array([], dtype=int)

    # reduction of from_nodes, to_nodes, node_geodata to intersection
    in_geo = np.isin(from_nodes, node_geodata.index.values) \
        & np.isin(to_nodes, node_geodata.index.values)
    fb_with_geo, tb_with_geo = from_nodes[in_geo], to_nodes[in_geo]
    if sum(~in_geo):
        logger.warning(
            f"No coords found for {table_name}s {np.array(element_indices)[~in_geo]}. {node_name} "
            f"geodata is missing for those {table_name}s!"
        )

    # reduction to not nans
    fb_geo_is_nan = node_geodata.loc[fb_with_geo].isnull().values
    tb_geo_is_nan = node_geodata.loc[tb_with_geo].isnull().values
    not_nan = ~(fb_geo_is_nan | tb_geo_is_nan)
    if n_nans := len(not_nan) - sum(not_nan):
        logger.warning(
            f"NaN coords are returned {n_nans} times for {table_name}s applying {node_name} geodata."
        )

    node_geodata = node_geodata.apply(_get_coords_from_geojson)
    if np.sum(not_nan):
        coords, no_geo_diff = zip(*[
            (f'{{"coordinates": [[{x_from}, {y_from}], [{x_to}, {y_to}]], "type": "LineString"}}',
            x_from == x_to and y_from == y_to) for [x_from, y_from], [x_to, y_to] in zip(
                node_geodata.loc[fb_with_geo[not_nan]], node_geodata.loc[tb_with_geo[not_nan]])])
                #   if not ignore_no_geo_diff or (ignore_no_geo_diff and not ())]
        coords, no_geo_diff = np.array(coords), np.array(no_geo_diff)
    else:
        coords, no_geo_diff = np.array([], dtype=object), np.array([], dtype=bool)

    if ignore_no_geo_diff:
        return coords, np.array(element_indices)[in_geo & not_nan]
    else:
        return coords[~no_geo_diff], np.array(element_indices)[in_geo & not_nan][~no_geo_diff]


def set_line_geodata_from_bus_geodata(net, line_index=None, overwrite=False, ignore_no_geo_diff=True):
    """
    Sets coordinates in net.line.geo based on the from_bus and to_bus coordinates
    in net.bus.geo
    :param net: pandapowerNet
    :param line_index: index of lines, coordinates of which will be set from bus geo (all lines if None)
    :param overwrite: whether the existing coordinates in net.line.geo must be overwritten
    :return: None
    """
    if 'geo' not in net.bus.columns or net.bus.geo.isnull().all():
        logger.warning("The function set_line_geodata_from_bus_geodata requires geodata to be "
                       "present in net.bus.geo")
        return
    if overwrite and line_index is None:
        line_index = net.line.index
    elif not overwrite and line_index is None:
        line_index = net.line.index[net.line.geo.isnull()]
    elif not overwrite and line_index is not None:
        line_index = pd.Index(line_index).intersection(net.line.index[net.line.geo.isnull()])

    geos, line_index_successful = coords_from_node_geodata(
        element_indices=line_index,
        from_nodes=net.line.loc[line_index, 'from_bus'].values,
        to_nodes=net.line.loc[line_index, 'to_bus'].values,
        node_geodata=net.bus.geo,
        table_name="line",
        node_name="bus",
        ignore_no_geo_diff=ignore_no_geo_diff)

    net.line.loc[line_index_successful, 'geo'] = geos

    num_failed = len(line_index) - len(line_index_successful)
    if num_failed > 0:
        logger.info(f"failed to set coordinates of {num_failed} lines")


@deprecated("Use of busbar is not by bus_geodata anymore.")
def position_on_busbar(net, bus, busbar_coords):
    """
    Checks if the first or the last coordinates of a line are on a bus

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus: ID of the target bus on one end of the line
    :type bus: int
    :param busbar_coords: The coordinates of the busbar (beginning and end point).
    :type busbar_coords: array, shape= (,2L)
    :return: intersection (tuple, shape= (2L,))- Intersection point of the line with the given bus.\
        Can be used for switch position
    """
    # If the line has no Intersection line will be returned and the bus coordinates can be used to
    # calculate the switch position
    intersection = None
    bus_coords = net.bus_geodata.loc[bus, "coords"]
    # Checking if bus has "coords" - if it is a busbar
    if bus_coords is not None and bus_coords is not np.nan and busbar_coords is not None:
        for i in range(len(bus_coords) - 1):
            try:
                # Calculating slope of busbar-line. If the busbar-line is vertical ZeroDivisionError
                # occurs
                m = (bus_coords[i + 1][1] - bus_coords[i][1]) / \
                    (bus_coords[i + 1][0] - bus_coords[i][0])
                # Clculating the off-set of the busbar-line
                b = bus_coords[i][1] - bus_coords[i][0] * m
                # Checking if the first end of the line is on the busbar-line
                if 0 == m * busbar_coords[0][0] + b - busbar_coords[0][1]:
                    # Checking if the end of the line is in the Range of the busbar-line
                    if bus_coords[i + 1][0] <= busbar_coords[0][0] <= bus_coords[i][0] \
                            or bus_coords[i][0] <= busbar_coords[0][0] <= bus_coords[i + 1][0]:
                        # Intersection found. Breaking for-loop
                        intersection = busbar_coords[0]
                        break
                # Checking if the second end of the line is on the busbar-line
                elif 0 == m * busbar_coords[-1][0] + b - busbar_coords[-1][1]:
                    if bus_coords[i][0] >= busbar_coords[-1][0] >= bus_coords[i + 1][0] \
                            or bus_coords[i][0] <= busbar_coords[-1][0] <= bus_coords[i + 1][0]:
                        # Intersection found. Breaking for-loop
                        intersection = busbar_coords[-1]
                        break
            # If the busbar-line is a vertical line and the slope is infinitely
            except ZeroDivisionError:
                # Checking if the first line-end is at the same position
                if bus_coords[i][0] == busbar_coords[0][0]:
                    # Checking if the first line-end is in the Range of the busbar-line
                    if bus_coords[i][1] >= busbar_coords[0][1] >= bus_coords[i + 1][1] \
                            or bus_coords[i][1] <= busbar_coords[0][1] <= bus_coords[i + 1][1]:
                        # Intersection found. Breaking for-loop
                        intersection = busbar_coords[0]
                        break
                # Checking if the second line-end is at the same position
                elif bus_coords[i][0] == busbar_coords[-1][0]:
                    if bus_coords[i][1] >= busbar_coords[-1][1] >= bus_coords[i + 1][1] \
                            or bus_coords[i][1] <= busbar_coords[-1][1] <= bus_coords[i + 1][1]:
                        # Intersection found. Breaking for-loop
                        intersection = busbar_coords[-1]
                        break
    # If the bus has no "coords" it mus be a normal bus
    elif bus_coords is np.nan:
        bus_geo = (net["bus_geodata"].loc[bus, "x"], net["bus_geodata"].loc[bus, "y"])
        # Checking if the first end of the line is on the bus
        if bus_geo == busbar_coords[0]:
            intersection = busbar_coords[0]
        # Checking if the second end of the line is on the bus
        elif bus_geo == busbar_coords[-1]:
            intersection = busbar_coords[-1]

    return intersection

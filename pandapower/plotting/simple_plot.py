# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import matplotlib.pyplot as plt

from pandapower.plotting.collections import create_bus_collection, create_line_collection, \
    create_trafo_collection, draw_collections
from pandapower.plotting.generic_geodata import create_generic_coordinates

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0,
                scale_size=True, bus_color="b", line_color='grey', trafo_color='g',
                ext_grid_color='y', library = "igraph"):
    """
    Plots a pandapower network as simple as possible. If no geodata is available, artificial
    geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network.

    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches if artificial geodata is created

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 1.0) - Relative size of buses to plot.

            The value bus_size is multiplied with mean_distance_between_buses, which equals the
            distance between
            the max geoocord and the min divided by 200.
            mean_distance_between_buses = sum((net['bus_geodata'].max()
                                          - net['bus_geodata'].min()) / 200)

        **ext_grid_size** (float, 1.0) - Relative size of ext_grids to plot.

            See bus sizes for details. Note: ext_grids are plottet as rectangles

        **scale_size** (bool, True) - Flag if bus_size and ext_grid_size will be scaled with
        respect to grid mean distances

        **bus_color** (String, colors[0]) - Bus Color. Init as first value of color palette.
        Usually colors[0] = "b".

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'g') - Trafo Color. Init is green

        **ext_grid_color** (String, 'y') - External Grid Color. Init is yellow
    """
    # create geocoord if none are available
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches, library=library)

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        mean_distance_between_buses = sum((net['bus_geodata'].max() - net[
            'bus_geodata'].min()) / 200)
        # set the bus / ext_grid sizes accordingly
        # Comment: This is implemented because if you would choose a fixed values
        # (e.g. bus_size = 0.2), the size
        # could be to small for large networks and vice versa
        bus_size *= mean_distance_between_buses
        ext_grid_size *= mean_distance_between_buses * 1.5


    # create bus collections ti plot
    bc = create_bus_collection(net, net.bus.index, size=bus_size, color=bus_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_line_geodata = False if len(net.line_geodata) == 0 else True
    in_service_lines = net.line[net.line.in_service==True].index
    lc = create_line_collection(net, in_service_lines, color=line_color, linewidths=line_width,
                                use_line_geodata=use_line_geodata)
    collections = [bc, lc]
    eg_buses_with_geo_coordinates = set(net.ext_grid.bus.values) & set(net.bus_geodata.index)
    if len(eg_buses_with_geo_coordinates) > 0:
        sc = create_bus_collection(net, eg_buses_with_geo_coordinates, patch_type="rect",
                                   size=ext_grid_size, color=ext_grid_color, zorder=11)
        collections.append(sc)
    # create trafo collection if trafo is available

    trafo_buses_with_geo_coordinates = [t for t, trafo in net.trafo.iterrows()
                                        if trafo.hv_bus in net.bus_geodata.index and
                                        trafo.lv_bus in net.bus_geodata.index]
    if len(trafo_buses_with_geo_coordinates) > 0:
        tc = create_trafo_collection(net, trafo_buses_with_geo_coordinates, color=trafo_color)
        collections.append(tc)

    draw_collections(collections)
    plt.show()


if __name__ == "__main__":
    simple_plot()

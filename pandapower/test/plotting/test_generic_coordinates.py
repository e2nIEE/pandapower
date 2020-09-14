# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
from pandapower.test.toolbox import create_test_network
from pandapower.plotting.generic_geodata import create_generic_coordinates, build_igraph_from_pp
import pandapower as pp
import pandapower.networks as nw
import numpy as np
try:
    import igraph
    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False


@pytest.mark.skipif(IGRAPH_INSTALLED is False, reason="Requires python-igraph.")
def test_create_generic_coordinates_igraph():
    net = create_test_network()
    net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
    create_generic_coordinates(net, library="igraph")
    assert len(net.bus_geodata) == len(net.bus)


@pytest.mark.xfail(reason="The current implementation is not working properly, as multigraph edges "
                          "as AtlasViews are accessed with list logic.")
def test_create_generic_coordinates_nx():
    net = create_test_network()
    net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
    create_generic_coordinates(net, library="networkx")
    assert len(net.bus_geodata) == len(net.bus)

@pytest.mark.skipif(IGRAPH_INSTALLED is False, reason="Requires python-igraph.")
def test_create_generic_coordinates_igraph_custom_table_index():
    net = nw.simple_four_bus_system()
    for buses in [[0,1], [0,2], [0,1,2]]:
        create_generic_coordinates(net, buses=buses, geodata_table="test", 
                                   overwrite=True)
        assert np.all(net.test.index == buses)

if __name__ == "__main__":
    net = nw.mv_oberrhein()
    g, meshed, roots = build_igraph_from_pp(net, buses=[1,2,3])
    g.vcount()
#    pytest.main(["test_generic_coordinates.py"])

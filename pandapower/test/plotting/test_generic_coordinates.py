# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.networks.simple_pandapower_test_networks import simple_four_bus_system
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.test.helper_functions import create_test_network

try:
    import igraph

    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False


@pytest.mark.skipif(IGRAPH_INSTALLED is False, reason="Requires igraph.")
def test_create_generic_coordinates_igraph():
    net = create_test_network()
    net.bus = net.bus.drop("geo", axis=1)
    create_generic_coordinates(net, library="igraph")
    assert len(net.bus.geo.dropna()) == len(net.bus)


@pytest.mark.xfail(reason="The current implementation is not working properly, as multigraph edges "
                          "as AtlasViews are accessed with list logic.")
def test_create_generic_coordinates_nx():
    net = create_test_network()
    net.bus_geodata = net.bus_geodata.drop(net.bus_geodata.index)
    create_generic_coordinates(net, library="networkx")
    assert len(net.bus_geodata) == len(net.bus)


@pytest.mark.skipif(IGRAPH_INSTALLED is False, reason="Requires igraph.")
def test_create_generic_coordinates_igraph_custom_table_index():
    net = simple_four_bus_system()
    for buses in [[0, 1], [0, 2], [0, 1, 2]]:
        create_generic_coordinates(net, buses=buses, geodata_table="test", overwrite=True)
        assert np.all(net.test.index == buses)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

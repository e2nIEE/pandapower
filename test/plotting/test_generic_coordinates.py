# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
import pytest

from pandapower.networks.simple_pandapower_test_networks import simple_four_bus_system
from pandapower.plotting.generic_geodata import create_generic_coordinates

from test.helper_functions import create_test_network

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
    create_generic_coordinates(net, geodata_table="bus", overwrite=True)
    assert pd.notna(net.bus.geo).all()
    net.bus.geo[[0, 2]] = pd.NA
    create_generic_coordinates(net, geodata_table="bus", buses=[0, 2])
    assert pd.notna(net.bus.geo).all()
    net.bus.geo.at[0] = "Hallo"
    create_generic_coordinates(net, geodata_table="bus", buses=[0], overwrite=True)
    assert net.bus.geo.at[0] != "Hallo"

    net["test"] = pd.DataFrame(data=["T1", "T2", "T3"], columns=["name"])
    create_generic_coordinates(net, geodata_table="test")
    assert pd.notna(net.test.geo).all()


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import create_empty_network, create_bus, create_line, create_ext_grid, create_sgen


@pytest.fixture
def feeder_network():
    net = create_empty_network(sn_mva=11)
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    b4 = create_bus(net, 110)
    b5 = create_bus(net, 110)

    create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.)
    create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=15.)
    create_line(net, b1, b4, std_type="305-AL1/39-ST1A 110.0", length_km=12.)
    create_line(net, b4, b5, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=8.)
    net.line["endtemp_degree"] = 80
    for b in [b2, b3, b4, b5]:
        create_sgen(net, b, sn_mva=2000, p_mw=0)
    net.sgen["k"] = 1.2
    return net


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

from pandapower.create import create_buses, create_ext_grid, create_lines, create_sgen
from pandapower.network import pandapowerNet


@pytest.fixture
def feeder_network():
    net = pandapowerNet(name="feeder_network", sn_mva=11)
    b1, b2, b3, b4, b5 = create_buses(net, 5, 110)

    create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    create_lines(
        net,
        [b1, b2, b1, b4],
        [b2, b3, b4, b5],
        line_params=[
            "305-AL1/39-ST1A 110.0",
            "N2XS(FL)2Y 1x185 RM/35 64/110 kV",
            "305-AL1/39-ST1A 110.0",
            "N2XS(FL)2Y 1x185 RM/35 64/110 kV",
        ],
        length_km=[20.0, 15.0, 12.0, 8.0],
    )
    net.line["endtemp_degree"] = 80
    for b in [b2, b3, b4, b5]:
        create_sgen(net, b, sn_mva=2000, p_mw=0)
    net.sgen["k"] = 1.2
    return net


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

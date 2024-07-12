# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def test_lv_schutterwald():
    include_heat_pumps = [False, True]
    separation_by_sub = [False, True]

    for j in include_heat_pumps:
        for k in separation_by_sub:
            net = pn.lv_schutterwald(include_heat_pumps=j)
            pp.runpp(net)

            if j is False:
                assert len(net.load.bus) == 1506
            elif j is True:
                assert len(net.load.bus) == 2757

            assert len(net.line) == 3000
            assert len(net.switch) == 378
            assert net.converged

            if k is True:
                subnets = pn.lv_schutterwald(include_heat_pumps=j, separation_by_sub=k)
                assert all(len(subnets[0].keys()) == len(subnet.keys()) for subnet in subnets[1:])
                assert len(net.bus) == sum(len(subnet.bus) for subnet in subnets)
                assert all(subnet.converged for subnet in subnets)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

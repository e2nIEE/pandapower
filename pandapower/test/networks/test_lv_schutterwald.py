# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

from pandapower.networks import lv_schutterwald
from pandapower.run import runpp


@pytest.mark.parametrize("include_heat_pumps", [False, True])
@pytest.mark.parametrize("separation_by_sub", [False, True])
def test_lv_schutterwald(include_heat_pumps, separation_by_sub):
    net = lv_schutterwald(include_heat_pumps=include_heat_pumps)
    runpp(net)

    if include_heat_pumps:
        assert len(net.load.bus) == 2757
    else:
        assert len(net.load.bus) == 1506

    assert len(net.line) == 3000
    assert len(net.switch) == 378
    assert net.converged

    if separation_by_sub:
        subnets = lv_schutterwald(include_heat_pumps=include_heat_pumps, separation_by_sub=separation_by_sub)
        assert all(len(subnets[0].keys()) == len(subnet.keys()) for subnet in subnets[1:])
        assert len(net.bus) == sum(len(subnet.bus) for subnet in subnets)
        assert all(subnet.converged for subnet in subnets)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

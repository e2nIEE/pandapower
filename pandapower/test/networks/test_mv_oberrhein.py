# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

from pandapower.networks.mv_oberrhein import mv_oberrhein
from pandapower.run import runpp


@pytest.mark.parametrize("scenarios", ["load", "generation"])
@pytest.mark.parametrize("include_substations", [False, True])
@pytest.mark.parametrize("separation_by_sub", [False, True])
def test_mv_oberrhein(scenarios, include_substations, separation_by_sub):
    net = mv_oberrhein(scenario=scenarios, include_substations=include_substations)
    runpp(net)

    if scenarios == "load":
        assert net.sgen.scaling.mean() < 0.2
        assert net.load.scaling.mean() > 0.5
    elif scenarios == "generation":
        assert net.sgen.scaling.mean() > 0.6
        assert net.load.scaling.mean() < 0.2

    if include_substations:
        assert len(net.bus) == 320
        assert len(net.trafo) == 143
    else:
        assert len(net.bus) == 179
        assert len(net.trafo) == 2

    assert len(net.line) == 181
    assert len(net.switch) == 322
    assert net.converged

    if separation_by_sub:
        net0, net1 = mv_oberrhein(
            scenario=scenarios, include_substations=include_substations, separation_by_sub=separation_by_sub
        )
        assert len(net1.keys()) == len(net0.keys()) == len(net.keys())
        assert net1.res_ext_grid.loc[1].all() == net.res_ext_grid.loc[1].all()
        assert net0.res_ext_grid.loc[0].all() == net.res_ext_grid.loc[0].all()
        assert len(net.bus) == len(net0.bus) + len(net1.bus)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

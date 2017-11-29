# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pytest

import pandapower as pp
import pandapower.networks as pn


def test_mv_oberrhein():
    scenarios = ["load", "generation"]
    include_substations = [False, True]

    for i in scenarios:
        for j in include_substations:
            net = pn.mv_oberrhein(scenario=i, include_substations=j)
            pp.runpp(net)

            if i == "load":
                assert net.sgen.scaling.mean() < 0.2
                assert net.load.scaling.mean() > 0.5
            elif i == "generation":
                net.sgen.scaling.mean() > 0.6
                net.load.scaling.mean() < 0.2

            if j is False:
                assert len(net.bus) == 179
                assert len(net.trafo) == 2
            elif j is True:
                assert len(net.bus) == 320
                assert len(net.trafo) == 143

            assert len(net.line) == 181
            assert len(net.switch) == 322
            assert net.converged

if __name__ == '__main__':
    pytest.main(['-x', "test_mv_oberrhein.py"])

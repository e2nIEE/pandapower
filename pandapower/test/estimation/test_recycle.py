# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import StateEstimation, estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow
from copy import deepcopy


def test_recycle_case14():
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    se = StateEstimation(net, recycle=True)
    se.estimate(net)

    # Run SE again
    assert se.estimate(net)

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

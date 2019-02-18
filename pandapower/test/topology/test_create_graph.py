# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower as pp
import pytest
import pandapower.topology as top
import pandapower.networks as nw



if __name__ == '__main__':
    net = nw.example_multivoltage()
    mg = top.create_nxgraph(net, calc_r_ohm=True)
#    pytest.main([__file__])

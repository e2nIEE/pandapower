# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:33:50 2016

@author: thurner
"""

import pandapower as pp

from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.result_test_network_generator import add_test_shunt

net = pp.create_empty_network()
add_test_shunt(net)
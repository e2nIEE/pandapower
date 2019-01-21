# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate

def test_h_matrix():
    pass

def test_h_jacobian_matrix():
    pass

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
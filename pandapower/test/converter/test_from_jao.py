# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy
import os
import pytest
import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.converter import from_jao


def test_from_jao_with_testfile():
    testfile = os.path.join(pp.pp_dir, 'test', 'converter', "jao_testfiles", "testfile.xlsx")
    assert os.path.isfile(testfile)

    # --- net1
    net1 = from_jao(testfile, None, False)

    assert len(net1.bus) == 10
    assert len(net1.line) == 7
    assert net1.line.Tieline.sum() == 2
    assert len(net1.trafo) == 1

    # line data conversion
    assert np.all((0.01 < net1.line[['r_ohm_per_km', 'x_ohm_per_km']]) & (
        net1.line[['r_ohm_per_km', 'x_ohm_per_km']] < 0.4))
    assert np.all((0.5 < net1.line['c_nf_per_km']) & (net1.line['c_nf_per_km'] < 25))
    assert np.all(net1.line['g_us_per_km'] < 1)
    assert np.all((0.2 < net1.line['max_i_ka']) & (net1.line['max_i_ka'] < 5))

    # trafo data conversion
    assert 100 < net1.trafo.sn_mva.iat[0] < 1000
    assert 6 < net1.trafo.vk_percent.iat[0] < 65
    assert 0.25 < net1.trafo.vkr_percent.iat[0] < 1.2
    assert 10 < net1.trafo.pfe_kw.iat[0] < 1000
    assert net1.trafo.i0_percent.iat[0] < 0.1
    assert np.isclose(net1.trafo.shift_degree.iat[0], 90)
    assert np.isclose(net1.trafo.tap_step_degree.iat[0], 1.794)
    assert net1.trafo.tap_min.iat[0] == -17
    assert net1.trafo.tap_max.iat[0] == 17

    # --- net2
    net2 = from_jao(testfile, None, True)
    pp.nets_equal(net1, net2)  # extend_data_for_grid_group_connections makes no difference here

    # --- net3
    net3 = from_jao(testfile, None, True, drop_grid_groups_islands=True)
    assert len(net3.bus) == 6
    assert len(net3.line) == 5
    assert net3.line.Tieline.sum() == 1
    assert len(net3.trafo) == 1


if __name__ == '__main__':
    test_from_jao_with_testfile()
    # pytest.main([__file__, "-xs"])
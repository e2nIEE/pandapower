# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import numpy as np
import pytest

from pandapower import pp_dir
from pandapower.toolbox.comparison import nets_equal
from pandapower.converter.jao import from_jao


def test_from_jao_with_testfile():
    testfile = os.path.join(pp_dir, "test", "converter", "jao_testfiles", "testfile.xlsx")
    assert os.path.isfile(testfile)

    # --- net1
    net1 = from_jao(testfile, None, False)

    assert len(net1.bus) == 10
    assert len(net1.line) == 7
    assert net1.line.Tieline.sum() == 2
    assert len(net1.trafo) == 1

    # line data conversion
    assert np.all(
        (0.01 < net1.line[["r_ohm_per_km", "x_ohm_per_km"]]) & (net1.line[["r_ohm_per_km", "x_ohm_per_km"]] < 0.4)
    )
    # The JAO sheet gives a susceptance B in µS while c_nf_per_km expects a capacitance in nF/km,
    # so the converter has to divide by omega: C[nF] = B[µS] * 1e-6 / (2*pi*f) * 1e9. All lines in
    # the test file run at 380 kV, so the result has to land in the band of pandapower's own 380 kV
    # overhead line standard types (11 ... 14.6 nF/km). The old 0.5 ... 25 window was wide enough to
    # accept the raw susceptance as well and therefore could not catch the missing conversion.
    assert np.all((6 < net1.line["c_nf_per_km"]) & (net1.line["c_nf_per_km"] < 20))
    # first line: B = 312.274 µS over 70.53 km at 50 Hz -> 312.274e-6 / (2*pi*50) * 1e9 / 70.53
    assert np.isclose(net1.line["c_nf_per_km"].iat[0], 14.0933, rtol=1e-4)
    assert np.all(net1.line["g_us_per_km"] < 1)
    assert np.all((0.2 < net1.line["max_i_ka"]) & (net1.line["max_i_ka"] < 5))

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
    nets_equal(net1, net2)  # extend_data_for_grid_group_connections makes no difference here

    # --- net3
    net3 = from_jao(testfile, None, True, drop_grid_groups_islands=True)
    assert len(net3.bus) == 6
    assert len(net3.line) == 5
    assert net3.line.Tieline.sum() == 1
    assert len(net3.trafo) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

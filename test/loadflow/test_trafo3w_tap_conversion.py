# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

from pandapower.build_branch import _calculate_3w_tap_changers
from pandapower.create import create_empty_network


def test_three_winding_star_tap_nullable_type():
    trafos = pd.DataFrame(
        {
            "tap_side": ["hv", "hv", "hv", "hv"],
            "tap_at_star_point": [True, True, True, True],
            "tap_pos": [6, 6, 6, 6],
            "tap_neutral": [4, 4, 4, 4],
            "tap_step_percent": [0, 10, 10, 10],
            "tap_step_degree": [7, 0, 0, 90],
            "tap_changer_type": pd.Series(["Ideal", pd.NA, "Ratio", "Symmetrical"], dtype="string"),
        }
    )
    net = create_empty_network()
    net.trafo3w = trafos
    symmetrical = trafos.iloc[[3]].copy()
    net.trafo3w = symmetrical
    symmetrical_equivalent = {}
    _calculate_3w_tap_changers(net, symmetrical_equivalent, ["hv", "mv", "lv"])
    np.testing.assert_array_equal(symmetrical_equivalent["tap_side"]["hv"], ["lv"])
    np.testing.assert_allclose(symmetrical_equivalent["tap_step_percent"]["hv"], [50 / np.sqrt(26)])
    np.testing.assert_allclose(symmetrical_equivalent["tap_step_degree"]["hv"], [-101.30993247402021])

    net.trafo3w = trafos
    equivalent = {}
    _calculate_3w_tap_changers(net, equivalent, ["hv", "mv", "lv"])

    np.testing.assert_array_equal(equivalent["tap_side"]["hv"], ["lv", "lv", "lv", "lv"])
    np.testing.assert_allclose(equivalent["tap_step_percent"]["hv"], [0, 1000 / 120, 1000 / 120, 50 / np.sqrt(26)])
    np.testing.assert_allclose(equivalent["tap_step_degree"]["hv"], [-7, -180, -180, -101.30993247402021])

    without_type = trafos.iloc[[1]].drop(columns="tap_changer_type")
    net.trafo3w = without_type
    legacy_equivalent = {}
    _calculate_3w_tap_changers(net, legacy_equivalent, ["hv", "mv", "lv"])
    np.testing.assert_array_equal(legacy_equivalent["tap_side"]["hv"], ["lv"])
    np.testing.assert_allclose(legacy_equivalent["tap_step_percent"]["hv"], [1000 / 120])
    np.testing.assert_allclose(legacy_equivalent["tap_step_degree"]["hv"], [-180])

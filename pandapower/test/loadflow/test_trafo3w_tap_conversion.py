# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

from pandapower.build_branch import _calculate_3w_tap_changers


def test_three_winding_star_tap_nullable_type():
    trafos = pd.DataFrame({
        "tap_side": ["hv", "hv", "hv"],
        "tap_at_star_point": [True, True, True],
        "tap_pos": [6, 6, 6],
        "tap_neutral": [4, 4, 4],
        "tap_step_percent": [0, 10, 10],
        "tap_step_degree": [7, 0, 0],
        "tap_changer_type": pd.Series(["Ideal", pd.NA, "Ratio"], dtype="string"),
    })
    equivalent = {}
    _calculate_3w_tap_changers(trafos, equivalent, ["hv", "mv", "lv"])

    np.testing.assert_array_equal(equivalent["tap_side"]["hv"], ["lv", "lv", "lv"])
    np.testing.assert_allclose(equivalent["tap_step_percent"]["hv"], [0, 1000 / 120, 1000 / 120])
    np.testing.assert_allclose(equivalent["tap_step_degree"]["hv"], [-7, -180, -180])

    without_type = trafos.iloc[[1]].drop(columns="tap_changer_type")
    legacy_equivalent = {}
    _calculate_3w_tap_changers(without_type, legacy_equivalent, ["hv", "mv", "lv"])
    np.testing.assert_array_equal(legacy_equivalent["tap_side"]["hv"], ["lv"])
    np.testing.assert_allclose(legacy_equivalent["tap_step_percent"]["hv"], [1000 / 120])
    np.testing.assert_allclose(legacy_equivalent["tap_step_degree"]["hv"], [-180])

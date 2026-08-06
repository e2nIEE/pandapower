# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import os

import pandas as pd
import numpy as np
import pytest

from pandapower.timeseries.data_sources.frame_data import DFData

from test import test_path

epsilon = 0.00000000000001


def test_data_source():
    """
    Testing simply reading from file and checking the data.
    """
    # load file

    filename = os.path.join(test_path, "timeseries", "test_files", "small_profile.csv")
    df = pd.read_csv(filename, sep=";")
    my_data_source = DFData(df)
    copy.deepcopy(my_data_source)

    assert np.isclose(my_data_source.get_time_step_value(time_step=0, profile_name="my_profilename"), 0.0)
    assert np.isclose(my_data_source.get_time_step_value(time_step=3, profile_name="my_profilename"), 0.0)
    assert abs(my_data_source.get_time_step_value(time_step=4, profile_name="my_profilename")
               - -3.97E-1) < epsilon
    assert abs(my_data_source.get_time_step_value(time_step=8, profile_name="constload3")
               - -5.37E-3) < epsilon


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

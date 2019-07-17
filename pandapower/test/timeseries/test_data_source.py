__author__ = 'jdollichon'

import os
import pandas as pd
import pytest
import copy
import pandapower.networks as nw
import pandapower.control
import pandapower.timeseries
epsilon = 0.00000000000001


def test_data_source():
    """
    Testing simply reading from file and checking the data.
    """
    # load file

    df = pd.read_csv("test_files\\small_profile.csv", sep=";")
    my_data_source = pandapower.timeseries.DFData(df)
    copy.deepcopy(my_data_source)

    # # print data_sources.time_df
    # for i in xrange(len(data_sources.time_df.index)):
    #     for j in xrange(len(data_sources.time_df.columns)):
    #         print data_sources.get_time_step_values(i, j)
    #         pass

    # check a few of the values
    # (profile_name can be the actual name but also the column number)
    assert my_data_source.get_time_step_value(time_step=0, profile_name="my_profilename") == 0.0
    assert my_data_source.get_time_step_value(time_step=3, profile_name="my_profilename") == 0.0
    assert abs(my_data_source.get_time_step_value(time_step=4, profile_name="my_profilename")
               - -3.97E-1) < epsilon
    assert abs(my_data_source.get_time_step_value(time_step=8, profile_name="constload3")
               - -5.37E-3) < epsilon



if __name__ == '__main__':
    pytest.main(['-x', '-s', __file__])
    # pytest.main(['-x', __file__])

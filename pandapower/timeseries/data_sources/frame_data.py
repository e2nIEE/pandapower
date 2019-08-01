# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.timeseries.data_source import DataSource

try:
    import pplog
except:
    import logging as pplog

from pandapower.timeseries.data_source import DataSource

logger = pplog.getLogger(__name__)


class DFData(DataSource):
    """
    a very basic implementation for a pandas.DataFrame data source, that uses index column as time
    step.
    the user should take care that the data is numeric and all the time steps are present,
    because we let it fail here.

    Please note: The columns should be integers for scalar accessing!
    You can enable casting of columns and indexes with the option multi =True!
    This is necessary e.g. if you read your data from csv.
    """

    def __init__(self, df, multi=False):
        super().__init__()
        self.update_initialized(locals())
        self.df = df
        if multi:
            # casting column and index to int for multi- columns accessing
            self.df.index = self.df.index.astype(int)
            self.df.columns = self.df.columns.astype(int)

    def __repr__(self):
        s = "%s with %d rows and %d columns" % (
            self.__class__.__name__, len(self.df), len(self.df.columns))

        if len(self.df.columns) <= 10:
            s += ": %s" % self.df.columns.values.__str__()
        return s

    def get_time_step_value(self, time_step, profile_name, scale_factor=1.0):
        res = self.df.loc[time_step, profile_name]
        if hasattr(res, 'values'):
            res = res.values
        res *= scale_factor
        return res

    def get_time_steps_len(self):
        return len(self.df)

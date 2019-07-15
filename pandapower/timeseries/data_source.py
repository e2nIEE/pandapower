# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pplog
import datetime
from pandapower.io_utils import JSONSerializableClass

logger = pplog.getLogger(__name__)


class DataSource(JSONSerializableClass):
    """
    This class may hold data from a profile, generate data at random or
    just return constant values for a time step. Controllers will call
    get_time_step_values(time) in each time step to get new values for e.g. P, Q
    """

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def get_time_step_value(self, time_step, profile_name, scale_factor=1.0):
        """
        This method retrieves values of the data source according to the given parameters.
        For actual parameters look into the DataSource you are actually using.
        """
        raise NotImplementedError("Subclasses should implement this!")


class FileData(DataSource):
    """
    This class may hold data from a profile, generate data at random or
    just return constant values for a time step. Controllers will call
    get_time_step_values(time) in each time step to get new values for e.g. P, Q

    |   **path** - The path to a HDF5-File, containing the data
    |   **time_res** - Desired time resolution in minutes (default = will infer from timestamps)
    """

    def __init__(self, resolution_sec=None, profile_span="epoch"):
        super().__init__()
        self.time_df = None
        self.profile_span = profile_span
        self.resolution_sec = resolution_sec
        self.update_initialized(locals())

    def add_file(self, path):
        """
        Adds the contents of a file to the dataframe of this DataSource
        """
        raise NotImplementedError("Subclasses should implement this!")

    def get_time_step_value(self, time_step, profile_name, scale_factor=1.0):
        """
        This method retrieves values of the data source according to the given parameters.
        |   **time_step** - The time step for which we want to look up values
        |   **profile_name** - The name of the columns for each profile
        |   **scale_factor** - A scale factor.
        """
        # we converse the time_step in case we only got e.g. monthly profiles
        if self.profile_span.lower() == "month":
            date_time = datetime.datetime.fromtimestamp(time_step)
            new_time_step = (date_time.day - 1) * 86400 + date_time.hour * 3600 + \
                            date_time.minute * 60 + date_time.second
            row = new_time_step
        elif self.profile_span.lower() == "day":
            date_time = datetime.datetime.fromtimestamp(time_step)
            new_time_step = date_time.hour * 3600 + \
                            date_time.minute * 60 + date_time.second
            row = new_time_step
        else:
            row = time_step

        try:
            res = self.time_df.at[row, profile_name] * scale_factor
            return res
        except Exception as e:
            logger.warning("Could not read value at profile %s, row %i. Error: %s" % (
                str(profile_name), row, e.args))
            return 0.0

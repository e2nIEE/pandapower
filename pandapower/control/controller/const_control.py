# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandas import Index
from pandapower.control.basic_controller import Controller
try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ConstControl(Controller):
    """
    Class representing a generic time series controller for a specified element and variable
    Control strategy: "No Control" -> just updates timeseries

    INPUT:

        **net** (attrdict) - The net in which the controller resides

        **element** - element table ('sgen', 'load' etc.)

        **variable** - variable ('p_mw', 'q_mvar', 'vm_pu', 'tap_pos' etc.)

        **element_index** (int[]) - IDs of the controlled elements

        **data_source** (obj) - The data source that provides profile data

        **profile_name** (str[]) - The profile names of the elements in the data source


    OPTIONAL:

        **scale_factor** (real, 1.0) - Scaling factor for time series input values

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **recycle** (bool, False) - Re-use of ppi-data (speeds-up time series simulation, experimental!)

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped

    .. note:: If multiple elements are represented with one controller, the data source must have integer columns. At the moment, only the DFData format is tested for the multiple const control.
    """

    def __init__(self, net, element, variable, element_index, profile_name=None, data_source=None,
                 scale_factor=1.0, in_service=True, recycle=False, order=0, level=0,
                 drop_same_existing_ctrl=False, set_q_from_cosphi=False, matching_params=None, initial_powerflow = False,
                 **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": element, "variable": variable,
                               "element_index": element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params
                        ,initial_powerflow = initial_powerflow, **kwargs)
        self.update_initialized(locals())
        self.matching_params = {"element": element, "variable": variable,
                                "element_index": element_index}

        # data source for time series values
        self.data_source = data_source
        # ids of sgens or loads
        self.element_index = element_index
        # element type
        self.element = element
        self.variable = variable
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        if set_q_from_cosphi:
            logger.error("Parameter set_q_from_cosphi deprecated!")
            raise ValueError
        self.set_q_from_cosphi_3ph = set_q_from_cosphi_3ph
        self.applied = False
        self.initial_powerflow = initial_powerflow
        # write functions faster, depending on type of self.element_index
        if isinstance(self.element_index, int):
            # use .at if element_index is integer for speedup
            self.write = self._write_to_single_index
        elif self.net[self.element].index.equals(Index(self.element_index)):
            # use : indexer if all elements are in index
            self.write = self._write_to_all_index
        else:
            # use common .loc
            self.write = self._write_with_loc

    def write_to_net(self):
        """
        Writes to self.element at index self.element_index in the column self.variable the data from self.values
        """
        self.write()
        if self.set_q_from_cosphi_3ph:
            self.net[self.element].loc[self.element_index, "q_a_mvar"] = \
                self.net[self.element].loc[self.element_index, "p_a_mw"].values * np.tan(
                    np.arccos(self.net[self.element].loc[self.element_index, "cos_phi"].values))
            self.net[self.element].loc[self.element_index, "q_b_mvar"] = \
                self.net[self.element].loc[self.element_index, "p_b_mw"].values * np.tan(
                    np.arccos(self.net[self.element].loc[self.element_index, "cos_phi"].values))        
            self.net[self.element].loc[self.element_index, "q_c_mvar"] = \
                self.net[self.element].loc[self.element_index, "p_c_mw"].values * np.tan(
                    np.arccos(self.net[self.element].loc[self.element_index, "cos_phi"].values))
                
    def time_step(self, time):
        """
        Get the values of the element from data source
        """
        self.values = self.data_source.get_time_step_value(time_step=time,
                                                           profile_name=self.profile_name,
                                                           scale_factor=self.scale_factor)
        self.write_to_net()

    def initialize_control(self):
        """
        At the beginning of each time step reset applied-flag
        """
        #
        if self.data_source is None:
            self.values = self.net[self.element][self.variable].loc[self.element_index]
        self.applied = False

    def is_converged(self):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        return self.applied

    def control_step(self):
        """
        Write to pandapower net by calling write_to_net()
        """
        if self.values is not None:
            self.write_to_net()
        self.applied = True

    def _write_to_single_index(self):
        self.net[self.element].at[self.element_index, self.variable] = self.values

    def _write_to_all_index(self):
        self.net[self.element].loc[:, self.variable] = self.values

    def _write_with_loc(self):
        self.net[self.element].loc[self.element_index, self.variable] = self.values

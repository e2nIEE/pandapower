# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.basic_controller import Controller
from pandapower.toolbox import _detect_read_write_flag, write_to_net

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ConstControl(Controller):
    """
    Class representing a generic time series controller for a specified element and variable.
    Control strategy: "No Control" -> updates values of specified elements according to timeseries input data.
    If ConstControl is used without timeseries input data, it will reset the controlled values to the initial values,
    preserving the initial net state.
    The timeseries values are written to net during time_step before the initial powerflow run and before other controllers' control_step.
    It is possible to set attributes of objects that are contained in a net table, e.g. attributes of other controllers. This can be helpful
    e.g. if a voltage setpoint of a transformer tap changer depends on the time step.
    An attribute of an object in the "object" column of a table (e.g. net.controller["object"] -> net.controller.object.at[0, "vm_set_pu"]
    can be set if the attribute is specified as "object.attribute" (e.g. "object.vm_set_pu").

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

        **recycle** (bool, True) - Re-use of internal-data in a time series loop.

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of
        the same type and with the same matching parameters (e.g. at same element) should be
        dropped

    .. note:: If multiple elements are represented with one controller, the data source must have
        integer columns. At the moment, only the DFData format is tested for the multiple const
        control.
    """

    def __init__(self, net, element, variable, element_index, profile_name=None, data_source=None,
                 scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=False, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": element, "variable": variable,
                               "element_index": element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        self.data_source = data_source
        self.element_index = element_index
        self.element = element
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.applied = False
        self.write_flag, self.variable = _detect_read_write_flag(net, element, element_index, variable)
        self.set_recycle(net)

    def set_recycle(self, net):
        pass

    def time_step(self, net, time):
        pass

    def is_converged(self, net):
        self.check_word = 'banana'
        return True

    def control_step(self, net):
        pass

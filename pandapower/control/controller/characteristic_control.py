# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net
from pandapower.control.basic_controller import Controller


class CharacteristicControl(Controller):
    """
    Controller that adjusts a certain parameter of an element in pandapower net based on a specified input parameter in pandapower net,
    according to a provided characteristic. The characteristic is specified by the index in the ``net.characteristic table``.
    Example: change the tap position of the transformers (``net.trafo.tap_pos``) based on transformer loading (``net.res_trafo.loading_percent``)
    according to a specified linear relationship. To this end, the input element is ``res_trafo``, the input variable is ``loading_percent``,
    the output element is ``trafo`` and the output variable is ``tap_pos``. The relationship between the values of the input and output
    variables is specified using the Characteristic class (or a scipy interpolator, e.g. ``scipy.interpolate.interp1d``).

    INPUT:
        **net** (attrdict) - Pandapower net

        **output_element** (str) - name of the element table in pandapower net where the values are adjusted by the controller

        **output_variable** (str) - variable in the output element table, values of which are adjusted by the controller. Can also be an attribute of an object (e.g. parameter of a controller object), for this case it must start with "object." (e.g. "object.vm_set_pu")

        **output_element_index** (int or list or numpy array) - index of the elements, values from which are adjusted

        **input_element** (str) - name of the element table or the element result table in pandapower net that provides input values for the controller

        **input_variable** (str) - name of the input variable in the input element table. Can also be an attribute of an object, similarly to output_variable

        **input_element_index** (int or list or numpy array) - index of elements in the input element table

        **characteristic_index** (int) - index of the characteristic curve that describes the relationship between the input and output values

        **tol** (float) - tolerance for convergence

    OPTIONAL:
        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """
    def __init__(self, net, output_element, output_variable, output_element_index, input_element,
                 input_variable, input_element_index, characteristic_index, tol=1e-3, in_service=True,
                 order=0, level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"element": output_element, "input_variable": input_variable,
                               "output_variable": output_variable,
                               "element_index": input_element_index}
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.input_element = input_element
        self.input_element_index = input_element_index
        self.output_element = output_element
        self.output_element_index = output_element_index
        self.characteristic_index = characteristic_index
        self.tol = tol
        self.applied = False
        self.values = None
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element, output_element_index,
                                                                        output_variable)
        self.read_flag, self.input_variable = _detect_read_write_flag(net, input_element, input_element_index,
                                                                      input_variable)

    def initialize_control(self, net):
        """
        At the beginning of each run_control call reset applied-flag
        """
        self.values = None
        self.applied = False

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # read input values
        input_values = read_from_net(net, self.input_element, self.input_element_index, self.input_variable,
                                     self.read_flag)
        # calculate set values
        self.values = net.characteristic.object.at[self.characteristic_index](input_values)
        # read previous set values
        output_values = read_from_net(net, self.output_element, self.output_element_index, self.output_variable,
                                      self.write_flag)
        # compare old and new set values
        diff = self.values - output_values
        # write new set values
        write_to_net(net, self.output_element, self.output_element_index, self.output_variable, self.values,
                     self.write_flag)
        return self.applied and np.all(np.abs(diff) < self.tol)

    def control_step(self, net):
        """
        Set applied to true to make sure it runs at least once
        """
        self.applied = True

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (self.input_element, self.input_variable, self.output_element, self.output_variable)

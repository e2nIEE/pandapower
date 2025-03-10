# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.controller.characteristic_control import CharacteristicControl


class TapDependentImpedance(CharacteristicControl):
    """
    Controller that adjusts the impedance of a transformer (or multiple transformers) depending to the actual tap position and
    according to a defined characteristic.

    INPUT:
        **net** (attrdict) - Pandapower net

        **element_index** (int or list or numpy array) - ID of the transformer or multiple transfromers

        **characteristic** (object of class Characteristic) - Characteristic that describes the relationship between transformer tap
                                                                position and transformer impedance

    OPTIONAL:

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, element_index, characteristic_index, element="trafo",
                 output_variable="vk_percent", tol=1e-3, restore=True, in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"element_index": element_index, 'output_variable': output_variable}
            super().__init__(
                net, output_element=element, output_variable=output_variable,
                output_element_index=element_index, input_element=element,
                input_variable="tap_pos", input_element_index=element_index,
                characteristic_index=characteristic_index, tol=tol, in_service=in_service,
                order=order, level=level, drop_same_existing_ctrl=drop_same_existing_ctrl,
                matching_params=matching_params, **kwargs)
        self.restore=restore
        self.initial_values = net[element].loc[element_index, output_variable].copy()

    def initialize_control(self, net):
        if self.restore:
            self.initial_values = net[self.output_element].loc[
                self.output_element_index, self.output_variable].copy()

    def finalize_control(self, net):
        if self.restore:
            net[self.output_element].loc[self.output_element_index, self.output_variable] = \
                self.initial_values

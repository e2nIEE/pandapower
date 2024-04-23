# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.controller.characteristic_control import CharacteristicControl

class USetTapControl(CharacteristicControl):
    def __init__(self, **kwargs):
        raise UserWarning("USetTapControl has been renamed. Use VmSetTapControl instead.")

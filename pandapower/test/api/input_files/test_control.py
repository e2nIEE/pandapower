# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.controller.const_control import ConstControl as ConstController

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class TestControl(ConstController):

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

class ConstControl(ConstController):

    def is_converged(self, net):
        self.check_word = 'banana'
        return super().is_converged(net)

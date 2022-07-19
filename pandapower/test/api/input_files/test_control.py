# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.basic_controller import Controller

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class TestControl(Controller):

    def is_converged(self, net):
        self.check_word = 'banana'
        return True

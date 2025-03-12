# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.auxiliary import read_from_net, write_to_net, _detect_read_write_flag
from pandapower.control.basic_controller import Controller

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


class TrafoController(Controller):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
       **net** (attrdict) - Pandapower struct

       **element_index** (int) - ID of the trafo that is controlled

       **side** (string) - Side of the transformer where the voltage is controlled ("hv", "mv"
       or "lv")

       **tol** (float) - Voltage tolerance band at bus in Percent

       **in_service** (bool) - Indicates if the element is currently active

       **element** (string) - Type of the controlled trafo ("trafo" or "trafo3w")

    OPTIONAL:
        **recycle** (bool, True) - Re-use of internal-data in a time series loop.
    """

    def __init__(self, net, element_index, side, tol, in_service, element, level=0, order=0,
                 recycle=True, **kwargs):
        super().__init__(net, in_service=in_service, level=level, order=order, recycle=recycle,
                         **kwargs)

        self.element = element
        self.element_index = element_index

        self._set_side(side)
        self._set_read_write_flag(net)
        # self._set_valid_controlled_index_and_bus(net)
        self._set_tap_parameters(net)
        self._set_tap_side_coeff(net)

        self.tol = tol

        self.set_recycle(net)

        self.trafobus = read_from_net(net, self.element, self.element_index, self.side + '_bus', self._read_write_flag)

    def _set_read_write_flag(self, net):
        # if someone changes indices of the controller from single index to array and vice versa
        self._read_write_flag, _ = _detect_read_write_flag(net, self.element, self.element_index, "tap_pos")
        if self._read_write_flag == 'loc':
            self.element_index = np.array(self.element_index)

    def initialize_control(self, net):
        # in case changes applied to net in the meantime:
        # the update occurs in case the in_service parameter of tranformers is changed in the meantime
        # update valid trafo and bus
        # update trafo tap parameters
        # we assume side does not change after the controller is created
        self._set_read_write_flag(net)
        # self._set_valid_controlled_index_and_bus(net)
        if self.nothing_to_do(net):
            return
        self._set_tap_parameters(net)
        self._set_tap_side_coeff(net)

    def nothing_to_do(self, net):
        element_in_service = read_from_net(net, self.element, self.element_index, 'in_service', self._read_write_flag)
        ext_grid_bus = np.isin(self.trafobus, net.ext_grid.loc[net.ext_grid.in_service, 'bus'].values)
        element_index_in_net = np.isin(self.element_index, net[self.element].index.values)
        self.controlled = np.logical_and(np.logical_and(element_in_service, element_index_in_net), np.logical_not(ext_grid_bus))
        if isinstance(self.element_index, np.int64) or isinstance(self.element_index, int):
            # if the controller shouldn't do anything, return True
            if not element_in_service or ext_grid_bus or not element_index_in_net or (
                    self._read_write_flag != 'single_index' and len(self.element_index) == 0):
                return True
            return False
        else:
            # if the controller shouldn't do anything, return True
            if np.all(~element_in_service[self.controlled]) or np.all(ext_grid_bus[self.controlled]) or np.all(~element_index_in_net[self.controlled]) or (
                    self._read_write_flag != 'single_index' and len(self.element_index) == 0):
                return True
            return False

    def _set_tap_side_coeff(self, net):
        tap_side = read_from_net(net, self.element, self.element_index, 'tap_side', self._read_write_flag)
        if (len(np.setdiff1d(tap_side, ['hv', 'lv'])) > 0 and self.element == "trafo") or \
            (len(np.setdiff1d(tap_side, ['hv', 'lv', 'mv'])) > 0 and self.element == "trafo3w"):
            raise ValueError("Trafo tap side (in net.%s) has to be either hv or lv, "
                             "but received: %s for trafo %s" % (self.element, tap_side, self.element_index))

        if self._read_write_flag == "single_index":
            self.tap_side_coeff = 1 if tap_side == 'hv' else -1
            if self.side == "hv":
                self.tap_side_coeff *= -1
            if self.tap_step_percent < 0:
                self.tap_side_coeff *= -1
        else:
            self.tap_side_coeff = np.where(tap_side=='hv', 1, -1)
            self.tap_side_coeff[self.side == "hv"] *= -1
            self.tap_side_coeff[self.tap_step_percent < 0] *= -1

    def _set_side(self, side):
        if self.element == "trafo":
            if side not in ["hv", "lv"]:
                raise UserWarning("side has to be 'hv' or 'lv' for high/low voltage, "
                                  "received %s" % side)
        elif self.element == "trafo3w":
            if side not in ["hv", "mv", "lv"]:
                raise UserWarning("side has to be 'hv', 'mv' or 'lv' for high/middle/low voltage, "
                                  "received %s" % side)
        else:
            raise UserWarning("unknown trafo type, received %s" % self.element)

        self.side = side

    def _set_valid_controlled_index_and_bus(self, net):
        element_in_service = read_from_net(net, self.element, self.element_index, 'in_service', self._read_write_flag)
        ext_grid_bus = np.isin(self.trafobus, net.ext_grid.loc[net.ext_grid.in_service, 'bus'].values)
        element_index_in_net = np.isin(self.element_index, net[self.element].index.values)
        self.controlled = np.logical_and(np.logical_and(element_in_service, element_index_in_net), np.logical_not(ext_grid_bus))
        if self._read_write_flag != 'single_index':
            self.element_index = self.element_index[self.controlled]
            self.trafobus = self.trafobus[self.controlled]

        if np.all(~self.controlled):
            logger.warning("All controlled buses are not valid: controller has no effect")

    def _set_tap_parameters(self, net):
        self.tap_min = read_from_net(net, self.element, self.element_index, "tap_min", self._read_write_flag)
        self.tap_max = read_from_net(net, self.element, self.element_index, "tap_max", self._read_write_flag)
        self.tap_neutral = read_from_net(net, self.element, self.element_index, "tap_neutral", self._read_write_flag)
        self.tap_step_percent = read_from_net(net, self.element, self.element_index, "tap_step_percent", self._read_write_flag)
        self.tap_step_degree = read_from_net(net, self.element, self.element_index, "tap_step_degree", self._read_write_flag)

        self.tap_pos = read_from_net(net, self.element, self.element_index, "tap_pos", self._read_write_flag)
        if self._read_write_flag == "single_index":
            self.tap_sign = 1 if np.isnan(self.tap_step_degree) else np.sign(np.cos(np.deg2rad(self.tap_step_degree)))
            if (self.tap_sign == 0) | (np.isnan(self.tap_sign)):
                self.tap_sign = 1
            if np.isnan(self.tap_pos):
                self.tap_pos = self.tap_neutral
        else:
            self.tap_sign = np.where(np.isnan(self.tap_step_degree), 1, np.sign(np.cos(np.deg2rad(self.tap_step_degree))))
            self.tap_sign = np.where((self.tap_sign == 0) | (np.isnan(self.tap_sign)), 1, self.tap_sign)
            self.tap_pos = np.where(np.isnan(self.tap_pos), self.tap_neutral, self.tap_pos)

        if np.any(np.isnan(self.tap_min)) or np.any(np.isnan(self.tap_max)) or np.any(np.isnan(self.tap_step_percent)):
            logger.error("Trafo-Controller has been initialized with NaN values, check "
                         "net.trafo.tap_pos etc. if they are set correctly!")

    def set_recycle(self, net):
        allowed_elements = ["trafo", "trafo3w"]
        if net.controller.at[self.index, 'recycle'] is False or self.element not in allowed_elements:
            # if recycle is set to False by the user when creating the controller it is deactivated or when
            # const control controls an element which is not able to be recycled
            net.controller.at[self.index, 'recycle'] = False
            return
        # these variables determine what is re-calculated during a time series run
        recycle = dict(trafo=True, gen=False, bus_pq=False)
        net.controller.at[self.index, 'recycle'] = recycle

    # def timestep(self, net):
    #     self.tap_pos = net[self.element].at[self.element_index, "tap_pos"]

    def __repr__(self):
        s = '%s of %s %s' % (self.__class__.__name__, self.element, self.element_index)
        return s

    def __str__(self):
        s = '%s of %s %s' % (self.__class__.__name__, self.element, self.element_index)
        return s

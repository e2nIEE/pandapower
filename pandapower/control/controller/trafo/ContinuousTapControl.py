# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.auxiliary import read_from_net, write_to_net
from pandapower.control.controller.trafo_control import TrafoController


class ContinuousTapControl(TrafoController):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
        **net** (attrdict) - Pandapower struct

        **tid** (int) - ID of the trafo that is controlled

        **vm_set_pu** (float) - Maximum OLTC target voltage at bus in pu

    OPTIONAL:

        **tol** (float, 0.001) - Voltage tolerance band at bus in percent (default: 1% = 0.01pu)

        **side** (string, "lv") - Side of the transformer where the voltage is controlled

        **trafo_type** (float, "2W") - Trafo type ("2W" or "3W")

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **check_tap_bounds** (bool, True) - In case of true the tap_bounds will be considered

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, tid, vm_set_pu, tol=1e-3, side="lv", trafotype="2W", in_service=True,
                 check_tap_bounds=True, level=0, order=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"tid": tid, 'trafotype': trafotype}
        super().__init__(net, tid=tid, side=side, tol=tol, in_service=in_service,
                         trafotype=trafotype, level=level, order=order,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)

        self._set_t_nom(net)
        self.check_tap_bounds = check_tap_bounds
        self.vm_set_pu = vm_set_pu

    def _set_t_nom(self, net):
        vn_hv_kv = read_from_net(net, self.trafotable, self.controlled_tid, 'vn_hv_kv', self._read_write_flag)
        hv_bus = read_from_net(net, self.trafotable, self.controlled_tid, 'hv_bus', self._read_write_flag)
        vn_hv_bus_kv = read_from_net(net, "bus", hv_bus, 'vn_kv', self._read_write_flag)

        if self.trafotype == "3W" and self.side == "mv":
            vn_mv_kv = read_from_net(net, self.trafotable, self.controlled_tid, 'vn_mv_kv', self._read_write_flag)
            mv_bus = read_from_net(net, self.trafotable, self.controlled_tid, 'mv_bus', self._read_write_flag)
            vn_mv_bus_kv = read_from_net(net, "bus", mv_bus, 'vn_kv', self._read_write_flag)
            self.t_nom = vn_mv_kv / vn_hv_kv * vn_hv_bus_kv / vn_mv_bus_kv
        else:
            vn_lv_kv = read_from_net(net, self.trafotable, self.controlled_tid, 'vn_lv_kv', self._read_write_flag)
            lv_bus = read_from_net(net, self.trafotable, self.controlled_tid, 'lv_bus', self._read_write_flag)
            vn_lv_bus_kv = read_from_net(net, "bus", lv_bus, 'vn_kv', self._read_write_flag)
            self.t_nom = vn_lv_kv / vn_hv_kv * vn_hv_bus_kv / vn_lv_bus_kv

    def initialize_control(self, net):
        super().initialize_control(net)
        if not self.nothing_to_do(net):
            self._set_t_nom(net)  # in case some of the trafo elements change their in_service in between runs

    def control_step(self, net):
        """
        Implements one step of the ContinuousTapControl
        """
        if self.nothing_to_do(net):
            return

        delta_vm_pu = read_from_net(net, "res_bus", self.controlled_bus, 'vm_pu', self._read_write_flag) - self.vm_set_pu
        tc = delta_vm_pu / self.tap_step_percent * 100 / self.t_nom
        self.tap_pos = self.tap_pos + tc * self.tap_side_coeff * self.tap_sign
        if self.check_tap_bounds:
            self.tap_pos = np.clip(self.tap_pos, self.tap_min, self.tap_max)

        # WRITE TO NET
        # necessary in case the dtype of the column is int
        if net[self.trafotable].tap_pos.dtype != "float":
            net[self.trafotable].tap_pos = net[self.trafotable].tap_pos.astype(float)
        write_to_net(net, self.trafotable, self.controlled_tid, "tap_pos", self.tap_pos, self._read_write_flag)

    def is_converged(self, net):
        """
        The ContinuousTapControl is converged, when the difference of the voltage between control steps is smaller
        than the Tolerance (tol).
        """
        if self.nothing_to_do(net):
            return True

        vm_pu = read_from_net(net, "res_bus", self.controlled_bus, "vm_pu", self._read_write_flag)
        # this is possible in case the trafo is set out of service by the connectivity check
        is_nan =  np.isnan(vm_pu)
        self.tap_pos = read_from_net(net, self.trafotable, self.controlled_tid, "tap_pos", self._read_write_flag)
        difference = 1 - self.vm_set_pu / vm_pu

        if self.check_tap_bounds:
            reached_limit = np.where(self.tap_side_coeff * self.tap_sign == 1,
                                     (vm_pu < self.vm_set_pu) & (self.tap_pos == self.tap_min) |
                                     (vm_pu > self.vm_set_pu) & (self.tap_pos == self.tap_max),
                                     (vm_pu < self.vm_set_pu) & (self.tap_pos == self.tap_max) |
                                     (vm_pu > self.vm_set_pu) & (self.tap_pos == self.tap_min))
            converged = np.logical_or(reached_limit, np.abs(difference) < self.tol)
        else:
            converged = np.abs(difference) < self.tol

        return np.all(np.logical_or(converged, is_nan))

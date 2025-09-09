# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.auxiliary import pandapowerNet
from pandapower.control.basic_controller import Controller
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DmrControl(Controller):
    """
    Class to calculate dmr currents. It will take the i_ka from the dc_plus_line, subtract the i_ka from the dc_minus_line
    and write the result in the dmr_line. This is a workaround since pp is not able to calculate the dmr current out-of-the-box.
    """

    def __init__(self, net: pandapowerNet, dmr_line: int, dc_plus_line: int, dc_minus_line: int,
                 in_service=True, order=0, level=0,
                 drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"dmr_line": dmr_line, "dc_plus_line": dc_plus_line, "dc_minus_line": dc_minus_line}

        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)

        self.dmr_line = dmr_line
        self.dc_plus_line = dc_plus_line
        self.dc_minus_line = dc_minus_line

        if not np.all(net.line_dc.index.isin([dmr_line, dc_plus_line, dc_minus_line])):
            raise ValueError("Wrong dc line index given. Please check if all lines are in line_dc!")

        self.applied = False

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # return not np.isclose(net.res_line_dc.loc[self.dmr_line, 'i_ka'], 0.)
        return True

    def finalize_control(self, net):
    #def control_step(self, net):
        """
        Set applied to True, which means that the values set in time_step have been included in the
        load flow calculation.
        """
        # dmr.i_ka = dcp.i_ka - dcm.i_ka
        max_i_ka = net.line_dc.loc[self.dmr_line, 'max_i_ka']
        parallel = net.line_dc.loc[self.dmr_line, 'parallel']

        dcp = net.res_line_dc.loc[self.dc_plus_line, ['i_from_ka', 'i_to_ka']]
        dcm = net.res_line_dc.loc[self.dc_minus_line, ['i_from_ka', 'i_to_ka']]
        net.res_line_dc.loc[self.dmr_line, ['i_from_ka', 'i_to_ka']] = dcp - dcm

        dcp = net.res_line_dc.loc[self.dc_plus_line, 'i_ka']
        dcm = net.res_line_dc.loc[self.dc_minus_line, 'i_ka']
        net.res_line_dc.loc[self.dmr_line, 'i_ka'] = np.abs(dcp - dcm)

        net.res_line_dc.loc[self.dmr_line, 'loading_percent'] = np.abs(dcp - dcm) / (max_i_ka * parallel) * 100.


    def __str__(self):
        return super().__str__() + f"{self.dmr_line}"

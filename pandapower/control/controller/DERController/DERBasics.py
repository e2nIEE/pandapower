# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.toolbox.power_factor import cosphi_to_pos, cosphi_from_pos

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Basic Classes
# -------------------------------------------------------------------------------------------------


class BaseModel:
    @classmethod
    def name(cls):
        return str(cls).split(".")[-1][:-2]


class QVCurve:
    """ Simple Q(V) controller. The characteristic curve is defined by 'v_points_pu' and
    'q_points' (relative to sn_mva).

                                   - Q(Vm)/sn (underexcited)
                                   ^
                                   |
                                   |
                                   |               _______
                                   |              /
                                   |             /
                                   |            /
                                   |           /
                                   |          /
             v[0] v[1]   v[2]      |         /
       --------+----+-----+--------+--------+-----+------+------>
                         /         |      v[3]   v[4]   v[5]    Vm
                        /          |
                       /           |
                      /            |
                     /             |
                ____/              |
                                   + Q(Vm)/sn (overexcited)
    """
    def __init__(self, v_points_pu, q_points):
        self.v_points_pu = v_points_pu
        self.q_points = q_points

    def step(self, vm_pu):
        return np.interp(vm_pu, self.v_points_pu, self.q_points)


class CosphiVCurve:
    """ Simple Q(V) controller. The characteristic curve is
    defined by 'v_points_pu' and 'cosphi_points' (pos. values -> overexcited, voltage increasing,
    neg. values -> underexcited, voltage decreasing).

                                   - cosphi(Vm) (underexcited)
                                   ^
                                   |
                                   |
                                   |               _______
                                   |              /
                                   |             /
                                   |            /
                                   |           /
                                   |          /
             v[0] v[1]   v[2]      |         /
       --------+----+-----+--------+--------+-----+------+------>
                         /         |      v[3]   v[4]   v[5]    Vm
                        /          |
                       /           |
                      /            |
                     /             |
                ____/              |
                                   + cosphi(Vm) (overexcited)
    """
    def __init__(self, v_points_pu, cosphi_points):
        self.v_points_pu = v_points_pu
        self.cosphi_points = cosphi_points
        self.cosphi_pos = cosphi_to_pos(self.cosphi_points)

    def step(self, vm_pu, p):
        cosphi = cosphi_from_pos(np.interp(vm_pu, self.v_points_pu, self.cosphi_pos))
        return np.tan(np.arccos(cosphi)) * p


class CosphiPCurve:
    """ CosphiPCurve is a Q(P) controller, more precisely a cosphi(P) controller. The characteristic
    curve is defined by 'p_points' and 'cosphi_points' (pos. values -> overexcited, voltage
    increasing, neg. values -> underexcited, voltage decreasing).

    **Exemplary Characteristic curve**::

        cosphi(P)
          ^
          |
          |________
          |        \
          |         \
          |          \
          |           \
          |            \ ___________
          |
          |
        0 +--+-----+----+-------------->
           p[0]    p[1]  p[2]
    INPUT:
        **p_points** (iterable of floats) - active power values (relative to sn_mva) on the
        cosphi(P) curve.

        **cosphi_points** (iterable of floats) - cosphi values on the cosphi(P) curve. positive
        values lead to positive reactive power values (inductive/overexcited generator), negative
        values mean capactive/underexcited generator
    """
    def __init__(self, p_points, cosphi_points):
        self.p_points = np.array(p_points)
        self.cosphi_points = np.array(cosphi_points)
        if any(self.cosphi_points > 1):
            raise ValueError("cosphi cannot be higher than 1")
        if any(self.cosphi_points < -1):
            raise ValueError("cosphi cannot be lower than -1")

        self.cosphi_pos = cosphi_to_pos(self.cosphi_points)

    def step(self, p):
        cosphi = cosphi_from_pos(np.interp(p, self.p_points, self.cosphi_pos))
        return np.tan(np.arccos(cosphi)) * p


if __name__ == "__main__":
    pass

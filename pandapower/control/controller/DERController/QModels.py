# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.auxiliary import ensure_iterability
from pandapower.control.controller.DERController.DERBasics import BaseModel, QVCurve, \
    CosphiVCurve, CosphiPCurve

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class QModel(BaseModel):
    """
    Base class to model how q is determined in step().
    """

    def __init__(self, **kwargs):
        pass

    def step(self, vm=None, p=None):
        pass

    def __str__(self):
        return self.__class__.name


class QModelConstQ(QModel):
    """
    Class to model a fixed q value.
    """

    def __init__(self, q):
        self.q = q

    def step(self, vm=None, p=None):
        if p is not None:
            len_ = len(p)
        elif vm is not None:
            len_ = len(vm)
        else:
            len_ = None
        return np.array(ensure_iterability(self.q, len_))

    def __str__(self):
        return self.__class__.name + " ,const_q:" + str(self.q)


class QModelCosphiVCurve(QModel):
    """
    Base class to model that q is determined in dependency of the voltage and active power.
    Class to model that q is determined in dependency of a voltage dependent cosphi curve.
    """

    def __init__(self, cosphi_v_curve):
        if isinstance(cosphi_v_curve, dict):
            self.cosphi_v_curve = CosphiVCurve(**cosphi_v_curve)
        else:
            self.cosphi_v_curve = cosphi_v_curve

    def step(self, vm, p=None):
        assert p is not None
        assert all(p >= 0)
        return self.cosphi_v_curve.step(vm, p)


class QModelCosphiP(QModel):
    """
    Class to model that q is determined in dependency of the momentary active power by a fixed
    cosphi. q is calculated based on: `q = sin(phi) * p` instead of `q = sin(phi) * s` -> It ignores
    that s changes by changing q.
    """

    def __init__(self, cosphi: float):
        assert -1 <= cosphi <= 1
        self.cosphi = cosphi
        self.q_setpoint = np.sign(self.cosphi) * np.sqrt(1-self.cosphi**2)

    def step(self, vm=None, p=None,):
        assert p is not None
        if any(p < 0):
            logger.warning("p < 0 is assumed as p=0 in QModelCosphiP.step()")
            p[p< 0] = 0
        return (p * self.q_setpoint)

    def __str__(self):
        return self.__class__.name + " ,cosphi:" + str(self.cosphi)


class QModelCosphiSn(QModel):
    """
    Class to model that q is determined in dependency of the apparent power by a fixed cosphi. q is
    calculated based on: `q = sin(phi) * s` only considering sn_mva and ignoring that s changes with
    p and q.
    """

    def __init__(self, cosphi=0.2):
        self.cosphi = cosphi

    def step(self, vm=None, p=None):
        return (1 * self.cosphi)

    def __str__(self):
        return self.__class__.name + " ,cosphi:" + str(self.cosphi)


class QModelCosphiPQ(QModel):
    """
    Class to model that q is determined in dependency of the momentary active power by a fixed
    cosphi. q is calculated based on: `q = sin(phi) * s` with `s = (p**2 + q**2)**0.5` -> the
    actual cosphi covers the requested cosphi.
    """

    def __init__(self, cosphi: float):
        assert -1 <= cosphi <= 1
        self.cosphi = cosphi
        self.q_setpoint = np.sign(self.cosphi) * np.sqrt(1-self.cosphi**2)

    def step(self, vm=None, p=None,):
        assert p is not None
        if any(p < 0):
            logger.warning("p < 0 is assumed as p=0 QModelCosphiPQ.step()")
            p[p < 0] = 0
        return (p/abs(self.cosphi) * self.q_setpoint)

    def __str__(self):
        return self.__class__.name + " ,cosphi:" + str(self.cosphi)


class QModelCosphiPCurve(QModel):
    """
    Class to model that q is determined in dependency of a cosphi which depends on the active power.
    """

    def __init__(self, cosphi_p_curve):
        if isinstance(cosphi_p_curve, dict):
            self.cosphi_p_curve = CosphiPCurve(**cosphi_p_curve)
        else:
            self.cosphi_p_curve = cosphi_p_curve

    def step(self, vm=None, p=None):
        assert p is not None
        assert all(p >= 0)
        return self.cosphi_p_curve.step(p)


class QModelQV(QModel):
    """
    Base class to model that q is determined in dependency of the voltage.
    """

    def __init__(self, qv_curve):
        if isinstance(qv_curve, dict):
            self.qv_curve = QVCurve(**qv_curve)
        else:
            self.qv_curve = qv_curve

    def step(self, vm, p=None):
        q = self.qv_curve.step(vm)
        return q


if __name__ == "__main__":
    pass

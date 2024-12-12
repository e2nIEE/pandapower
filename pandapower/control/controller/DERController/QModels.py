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

    def step(self, vm_pu=None, p_pu=None):
        pass

    def __str__(self):
        return self.__class__.name()


class QModelConstQ(QModel):
    """
    Class to model a fixed q value.
    """

    def __init__(self, q_pu):
        self.q_pu = q_pu

    def step(self, vm_pu=None, p_pu=None):
        if p_pu is not None:
            len_ = len(p_pu)
        elif vm_pu is not None:
            len_ = len(vm_pu)
        else:
            len_ = None
        return np.array(ensure_iterability(self.q_pu, len_))

    def __str__(self):
        return self.__class__.name() + f"(q={self.q_pu})"


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

    def step(self, vm_pu, p_pu=None):
        assert p_pu is not None
        assert all(p_pu >= 0)
        return self.cosphi_v_curve.step(vm_pu, p_pu)


class QModelCosphiP(QModel):
    """
    Class to model that q is determined in dependency of the momentary active power by a fixed
    cosphi. q is calculated based on: `q = sin(phi) * p` instead of `q = sin(phi) * s` -> It ignores
    that s changes by changing q.
    """

    def __init__(self, cosphi: float):
        assert -1 <= cosphi <= 1
        self.cosphi = cosphi
        self.q_setpoint_pu = np.sign(self.cosphi) * np.sqrt(1-self.cosphi**2)

    def step(self, vm_pu=None, p_pu=None,):
        assert p_pu is not None
        if any(p_pu < 0):
            logger.warning("p < 0 is assumed as p=0 in QModelCosphiP.step()")
            p_pu[p_pu< 0] = 0
        return (p_pu * self.q_setpoint_pu)

    def __str__(self):
        return self.__class__.name() + f"(cosphi={self.cosphi})"


class QModelCosphiSn(QModel):
    """
    Class to model that q is determined in dependency of the apparent power by a fixed cosphi. q is
    calculated based on: `q = sin(phi) * s` only considering sn_mva and ignoring that s changes with
    p and q.
    """

    def __init__(self, cosphi=0.2):
        self.cosphi = cosphi

    def step(self, vm_pu=None, p_pu=None):
        return (1 * self.cosphi)

    def __str__(self):
        return self.__class__.name() + f"(cosphi={self.cosphi})"


class QModelCosphiPQ(QModel):
    """
    Class to model that q is determined in dependency of the momentary active power by a fixed
    cosphi. q is calculated based on: `q = sin(phi) * s` with `s = (p**2 + q**2)**0.5` -> the
    actual cosphi covers the requested cosphi.
    """

    def __init__(self, cosphi: float):
        assert -1 <= cosphi <= 1
        self.cosphi = cosphi
        self.q_setpoint_pu = np.sign(self.cosphi) * np.sqrt(1-self.cosphi**2)

    def step(self, vm_pu=None, p_pu=None,):
        assert p_pu is not None
        if any(p_pu < 0):
            logger.warning("p_pu < 0 is assumed as p_pu=0 QModelCosphiPQ.step()")
            p_pu[p_pu < 0] = 0
        return (p_pu/abs(self.cosphi) * self.q_setpoint_pu)

    def __str__(self):
        return self.__class__.name() + f"(cosphi={self.cosphi})"


class QModelCosphiPCurve(QModel):
    """
    Class to model that q is determined in dependency of a cosphi which depends on the active power.
    """

    def __init__(self, cosphi_p_curve):
        if isinstance(cosphi_p_curve, dict):
            self.cosphi_p_curve = CosphiPCurve(**cosphi_p_curve)
        else:
            self.cosphi_p_curve = cosphi_p_curve

    def step(self, vm_pu=None, p_pu=None):
        assert p_pu is not None
        assert all(p_pu >= 0)
        return self.cosphi_p_curve.step(p_pu)


class QModelQVCurve(QModel):
    """
    Base class to model that q is determined in dependency of the voltage.
    """

    def __init__(self, qv_curve):
        if isinstance(qv_curve, dict):
            self.qv_curve = QVCurve(**qv_curve)
        else:
            self.qv_curve = qv_curve

    def step(self, vm_pu, p_pu=None):
        q_pu = self.qv_curve.step(vm_pu)
        return q_pu


if __name__ == "__main__":
    pass

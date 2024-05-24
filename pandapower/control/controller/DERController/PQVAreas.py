# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
try:
    from shapely.geometry import LineString, Point
    from shapely.geometry.polygon import Polygon
    shapely_imported = True
except ImportError:
    shapely_imported = False

from pandapower.auxiliary import soft_dependency_error
from pandapower.toolbox.power_factor import cosphi_to_pos, cosphi_from_pos
from pandapower.control.controller.DERController.DERBasics import BaseModel

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
""" This file defines PQ, QV, and PQV areas of reactive power provision capabilities including those
defined in VDE-AR-N technical connection rule standards"""
# -------------------------------------------------------------------------------------------------

""" Base class """

class BaseArea(BaseModel):
    """ Defines which functionality is common to all area (PQ or QV) classes. """
    def in_area(self, p, q, vm):
        min_max_q = self.q_flexibility(p, vm)
        return (min_max_q[:, 0] <= q) & (min_max_q[:, 1] >= q)

    def q_flexibility(self, p, vm):
        pass

    def __str__(self):
        return self.__class__.name


class BasePQVArea(BaseArea):
    """ Defines functionality common for mulitple PQVArea classes. """
    def in_area(self, p, q, vm):
        return self.pq_area.in_area(p, q, vm) & self.qv_area.in_area(p, q, vm)

    def q_flexibility(self, p, vm):
        min_max_q = self.pq_area.q_flexibility(p)
        min_max_q_qv = self.qv_area.q_flexibility(None, vm)
        min_max_q[:, 0] = np.maximum(min_max_q[:, 0], min_max_q_qv[:, 0])
        min_max_q[:, 1] = np.minimum(min_max_q[:, 1], min_max_q_qv[:, 1])
        no_flexibility = min_max_q[:, 0] > min_max_q[:, 1]
        if n_no_flex := sum(no_flexibility):
            logger.debug(f"For {n_no_flex} elements, min_q and max_q were set to the same point of "
                         "no flexibility.")
            min_max_q[no_flexibility, :] = np.tile(np.sum(min_max_q[no_flexibility], axis=1)/2,
                                                   (1, 2))
        return min_max_q


""" Polygon Areas """

class PQAreaPOLYGON(BaseArea):
    """
    Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'p_points' and 'q_points'.

    Note: Due to generator point of view, negative q values are correspond with underexcited behavior.

    Example
    -------
    >>> PQAreaDefault(p_points=(0.1, 0.2, 1, 1, 0.2, 0.1, 0.1),
    ...               q_points=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1))
    """
    def __init__(self, p_points, q_points):
        self.p_points = p_points
        self.q_points = q_points

        if not shapely_imported:
            soft_dependency_error("PQAreaDefault", "shapely")

        self.polygon = Polygon([(p, q) for p, q in zip(p_points, q_points)])

    def in_area(self, p, q, vm=None):
        return np.array([self.polygon.contains(Point(pi, qi)) for pi, qi in zip(p, q)])

    def q_flexibility(self, p, vm=None):
        def _q_flex(p):
            line = LineString([(p, -1), (p, 1)])
            if line.intersects(self.polygon):
                return [point[1] for point in LineString(
                    [(p, -1), (p, 1)]).intersection(self.polygon).coords]
            else:
                return [0, 0]
        return np.r_[[_q_flex(pi) for pi in p]]


class QVAreaPOLYGON(BaseArea):
    """
    Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'q_points' and 'vm_points'.

    Note: Due to generator point of view, negative q values are correspond with underexcited behavior.

    Example
    -------
    >>> QVAreaPOLYGON(q_points=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1),
    ...               vm_points=(0.9, 1.05, 1.1, 1.1, 1.05, 0.9, 0.9))
    """
    def __init__(self, q_points, vm_points):
        self.q_points = q_points
        self.vm_points = vm_points

        if not shapely_imported:
            soft_dependency_error("PQAreaDefault", "shapely")

        # note that the naming QVAreaPOLYGON might be confusing because, in fact, it is a VQAreaPOLYGON
        self.polygon = Polygon([(vm, q) for vm, q in zip(vm_points, q_points)])

    def in_area(self, p, q, vm):
        return np.array([self.polygon.contains(Point(vmi, qi)) for vmi, qi in zip(vm, q)])

    def q_flexibility(self, p, vm):
        assert all(vm >= 0) and all(vm <= 2)
        def _q_flex(vm):
            line = LineString([(vm, -1), (vm, 1)])
            if line.intersects(self.polygon):
                return [point[1] for point in LineString(
                    [(vm, -1), (vm, 1)]).intersection(self.polygon).coords]
            else:
                return [0, 0]
        return np.r_[[_q_flex(vmi) for vmi in vm]]


class PQVAreaPOLYGON(BasePQVArea):
    """
    Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'p_points' and 'q_pq_points' as well as 'q_qv_points' and 'vm_points'.

    Example
    -------
    >>> PQVAreaDefault(p_points=(0.1, 0.2, 1, 1, 0.2, 0.1, 0.1),
    ...                q_pq_points=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1),
    ...                q_qv_points=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1),
    ...                vm_points=(0.9, 1.05, 1.1, 1.1, 1.05, 0.9, 0.9))
    """
    def __init__(self, p_points, q_pq_points, q_qv_points, vm_points):
        self.pq_area = PQAreaPOLYGON(p_points, q_pq_points)
        self.qv_area = QVAreaPOLYGON(q_qv_points, vm_points)


""" STATCOM DERs: """


class PQAreaSTATCOM(BaseArea):
    """
    PQAreaSTATCOM provides a simple rectangular reactive power provision area without a dependency on the
    on the active power.

    Example
    -------
    >>> PQAreaSTATCOM(min_q=-0.328684, max_q=0.410775)
    """

    def __init__(self, min_q, max_q):
        self.min_q, self.max_q = min_q, max_q

    def in_area(self, p, q, vm=None):
        return (self.min_q <= q) & (q <= self.max_q)

    def q_flexibility(self, p, vm=None):
        return np.c_[[self.min_q]*len(p), [self.max_q]*len(p)]


""" HV DERs: """


class PQArea4120(BaseArea):
    """
    This class models the PQ area of flexible Q for high-voltage plants according to VDE AR-N-4120.
    It is used to be combined with Voltage dependencies in PQVArea4120V1, PQVArea4120V2 and
    PQVArea4120V3.
    """
    def __init__(self, min_q, max_q, version=2018):
        supported_versions = [2015, 2018]
        p_points = {2015: [0.1, 0.2], 2018: [0.05, 0.2]}
        if version not in supported_versions:
            raise ValueError(
                f"version '{version}' is not supported, it is not in {supported_versions}")
        self.version = version
        self.p_points = p_points[version]
        self.min_q, self.max_q = min_q, max_q

        self.linear_factor_ind = (self.min_q + self.p_points[0]) / (
            self.p_points[1] - self.p_points[0])
        self.linear_factor_cap = (self.max_q - self.p_points[0]) / (
            self.p_points[1] - self.p_points[0])

    def in_area(self, p, q, vm=None, q_max_under_p_point=0.):
        is_in_area = np.ones(len(p), dtype=bool)
        is_in_area[(p < self.p_points[0]) & ((q < -0.05) | (q > q_max_under_p_point))] = False
        if all(~is_in_area):
            return is_in_area
        is_in_area[(p > self.p_points[1]) & ((q < self.min_q) | (q > self.max_q))] = False
        if all(~is_in_area):
            return is_in_area
        is_in_area[(q < -self.p_points[0]-(p-self.p_points[0])*self.linear_factor_ind) | \
            (q > self.p_points[0]+(p-self.p_points[0])*self.linear_factor_cap)] = False
        return is_in_area

    def q_flexibility(self, p, vm=None):
        q_flex = np.c_[[self.min_q]*len(p), [self.max_q]*len(p)]

        part = p < self.p_points[1]
        q_flex[part] = np.c_[-self.p_points[0]+(p[part]-self.p_points[0])*self.linear_factor_ind,
                             self.p_points[0]+(p[part]-self.p_points[0])*self.linear_factor_cap]

        part = p < self.p_points[0]
        q_flex[part] = np.c_[[-0.05]*sum(part), [0]*sum(part)]

        return q_flex


class QVArea4120(BaseArea):
    """
    This class models the QV area of flexible Q for high-voltage power plants according to
    VDE AR-N-4120.
    It is used to be combined with active power dependencies in PQVArea4120V1, PQVArea4120V2, or
    PQVArea4120V3
    """
    def __init__(self, min_q, max_q):
        self.min_q, self.max_q = min_q, max_q

        self.max_vm = 127.0/110
        self.min_vm = 96.0/110
        self.delta_vm = 7.0/110
        self.linear_factor = (self.max_q - self.min_q) / self.delta_vm

    def q_flexibility(self, p, vm):

        # part = vm > self.max_vm
        len_ = len(vm)
        min_max_q = np.c_[[self.min_q]*len_, [self.min_q]*len_]

        part = vm < self.min_vm
        len_ = sum(part)
        min_max_q[part] = np.c_[[self.max_q]*len_, [self.max_q]*len_]

        part = (self.min_vm < vm) & (vm <= self.min_vm + self.delta_vm)
        min_max_q[part] = np.c_[self.max_q - self.linear_factor * (vm[part]-self.min_vm),
                                [self.max_q]*sum(part)]

        part = (self.min_vm+self.delta_vm < vm) & (vm <= self.max_vm-self.delta_vm)
        min_max_q[part] = np.repeat(np.array([[self.min_q, self.max_q]]), sum(part), axis=0)

        part = (self.max_vm-self.delta_vm < vm) & (vm <= self.max_vm)
        min_max_q[part] = np.c_[[self.min_q]*sum(part),
                                self.min_q + self.linear_factor * (self.max_vm-vm[part])]

        return min_max_q


class PQVArea4120Base(BasePQVArea):
    """
    This is the base class for the three variants of VDE AR-N-4120. This class is not for direct
    application by the user.
    """
    def __init__(self, min_q, max_q, version=2018):
        self.pq_area = PQArea4120(min_q, max_q, version=version)
        self.qv_area = QVArea4120(min_q, max_q)

class PQVArea4120V1(PQVArea4120Base):
    """
    This class models the PQV area of flexible Q for high-voltage plants according to variant 1 of
    VDE AR-N-4120.
    """
    def __init__(self, version=2018):
        super().__init__(-0.227902, 0.484322, version=version)

class PQVArea4120V2(PQVArea4120Base):
    """
    This class models the PQV area of flexible Q for high-voltage plants according to variant 2 of
    VDE AR-N-4120.
    """
    def __init__(self, version=2018):
        super().__init__(-0.328684, 0.410775, version=version)


class PQVArea4120V3(PQVArea4120Base):
    """
    This class models the PQV area of flexible Q for high-voltage plants according to variant 3 of
    VDE AR-N-4120.
    """
    def __init__(self, version=2018):
        super().__init__(-0.410775, 0.328684, version=version)


""" EHV DERs: """

class PQArea4130(PQArea4120):
    """
    This class models the PQ area of flexible Q for extra high-voltage plants according to VDE AR-N-4130.
    It is used to be combined with Voltage dependencies in PQVArea4130V1, PQVArea4130V2 and
    PQVArea4130V3.
    """
    def in_area(self, p, q, vm=None):
        return super().in_area(p, q, vm=vm, q_max_under_p_point=0.05)

    def q_flexibility(self, p, vm=None):
        q_flex = super().q_flexibility(p, vm)
        part = p < 0.1
        q_flex[part] = np.c_[[-0.05]*sum(part), [0.05]*sum(part)]
        return q_flex


class QVArea4130(QVArea4120):
    """
    This class models the QV area of flexible Q for extra high voltage power plants according to
    VDE AR-N-4130.
    It is used to be combined with active power dependencies in PQVArea4130V1, PQVArea4130V2, or
    PQVArea4130V3
    """
    def __init__(self, min_q, max_q, vn, variant):
        self.min_q, self.max_q, self.vn, self.variant = min_q, max_q, vn, variant
        epsilon = 1e-3
        if not np.any(np.isclose(np.array([380, 220]), vn)):
            raise ValueError(f"QVArea4130 is defined for 380kV and 220kV only, not for {vn}kV.")
        if variant == 1:
            if vn == 380:
                self.vm_min_points = np.array([350-epsilon, 350, 380, 400]) / vn
                self.vm_max_points = np.array([420, 440]) / vn
            elif vn == 220:
                self.vm_min_points = np.array([193-epsilon, 193, 220, 233.5]) / vn
                self.vm_max_points = np.array([245, 253]) / vn
            self.q_min_points = np.array([
                self.max_q, self.max_q*np.sin(np.arccos(0.95))/np.sin(np.arccos(0.9)), 0, self.min_q])
            self.q_max_points = np.array([self.max_q, self.min_q])
        elif variant == 2:
            if vn == 380:
                self.vm_min_points = np.array([350-epsilon, 350, 380, 410]) / vn
                self.vm_max_points = np.array([420, 440, 440+epsilon]) / vn
            elif vn == 220:
                self.vm_min_points = np.array([193-epsilon, 193, 220, 240]) / vn
                self.vm_max_points = np.array([245, 253, 253+epsilon]) / vn
            self.q_min_points = np.array([
                self.max_q, self.max_q*np.sin(np.arccos(0.95))/np.sin(np.arccos(0.925)), 0, self.min_q])
            self.q_max_points = np.array([self.max_q, 0, self.min_q])
        elif variant == 3:
            if vn == 380:
                self.vm_min_points = np.array([350, 380]) / vn
                self.vm_max_points = np.array([420, 440, 440+epsilon]) / vn
            elif vn == 220:
                self.vm_min_points = np.array([193, 220]) / vn
                self.vm_max_points = np.array([245, 253, 253+epsilon]) / vn
            self.q_min_points = np.array([self.max_q, self.min_q])
            self.q_max_points = np.array([self.max_q, 0, self.min_q])
        else:
            raise ValueError(
                f"QVArea4130 is defined for variants 1, 2, and 3 but not for {self.variant}.")

    def q_flexibility(self, p, vm):
        return np.c_[np.interp(vm, self.vm_min_points, self.q_min_points),
                     np.interp(vm, self.vm_max_points, self.q_max_points)]


class PQVArea4130Base(BasePQVArea):
    """
    This is the base class for the three variants of VDE AR-N-4130. This class is not for direct
    application by the user.
    """
    def __init__(self, min_q, max_q, variant, version=2018, vn=380):
        self.pq_area = PQArea4130(min_q, max_q, version=version)
        self.qv_area = QVArea4130(min_q, max_q, vn=vn, variant=variant)


class PQVArea4130V1(PQVArea4130Base):
    """
    This class models the PQV area of flexible Q for extra high voltage power plants according to
    variant 1 of VDE AR-N-4130.
    """
    def __init__(self, version=2018, vn=380):
        super().__init__(-0.227902, 0.484322, 1, version=version, vn=vn)

class PQVArea4130V2(PQVArea4130Base):
    """
    This class models the PQV area of flexible Q for extra high voltage power plants according to
    variant 2 of VDE AR-N-4130.
    """
    def __init__(self, version=2018, vn=380):
        super().__init__(-0.328684, 0.410775, 2, version=version, vn=vn)


class PQVArea4130V3(PQVArea4130Base):
    """
    This class models the PQV area of flexible Q for extra high voltage power plants according to
    variant 3 of VDE AR-N-4130.
    """
    def __init__(self, version=2018, vn=380):
        super().__init__(-0.410775, 0.328684, 3, version=version, vn=vn)


""" MV DERs: """


class PQArea4110(PQAreaPOLYGON):
    def __init__(self):
        p_points = (0.05, 1, 1, 0.05, 0.0)
        q_points = (-0.01961505, -0.484322, 0.484322, 0.01961505, -0.01961505)
        super().__init__(p_points, q_points)


class QVArea4110(QVAreaPOLYGON):
    """
    This class models the QV area of flexible Q for medium-voltage plants according to
    VDE AR-N-4110.
    """
    def __init__(self):
        q_points =  (0. , -0.484322, -0.484322, 0. , 0.484322, 0.484322, 0. )
        vm_points = (0.9, 0.95     , 1.1      , 1.1, 1.05    , 0.9     , 0.9)
        super().__init__(q_points, vm_points)


class PQVArea4110(BasePQVArea):
    def __init__(self):
        self.pq_area = PQArea4110()
        self.qv_area = QVArea4110()

""" LV DERs: """


class PQArea4105(PQAreaPOLYGON):
    """
    This class models the PQ area of flexible Q for low-voltage plants according to VDE AR-N-4105.
    Plants with S_E,max <= 4.6 kVA should apply variant 1 while S_E,max > 4.6 kVA should apply
    variant 2.

    Note: To be in line with the other VDE AR N standard (meaning that all values are relative to
    p_{b,install}), the free area between the minimial reactive power requirements and the maximum
    apparent power of the inverter is not modeled as 'in_area'.
    """
    def __init__(self, variant):
        if variant == 1:
            p_points = (0., 0.95, 0.95, 0.)
            q_points = (0., -0.328684, 0.328684, 0.)
        elif variant == 2:
            p_points = (0., 0.9, 0.9, 0.)
            q_points = (0., -0.484322, 0.484322, 0.)
        else:
            raise ValueError(f"{variant=}")
        super().__init__(p_points, q_points)


class QVArea4105(QVAreaPOLYGON):
    """
    This class models the QV area of flexible Q for low-voltage plants according to VDE AR-N-4105.
    Plants with S_E,max <= 4.6 kVA should apply variant 1 while S_E,max > 4.6 kVA should apply
    variant 2.
    """
    def __init__(self, variant):
        if variant == 1:
            q_points =  (0. , -0.328684, -0.328684, 0. , 0.328684, 0.328684, 0. )
            vm_points = (0.9, 0.95     , 1.1      , 1.1, 1.05    , 0.9     , 0.9)
        elif variant == 2:
            q_points =  (0. , -0.484322, -0.484322, 0. , 0.484322, 0.484322, 0. )
            vm_points = (0.9, 0.95     , 1.1      , 1.1, 1.05    , 0.9     , 0.9)
        else:
            raise ValueError(f"{variant=}")
        super().__init__(q_points, vm_points)


class PQVArea4105(BasePQVArea):
    """
    This class models the PQV area of flexible Q for low-voltage plants according to VDE AR-N-4105.
    Plants with S_E,max <= 4.6 kVA should apply variant 1 while S_E,max > 4.6 kVA should apply
    variant 2.
    """
    def __init__(self, variant, min_cosphi, max_cosphi):
        self.pq_area = PQArea4105(variant)
        self.qv_area = QVArea4105(variant)


if __name__ == "__main__":
    pass

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
    def in_area(self, p_pu, q_pu, vm_pu):
        min_max_q = self.q_flexibility(p_pu, vm_pu)
        return (min_max_q[:, 0] <= q_pu) & (min_max_q[:, 1] >= q_pu)

    def q_flexibility(self, p_pu, vm_pu):
        pass

    def __str__(self):
        return self.__class__.name()


class BasePQVArea(BaseArea):
    """ Defines functionality common for mulitple PQVArea classes. """
    def __init__(self, raise_merge_overlap=True):
        self.raise_merge_overlap = raise_merge_overlap

    def in_area(self, p_pu, q_pu, vm_pu):
        return self.pq_area.in_area(p_pu, q_pu, vm_pu) & self.qv_area.in_area(p_pu, q_pu, vm_pu)

    def q_flexibility(self, p_pu, vm_pu):
        # PQ area flexibility
        min_max_q = self.pq_area.q_flexibility(p_pu)
        no_flexibility = min_max_q[:, 0] > min_max_q[:, 1]
        if n_no_flex := sum(no_flexibility):
            raise ValueError(f"For {n_no_flex} elements, the q flexibility of the PQ area provides "
                             "max_q < min_q.")

        # QV area flexibility
        min_max_q_qv = self.qv_area.q_flexibility(None, vm_pu)
        no_flexibility = min_max_q_qv[:, 0] > min_max_q_qv[:, 1]
        if n_no_flex := sum(no_flexibility):
            raise ValueError(f"For {n_no_flex} elements, the q flexibility of the QV area provides "
                             "max_q < min_q.")

        # merge q flexibility of pq and qv areas
        min_max_q[:, 0] = np.maximum(min_max_q[:, 0], min_max_q_qv[:, 0])
        min_max_q[:, 1] = np.minimum(min_max_q[:, 1], min_max_q_qv[:, 1])

        no_flexibility = min_max_q[:, 0] > min_max_q[:, 1]
        if n_no_flex := sum(no_flexibility):
            error_msg = (f"For {n_no_flex} elements, max_q > min_q result from considering both "
                         "the PQ area flexibility and the QV area flexibility.")
            if self.raise_merge_overlap:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
            min_max_q[no_flexibility, 0] = np.sum(min_max_q[no_flexibility], axis=1)/2
            min_max_q[no_flexibility, 1] = min_max_q[no_flexibility, 0]

        return min_max_q


""" Polygon Areas """

class PQAreaPOLYGON(BaseArea):
    """ Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'p_points_pu' and 'q_points_pu'.

    Note: Due to generator point of view, negative q values are correspond with underexcited behavior.

    Example
    -------
    >>> PQAreaDefault(p_points_pu=(0.1, 0.2, 1, 1, 0.2, 0.1, 0.1),
    ...               q_points_pu=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1))
    """
    def __init__(self, p_points_pu, q_points_pu):
        self.p_points_pu = p_points_pu
        self.q_points_pu = q_points_pu

        if not shapely_imported:
            soft_dependency_error("PQAreaDefault", "shapely")

        self.polygon = Polygon([(p_pu, q_pu) for p_pu, q_pu in zip(p_points_pu, q_points_pu)])

    def in_area(self, p_pu, q_pu, vm_pu=None):
        return np.array([self.polygon.contains(Point(pi, qi)) for pi, qi in zip(p_pu, q_pu)])

    def q_flexibility(self, p_pu, vm_pu=None):
        def _q_flex(p_pu):
            line = LineString([(p_pu, -1), (p_pu, 1)])
            if line.intersects(self.polygon):
                points = [point[1] for point in LineString(
                    [(p_pu, -1), (p_pu, 1)]).intersection(self.polygon).coords]
                if len(points) == 1:
                    return [points[0]]*2
                elif len(points) == 2:
                    return points
                else:
                    raise ValueError(f"{len(points)=} is wrong. 2 or 1 is expected.")
            else:
                return [0, 0]
        return np.r_[[_q_flex(pi) for pi in p_pu]]


class QVAreaPOLYGON(BaseArea):
    """ Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'q_points_pu' and 'vm_points_pu'.

    Note: Due to generator point of view, negative q values are correspond with underexcited behavior.

    Example
    -------
    >>> QVAreaPOLYGON(q_points_pu=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1),
    ...               vm_points_pu=(0.9, 1.05, 1.1, 1.1, 1.05, 0.9, 0.9))
    """
    def __init__(self, q_points_pu, vm_points_pu):
        self.q_points_pu = q_points_pu
        self.vm_points_pu = vm_points_pu

        if not shapely_imported:
            soft_dependency_error("PQAreaDefault", "shapely")

        # note that the naming QVAreaPOLYGON might be confusing because, in fact, it is a VQAreaPOLYGON
        self.polygon = Polygon([(vm, q) for vm, q in zip(vm_points_pu, q_points_pu)])

    def in_area(self, p_pu, q_pu, vm_pu):
        return np.array([self.polygon.contains(Point(vmi, qi)) for vmi, qi in zip(vm_pu, q_pu)])

    def q_flexibility(self, p_pu, vm_pu):
        assert all(vm_pu >= 0) and all(vm_pu <= 2)
        def _q_flex(vm_pu):
            line = LineString([(vm_pu, -1), (vm_pu, 1)])
            if line.intersects(self.polygon):
                return [point[1] for point in LineString(
                    [(vm_pu, -1), (vm_pu, 1)]).intersection(self.polygon).coords]
            else:
                return [0, 0]
        return np.r_[[_q_flex(vmi) for vmi in vm_pu]]


class PQVAreaPOLYGON(BasePQVArea):
    """ Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'p_points_pu' and 'q_pq_points_pu' as well as 'q_qv_points_pu' and 'vm_points_pu'.

    Example
    -------
    >>> PQVAreaDefault(p_points_pu=(0.1, 0.2, 1, 1, 0.2, 0.1, 0.1),
    ...                q_pq_points_pu=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1),
    ...                q_qv_points_pu=(0.1, 0.410775, 0.410775, -0.328684, -0.328684, -0.1, 0.1),
    ...                vm_points_pu=(0.9, 1.05, 1.1, 1.1, 1.05, 0.9, 0.9))
    """
    def __init__(self, p_points_pu, q_pq_points_pu, q_qv_points_pu, vm_points_pu,
                 raise_merge_overlap=True):
        super().__init__(raise_merge_overlap=raise_merge_overlap)
        self.pq_area = PQAreaPOLYGON(p_points_pu, q_pq_points_pu)
        self.qv_area = QVAreaPOLYGON(q_qv_points_pu, vm_points_pu)


""" STATCOM DERs: """


class PQAreaSTATCOM(BaseArea):
    """ PQAreaSTATCOM provides a simple rectangular reactive power provision area without a
    dependency on the on the active power.

    Example
    -------
    >>> PQAreaSTATCOM(min_q=-0.328684, max_q=0.410775)
    """

    def __init__(self, min_q_pu, max_q_pu):
        self.min_q_pu, self.max_q_pu = min_q_pu, max_q_pu

    def in_area(self, p_pu, q_pu, vm_pu=None):
        return (self.min_q_pu <= q_pu) & (q_pu <= self.max_q_pu)

    def q_flexibility(self, p_pu, vm_pu=None):
        return np.c_[[self.min_q_pu]*len(p_pu), [self.max_q_pu]*len(p_pu)]


""" HV DERs: """


class PQArea4120(BaseArea):
    """ This class models the PQ area of flexible Q for high-voltage plants according to
    VDE AR-N-4120. It is used to be combined with Voltage dependencies in PQVArea4120V1,
    PQVArea4120V2 and PQVArea4120V3.
    """
    def __init__(self, min_q_pu, max_q_pu, version=2018, **kwargs):
        supported_versions = [2015, 2018]
        p_points_pu = {2015: [0.1, 0.2], 2018: [0.05, 0.2]}
        if version not in supported_versions:
            raise ValueError(
                f"version '{version}' is not supported, it is not in {supported_versions}")
        self.version = version
        self.p_points_pu = p_points_pu[version]
        self.min_q_pu, self.max_q_pu = min_q_pu, max_q_pu
        self.q_max_under_p_point = kwargs.get("q_max_under_p_point", 0.)

        self.linear_factor_ind = (self.min_q_pu + 0.1) / (
            self.p_points_pu[1] - self.p_points_pu[0])
        self.linear_factor_cap = (self.max_q_pu - 0.1) / (
            self.p_points_pu[1] - self.p_points_pu[0])

    def in_area(self, p_pu, q_pu, vm_pu=None):
        is_in_area = np.ones(len(p_pu), dtype=bool)
        is_in_area[(p_pu < self.p_points_pu[0]) & ((q_pu < -0.05) | (q_pu > self.q_max_under_p_point))] = False
        if all(~is_in_area):
            return is_in_area
        is_in_area[(p_pu > self.p_points_pu[1]) & ((q_pu < self.min_q_pu) | (q_pu > self.max_q_pu))] = False
        if all(~is_in_area):
            return is_in_area
        is_in_area[(q_pu < -0.1-(p_pu-self.p_points_pu[0])*self.linear_factor_ind) | \
            (q_pu > 0.1+(p_pu-self.p_points_pu[0])*self.linear_factor_cap)] = False
        return is_in_area

    def q_flexibility(self, p_pu, vm_pu=None):
        q_flex = np.c_[[self.min_q_pu]*len(p_pu), [self.max_q_pu]*len(p_pu)]

        part = p_pu < self.p_points_pu[1]
        q_flex[part] = np.c_[-0.1+(p_pu[part]-self.p_points_pu[0])*self.linear_factor_ind,
                             0.1+(p_pu[part]-self.p_points_pu[0])*self.linear_factor_cap]

        part = p_pu < self.p_points_pu[0]
        q_flex[part] = np.c_[[-0.05]*sum(part), [self.q_max_under_p_point]*sum(part)]

        return q_flex


class QVArea4120(BaseArea):
    """ This class models the QV area of flexible Q for high-voltage power plants according to
    VDE AR-N-4120.
    It is used to be combined with active power dependencies in PQVArea4120V1, PQVArea4120V2, or
    PQVArea4120V3
    """
    def __init__(self, min_q_pu, max_q_pu):
        self.min_q_pu, self.max_q_pu = min_q_pu, max_q_pu

        self.max_vm_pu = 127.0/110
        self.min_vm_pu = 96.0/110
        self.delta_vm_pu = 7.0/110
        self.linear_factor = (self.max_q_pu - self.min_q_pu) / self.delta_vm_pu

    def q_flexibility(self, p_pu, vm_pu):

        # part = vm_pu > self.max_vm_pu
        len_ = len(vm_pu)
        min_max_q = np.c_[[self.min_q_pu]*len_, [self.min_q_pu]*len_]

        part = vm_pu < self.min_vm_pu
        len_ = sum(part)
        min_max_q[part] = np.c_[[self.max_q_pu]*len_, [self.max_q_pu]*len_]

        part = (self.min_vm_pu < vm_pu) & (vm_pu <= self.min_vm_pu + self.delta_vm_pu)
        min_max_q[part] = np.c_[self.max_q_pu - self.linear_factor * (vm_pu[part]-self.min_vm_pu),
                                [self.max_q_pu]*sum(part)]

        part = (self.min_vm_pu+self.delta_vm_pu < vm_pu) & (vm_pu <= self.max_vm_pu-self.delta_vm_pu)
        min_max_q[part] = np.repeat(np.array([[self.min_q_pu, self.max_q_pu]]), sum(part), axis=0)

        part = (self.max_vm_pu-self.delta_vm_pu < vm_pu) & (vm_pu <= self.max_vm_pu)
        min_max_q[part] = np.c_[[self.min_q_pu]*sum(part),
                                self.min_q_pu + self.linear_factor * (self.max_vm_pu-vm_pu[part])]

        return min_max_q


class PQVArea4120Base(BasePQVArea):
    """ This is the base class for the three variants of VDE AR-N-4120. This class is not for direct
    application by the user.
    """
    def __init__(self, min_q_pu, max_q_pu, version=2018, raise_merge_overlap=True):
        super().__init__(raise_merge_overlap=raise_merge_overlap)
        self.pq_area = PQArea4120(min_q_pu, max_q_pu, version=version)
        self.qv_area = QVArea4120(min_q_pu, max_q_pu)

class PQVArea4120V1(PQVArea4120Base):
    """ This class models the PQV area of flexible Q for high-voltage plants according to variant 1
    of VDE AR-N-4120.
    """
    def __init__(self, version=2018, raise_merge_overlap=True):
        super().__init__(-0.227902, 0.484322, version=version,
                         raise_merge_overlap=raise_merge_overlap)

class PQVArea4120V2(PQVArea4120Base):
    """ This class models the PQV area of flexible Q for high-voltage plants according to variant 2
    of VDE AR-N-4120.
    """
    def __init__(self, version=2018, raise_merge_overlap=True):
        super().__init__(-0.328684, 0.410775, version=version,
                         raise_merge_overlap=raise_merge_overlap)


class PQVArea4120V3(PQVArea4120Base):
    """ This class models the PQV area of flexible Q for high-voltage plants according to variant 3
    of VDE AR-N-4120.
    """
    def __init__(self, version=2018, raise_merge_overlap=True):
        super().__init__(-0.410775, 0.328684, version=version,
                         raise_merge_overlap=raise_merge_overlap)


""" EHV DERs: """

class PQArea4130(PQArea4120):
    """ This class models the PQ area of flexible Q for extra high-voltage plants according to
    VDE AR-N-4130. It is used to be combined with Voltage dependencies in PQVArea4130V1,
    PQVArea4130V2 and PQVArea4130V3.
    """
    def __init__(self, min_q_pu, max_q_pu):
        super().__init__(min_q_pu, max_q_pu, version=2018, q_max_under_p_point=0.05)


class QVArea4130(QVArea4120):
    """ This class models the QV area of flexible Q for extra high voltage power plants according to
    VDE AR-N-4130.
    It is used to be combined with active power dependencies in PQVArea4130V1, PQVArea4130V2, or
    PQVArea4130V3
    """
    def __init__(self, min_q_pu, max_q_pu, vn_kv, variant):
        self.min_q_pu, self.max_q_pu, self.vn_kv, self.variant = min_q_pu, max_q_pu, vn_kv, variant
        epsilon = 1e-3
        if not np.any(np.isclose(np.array([380, 220]), vn_kv)):
            raise ValueError(f"QVArea4130 is defined for 380kV and 220kV only, not for {vn_kv} kV.")
        if variant == 1:
            if vn_kv == 380:
                self.min_vm_points_pu = np.array([350-epsilon, 350, 380, 400]) / vn_kv
                self.max_vm_points_pu = np.array([420, 440]) / vn_kv
            elif vn_kv == 220:
                self.min_vm_points_pu = np.array([193-epsilon, 193, 220, 233.5]) / vn_kv
                self.max_vm_points_pu = np.array([245, 253]) / vn_kv
            self.min_q_points_pu = np.array([
                self.max_q_pu, self.max_q_pu*np.sin(np.arccos(0.95))/np.sin(np.arccos(0.9)), 0,
                                                    self.min_q_pu])
            self.max_q_points_pu = np.array([self.max_q_pu, self.min_q_pu])
        elif variant == 2:
            if vn_kv == 380:
                self.min_vm_points_pu = np.array([350-epsilon, 350, 380, 410]) / vn_kv
                self.max_vm_points_pu = np.array([420, 440, 440+epsilon]) / vn_kv
            elif vn_kv == 220:
                self.min_vm_points_pu = np.array([193-epsilon, 193, 220, 240]) / vn_kv
                self.max_vm_points_pu = np.array([245, 253, 253+epsilon]) / vn_kv
            self.min_q_points_pu = np.array([
                self.max_q, self.max_q*np.sin(np.arccos(0.95))/np.sin(np.arccos(0.925)), 0,
                                              self.min_q])
            self.max_q_points_pu = np.array([self.max_q, 0, self.min_q])
        elif variant == 3:
            if vn_kv == 380:
                self.min_vm_points_pu = np.array([350, 380]) / vn_kv
                self.max_vm_points_pu = np.array([420, 440, 440+epsilon]) / vn_kv
            elif vn_kv == 220:
                self.min_vm_points_pu = np.array([193, 220]) / vn_kv
                self.max_vm_points_pu = np.array([245, 253, 253+epsilon]) / vn_kv
            self.min_q_points_pu = np.array([self.max_q_pu, self.min_q_pu])
            self.max_q_points_pu = np.array([self.max_q_pu, 0, self.min_q_pu])
        else:
            raise ValueError(
                f"QVArea4130 is defined for variants 1, 2, and 3 but not for {self.variant}.")

    def q_flexibility(self, p_pu, vm_pu):
        return np.c_[np.interp(vm_pu, self.min_vm_points_pu, self.min_q_points_pu),
                     np.interp(vm_pu, self.max_vm_points_pu, self.max_q_points_pu)]


class PQVArea4130Base(BasePQVArea):
    """ This is the base class for the three variants of VDE AR-N-4130. This class is not for direct
    application by the user.
    """
    def __init__(self, min_q_pu, max_q_pu, variant, vn_kv=380, raise_merge_overlap=True):
        super().__init__(raise_merge_overlap=raise_merge_overlap)
        self.pq_area = PQArea4130(min_q_pu, max_q_pu)
        self.qv_area = QVArea4130(min_q_pu, max_q_pu, vn_kv=vn_kv, variant=variant)


class PQVArea4130V1(PQVArea4130Base):
    """ This class models the PQV area of flexible Q for extra high voltage power plants according
    to variant 1 of VDE AR-N-4130.
    """
    def __init__(self, vn_kv=380, raise_merge_overlap=True):
        super().__init__(-0.227902, 0.484322, 1, vn_kv=vn_kv, raise_merge_overlap=raise_merge_overlap)

class PQVArea4130V2(PQVArea4130Base):
    """ This class models the PQV area of flexible Q for extra high voltage power plants according
    to variant 2 of VDE AR-N-4130.
    """
    def __init__(self, vn_kv=380, raise_merge_overlap=True):
        super().__init__(-0.328684, 0.410775, 2, vn_kv=vn_kv, raise_merge_overlap=raise_merge_overlap)


class PQVArea4130V3(PQVArea4130Base):
    """ This class models the PQV area of flexible Q for extra high voltage power plants according
    to variant 3 of VDE AR-N-4130.
    """
    def __init__(self, vn_kv=380, raise_merge_overlap=True):
        super().__init__(-0.410775, 0.328684, 3, vn_kv=vn_kv, raise_merge_overlap=raise_merge_overlap)


""" MV DERs: """


class PQArea4110(PQAreaPOLYGON):
    def __init__(self):
        p_points_pu = (-1e-7,  0.05, 0.05       , 1.       , 1.      , 0.05      , 0.05, 1e-7, -1e-7)
        q_points_pu = (-1e-7, -1e-7, -0.01961505, -0.484322, 0.484322, 0.01961505, 0.  , 1e-7, -1e-7)
        super().__init__(p_points_pu, q_points_pu)


class QVArea4110(QVAreaPOLYGON):
    """
    This class models the QV area of flexible Q for medium-voltage plants according to
    VDE AR-N-4110.
    """
    def __init__(self):
        q_points_pu =  (0. , -0.484322, -0.484322, 0. , 0.484322, 0.484322, 0. )
        vm_points_pu = (0.9, 0.95     , 1.1      , 1.1, 1.05    , 0.9     , 0.9)
        super().__init__(q_points_pu, vm_points_pu)


class PQVArea4110(BasePQVArea):
    def __init__(self, raise_merge_overlap=True):
        super().__init__(raise_merge_overlap=raise_merge_overlap)
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
            # commented due to the note of the docstring
            # p_points_pu = (0., 0.95, 0.95, 0.)
            # q_points_pu = (0., -0.312, 0.312, 0.)
            p_points_pu = (0., 1, 1, 0.)
            q_points_pu = (0., -0.328684, 0.328684, 0.)
        elif variant == 2:
            # commented due to the note of the docstring
            # p_points_pu = (0., 0.9, 0.9, 0.)
            # q_points_pu = (0., -0.36, 0.36, 0.)
            p_points_pu = (0., 1, 1, 0.)
            q_points_pu = (0., -0.484322, 0.484322, 0.)
        else:
            raise ValueError(f"{variant=}")
        super().__init__(p_points_pu, q_points_pu)


class QVArea4105(QVAreaPOLYGON):
    """ This class models the QV area of flexible Q for low-voltage plants according to
    VDE AR-N-4105. Plants with S_E,max <= 4.6 kVA should apply variant 1 while S_E,max > 4.6 kVA
    should apply variant 2.
    """
    def __init__(self, variant):
        if variant == 1:
            q_points_pu =  (0. , -0.328684, -0.328684, 0. , 0.328684, 0.328684, 0. )
            vm_points_pu = (0.9, 0.95     , 1.1      , 1.1, 1.05    , 0.9     , 0.9)
        elif variant == 2:
            q_points_pu =  (0. , -0.484322, -0.484322, 0. , 0.484322, 0.484322, 0. )
            vm_points_pu = (0.9, 0.95     , 1.1      , 1.1, 1.05    , 0.9     , 0.9)
        else:
            raise ValueError(f"{variant=}")
        super().__init__(q_points_pu, vm_points_pu)


class PQVArea4105(BasePQVArea):
    """ This class models the PQV area of flexible Q for low-voltage plants according to
    VDE AR-N-4105. Plants with S_E,max <= 4.6 kVA should apply variant 1 while S_E,max > 4.6 kVA
    should apply variant 2.
    """
    def __init__(self, variant, raise_merge_overlap=True):
        super().__init__(raise_merge_overlap=raise_merge_overlap)
        self.pq_area = PQArea4105(variant)
        self.qv_area = QVArea4105(variant)


if __name__ == "__main__":
    pass

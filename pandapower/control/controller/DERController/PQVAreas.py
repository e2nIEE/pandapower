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
""" All defined PQV areas of reactive power provision capabilities including those defined in
VDE-AR-N technical connection rule standards"""
# -------------------------------------------------------------------------------------------------


class BasePQVArea(BaseModel):
    def pq_in_range(self, p, q, vm_pu):
        pass

    def q_flexibility(self, p, vm_pu):
        pass

    def __str__(self):
        return self.__class__.name


class PQAreaPOLYGON(BasePQVArea):
    """
    Provides a polygonal area of feasible reactive power provision. The polygonal area can be
    defined by 'p_points' and 'q_points'.

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

    def pq_in_range(self, p, q, vm_pu=None):
        return self.polygon.contains(Point(p, q))

    def q_flexibility(self, p, vm_pu=None):
        assert p >= 0 and p <= 1
        line = LineString([(p, -1), (p, 1)])
        if line.intersects(self.polygon):
            return [point[1] for point in LineString([(p, -1),
                    (p, 1)]).intersection(self.polygon).coords]
        else:
            return (0, 0)


""" STATCOM DERs: """


class PQAreaSTATCOM(BasePQVArea):
    """
    TODO: Description
    """

    def __init__(self, min_q=-0.328684, max_q=0.410775):
        self.min_q, self.max_q = min_q, max_q

    def pq_in_range(self, p, q, vm_pu=None):
        if self.min_q <= q <= self.max_q:
            return True
        else:
            return False

    def q_flexibility(self, p, vm_pu=None):
        return (self.min_q, self.max_q)


class PQVArea_STATCOM(BasePQVArea):
    """
    TODO: Description
    """

    def __init__(self, min_q=-0.328684, max_q=0.328684):
        self.min_q, self.max_q = min_q, max_q
        self.pq_area = PQAreaSTATCOM(min_q=min_q, max_q=max_q)
        self.qu_area = QVArea4110UserDefined(min_q=min_q, max_q=max_q)
        # self.linear_factor = -(max_q - min_q) / 0.04

    def pq_in_range(self, p, q, vm_pu):
        if self.pq_area.pq_in_range(p, q, vm_pu) and self.qu_area.pq_in_range(p, q, vm_pu):
            return True
        else:
            return False

    def q_flexibility(self, p, vm_pu=None):
        min_q, max_q = self.pq_area.q_flexibility(p=p, vm_pu=vm_pu)
        q_range_qu_area = self.qu_area.q_flexibility(p=p, vm_pu=vm_pu)

        if max_q < q_range_qu_area[0] or min_q > q_range_qu_area[1]:
            # when q is not available due to pq_area then no q
            return min_q, max_q
        else:
            return (np.clip(min_q, *q_range_qu_area), np.clip(max_q, *q_range_qu_area))


""" HV DERs: """


class PQArea4120(BasePQVArea):
    """
    This class models the PQ area of flexible Q for high-voltage plants according to VDE AR-N-4120.
    It is used to be combined with Voltage dependencies in PQVArea4120V1, PQVArea4120V2 and
    PQVArea4120V3
    """

    def __init__(self, min_q=-0.328684, max_q=0.410775, version=2018):
        supported_versions = [2015, 2018]
        p_points = {2015: [0.1, 0.2], 2018: [0.05, 0.2]}
        if version not in supported_versions:
            raise ValueError(
                f"version '{version}' is not supported, it is not in {supported_versions}")
        self.version = version
        self.p_points = p_points[version]
        self.min_q, self.max_q = min_q, max_q

        self.linear_factor_ind = (self.min_q + self.p_points[0]) / self.p_points[0]
        self.linear_factor_cap = (self.max_q - self.p_points[0]) / self.p_points[0]

    def pq_in_range(self, p, q, vm_pu=None):
        if p < self.p_points[0]:
            return False
        elif p > self.p_points[1]:
            if q < self.min_q or q > self.max_q:
                return False
        else:
            if q < -self.p_points[0]-(p-self.p_points[0])*self.linear_factor_ind or \
                    q > self.p_points[0]+(p-self.p_points[0])*self.linear_factor_cap:
                return False
        return True

    def q_flexibility(self, p, vm_pu=None):
        if p < self.p_points[0]:
            return (0, 0)
        elif p > self.p_points[1]:
            return (self.min_q, self.max_q)
        else:
            return (-self.p_points[0]+(p-self.p_points[0])*self.linear_factor_ind,
                    self.p_points[0]+(p-self.p_points[0])*self.linear_factor_cap)


class QVArea4120(BasePQVArea):
    """
    This class models the PV area of flexible Q for high-voltage plants according to VDE AR-N-4120.
    It is used to be combined with active power dependencies in PQVArea4120V1, PQVArea4120V2 and
    PQVArea4120V3
    """

    def __init__(self, min_q, max_q):
        self.min_q, self.max_q = min_q, max_q

        self.max_u = 127.0/110
        self.min_u = 96.0/110
        self.delta_u = 7.0/110
        self.linear_factor = (self.max_q - self.min_q) / self.delta_u

    def pq_in_range(self, p, q, vm_pu):
        q_flex_min, q_flex_max = self.q_flexibility(p, vm_pu)
        if q < q_flex_min or q > q_flex_max:
            return False
        else:
            return True

    def q_flexibility(self, p, vm_pu):
        if vm_pu < self.min_u:
            q_range_qu_area = (self.max_q, self.max_q)
        elif self.min_u < vm_pu <= self.min_u + self.delta_u:
            q_range_qu_area = (self.max_q - self.linear_factor * (vm_pu-self.min_u), self.max_q)
        elif self.min_u+self.delta_u < vm_pu <= self.max_u-self.delta_u:
            q_range_qu_area = (self.min_q, self.max_q)
        elif self.max_u-self.delta_u < vm_pu <= self.max_u:
            q_range_qu_area = (self.min_q, self.min_q + self.linear_factor * (self.max_u-vm_pu))
        else:
            assert vm_pu > self.max_u
            q_range_qu_area = (self.min_q, self.min_q)

        return q_range_qu_area


class PQVArea4120V1(BasePQVArea):
    """
    This class models the PQV area of flexible Q for high-voltage plants according to variant 1 of
    VDE AR-N-4120.
    """

    def __init__(self, min_q=-0.227902, max_q=0.484322, version=2018):
        self.min_q, self.max_q, self.version = min_q, max_q, version
        self.pq_area = PQArea4120(self.min_q, self.max_q, version=self.version)
        self.qu_area = QVArea4120(self.min_q, self.max_q)

    def pq_in_range(self, p, q, vm_pu):
        # Check PQ Area state (Dach-Kurve)
        if self.pq_area.pq_in_range(p, q, vm_pu) and self.qu_area.pq_in_range(p, q, vm_pu):
            return True
        else:
            return False

    def q_flexibility(self, p, vm_pu=None):
        min_q, max_q = self.pq_area.q_flexibility(p=p, vm_pu=vm_pu)
        q_range_qu_area = self.qu_area.q_flexibility(p=p, vm_pu=vm_pu)

        if max_q < q_range_qu_area[0] or min_q > q_range_qu_area[1]:
            # when q is not available due to pq_area then no q
            return min_q, max_q
        else:
            return (np.clip(min_q, *q_range_qu_area), np.clip(max_q, *q_range_qu_area))


class PQVArea4120V2(PQVArea4120V1):
    """
    This class models the PQV area of flexible Q for high-voltage plants according to variant 2 of
    VDE AR-N-4120.
    """

    def __init__(self, min_q=-0.328684, max_q=0.410775, version=2018):
        super().__init__(min_q, max_q, version=version)


class PQVArea4120V3(PQVArea4120V1):
    """
    This class models the PQV area of flexible Q for high-voltage plants according to variant 3 of
    VDE AR-N-4120.
    """

    def __init__(self, min_q=-0.410775, max_q=0.328684, version=2018):
        super().__init__(min_q, max_q, version=version)


""" EHV DERs: """

class PQArea4130(PQArea4120):
    pass


class QVArea4130(QVArea4120):

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

    def q_flexibility(self, p, vm_pu):
        return (np.interp(vm_pu, self.vm_min_points, self.q_min_points),
                np.interp(vm_pu, self.vm_max_points, self.q_max_points))

class PQVArea4130V1(PQVArea4120V1):

    def __init__(self, min_q=-0.227902, max_q=0.484322, version=2018, vn=380):
        self.min_q, self.max_q, self.version, self.vn = min_q, max_q, version, vn
        self.pq_area = PQArea4130(self.min_q, self.max_q, version=self.version)
        self.qu_area = QVArea4130(self.min_q, self.max_q, vn=self.vn, variant=1)

class PQVArea4130V2(PQVArea4130V1):

    def __init__(self, min_q=-0.328684, max_q=0.410775, version=2018, vn=380):
        super().__init__(min_q, max_q, version=version, vn=vn)
        self.qu_area = QVArea4130(self.min_q, self.max_q, vn=self.vn, variant=2)


class PQVArea4130V3(PQVArea4130V1):

    def __init__(self, min_q=-0.410775, max_q=0.328684, version=2018, vn=380):
        super().__init__(min_q, max_q, version=version, vn=vn)
        self.qu_area = QVArea4130(self.min_q, self.max_q, vn=self.vn, variant=3)


""" MV DERs: """


class QVArea4110(QVArea4120):
    """
    This class models the QV area of flexible Q for medium-voltage plants according to
    VDE AR-N-4110.
    """

    def __init__(self, min_q=-0.328684, max_q=0.328684):
        self.min_q, self.max_q = min_q, max_q
        self.min_u, self.max_u = 0.9, 1.1
        self.delta_u = 0.025
        self.linear_factor_ind = (0 - self.min_q) / self.delta_u
        self.linear_factor_cap = (self.max_q - 0) / self.delta_u

    def q_flexibility(self, p, vm_pu):
        if vm_pu < self.min_u:
            q_range_qu_area = (0, self.max_q)
        elif self.min_u < vm_pu <= self.min_u + self.delta_u:
            q_range_qu_area = (0 - self.linear_factor_ind * (vm_pu-self.min_u),
                               self.max_q)
        elif self.min_u+self.delta_u < vm_pu <= self.max_u-self.delta_u:
            q_range_qu_area = (self.min_q, self.max_q)
        elif self.max_u-self.delta_u < vm_pu <= self.max_u:
            q_range_qu_area = (self.min_q, 0 + self.linear_factor_cap * (self.max_u-vm_pu))
        else:
            assert vm_pu > self.max_u
            q_range_qu_area = (self.min_q, 0)

        return q_range_qu_area


class QVArea4110UserDefined(BasePQVArea):
    """
    TODO: Description (what difference to QVArea4110 and for what reason/use case?)
    """

    def __init__(self, min_q, max_q):
        self.min_q, self.max_q = min_q, max_q
        self.min_u, self.max_u = 0.92, 1.08
        self.delta_u = 0.04
        self.linear_factor = - (self.max_q - self.min_q) / self.delta_u

    def pq_in_range(self, p, q, vm_pu):
        q_flex_min, q_flex_max = self.q_flexibility(p, vm_pu)
        if q < q_flex_min or q > q_flex_max:
            return False
        else:
            return True

    def q_flexibility(self, p, vm_pu):
        if vm_pu < self.min_u:
            q_range_qu_area = (self.max_q-0.002, self.max_q+0.002)
        elif self.min_u < vm_pu <= self.min_u + self.delta_u:
            q_range_qu_area = (self.max_q + self.linear_factor * (vm_pu-self.min_u), self.max_q)
        elif self.min_u+self.delta_u < vm_pu <= self.max_u-self.delta_u:
            q_range_qu_area = (self.min_q, self.max_q)
        elif self.max_u-self.delta_u < vm_pu <= self.max_u:
            q_range_qu_area = (self.min_q, self.min_q - self.linear_factor * (self.max_u-vm_pu))
        else:
            assert vm_pu > self.max_u
            q_range_qu_area = (self.min_q-0.002, self.min_q+0.002)

        return q_range_qu_area


class PQVArea4110User(BasePQVArea):
    """
    TODO: Description (what difference to QVArea4110 and for what reason/use case?)
    """

    def __init__(self):
        self.pq_area = PQArea4120(min_q=-0.328684, max_q=0.328684)
        self.qu_area = QVArea4110UserDefined(min_q=-0.328684, max_q=0.328684)

    def pq_in_range(self, p, q, vm_pu):
        if self.pq_area.pq_in_range(p, q) and self.qu_area.pq_in_range(p, q, vm_pu):
            return True
        else:
            return False

    def q_flexibility(self, p, vm_pu):
        min_q, max_q = self.pq_area.q_flexibility(p=p, vm_pu=vm_pu)
        q_range_qu_area = self.qu_area.q_flexibility(p=p, vm_pu=vm_pu)

        if max_q < q_range_qu_area[0] or min_q > q_range_qu_area[1]:
            # when q is not available due to pq_area then no q
            return min_q, max_q
        else:
            return (np.clip(min_q, *q_range_qu_area), np.clip(max_q, *q_range_qu_area))


""" LV DERs: """


class PQArea4105(BasePQVArea):
    """
    This class models the PQ area of flexible Q for low-voltage plants according to VDE AR-N-4105.
    """

    def __init__(self, min_q=-0.328684, max_q=0.328684):
        self.min_q, self.max_q = min_q, max_q
        self.linear_factor_ind = self.min_q / 0.9
        self.linear_factor_cap = self.max_q / 0.9

    def pq_in_range(self, p, q, vm_pu=None):
        if p < 0.1:
            return False
        elif q < (p-0.1)*self.linear_factor_ind or q > (p-0.1)*self.linear_factor_cap:
            return False
        else:
            return True

    def q_flexibility(self, p, vm_pu=None):
        if p < 0.1:
            return (0, 0)
        else:
            return ((p-0.1)*self.linear_factor_ind, (p-0.1)*self.linear_factor_cap)


class QVAreaLV(QVArea4120):
    """
    TODO: Description
    """

    def __init__(self, min_cosphi, max_cosphi):
        self.min_cosphi, self.max_cosphi = min_cosphi, max_cosphi
        self.min_u, self.max_u = 0.9, 1.1
        self.delta_u = 0.05
        self.linear_factor_ind = (1 - self.min_cosphi) / self.delta_u
        self.linear_factor_cap = (1 - self.max_cosphi) / self.delta_u

    def q_flexibility(self, p, vm_pu):
        if vm_pu < self.min_u:
            cosphi_range_qu_area = (1, self.max_cosphi)
        elif self.min_u < vm_pu <= self.min_u + self.delta_u:
            cosphi_range_qu_area = (1 - self.linear_factor_ind * (vm_pu-self.min_u),
                                    self.max_cosphi)
        elif self.min_u+self.delta_u < vm_pu <= self.max_u-self.delta_u:
            cosphi_range_qu_area = (self.min_cosphi, self.max_cosphi)
        elif self.max_u-self.delta_u < vm_pu <= self.max_u:
            cosphi_range_qu_area = (self.min_cosphi,
                                    1 - self.linear_factor_cap * (self.max_u-vm_pu))
        else:
            assert vm_pu > self.max_u
            cosphi_range_qu_area = (self.min_cosphi, 1)

        q_range_qu_area = (-1 * np.tan(np.arccos(cosphi_range_qu_area[0])),
                           1 * np.tan(np.arccos(cosphi_range_qu_area[1])))
        return q_range_qu_area


class PQUArea_PFcapability(BasePQVArea):
    """
    TODO: Description
    """

    def __init__(self, min_q=-0.328684, max_q=0.328684):
        self.min_q, self.max_q = min_q, max_q
        self.pq_area = PQArea4105(min_q=min_q, max_q=max_q)
        # self.qu_area = QVArea4110(min_q=min_q, max_q=max_q)
        self.qu_area = QVAreaLV(min_cosphi=0.95, max_cosphi=0.95)

    def pq_in_range(self, p, q, vm_pu):
        if self.pq_area.pq_in_range(p, q, vm_pu) and\
                self.qu_area.pq_in_range(p, q, vm_pu):
            return True
        else:
            return False

    def q_flexibility(self, p, vm_pu=None):
        min_q, max_q = self.pq_area.q_flexibility(p=p, vm_pu=vm_pu)
        q_range_qu_area = self.qu_area.q_flexibility(p=p, vm_pu=vm_pu)

        if max_q < q_range_qu_area[0] or min_q > q_range_qu_area[1]:
            # when q is not available due to pq_area then no q
            return min_q, max_q
        else:
            return (np.clip(min_q, *q_range_qu_area), np.clip(max_q, *q_range_qu_area))


# TODO Abdullah: Add LV DERs


if __name__ == "__main__":
    pass

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from builtins import zip
from builtins import object
import numpy as np

from numpy import interp
from scipy.interpolate import interp1d, PchipInterpolator
from pandapower.io_utils import JSONSerializableClass

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)


class Characteristic(JSONSerializableClass):
    """
    This class represents a characteristics curve. The curve is described as a piecewise linear function.

    INPUT:
        **pts** - Expects two (or more) points of the function (i.e. kneepoints)

    OPTIONAL:
        **eps** - An epsilon to compare the difference to

    The class has an implementation of the ``__call__`` method, which allows using it interchangeably with other interpolator objects,
    e.g. ``scipy.interpolate.interp1d``, ``scipy.interpolate.CubicSpline``, ``scipy.interpolate.PPoly``, etc.

    Example usage:
        Create a simple function from two points and ask for the target y-value for a
        given x-value.
        Assume a characteristics curve in which for voltages < 0.95pu a power of 10kW
        is desired, linear rising to a max. of 20 kW at 1.05 pu

    ::

        You can give points by lists of x/y-values
        >>> c = Characteristic(net, x_values=[0.95, 1.05],y_values=[10, 20])
        >>> c(x=1.0)
        15.0

        or pass a list of points (x,y)
        >>> c = Characteristic.from_points(net,points=[(0.95, 10), (1.05, 20)])
        >>> c(x=1.0)
        15.0

        or in a simple case from a gradient, its zero crossing and the maximal values for y
        >>> c = Characteristic.from_gradient(net,zero_crossing=-85,gradient=100,y_min=10,y_max=20)
        >>> c(x=1.0)
        15.0

        Values are constant beyond the first and last defined points
        >>> c(x=42)
        20.0
        >>> c(x=-42)
        10.0

        Create a curve with many points and ask for the difference between the y-value being measured
        and the expected y-value for a given x-value
        >>> c = Characteristic.from_points(net,points=[(1,2),(2,4),(3,2),(42,24)])
        >>> c.diff(x=2.5, measured=3)
        0.0

        You can also ask if a y-values satisfies the curve at a certain x-value. Note how the use of
        an epsilon behaves (for x=2.5 we expect 3.0):
        >>> c.satisfies(x=2.5, measured=3.099999999, epsilon=0.1)
        True
        >>> c.satisfies(x=2.5, measured=3.1, epsilon=0.1)
        False
    """
    def __init__(self, net, x_values, y_values, **kwargs):
        super().__init__()
        self.x_vals = x_values
        self.y_vals = y_values
        self.index = super().add_to_net(net, "characteristic")

    # @property
    # def x_vals(self):
    #     return self._x_vals
    #
    # @property
    # def y_vals(self):
    #     return self._y_vals

    @classmethod
    def from_points(cls, net, points, **kwargs):
        unzipped = list(zip(*points))
        return cls(net, unzipped[0], unzipped[1], **kwargs)

    @classmethod
    def from_gradient(cls, net, zero_crossing, gradient, y_min, y_max, **kwargs):
        x_left = (y_min - zero_crossing) / float(gradient)
        x_right = (y_max - zero_crossing) / float(gradient)
        return cls(net, [x_left, x_right], [y_min, y_max], **kwargs)

    def diff(self, x, measured):
        """

        INPUT:
            **x** - The x-value at which the current y-value is measured
            **actual** - The actual y-value being measured.
            **return** - The difference between actual and expected value.
        """
        return measured - self(x)

    def satisfies(self, x, measured, epsilon):
        """

        INPUT:
            **x** - The x-value at which the current y-value is measured

            **measured** - The actual y-value being measured.

        OUTPUT:
            Whether or not the point satisfies the characteristics curve with respect to the
            epsilon being set
        """
        if abs(self.diff(x, measured)) < epsilon:
            return True
        else:
            return False

    def __call__(self, x):
        """

        INPUT:
            **x** - An x-value

        OUTPUT:
            The corresponding target value of this characteristics
        """
        return interp(x, self.x_vals, self.y_vals)

    def __repr__(self):
        return self.__class__.__name__


class SplineCharacteristic(Characteristic):
    """
    SplineCharacteristic interpolates the y-value(s) for the given x-value(s) according to a non-linear function.
    Internally the interpolator object interp1d from scipy.interpolate is used.
    By default, the function is quadratic, but the user can specify other methods (refer to the documentation of
    interp1d). The fill_value can be specified as "extrapolate" so that even x-values outside of the specified
    range can be used and yield y-values outside the specified y range. Alternatively, the behavior of
    Characteristic can be followed by providing a tuple for the fill value for x outside the specified range,
    refer to the documentation of interp1d for more details. We set the parameter bounds_error to False.

    INPUT:
        **net**

        **x_values**

        **y_values**

        **fill_value**
    """
    json_excludes = ["self", "__class__", "_interpolator"]

    def __init__(self, net, x_values, y_values, interpolator_kind="interp1d", **kwargs):
        super().__init__(net, x_values=x_values, y_values=y_values)
        self.kwargs = kwargs
        self.interpolator_kind = interpolator_kind

    @property
    def interpolator(self):
        """
        We need to store the interpolator in a property because we need to serialize
        the characteristic. Instead of storing the serialized interpolator, we store the
        x_values and y_values (the attribute _interpolator is ecluded from serialization by
        adding it to json_excludes). For it to work, we need to recreate the interpolator on
        demand. As soon as the characteristic is called, if the interpolator is there,
        we can use it. If not, we recreate it.
        """
        return self._interpolator

    @interpolator.getter
    def interpolator(self):
        if not hasattr(self, '_interpolator'):
            if self.interpolator_kind == "interp1d":
                self._interpolator = default_interp1d(self.x_vals, self.y_vals, **self.kwargs)
            elif self.interpolator_kind == "Pchip":
                self._interpolator = PchipInterpolator(self.x_vals, self.y_vals, **self.kwargs)
            else:
                raise NotImplementedError(f"Interpolator {self.interpolator_kind} not implemented!")
        return self._interpolator

    def __call__(self, x):
        """
        This method allows calling the SciPy interpolator object directly.
        Codacy is complaining about this, but it is not a problem.

        INPUT:
            **x** (float) -  The x-value at which the current y-value is interpolated for.

        OUTPUT:
            The interpolated y-value.
        """
        return self.interpolator(x)


class LogSplineCharacteristic(SplineCharacteristic):

    def __init__(self, net, x_values, y_values, **kwargs):
        super().__init__(net, x_values, y_values, **kwargs)

    @property
    def x_vals(self):
        return self._x_vals

    @property
    def y_vals(self):
        return self._y_vals

    @x_vals.setter
    def x_vals(self, x_values):
        if np.any(x_values == 0):
            logger.warning("zero-values not supported in x_values")
        self._x_vals = np.log10(x_values)

    @y_vals.setter
    def y_vals(self, y_values):
        if np.any(y_values == 0):
            logger.warning("zero-values not supported in y_values")
        self._y_vals = np.log10(y_values)

    def __call__(self, x):
        return np.power(10, self.interpolator(np.log10(x)))


def default_interp1d(x, y, kind="quadratic", bounds_error=False, fill_value="extrapolate", **kwargs):
    return interp1d(x, y, kind=kind, bounds_error=bounds_error, fill_value=fill_value, **kwargs)

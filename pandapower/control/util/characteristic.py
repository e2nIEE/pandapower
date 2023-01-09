# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from builtins import zip
from builtins import object

from numpy import interp
from scipy.interpolate import interp1d
from pandapower.io_utils import JSONSerializableClass


class Characteristic(JSONSerializableClass):
    """
    This class represents a characteristics curve. The curve is described as a
    piecewise linear function.

    |   **pts** - Expects two (or more) points of the function (i.e. kneepoints)
    |   **eps** - Optional: An epsilon to compare the difference to.

    The class has an implementation of the __call__ method, which allows using it interchangeably with other interpolator objects,
    e.g. scipy.interpolate.interp1d, scipy.interpolate.CubicSpline, scipy.interpolate.PPoly, etc.

    Example usage:

    Create a simple function from two points and ask for the target y-value for a
    given x-value.
    Assume a characteristics curve in which for voltages < 0.95pu a power of 10kW
    is desired, linear rising to a max. of 20kW at 1.05pu

    # You can give points by lists of x/y-values
    >>> c = Characteristic(net, x_values=[0.95, 1.05],y_values=[10, 20])
    >>> c.target(x=1.0)
    15.0

    # or pass a list of points (x,y)
    >>> c = Characteristic.from_points(net,points=[(0.95, 10), (1.05, 20)])
    >>> c.target(x=1.0)
    15.0

    # or in a simple case from a gradient, its zero crossing and the maximal values for y
    >>> c = Characteristic.from_gradient(net,zero_crossing=-85,gradient=100,y_min=10,y_max=20)
    >>> c.target(x=1.0)
    15.0

    # Values are constant beyond the first and last defined points
    >>> c.target(x=42)
    20.0
    >>> c.target(x=-42)
    10.0

    # Create a curve with many points and ask for the difference between the y-value being measured
    and the expected y-value for a given x-value
    >>> c = Characteristic.from_points(net,points=[(1,2),(2,4),(3,2),(42,24)])
    >>> c.diff(x=2.5, measured=3)
    0.0

    # You can also ask if a y-values satisfies the curve at a certain x-value. Note how the use of
    an epsilon behaves (for x=2.5 we expect 3.0):
    >>> c.satisfies(x=2.5, measured=3.099999999, epsilon=0.1)
    True
    >>> c.satisfies(x=2.5, measured=3.1, epsilon=0.1)
    False
    """

    def __init__(self, net, x_values, y_values):
        super().__init__()
        self.x_vals = x_values
        self.y_vals = y_values
        self.index = super().add_to_net(net, "characteristic")

    @classmethod
    def from_points(cls, net, points):
        unzipped = list(zip(*points))
        return cls(net, unzipped[0], unzipped[1])

    @classmethod
    def from_gradient(cls, net, zero_crossing, gradient, y_min, y_max):
        x_left = (y_min - zero_crossing) / float(gradient)
        x_right = (y_max - zero_crossing) / float(gradient)
        return cls(net, [x_left, x_right], [y_min, y_max])

    def diff(self, x, measured):
        """
        :param x: The x-value at which the current y-value is measured
        :param actual: The actual y-value being measured.
        :return: The difference between actual and expected value.
        """
        return measured - interp(x, self.x_vals, self.y_vals)

    def target(self, x):
        """
        Note: Deprecated. Use the __call__ interface instead.
        :param x: An x-value
        :return: The corresponding target value of this characteristics
        """
        # return interp(x, self.x_vals, self.y_vals)
        raise DeprecationWarning("target method is deprecated. Use the __call__ interface instead.")

    def satisfies(self, x, measured, epsilon):
        """

        :param x: The x-value at which the current y-value is measured
        :param measured: The actual y-value being measured.
        :return: Whether or not the point satisfies the characteristics curve with respect to the
        epsilon being set
        """
        if abs(self.diff(x, measured)) < epsilon:
            return True
        else:
            return False

    def __call__(self, x):
        """
        :param x: An x-value
        :return: The corresponding target value of this characteristics
        """
        return interp(x, self.x_vals, self.y_vals)


    def __repr__(self):
        return self.__class__.__name__


class SplineCharacteristic(Characteristic):
    json_excludes = ["self", "__class__", "_interpolator"]
    def __init__(self, net, x_values, y_values):
        super().__init__(net, x_values=x_values, y_values=y_values)

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
            self._interpolator = interp1d(self.x_vals, self.y_vals, kind='quadratic', fill_value='extrapolate')
        return self._interpolator

    def __call__(self, x):
        """
        This method allows calling the SciPy interpolator object directly.
        Codacy is complaining about this, but it is not a problem.
        Parameters
        ----------
        x : float
            The x-value at which the current y-value is interpolated for.

        Returns
        -------
        float
            The interpolated y-value.
        """
        return self.interpolator(x)


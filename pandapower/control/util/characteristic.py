
from builtins import zip
from builtins import object

from numpy import interp
from pandapower.io_utils import JSONSerializableClass

__author__ = 'jdollichon'


# TODO option eine Exception zu raisen wenn Wert ausserhalb des def. Intervalls gefragt wird.
class Characteristic(JSONSerializableClass):
    """
    This class represents a characteristics curve. The curve is described as a
    piecewise linear function.

    |   **pts** - Expects two (or more) points of the function (i.e. kneepoints)
    |   **eps** - Optional: An epsilon to compare the difference to.

    Example usage:

    Create a simple function from two points and ask for the target y-value for a
    given x-value.
    Assume a characteristics curve in which for voltages < 0.95pu a power of 10kW
    is desired, linear rising to a max. of 20kW at 1.05pu

    # You can give points by lists of x/y-values
    >>> c = Characteristic(x_values=[0.95, 1.05], y_values=[10, 20])
    >>> c.target(x=1.0)
    15.0

    # or pass a list of points (x,y)
    >>> c = Characteristic.from_points(points=[(0.95, 10), (1.05, 20)])
    >>> c.target(x=1.0)
    15.0

    # or in a simple case from a gradient, its zero crossing and the maximal values for y
    >>> c = Characteristic.from_gradient(zero_crossing=-85, gradient=100, y_min=10, y_max=20)
    >>> c.target(x=1.0)
    15.0

    # Values are constant beyond the first and last defined points
    >>> c.target(x=42)
    20.0
    >>> c.target(x=-42)
    10.0

    # Create a curve with many points and ask for the difference between the y-value being measured
    and the expected y-value for a given x-value
    >>> c = Characteristic.from_points(points=[(1,2),(2,4),(3,2),(42,24)])
    >>> c.diff(x=2.5, measured=3)
    0.0

    # You can also ask if a y-values satisfies the curve at a certain x-value. Note how the use of
    an epsilon behaves (for x=2.5 we expect 3.0):
    >>> c.satisfies(x=2.5, measured=3.099999999, epsilon=0.1)
    True
    >>> c.satisfies(x=2.5, measured=3.1, epsilon=0.1)
    False

    """

    def __init__(self, x_values, y_values):
        super().__init__()
        self.x_vals = x_values
        self.y_vals = y_values


    @classmethod
    def from_points(self, points):
        unzipped = list(zip(*points))
        return Characteristic(unzipped[0], unzipped[1])

    @classmethod
    def from_gradient(self, zero_crossing, gradient, y_min, y_max):
        x_left = (y_min - zero_crossing) / float(gradient)
        x_right = (y_max - zero_crossing) / float(gradient)
        return Characteristic([x_left, x_right], [y_min, y_max])

    def diff(self, x, measured):
        """
        :param x: The x-value at which the current y-value is measured
        :param actual: The actual y-value being measured.
        :return: The difference between actual and expected value.
        """
        return measured-interp(x, self.x_vals, self.y_vals)

    def target(self, x):
        """
        :param x: An x-value
        :return: The corresponding target value of this characteristics
        """
        return interp(x, self.x_vals, self.y_vals)

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

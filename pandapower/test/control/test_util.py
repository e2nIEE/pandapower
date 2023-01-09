# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import numpy as np
import pytest
import logging

import pandapower as pp
from pandapower.control import Characteristic, SplineCharacteristic, trafo_characteristics_diagnostic


def test_characteristic():
    net = pp.create_empty_network()
    x_points = [0, 1, 2]
    y_points = [3, 4, 5]
    c = Characteristic(net, x_points, y_points)

    assert np.array_equal(c(x_points), y_points)
    # bounds are fixed:
    assert c(-1) == 3
    assert c(3) == 5

    assert c.satisfies(0.5, 3.5, 1e-9)
    assert not c.satisfies(0.5, 4, 1e-9)

    assert c.diff(1, 5) == 1

    # testing alternative constructors:
    c1 = Characteristic.from_points(net, ((0, 3), (1, 4), (2, 5)))
    assert np.array_equal(c1(x_points), y_points)

    c2 = Characteristic.from_gradient(net, 3, 1, 3, 5)
    assert np.array_equal(c2(x_points), y_points)

    x_spline = [0, 1, 2]
    y_spline = [0, 1, 4]
    c3 = SplineCharacteristic(net, x_points, y_spline)

    assert np.array_equal(c3(x_spline), y_spline)
    assert c3(1.5) == 2.25
    assert c3(3) == 9

    c4 = SplineCharacteristic(net, x_points, y_spline, fill_value=(y_spline[0], y_spline[-1]))
    assert c4(2) == 4


def test_false_alarm_trafos(simple_test_net):
    net = simple_test_net

    import io
    s = io.StringIO()
    h = logging.StreamHandler(stream=s)
    pp.control.util.diagnostic.logger.addHandler(h)

    ContinuousTapControl(net, 0, 1)
    ContinuousTapControl(net, 0, 1, trafotype='3W')

    if 'convergence problems' in s.getvalue():
        raise UserWarning('Control diagnostic raises false alarm! Controllers are fine, '
                          'but warning is raised: %s' % s.getvalue())

    trafo_characteristics_diagnostic(net)
    if 'convergence problems' in s.getvalue():
        raise UserWarning('Control diagnostic raises false alarm! Controllers are fine, '
                          'but warning is raised: %s' % s.getvalue())

    pp.control.util.diagnostic.logger.removeHandler(h)
    del h
    del s


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

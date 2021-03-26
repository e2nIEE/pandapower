import pytest
from pandapower.estimation import estimate
from pandapower.test.estimation.test_wls_estimation import create_net_with_bb_switch
import pandapower as pp


def test_none_net():
    with pytest.raises(UserWarning):
        estimate(None)


def test_nonexistant_alg():
    with pytest.raises(UserWarning):
        estimate(None, "superalg")


def test_wrong_fuse_setting():
    grid = create_net_with_bb_switch()
    with pytest.raises(UserWarning):
        estimate(grid, fuse_buses_with_bb_switch="no")


def test_wrong_init():
    grid = create_net_with_bb_switch()
    with pytest.raises(UserWarning):
        estimate(grid, init="no")


def test_wrong_zero_inj():
    grid = create_net_with_bb_switch()
    with pytest.raises(UserWarning):
        estimate(grid, zero_injection="no")


def test_no_observability():
    grid = create_net_with_bb_switch()
    grid.measurement.drop(grid.measurement.index[int(len(grid.measurement) * 0.1):], inplace=True)
    with pytest.raises(UserWarning):
        estimate(grid)

import pytest
from numpy import array

import pandapower.converter as cv
import pandapower.networks as nw


def test_convert_voltlvl_names():
    voltlvl_names = cv.convert_voltlvl_names([1, 2, "hv", 4, 5, "ehv", 7], str)
    assert voltlvl_names == ["EHV", "EHV-HV", "HV", "HV-MV", "MV", "EHV", "LV"]
    voltlvl_names = cv.convert_voltlvl_names([1, 2, "hv", 4, 5, "ehv", 7], int)
    assert voltlvl_names == [1, 2, 3, 4, 5, 1, 7]


def test_voltlvl_idx():
    net = nw.example_multivoltage()
    hv_and_mv_buses = list(range(16, 45))
    assert hv_and_mv_buses == cv.voltlvl_idx(net, "bus", 4)
    assert hv_and_mv_buses == cv.voltlvl_idx(net, "bus", ["HV-MV"])
    assert hv_and_mv_buses == cv.voltlvl_idx(net, "bus", [3, 5])
    assert hv_and_mv_buses == cv.voltlvl_idx(net, "bus", ["HV", "MV"])
    mv_loads = list(range(5, 13))
    assert mv_loads == cv.voltlvl_idx(net, "load", "MV")
    hvmv_trafo3ws = [0]
    assert hvmv_trafo3ws == cv.voltlvl_idx(net, "trafo3w", "HV", branch_bus="hv_bus")
    assert hvmv_trafo3ws == cv.voltlvl_idx(net, "trafo3w", "MV", branch_bus="mv_bus")
    assert hvmv_trafo3ws == cv.voltlvl_idx(net, "trafo3w", "MV", branch_bus="lv_bus")
    ehvhv_trafos = [0]
    assert ehvhv_trafos == cv.voltlvl_idx(net, "trafo", 1, branch_bus="hv_bus")
    assert ehvhv_trafos == cv.voltlvl_idx(net, "trafo", 3, branch_bus="lv_bus")
    ehvhv_and_hvmv_trafos = [0]
    assert ehvhv_and_hvmv_trafos == cv.voltlvl_idx(net, "trafo", 2, branch_bus="hv_bus")
    assert ehvhv_and_hvmv_trafos == cv.voltlvl_idx(net, "trafo", 4, branch_bus="lv_bus")
    hvmv_trafos = []
    assert hvmv_trafos == cv.voltlvl_idx(net, "trafo", 5, branch_bus="lv_bus")
    mvlv_trafos = [1]
    assert mvlv_trafos == cv.voltlvl_idx(net, "trafo", 5, branch_bus="hv_bus")
    lv_loads = list(range(13, 25))
    assert lv_loads == cv.voltlvl_idx(net, "load", 7)


def test_get_voltlvl():
    input1 = [146, 145, 144, 61, 60, 59, 2, 1, 0.8]
    input2 = 0.4
    assert all(cv.get_voltlvl(input1) == array([1, 3, 3, 3, 5, 5, 5, 7, 7]))
    assert cv.get_voltlvl(input2) == 7


if __name__ == "__main__":
    if 0:
        pytest.main(["test_voltLvl.py", "-xs"])
    else:
#        test_convert_voltlvl_names()
#        test_voltlvl_idx()
#        test_get_voltlvl()
        pass

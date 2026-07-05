# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Smoke test keeping the station-control benchmark harness functional."""

import pytest

from pandapower.test.control.benchmarks.bench_station_control import MODES, bench


@pytest.mark.parametrize("mode", MODES)
def test_bench_smoke(mode):
    result = bench(10, mode)
    assert result["converged"]
    assert result["runpp_calls"] >= 1
    assert result["wall_s"] > 0


if __name__ == '__main__':
    pytest.main(['-s', __file__])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from tempfile import gettempdir
from os.path import join
import pytest
from pandapower.plotting.plotly import simple_plotly
import pandapower.networks as nw
try:
    from plotly import __version__
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


@pytest.mark.slow
@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")
def test_simple_plotly_coordinates():
    net = nw.mv_oberrhein(include_substations=True)
    fig = simple_plotly(net, filename=join(gettempdir(), "temp-plot.html"), auto_open=False)
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + 2
                            # +1 for the infofunc traces, +2 = 1 bus trace + 1 ext_grid trace


@pytest.mark.slow
@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")
def test_simple_plotly_3w():
    # net with 3W-transformer
    net = nw.example_multivoltage()
    fig = simple_plotly(net, filename=join(gettempdir(), "temp-plot.html"), auto_open=False)
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + (len(net.trafo3w)*3 + 1)\
                            + 2  # +1 is for infofunc traces, +2 = 1 bus trace + 1 ext_grid trace


if __name__ == '__main__':
    pytest.main(['-s', __file__])

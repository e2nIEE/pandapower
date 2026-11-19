# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from os.path import join
from tempfile import gettempdir

import numpy as np
import pytest

import pandapower as pp
from pandapower.networks.create_examples import example_multivoltage
from pandapower.networks.mv_oberrhein import mv_oberrhein
from pandapower.plotting import create_weighted_marker_trace, pf_res_plotly
from pandapower.plotting.plotly import get_plotly_cmap, get_plotly_colorscale, simple_plotly

matplotlib = pytest.importorskip("matplotlib", reason="Matplotlib not installed")
plotly = pytest.importorskip("plotly", reason="plotly functions require the plotly package")


@pytest.mark.slow
def test_simple_plotly_coordinates():
    net = mv_oberrhein(include_substations=True)
    net.load.scaling, net.sgen.scaling = 1, 1
    # different markers and sizemodes as examples
    markers_load = create_weighted_marker_trace(
        net, elm_type="load", color="red", patch_type="triangle-up", sizemode="area", marker_scaling=100
    )
    markers_sgen = create_weighted_marker_trace(
        net,
        elm_type="sgen",
        color="green",
        patch_type="circle-open",
        sizemode="diameter",
        marker_scaling=100,
        scale_marker_size=0.5,
    )
    fig = simple_plotly(
        net,
        filename=join(gettempdir(), "temp-plot.html"),
        auto_open=False,
        additional_traces=[markers_sgen, markers_load],
    )
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + 6
    # +1 for the infofunc traces,
    # +6 = 1 bus trace + 1 ext_grid trace + 2 weighted marker traces + 2 scale traces


@pytest.mark.slow
def test_simple_plotly_3w():
    # net with 3W-transformer
    net = example_multivoltage()
    fig = simple_plotly(net, filename=join(gettempdir(), "temp-plot.html"), auto_open=False)
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + (len(net.trafo3w) * 3 + 1) + 2
    # +1 is for infofunc traces, +2 = 1 bus trace + 1 ext_grid trace


@pytest.mark.slow
def test_simple_plotly_no_html():
    net = example_multivoltage()
    # fig without generating a HTML
    fig = simple_plotly(net, filename=None, auto_open=False)
    assert len(fig.data) == (len(net.line) + 1) + (len(net.trafo) + 1) + (len(net.trafo3w) * 3 + 1) + 2
    # +1 is for infofunc traces, +2 = 1 bus trace + 1 ext_grid trace


@pytest.mark.slow
def test_pf_res_plotly_plotly_only_cmap():
    # https://github.com/e2nIEE/pandapower/issues/2567: 'algae' only exists in the plotly
    # colorscale registry, not in matplotlib's. The line traces used to look the name up
    # in the matplotlib registry and raise a KeyError.
    net = mv_oberrhein()
    pp.runpp(net)
    fig = pf_res_plotly(
        net,
        cmap="algae",
        line_width=3,
        bus_size=15,
        climits_volt=(0.95, 1.05),
        climits_load=(0, 150),
        filename=None,
        auto_open=False,
    )
    assert len(fig.data) == 188
    bus_traces = [
        trace
        for trace in fig.data
        if trace.marker is not None
        and trace.marker.colorbar is not None
        and trace.marker.colorbar.title.text == "Bus Voltage [pu]"
    ]
    assert len(bus_traces) == 1
    # plotly expands a named colorscale to its stops when read back (positions may
    # differ in the last float bit)
    from plotly.colors import get_colorscale

    expected_scale = get_colorscale("algae")
    actual_scale = bus_traces[0].marker.colorscale
    assert len(actual_scale) == len(expected_scale)
    assert all(
        abs(actual[0] - expected[0]) < 1e-9 and actual[1] == expected[1]
        for actual, expected in zip(actual_scale, expected_scale)
    )


@pytest.mark.slow
def test_pf_res_plotly_plotly_only_cmap_reversed():
    # reversed plotly-only colorscale names ('_r' suffix) must work as well, and the
    # bus markers must use the reversed scale, not the forward one
    from plotly.colors import get_colorscale

    net = mv_oberrhein()
    pp.runpp(net)
    fig = pf_res_plotly(net, cmap="algae_r", filename=None, auto_open=False)
    assert len(fig.data) == 188
    bus_traces = [
        trace
        for trace in fig.data
        if trace.marker is not None
        and trace.marker.colorbar is not None
        and trace.marker.colorbar.title.text == "Bus Voltage [pu]"
    ]
    assert len(bus_traces) == 1
    expected_scale = get_colorscale("algae_r")
    actual_scale = bus_traces[0].marker.colorscale
    assert len(actual_scale) == len(expected_scale)
    assert all(
        abs(actual[0] - expected[0]) < 1e-9 and actual[1] == expected[1]
        for actual, expected in zip(actual_scale, expected_scale)
    )


@pytest.mark.slow
def test_pf_res_plotly_matplotlib_only_cmap():
    # mirror case of #2567: 'gnuplot' only exists in the matplotlib registry. The bus
    # markers used to pass the name to plotly unchanged, which raised a ValueError.
    net = mv_oberrhein()
    pp.runpp(net)
    fig = pf_res_plotly(net, cmap="gnuplot", filename=None, auto_open=False)
    assert len(fig.data) == 188
    bus_traces = [
        trace
        for trace in fig.data
        if trace.marker is not None
        and trace.marker.colorbar is not None
        and trace.marker.colorbar.title.text == "Bus Voltage [pu]"
    ]
    assert len(bus_traces) == 1
    # a matplotlib-only name is sampled to an explicit colorscale list for plotly
    expected_scale = get_plotly_colorscale("gnuplot")
    actual_scale = bus_traces[0].marker.colorscale
    assert len(actual_scale) == len(expected_scale)
    assert all(
        abs(actual[0] - expected[0]) < 1e-9 and actual[1] == expected[1]
        for actual, expected in zip(actual_scale, expected_scale)
    )


def test_get_plotly_cmap_plotly_only_name():
    # line colors for a plotly-only name are exactly plotly's own sampling
    from plotly.colors import sample_colorscale

    values = np.array([0.0, 75.0, 150.0])
    assert get_plotly_cmap(values, cmap_name="algae", cmin=0, cmax=150) == sample_colorscale("algae", [0.0, 0.5, 1.0])
    assert get_plotly_cmap(values, cmap_name="algae_r", cmin=0, cmax=150) == sample_colorscale(
        "algae_r", [0.0, 0.5, 1.0]
    )


def test_get_plotly_cmap_plotly_only_name_out_of_range():
    # values outside [cmin, cmax] clamp to the end colors, like the matplotlib branch;
    # sample_colorscale itself raises IndexError for points outside [0, 1]
    from plotly.colors import sample_colorscale

    values = np.array([-50.0, 75.0, 200.0])
    assert get_plotly_cmap(values, cmap_name="algae", cmin=0, cmax=150) == sample_colorscale("algae", [0.0, 0.5, 1.0])
    # degenerate range (cmin == cmax) maps everything to the low end, as Normalize does
    assert get_plotly_cmap(values, cmap_name="algae", cmin=100, cmax=100) == sample_colorscale("algae", [0.0, 0.0, 0.0])


@pytest.mark.slow
def test_pf_res_plotly_plotly_only_cmap_overloaded_line():
    # overloaded lines exceed climits_load, so their normalized value is above cmax;
    # this used to raise IndexError in sample_colorscale on the plotly path
    net = mv_oberrhein()
    pp.runpp(net)
    net.res_line.loc[net.res_line.index[0], "loading_percent"] = 200.0
    fig = pf_res_plotly(net, cmap="algae", climits_load=(0, 150), filename=None, auto_open=False)
    assert len(fig.data) == 188


def test_get_plotly_cmap_matplotlib_name_unchanged():
    # matplotlib-only names still go through the matplotlib registry as before
    values = np.array([0.0, 150.0])
    assert get_plotly_cmap(values, cmap_name="gnuplot", cmin=0, cmax=150)[0] == "rgb(0,0,0)"


def test_get_plotly_cmap_unknown_name():
    with pytest.raises(ValueError, match="not a known matplotlib or plotly colormap"):
        get_plotly_cmap(np.array([0.0, 1.0]), cmap_name="not_a_colormap")


def test_get_plotly_colorscale():
    # plotly names pass through unchanged, lists pass through unchanged
    assert get_plotly_colorscale("viridis") == "viridis"
    assert get_plotly_colorscale("Jet") == "Jet"
    assert get_plotly_colorscale([[0, "red"], [1, "blue"]]) == [[0, "red"], [1, "blue"]]
    # a matplotlib-only name is sampled to an explicit colorscale list
    scale = get_plotly_colorscale("gnuplot")
    assert len(scale) == 11
    assert scale[0] == [0.0, "rgb(0, 0, 0)"]
    assert scale[-1] == [1.0, "rgb(255, 255, 0)"]
    with pytest.raises(ValueError, match="not a known matplotlib or plotly colormap"):
        get_plotly_colorscale("not_a_colormap")


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

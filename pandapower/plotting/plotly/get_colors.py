# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = ["get_plotly_cmap", "get_plotly_color", "get_plotly_color_palette", "get_plotly_colorscale"]

import sys

import numpy as np

try:
    import matplotlib.colors as mplc
    from matplotlib import colormaps as mpl_colormaps

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

try:
    import seaborn
except ImportError:
    pass

from pandapower.auxiliary import soft_dependency_error


def get_plotly_color(color_string):
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    try:
        converted = _to_plotly_color(mplc.to_rgba(color_string))
        return converted
    except ValueError:
        return color_string


def get_plotly_color_palette(n):
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    if "seaborn" in sys.modules:
        return _to_plotly_palette(seaborn.color_palette("hls", n))
    else:
        hsv = mpl_colormaps["hsv"]
        return _to_plotly_palette(hsv(np.linspace(0, 1.0, n)))


def _to_plotly_palette(scl, transparence=None):
    """
    converts a rgb color palette in format (0-1,0-1,0-1) to a plotly color palette
    'rgb(0-255,0-255,0-255)'
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    _out = []
    for color in scl:
        plotly_col = [255 * _c for _c in mplc.to_rgba(color)]
        if transparence:
            if not (0 <= transparence <= 1):
                raise AssertionError("transparence should be between 0 and 1")
            plotly_col[3] = transparence
            plotly_col = "rgba({:.0f}, {:.0f}, {:.0f}, {:.4f})".format(*plotly_col)
        else:
            plotly_col = "rgb({:.0f}, {:.0f}, {:.0f})".format(*plotly_col[:3])
        _out.append(plotly_col)
    return _out


def _to_plotly_color(scl, transparence=None):
    """
    converts a rgb color in format (0-1,0-1,0-1) to a plotly color 'rgb(0-255,0-255,0-255)'
    """
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    plotly_col = [255 * _c for _c in mplc.to_rgba(scl)] if len(scl) == 3 else [255 * _c for _c in mplc.to_rgb(scl)]
    if transparence is not None:
        assert 0.0 <= transparence <= 1.0
        plotly_col[3] = transparence
        return "rgba({:.0f}, {:.0f}, {:.0f}, {:.4f})".format(*plotly_col)
    else:
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(*plotly_col[:3])


def get_plotly_cmap(values, cmap_name="jet", cmin=None, cmax=None):
    if MATPLOTLIB_INSTALLED:
        try:
            cmap = mpl_colormaps[cmap_name]
        except KeyError:
            # plotly-only colorscale name (e.g. 'algae'): sample the plotly colorscale instead
            return _get_plotly_cmap_from_plotly(values, cmap_name, cmin, cmax)
    else:
        # without matplotlib only plotly colorscale names can be resolved
        return _get_plotly_cmap_from_plotly(values, cmap_name, cmin, cmax)
    if cmin is None:
        cmin = values.min()
    if cmax is None:
        cmax = values.max()
    norm = mplc.Normalize(vmin=cmin, vmax=cmax)
    bus_fill_colors_rgba = cmap(norm(values).data)[:, 0:3] * 255.0
    # integer components: plotly rejects the scientific notation that float formatting
    # can produce for very small components (e.g. 'rgb(15.9,1.5e-05,6.2)')
    return [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in bus_fill_colors_rgba]


def _get_plotly_cmap_from_plotly(values, cmap_name, cmin, cmax):
    try:
        from plotly.colors import get_colorscale, sample_colorscale
        from plotly.exceptions import PlotlyError
    except ImportError:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "plotly")
    try:
        colorscale = get_colorscale(cmap_name)
    except (PlotlyError, KeyError, ValueError):
        raise ValueError(f"Colormap {cmap_name!r} is not a known matplotlib or plotly colormap") from None
    values = np.asarray(values, dtype=float)
    if cmin is None:
        cmin = values.min()
    if cmax is None:
        cmax = values.max()
    if cmax == cmin:
        # mirror matplotlib.colors.Normalize, which maps everything to 0 for a degenerate range
        sample_points = np.zeros(values.shape)
    else:
        sample_points = (values - cmin) / (cmax - cmin)
    # sample_colorscale requires points in [0, 1] and raises IndexError outside of it, so
    # clamp like the matplotlib colormap does (out-of-range values take the end colors);
    # NaN values fall back to the low end instead of raising
    sample_points = np.clip(np.nan_to_num(sample_points, nan=0.0), 0.0, 1.0)
    return sample_colorscale(colorscale, sample_points.tolist())


def get_plotly_colorscale(cmap_name):
    """
    Returns a colorscale that plotly accepts for a matplotlib or plotly colormap name.

    A name known to plotly is returned unchanged, so plotly keeps using its own (named)
    colorscale. A name only known to matplotlib is sampled to an explicit list of
    [position, color] stops, which plotly accepts as a colorscale. A name unknown to both
    registries raises a ValueError.
    """
    if not isinstance(cmap_name, str):
        return cmap_name
    try:
        from plotly.colors import get_colorscale
        from plotly.exceptions import PlotlyError
    except ImportError:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "plotly")
    try:
        get_colorscale(cmap_name)
        return cmap_name
    except (PlotlyError, KeyError, ValueError):
        pass
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    try:
        cmap = mpl_colormaps[cmap_name]
    except KeyError:
        raise ValueError(f"Colormap {cmap_name!r} is not a known matplotlib or plotly colormap") from None
    positions = np.linspace(0, 1, 11)
    colors_rgba = cmap(positions)[:, 0:3] * 255.0
    return [[float(position), f"rgb({r:.0f}, {g:.0f}, {b:.0f})"] for position, (r, g, b) in zip(positions, colors_rgba)]

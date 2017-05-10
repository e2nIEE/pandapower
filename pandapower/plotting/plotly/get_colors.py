# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

try:
    import seaborn
except ImportError:
    pass



def get_plotly_color(color_string):
    colors_names = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan']
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    colors_plotly = []
    if 'seaborn' in sys.modules:
        for c in colors:
            colors_plotly.append(_to_plotly_color(c))
        colors_dict = dict(zip(colors_names, colors_plotly))
    else:
        colors_dict = dict(zip(colors_names, colors_names))

    return color_string if colors_dict.get(color_string) is None else colors_dict.get(color_string)


def get_plotly_color_palette(n):
    if 'seaborn' in sys.modules:
        return _to_plotly_palette(seaborn.color_palette("hls", n))
    else:
        hsv = plt.get_cmap('hsv')
        return _to_plotly_palette(hsv(np.linspace(0, 1.0, n)))


def _to_plotly_palette(scl, transparence=None):
    """
    converts a rgb color palette in format (0-1,0-1,0-1) to a plotly color palette 'rgb(0-255,0-255,0-255)'
    """
    if transparence:
        return ['rgb({0},{1},{2},{3})'.format(r*255, g*255, b*255, transparence) for r, g, b in scl]
    else:
        return ['rgb({0},{1},{2})'.format(r*255, g*255, b*255) for r, g, b in scl]



def _to_plotly_color(scl, transparence=None):
    """
    converts a rgb color in format (0-1,0-1,0-1) to a plotly color 'rgb(0-255,0-255,0-255)'
    """
    if transparence:
        return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, transparence))
    elif len(scl) > 3:
        return 'rgb' + str((scl[0] * 255, scl[1] * 255, scl[2] * 255, scl[3]))
    else:
        return 'rgb'+str((scl[0]*255, scl[1]*255, scl[2]*255))


def get_plotly_cmap(values, cmap_name='jet', cmin=None, cmax=None):
    cmap = cm.get_cmap(cmap_name)
    if cmin is None:
        cmin = values.min()
    if cmax is None:
        cmax = values.max()
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    bus_fill_colors_rgba = cmap(norm(values).data)[:, 0:3] * 255.
    return ['rgb({0},{1},{2})'.format(r, g, b) for r, g, b in bus_fill_colors_rgba]
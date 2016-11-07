# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from decimal import Decimal

def cmap_discrete(cmap_list):
    """
    Can be used to create a discrete colormap.
    
    Input:
        
        - cmap_list (list) - list of tuples, where each tuple represents one range. Each tuple has the form of ((from, to), color).
        
    Return:
        
        - cmap - matplotlib colormap

        - norm - matplotlib norm object
        
    Example:
        
        >>> from pandapower.plotting import cmap_discrete, create_line_collection, draw_collections
        >>> cmap_list = [((20, 50), "green"), ((50, 70), "yellow"), ((70, 100), "red")]
        >>> cmap, norm = cmap_discrete(cmap_list)
        >>> lc = create_line_collection(net, cmap=cmap, norm=norm)
        >>> draw_collections([lc])
    """
    #TODO: this implementation is extremely hacky, should be possible to implement this more
    # elegenatly with BoundaryNorm, but I failed in doing so. Works, but should be refactored. LT
    cmap_colors = []
    min_loading = cmap_list[0][0][0]
    max_loading = cmap_list[-1][0][1]
    max_decimal = max([abs(Decimal(str(x1)).as_tuple().exponent) for (x1, x2), color in cmap_list])
    x2_before = None
    for (x1, x2), color in cmap_list:
        if x2_before and x2_before != x1:
            raise ValueError("Ranges for colormap must be continous")
        cmap_colors += [color]*int((x2-x1)*10**(max_decimal+1))
        x2_before = x2
    cmap = ListedColormap(cmap_colors)
    norm = Normalize(min_loading, max_loading)
    return cmap, norm

def cmap_continous(cmap_list):
    """
    Can be used to create a continous colormap.
    
    Input:
        
        - cmap_list (list) - list of tuples, where each tuple represents one color. Each tuple has the form of (center, color). The colorbar is a linear segmentation of the colors between the centers.

    Return:
        
        - cmap - matplotlib colormap

        - norm - matplotlib norm object
        
    Example:
        
        >>> from pandapower.plotting import cmap_continous, create_bus_collection, draw_collections
        >>> cmap_list = [(0.97, "blue"), (1.0, "green"), (1.03, "red")]
        >>> cmap, norm = cmap_continous(cmap_list)
        >>> bc = create_bus_collection(net, cmap=cmap, norm=norm)
        >>> draw_collections([bc])
    """
    min_loading = cmap_list[0][0]
    max_loading = cmap_list[-1][0]
    cmap_colors = [((loading-min_loading)/(max_loading - min_loading), color) for
                 (loading, color) in cmap_list]
    cmap = LinearSegmentedColormap.from_list('name', cmap_colors)
    norm = Normalize(min_loading, max_loading)
    return cmap, norm

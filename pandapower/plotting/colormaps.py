# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

from matplotlib.colors import LinearSegmentedColormap
    
def set_cmap_line_loading(net, collection, min_loading, max_loading, 
                          colors=['green', 'yellow', 'orange'], max_color=None, min_color=None, 
                          title="Line Loading [%]"):
    """
    Sets a colormap for a line collection.
    
    Input:

        **net** (PandapowerNet) - The pandapower network
        
        **collection** (matplotlib collection) - pandapower line collection
        
        **min_loading** (float) - lower line loading limit of the colorbar

        **max_loading** (loat) - upper line loading limit of the colorbar

               
    Optional:
    
        **colors** (list) - list of colors for matpolotlib linear segmented colormap

        **max_color** (color) - color for all lines above max_loading
        
        **min_color** (color) - color for all lines below min_loading

        **min_color** (color) - color for all lines below min_loading
        
        *title** (str) - colorbar title
    """
    cmap = LinearSegmentedColormap.from_list('name', colors)
    collection.set_cmap(cmap)
    collection.set_array(net.res_line.loading_percent)
    collection.set_clim(min_loading, max_loading)
    collection.cbar_title = title
    collection.has_colormap = True
    if max_color:
        cmap.set_over(max_color)
        collection.extend = "max"
    if min_color:
        cmap.set_under(min_color)
        collection.extend = "min"
    if max_color and min_color:
        collection.extend = "both"
                    
def set_cmap_bus_voltage(net, collection, colors=["blue", "yellow", "red"], min_voltage=0.95,
                         max_voltage = 1.05, max_color=None, min_color=None,
                         title="Bus Voltage [pu]"):
    """
    Sets a colormap for a bus collection.
    
    Input:

        **net** (PandapowerNet) - The pandapower network
        
        **collection** (matplotlib collection) - pandapower bus collection
        
        **min_loading** (float) - lower voltage limit of the colorbar

        **max_loading** (loat) - upper voltage limit of the colorbar

               
    Optional:
    
        **colors** (list) - list of colors for matpolotlib linear segmented colormap

        **max_color** (color) - color for all buses above max_voltage
        
        **min_color** (color) - color for all buses below min_voltage

        **min_color** (color) - color for all buses below min_voltage
        
        *title** (str) - colorbar title
    """
    cmap = LinearSegmentedColormap.from_list('name', colors)
    collection.set_cmap(cmap)
    collection.set_array(net.res_bus.vm_pu)
    collection.set_clim(min_voltage, max_voltage)
    collection.cbar_title = title
    collection.has_colormap = True
    if max_color:
        cmap.set_over(max_color)
        collection.extend = "max"
    if min_color:
        cmap.set_under(min_color)
        collection.extend = "min"
    if max_color and min_color:
        collection.extend = "both"
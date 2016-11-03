# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.


from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
import matplotlib.pyplot as plt

def cmap_loading_discrete(cmap_list=[((20, 50), "green"), ((50, 70), "yellow"), 
                                         ((70, 100), "red")]):
    cmap_colors = []
    min_loading = cmap_list[0][0][0]
    max_loading = cmap_list[-1][0][1]
    x2_before = None
    for (x1, x2), color in cmap_list:
        if x2_before and x2_before != x1:
            raise ValueError("Ranges for colormap must be continous")
        cmap_colors += [color]*(x2-x1)
        x2_before = x2
    cmap = ListedColormap(cmap_colors)
    norm = Normalize(min_loading, max_loading)
    return cmap, norm

def cmap_voltage_discrete(cmap_list=[((0.95, 0.97), "blue"), ((0.97, 1.03), "green"), 
                                         ((1.03, 1.05), "red")]):
    cmap_colors = []
    min_loading = cmap_list[0][0][0]
    max_loading = cmap_list[-1][0][1]
    x2_before = None
    for (x1, x2), color in cmap_list:
        if x2_before and x2_before != x1:
            raise ValueError("Ranges for colormap must be continous")
        cmap_colors += [color]*int((x2-x1)*1000)
        x2_before = x2
    cmap = ListedColormap(cmap_colors)
    norm = Normalize(min_loading, max_loading)
    return cmap, norm
    
def cmap_loading_continous(cmap_list=[(0, "green"), (50, "yellow"), (100, "red")]):
    min_loading = cmap_list[0][0]
    max_loading = cmap_list[-1][0]
    cmap_colors = [((loading-min_loading)/(max_loading - min_loading), color) for
                 (loading, color) in cmap_list]
    cmap = LinearSegmentedColormap.from_list('name', cmap_colors)
    norm = Normalize(min_loading, max_loading)
    return cmap, norm

def cmap_voltage_continous(cmap_list=[(0.975, "blue"), (1.0, "green"), (1.03, "red")]):
    min_voltage = cmap_list[0][0]
    max_voltage = cmap_list[-1][0]
    cmap_colors = [((voltage-min_voltage)/(max_voltage - min_voltage), color) for
                 (voltage, color) in cmap_list]
    cmap = LinearSegmentedColormap.from_list('name', cmap_colors)
    norm = Normalize(min_voltage, max_voltage)
    return cmap, norm
        
if __name__ == '__main__':
    import pandapower as pp
    import pandapower.plotting as plot
    import pandapower.networks as nw
    from matplotlib.colors import LogNorm
    net = nw.mv_oberrhein("load")
    pp.runpp(net)
    
    cmap = plt.get_cmap('PuBu_r')
    norm = LogNorm(vmin=20, vmax=100)
    lc = plot.create_line_collection(net, net.line.index, zorder=1, color="grey", linewidths=2,
                                     cmap=cmap, norm=norm)
    bc = plot.create_bus_collection(net, net.bus.index, size=80, zorder=2)
    plot.draw_collections([lc, bc], figsize=(12,10))
#    cb = plt.colorbar(lc)
#    cb.set_ticks([40, 50])
#    cb.ax.set_ylabel("This is a individual colorbar title")
    
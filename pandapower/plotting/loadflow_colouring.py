# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:59:38 2016

@author: ulffers
"""
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandapower as pp


def colour_line_loadings(net, lc, loading_ticks_percent=[80, 90, 100]):
    pp.runpp(net)
    line_loadings = net.res_line.loading_percent
    ticks_yellow = (loading_ticks_percent[1] - loading_ticks_percent[0])*10
    ticks_orange = (loading_ticks_percent[2] - loading_ticks_percent[1])*10
    cmap_list_loading = ['yellow']*ticks_yellow + ['orange']*ticks_orange
    loading_cmap = ListedColormap(cmap_list_loading)
    loading_cmap.set_over('r')
    loading_cmap.set_under('grey')
    lc.set_array(line_loadings)
    lc.set_clim(loading_ticks_percent[-1], loading_ticks_percent[0])
    lc.set_cmap(loading_cmap)

    return lc


def colour_bus_voltages(net, bc, voltage_ticks_pu=[0.92, 0.96, 1, 1.04, 1.08]):
    pp.runpp(net)
    busses_voltages = net.res_bus[net.res_bus.index.isin(net.bus_geodata.index)].vm_pu
    ticks_blue = round((voltage_ticks_pu[1] - voltage_ticks_pu[0])*100)
    ticks_green = round((voltage_ticks_pu[2] - voltage_ticks_pu[1])*100)
    ticks_green = round((voltage_ticks_pu[3] - voltage_ticks_pu[2])*100)
    ticks_orange = round((voltage_ticks_pu[4] - voltage_ticks_pu[3])*100)
    cmap_list_voltage = ['blue']*ticks_blue + ['green']*ticks_green + ['green']*ticks_green \
                        + ['orange']*ticks_orange
    voltage_cmap = ListedColormap(cmap_list_voltage)
    voltage_cmap.set_under('violet')
    voltage_cmap.set_over('r')
    bc.set_array(busses_voltages)
    bc.set_clim(vmax=voltage_ticks_pu[-1], vmin=voltage_ticks_pu[0])
    bc.set_cmap(voltage_cmap)

    return bc


def add_line_loadings_colourbar(lc, loading_ticks_percent=[80, 90, 100]):
    cbar_load = plt.colorbar(lc, extend='max')
    cbar_load.ax.set_ylabel('line loading [%]')
    cbar_load.set_ticks(loading_ticks_percent)


def add_bus_voltages_colourbar(bc, voltage_ticks_pu=[0.92, 0.96, 1, 1.04, 1.08]):
    cbar_volt = plt.colorbar(bc, extend='both')
    cbar_volt.ax.set_ylabel('bus voltage [p.u.]')
    cbar_volt.set_ticks(voltage_ticks_pu)

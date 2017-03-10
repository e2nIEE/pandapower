# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import pandapower.topology as top
import numpy as np

def plot_voltage_profile(net, plot_transformers=True, ax=None, xlabel="Distance from Slack [km]",
                         ylabel="Voltage [pu]", x0=0, trafocolor="r", bus_colors=None, **kwargs):
    if ax is None:
        plt.figure(facecolor="white", dpi=120)
        ax = plt.gca()
    if not net.converged:
        raise ValueError("no results in this pandapower network")
    for eg in net.ext_grid[net.ext_grid.in_service==True].bus:
        d = top.calc_distance_to_bus(net, eg)
        for _, line in net.line[net.line.in_service == True].iterrows():
            if not line.from_bus in d.index:
                continue
            if not ((net.switch.element==line.name) & (net.switch.closed==False)
                    & (net.switch.et=='l')).any():
                fbus = line.from_bus
                tbus = line.to_bus
                x = [x0 + d.at[fbus], x0 + d.at[tbus]]
                try:
                    y = [net.res_bus.vm_pu.at[fbus], net.res_bus.vm_pu.at[tbus]]
                except:
                    raise UserWarning
                ax.plot(x, y, **kwargs)
                if bus_colors is not None:
                    for bus, x, y in zip((fbus, tbus), x, y):
                        if bus in bus_colors:
                            ax.plot(x, y, 'or', color=bus_colors[bus], ms=3)

                kwargs = {k:v for k,v in kwargs.items() if not k == "label"}
        if plot_transformers:
            if hasattr(plot_transformers, "__iter__"):  # if is a list for example
                trafos = net.trafo.loc[list(plot_transformers)]
            else:
                trafos = net.trafo[net.trafo.in_service == True]
            for _, trafo in trafos.iterrows():
                if trafo.hv_bus not in d.index:
                    continue
                ax.plot([x0 + d.loc[trafo.hv_bus],
                         x0 + d.loc[trafo.lv_bus]],
                        [net.res_bus.vm_pu.loc[trafo.hv_bus],
                         net.res_bus.vm_pu.loc[trafo.lv_bus]], color=trafocolor,
                        **{k:v for k,v in kwargs.items() if not k == "color"})
        if xlabel:
            ax.set_xlabel(xlabel, fontweight="bold", color=(.4, .4, .4))
        if ylabel:
            ax.set_ylabel(ylabel, fontweight="bold", color=(.4, .4, .4))
    return ax


def plot_loading(net, element="line", boxcolor="b", mediancolor="r", whiskercolor="k", ax = None):
    if ax is None:
        plt.figure(facecolor="white", dpi=80)
        ax = plt.gca()
    loadings = net["res_%s"%element].loading_percent.values
    boxplot = ax.boxplot(loadings[~np.isnan(loadings)], whis = "range")
    for l in list(boxplot.keys()):
        plt.setp(boxplot[l], lw=3)
        if l == "medians":
            plt.setp(boxplot[l], color=mediancolor)
        elif l == "boxes" or l == "whiskers":
            plt.setp(boxplot[l], color=boxcolor)
        else:
            plt.setp(boxplot[l], color=whiskercolor)

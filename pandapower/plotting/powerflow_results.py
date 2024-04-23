# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import sys
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import networkx as nx
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

from pandapower.auxiliary import soft_dependency_error, warn_and_fix_parameter_renaming
import pandapower.topology as top


def plot_voltage_profile(net, ax=None, plot_transformers=True, xlabel="Distance from Slack [km]",
                         ylabel="Voltage [pu]", x0=0, line_color="grey", trafo_color="r",
                         bus_colors="b", line_loading_weight=False, voltage_column=None,
                         bus_size=3, lines=None, **kwargs):
    """Plot the voltage profile depending on the distance from the slack.

    Parameters
    ----------
    net : pp.PandapowerNet
        net including power flow results
    ax : matplotlib.axes, optional
        axis to plot to, by default None
    plot_transformers : bool, optional
        Whether vertical lines should be plotted to display the voltage drop of the transformers,
        by default True
    xlabel : str, optional
        xlable of the figure, by default "Distance from Slack [km]"
    ylabel : str, optional
        ylable of the figure, by default "Voltage [pu]"
    x0 : int, optional
        slack position at the xaxis, by default 0
    line_color : str, optional
        color used to plot the lines, by default "grey"
    trafo_color : str, optional
        color used to plot the trafos, by default "r"
    bus_colors : [str, dict[int, str]], optional
        colors used to plot the buses. Can be passed as string (to give all buses the same color),
        or as dict, by default "b"
    line_loading_weight : bool, optional
        enables loading dependent width of the lines, by default False
    bus_size : int, optional
        size of bus representations, by default 3
    lines : Any[list[int], pd.Index[int]], optional
        list of line indices which should be plottet. If None, all lines are plotted, by default None

    Returns
    -------
    matplotlib.axes
        axis of the plot

    """
    # handle exceptions and inputs
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "matplotlib")
    if "voltage_column" in kwargs:
        raise DeprecationWarning("Parameter 'voltage_column' has been removed.")
    trafo_color = warn_and_fix_parameter_renaming(
        "trafocolor", "trafo_color", trafo_color, "r", **kwargs)
    if ax is None:
        plt.figure(facecolor="white", dpi=120)
        ax = plt.gca()
    if not net.converged and not net.OPF_converged:
        raise ValueError("no results in this pandapower network")
    if lines is None:
        lines = net.line.index

    # run plotting code
    sl_buses = np.union1d(
        net.ext_grid.loc[net.ext_grid.in_service, "bus"].values,
        net.gen.loc[net.gen.slack & net.gen.in_service, "bus"].values)
    for eg in sl_buses:
        d = top.calc_distance_to_bus(net, eg)
        for lix, line in net.line[net.line.in_service & net.line.index.isin(lines)].iterrows():
            if line.from_bus not in d.index:
                continue
            if not ((net.switch.element == line.name) & ~net.switch.closed & (
                    net.switch.et == 'l')).any():
                from_bus = line.from_bus
                to_bus = line.to_bus
                x = [x0 + d.at[from_bus], x0 + d.at[to_bus]]
                try:
                    y = [net.res_bus.vm_pu.at[from_bus], net.res_bus.vm_pu.at[to_bus]]
                except:
                    raise UserWarning
                if "linewidth" in kwargs or not line_loading_weight:
                    ax.plot(x, y, color=line_color, **kwargs)
                else:
                    ax.plot(x, y, linewidth=0.4 * np.sqrt(net.res_line.loading_percent.at[lix]),
                            color=line_color, **kwargs)
                if bus_colors is not None:
                    if isinstance(bus_colors, str):
                        bus_colors = {b: bus_colors for b in net.bus.index}
                    for bus, x, y in zip((from_bus, to_bus), x, y):
                        if bus in bus_colors:
                            ax.plot(x, y, 'o', color=bus_colors[bus], ms=bus_size)
                kwargs = {k: v for k, v in kwargs.items() if not k == "label"}

        # trafo geodata
        if plot_transformers:
            for trafo_table in ['trafo', 'trafo3w']:
                if trafo_table not in net.keys():
                    continue
                transformers = net[trafo_table].query('in_service')
                for tid, tr in transformers.iterrows():
                    t_buses = [tr[b_col] for b_col in ('lv_bus', 'mv_bus', 'hv_bus') if
                               b_col in tr.index]
                    if any([b not in d.index.values or b not in net.res_bus.index.values for b in
                            t_buses]):
                        # logger.info('cannot add trafo %d to plot' % tid)
                        continue

                    for bi, bj in combinations(t_buses, 2):
                        tr_coords = ([x0 + d.loc[bi], x0 + d.loc[bj]],
                                     [net.res_bus.at[bi, 'vm_pu'], net.res_bus.at[bj, 'vm_pu']])
                        ax.plot(*tr_coords, color=trafo_color,
                                **{k: v for k, v in kwargs.items() if not k == "color"})

        if xlabel:
            ax.set_xlabel(xlabel, fontweight="bold", color=(.4, .4, .4))
        if ylabel:
            ax.set_ylabel(ylabel, fontweight="bold", color=(.4, .4, .4))
    return ax


def plot_loading(net, ax=None, element_type="line", box_color="b", median_color="r",
                 whisker_color="k", index_subset=None, **kwargs):
    """Plot a boxplot of loadings

    Parameters
    ----------
    net : pp.PandapowerNet
        net including power flow results
    ax : matplotlib.axes, optional
        axis to plot to, by default None
    element : str, optional
        name of element which loadings should be plotted, by default "line"
    box_color : str, optional
        color of the box, by default "b"
    median_color : str, optional
        color of the median line, by default "r"
    whisker_color : str, optional
        color of the whiskers, by default "k"
    index_subset : Any[list[int], pd.Index[int]], optional
        list of element indices which should be considered. If None, all elements are considered,
        by default None

    Returns
    -------
    matplotlib.axes
        axis of the plot
    """
    # handle exceptions and inputs
    deprecated = [("boxcolor", "box_color", "b"),
                  ("mediancolor", "median_color", "r"),
                  ("whiskercolor", "whisker_color", "k"),
                  ("element", "element_type", "line")]
    for (old, new, default) in deprecated:
        locals()[new] = warn_and_fix_parameter_renaming(
            old, new, locals()[new], default, **kwargs)
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "matplotlib")
    if ax is None:
        plt.figure(facecolor="white", dpi=80)
        ax = plt.gca()
    if index_subset is None:
        index_subset = net[element_type].index

    # run plotting code
    loadings = net["res_%s" % element_type].loading_percent.values[net[
        "res_%s" % element_type].index.isin(index_subset)]
    boxplot = ax.boxplot(loadings[~np.isnan(loadings)], whis=[0, 100])
    for l in list(boxplot.keys()):
        plt.setp(boxplot[l], lw=3)
        if l == "medians":
            plt.setp(boxplot[l], color=median_color)
        elif l == "boxes" or l == "whiskers":
            plt.setp(boxplot[l], color=box_color)
        else:
            plt.setp(boxplot[l], color=whisker_color)
    ax.set_ylabel(f"{element_type.capitalize()} Loading [%]", fontweight="bold", color=(.4, .4, .4))
    ax.set_xticks([1], [""])
    return ax


def voltage_profile_to_bus_geodata(net, voltages=None, root_bus=None):
    if voltages is None:
        if not net.converged:
            raise ValueError("no results in this pandapower network")
        voltages = net.res_bus.vm_pu

    mg = top.create_nxgraph(net, respect_switches=True)
    sl_buses = np.r_[
        net.ext_grid.loc[net.ext_grid.in_service, "bus"].values,
        net.gen.loc[net.gen.slack & net.gen.in_service, "bus"].values]
    first_eg = sl_buses[0] if root_bus is None else root_bus
    other_eg = np.setdiff1d(sl_buses, first_eg)
    mg.add_edges_from([(first_eg, y, {"weight": 0}) for y in other_eg])
    dist = pd.Series(nx.single_source_dijkstra_path_length(mg, first_eg))

    bgd = pd.DataFrame({"x": dist.loc[net.bus.index.values].values,
                        "y": voltages.loc[net.bus.index.values].values},
                       index=net.bus.index)
    return bgd


if __name__ == "__main__":
    import pandapower as pp
    import pandapower.networks as nw

    net = nw.mv_oberrhein()
    pp.runpp(net)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 5), gridspec_kw={"width_ratios": [4, 1]})
    plot_voltage_profile(net, ax=axs[0])
    plot_loading(net, ax=axs[1])
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:02:21 2021

@author: x230
"""
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks as nw


net = nw.create_cigre_network_mv("pv_wind")
# set some limits
min_vm_pu = 0.95
max_vm_pu = 1.05

net["bus"].loc[:, "min_vm_pu"] = min_vm_pu
net["bus"].loc[:, "max_vm_pu"] = max_vm_pu

net["line"].loc[:, "max_loading_percent"] = 100.

# close all switches
net.switch.loc[:, "closed"] = True
# add storage to bus 10
pp.create_storage(net, 10, p_mw=0.5, max_e_mwh=.2, soc_percent=0., q_mvar=0., controllable=True)


# set the load type in the cigre grid, since it is not specified
net["load"].loc[:, "type"] = "residential"
# change the type of the last sgen to wind
net.sgen.loc[:, "type"] = "pv"
net.sgen.loc[8, "type"] = "wind"

# read the example time series
input_file = "cigre_timeseries_15min.json"
time_series = pd.read_json(input_file)
time_series.sort_index(inplace=True)
# this example time series has a 15min resolution with 96 time steps for one day
n_timesteps = time_series.shape[0]

n_load = len(net.load)
n_sgen = len(net.sgen)
p_timeseries = np.zeros((n_timesteps, n_load + n_sgen), dtype=float)
# p
load_p = net["load"].loc[:, "p_mw"].values
sgen_p = net["sgen"].loc[:7, "p_mw"].values
wind_p = net["sgen"].loc[8, "p_mw"]

p_timeseries_dict = dict()
for t in range(n_timesteps):
    # print(time_series.at[t, "residential"])
    p_timeseries[t, :n_load] = load_p * time_series.at[t, "residential"]
    p_timeseries[t, n_load:-1] = - sgen_p * time_series.at[t, "pv"]
    p_timeseries[t, -1] = - wind_p * time_series.at[t, "wind"]
    p_timeseries_dict[t] = p_timeseries[t, :].tolist()

time_series_file = os.path.join(tempfile.gettempdir(), "timeseries.json")
with open(time_series_file, 'w') as fp:
    json.dump(p_timeseries_dict, fp)


storage_results = pp.runpm_storage_opf(net, n_timesteps=96, time_elapsed=0.25)

# storage_results = pp.runpm_storage_opf(net, calculate_voltage_angles=True,
#                       trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
#                       n_timesteps=24, time_elapsed=1.0, correct_pm_network_data=True,
#                       pm_model="ACPPowerModel", pm_time_limits=None, pm_log_level=0, opf_flow_lim="S")

# def store_results(storage_results, grid_name):
#     for key, val in storage_results.items():
#         file = grid_name + "_strg_res" + str(key) + ".json"
#         print("Storing results to file {}".format(file))
#         print(val)
#         val.to_json(file)
# # store the results to disk optionally
# store_results(storage_results, "cigre_ts")


# def plot_storage_results(storage_results):
#     n_res = len(storage_results.keys())
#     fig, axes = plt.subplots(n_res, 2)
#     if n_res == 1:
#         axes = [axes]
#     for i, (key, val) in enumerate(storage_results.items()):
#         res = val
#         axes[i][0].set_title("Storage {}".format(key))
#         el = res.loc[:, ["p_mw", "q_mvar", "soc_mwh"]]
#         el.plot(ax=axes[i][0])
#         axes[i][0].set_xlabel("time step")
#         axes[i][0].legend(loc=4)
#         axes[i][0].grid()
#         ax2 = axes[i][1]
#         patch = plt.plot([], [], ms=8, ls="--", mec=None, color="grey", label="{:s}".format("soc_percent"))
#         ax2.legend(handles=patch)
#         ax2.set_label("SOC percent")
#         res.loc[:, "soc_percent"].plot(ax=ax2, linestyle="--", color="grey")
#         ax2.grid()

#     plt.show()
# # plot the result
# plot_storage_results(storage_results)



# import pandapower as pp
# import pandapower.networks as nw
# net = nw.case5()
# pp.create_storage(net, 2, p_mw=1., max_e_mwh=.2, soc_percent=100., q_mvar=1.)
# pp.create_storage(net, 3, p_mw=1., max_e_mwh=.3, soc_percent=100., q_mvar=1.)

# # optimize for 24 time steps. At the end the SOC is 0%
# storage_results = pp.runpm_storage_opf(net, n_timesteps=24)
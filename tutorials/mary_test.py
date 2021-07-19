from pandapower.converter.powermodels.to_pm import convert_pp_to_pm

import json
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData

from timeseries.run_profile_cython.run_cython import profiles_from_ds

net = nw.create_cigre_network_mv("pv_wind")
pp.runpp(net)
# set some limits
min_vm_pu = 0.95
max_vm_pu = 1.05

net["bus"].loc[:, "min_vm_pu"] = 0.9
net["bus"].loc[:, "max_vm_pu"] = 1.1

net["line"].loc[:, "max_loading_percent"] = 100.

# close all switches
net.switch.loc[:, "closed"] = True
# add storage to bus 10
pp.create_storage(net, 10, p_mw=0.5, max_e_mwh=.2, soc_percent=0., q_mvar=0., controllable=True)

# net.load["controllable"] = False
# net.sgen["controllable"] = False


pp.runpp(net)


net.ext_grid.loc[:, "max_p_mw"] = 99999.
net.ext_grid.loc[:, "min_p_mw"] = -99999.

net.ext_grid.loc[:, "max_q_mvar"] = 99999.
net.ext_grid.loc[:, "min_q_mvar"] = -99999.


net["load"].loc[:, "type"] = "residential"
net.sgen.loc[:, "type"] = "pv"
net.sgen.loc[8, "type"] = "wind"

input_file = r"C:\Users\x230\pandapower\tutorials\cigre_timeseries_15min.json"
time_series = pd.read_json(input_file)


time_series.sort_index(inplace=True)

load_df = pd.DataFrame(time_series["residential"]).dot([net["load"]["p_mw"].values])

ConstControl(net, "load", "p_mw", element_index=net.load.index[net.load.type == "residential"],
             profile_name=net.load.index[net.load.type == "residential"], data_source=DFData(load_df))

pv_df = pd.DataFrame(time_series["pv"]).dot([net["sgen"].loc[:7, "p_mw"].values])

ConstControl(net, "sgen", "p_mw", element_index=net.sgen.index[net.sgen.type == "pv"],
             profile_name=net.sgen.index[net.sgen.type == "pv"], data_source=DFData(pv_df))

wind_df = pd.DataFrame(time_series["wind"]).dot([net["sgen"].loc[8, "p_mw"]])

ConstControl(net, "sgen", "p_mw", element_index=net.sgen.index[net.sgen.type == "wind"],
             profile_name=net.sgen.index[net.sgen.type == "wind"], data_source=DFData(time_series["wind"]))


storage_results = pp.runpm_storage_opf(net, n_timesteps=96, time_elapsed=0.25)

# time_series.sort_values(by="timestep", inplace=True)
# time_series["residential"].plot()
# time_series["pv"].plot()
# time_series["wind"].plot()
# plt.xlabel("time step")
# plt.xlabel("relative value")
# plt.legend()
# plt.grid()
# plt.show()
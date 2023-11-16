import matplotlib
import matplotlib.pyplot as plt

from copy import deepcopy
import pandapower as pp
import pandapower.plotting as plot
from pandapower import networks as nw
from pandapower.networks import example_simple
from IPython.display import display, Markdown

net = nw.mv_oberrhein()
pp.create_sgens(net, buses=net.bus.index[0:50], p_mw=1, type=["wye"]*50)

net.sgen.type.loc[0:50] = "WT"
bc = plot.create_bus_collection(net, net.bus.index, size=80, color="red", zorder=1)
# sc= plot.create_sgen_collection(net, sgens=net.sgen.index[0:50], size=80,orientation=0, picker=False)
#sc= plot.create_sgen_collection(net, sgens=net.sgen.index, size=80,orientation=1.4, picker=False, patch_type=["wye", "WT", "PV"])

# wind_turbine_sgen = net.sgen.loc[net.sgen.type == "WT"].index
# wt_c = plot.create_sgen_collection(net, sgens=wind_turbine_sgen, size=80, orientation=1.4, picker=False, patch_type="WT")


# pc_sgen = net.sgen.loc[net.sgen.type == "PV"].index
# pv_c = plot.create_sgen_collection(net, sgens=pc_sgen, size=80, orientation=1.4, picker=False, patch_type="PV")


#todo for simple plot
# orientation_list = [0, 180]
# for type in net.sgen.type.unique():
#     sgen_indices = net.sgen.loc[net.sgen.type == type].index
#     s_c = plot.create_sgen_collection(net, sgens=sgen_indices, size=80, orientation="insert angles from your method", picker=False, patch_type=type)
#     collections.append(s_c)

plot.draw_collections([bc], figsize=(8,6))

# pp.plotting.simple_plot(net, plot_loads=True, plot_gens=False, plot_sgens=True,orientation=0, load_size=2.5, gen_size=2.0,sgen_size=2,additional_patches=False);


# %% [markdown]
# # Tutorial: DERController
#

# %% [markdown]
# **This tutorial introduces the pandapower `DERController`**
#
# The `DERController` can be used to model Distributed Energy Resource (DER) capabilities for reactive power (Q) provision. The capabilities of reactive power provision may have differnent dependencies:
# + Q(P) capability: defines the reactive power capability depending on the DER active power generation
# + Q(V) capability: defines the reactive power capability depending on the voltage at the DER network connection point
#
# Different local control characteristics are included and can be applied via `DERController`, in particular:
# + const Q: fixed Q setpoint
# + fixed cosphi: fixed power factor
# + cosphi(P): power factor-watt function
# + Q(V): volt- var function
# + Voltage-Limitation function (in German Spannungsbegrenzungsfunktion)
#
# Common DER capabilities and local control characteristics are predefined in DERController according to German Grid Codes.

# %% [markdown]
# **First of all we import all needed packages and libraries:**

# %%
# Importing necessary packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.control.controller.DERController as DERModels
from pandapower.control.controller.DERController.der_control_plot import verify_pq_area, verify_qv_area

try:
    from cycler import cycler
    cycler_imported = True
except ImportError:
    cycler_imported = False

plt.rcParams['text.usetex'] = 'true'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb} \usepackage{amsmath} \usepackage{xfrac}'

# %% [markdown]
# **Then, we create a simple grid example:**

# %%
def simple_test_net():
    net = pp.create_empty_network()

    pp.create_buses(net, 2, vn_kv=20)

    pp.create_ext_grid(net, 0)
    pp.create_sgen(net, 1, p_mw=1., sn_mva=1.)

    pp.create_line(net, 0, 1, length_km=5.0, std_type="NAYY 4x50 SE")

    return net

# %% [markdown]
# Define some timeseries data to check multiple active power injection states.

# %%
ts_data = pd.DataFrame({"p_0": np.arange(0, -1.21, -0.05)})
data = pp.timeseries.DFData(ts_data)

# %% [markdown]
# **And finally, we define a function that is to plot the P/Q curves and Q/Vm curves from timeseries results:**

# %%
def plot_function(net, der_ctrl, ow):

    # --- Plot P/Q diagram of allowed area and resulting points
    fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4.5))
    pq_area = der_ctrl.pqv_area.pq_area
    verify_pq_area(pq_area, pq_area.name(), ax=ax1, circle_segment=28)
    S_N = net.sgen.sn_mva.at[0]
    xs = ow.output["res_sgen.q_mvar"]/S_N
    ys = ow.output["res_sgen.p_mw"]/S_N
    for x, y in zip(xs[0], ys[0]):
        ax1.scatter(x, y, label='Operation points', linestyle='-')
    ax1.set_ylim(0, max(np.max(ys.values), 1)+0.1)

    # --- Plot Q/V diagram of allowed area and resulting points
    qv_area = net.controller.object.at[0].pqv_area.qv_area
    verify_qv_area(qv_area, qv_area.name(), ax=ax2)
    xs = ow.output['res_bus.vm_pu'][1]
    ys = ow.output["res_sgen.q_mvar"]/S_N
    for x, y in zip(xs, ys[0]):
        ax2.scatter(x, y, label='Operation points')

    # --- remove legends of both parts and add a common legend
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(handles, ['$Q_{min}$','$Q_{max}$','$S_{max}$','Operation points'],
                ncol=4, framealpha=0, bbox_to_anchor=[0.7, 1.05])
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Q(P) Capability
# The Q(P) capability defines the reactive power capability of the DERs depending on their active power generation. The Q(P) capability of the DERs are usually defined in the national Requirements for Generators (RfG) or grid codes.
#
# In this simple example, a Q(P) cpability according to the German medium voltage RfG VDE AR-N 4110 is applied. We set a fixed reactive power setpoint `q=0.5` which is outside of the Q(P)-capability. Consequently, the `DERController` is needed to set the Q provision within its limits.
#
# The results of the following code, which gets plotted (first look at the plot left!), show that the reactive power is usually limited to `0.33 Sn` which is a power factor of `0.95 overexceited`. If the active power gets lower than `0.2 Sn`, also the reactive power gets reduced.

# %%
net = simple_test_net()

# define outputs and outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")

# define DERController
pqv_area = DERModels.PQVArea4120V2()
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=pqv_area, saturate_sn_mva=False,
    data_source=data, p_profile="p_0", profile_scale=-1)
pp.timeseries.run_timeseries(net, time_steps=range(0, 25)) # constQ: Attetion q is NOT in MVAR but relative to SnMVA

# plot results
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ## Inverter apparent power limitation
#
# The parameter `saturate_sn_mva` can be used to consider limitations to the apparent power of the DER.
#
# + saturate_sn_mva = False: active and reactive power will not be limited by the DER nominal apparent power (sn_mva)
#
# + saturate_sn_mva = True (default): active and reactive power will be limited by the DER nominal apparent power (sn_mva)
#
# `q_prio` defines which part of apparent power gets limited to achieve the apparent power limitation. The parameter is only active if `saturate_sn_mva is True`.
#
# + q_prio = True (default): DER active power is curtailed
#
# + q_prio = False: DER reactive power is curtailed
#

# %% [markdown]
# Let's repeat the last calculation with `saturate_sn_mva=True` and `q_prio=True` to see the difference to the last results (again focus on left plot): There are no dots outside the apparent power circle due to reduced active powers.

# %%
# change saturate_sn_mva using a new controller
net = simple_test_net()

# define outputs and outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")

# define DERController
pq_area = DERModels.PQVArea4110User()
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=True,
    data_source=data, p_profile="p_0", profile_scale=-1)

# run timeseries and plot results
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# If we consider `q_prio=False`, we can see in the left plot, that, for some operation points, q is reduced to zero to hold apparent power limits:

# %%
# change q_prio using a new controller
net = simple_test_net()

# define outputs and outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")

# define DERController
pq_area = DERModels.PQVArea4110User()
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=False,
    data_source=data, p_profile="p_0", profile_scale=-1)

# run timeseries and plot results
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ## Q(V) capability and voltage limitation function
#
# To visualize limiting the reactive power depending on the voltage of the RES connection point, we set the line length to 40 km. Then there is sufficient voltage rise to show that the controller reacts on over voltages via decreasing the inductive reactive power provision (overexcited behavior). That reduction is not used if the active power is low since then no overvoltages occur.
#
# Note that the apparent power limit is considered simultaneously.

# %%
# change line length to cause voltage issues
# run controller
net = simple_test_net()
net.line["length_km"] = 40

# define outputs and outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")

# define DERController
pq_area = DERModels.PQVArea4110User()
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.5), pqv_area=pq_area,
    data_source=data, p_profile="p_0", profile_scale=-1)
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ### Cosphi(P) curve
# ## "explination here"

# %%
net = simple_test_net()

# Create, add output and set outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")
pq_area = DERModels.PQVArea4110User()

# Cosphi(P) curve
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelCosphiPCurve(cosphi_p_curve={
        'p_points':(0, 0.5, 1), 'cosphi_points':(1, 1, 0.95)}),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=True,
    data_source=data, p_profile="p_0", profile_scale=-1)
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ### Cosphi(P)
# ## "explination here"

# %%
net = simple_test_net()

# Create, add output and set outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")
pq_area = DERModels.PQVArea4110User()

# Cosphi(P)
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelCosphiP(cosphi=0.95),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=True,
    data_source=data, p_profile="p_0", profile_scale=-1)

pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ### Q(U) curve
# ## "explination here"

# %%
net = simple_test_net()

# Create, add output and set outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")
pq_area = DERModels.PQVArea4110User()

# Q(U) curve
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelQV(qv_curve=DERModels.QVCurve(
        v_points_pu=(0, 0.90, 1.00, 1.02, 2),
        q_points=(0.312, 0.312, 0, -0.312, -0.312))),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=True, data_source=data,
    p_profile="p_0", profile_scale=-1)

pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ### Statcom  capability
# This capability does not depend on active power P, it can provide reactive power Q without active power.
# <br>
# min_q, max_q : reactive power limits, adjustable and user defined parameters.
#

# %%
net = simple_test_net()

# Create, add output and set outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")
pq_area = DERModels.PQVArea_STATCOM(min_q=-0.33, max_q=0.33)
net.sgen.sn_mva *= 8
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=True,
    data_source=data, p_profile="p_0", profile_scale=-8)
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %% [markdown]
# ### Fix PF capability

# %%
net = simple_test_net()

# Create, add output and set outputwriter
ow = pp.timeseries.OutputWriter(net)
ow.log_variable("res_sgen", "p_mw")
ow.log_variable("res_sgen", "q_mvar")
pq_area = DERModels.PQUArea_PFcapability(min_q=-0.33, max_q=0.33)
ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.2),
    pqv_area=pq_area, saturate_sn_mva=True, q_prio=False,
    data_source=data, p_profile="p_0", profile_scale=-1)
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, ctrl_sgen_new, ow)

# %%




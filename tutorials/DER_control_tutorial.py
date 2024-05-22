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
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.control.controller.DERController as DERModels
from pandapower.control.controller.DERController.der_control_plot import \
    verify_pq_area, verify_qv_area

plt.rcParams['text.usetex'] = 'true'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb} \usepackage{amsmath} \usepackage{xfrac}'

# %% [markdown]
# Then, we create a simple **grid example** and with identically connected sgens, but with different active power feed-in. This enables to **check multiple active power injection states**.

# %%
p_vals = np.arange(0, 1.21, 0.05)

def simple_test_net(p_vals):
    net = pp.create_empty_network()
    pp.create_buses(net, 1+len(p_vals), vn_kv=20)
    pp.create_ext_grid(net, 0)
    sgen_buses = list(range(1, len(p_vals)+1))
    pp.create_sgens(net, sgen_buses, p_mw=p_vals, sn_mva=1.)
    pp.create_lines(net, [0]*len(p_vals), sgen_buses, length_km=5.0, std_type="NAYY 4x50 SE")
    return net

# %% [markdown]
# **And finally, we define a function that is to plot the P/Q curves and Q/Vm curves from timeseries results:**

# %%
def plot_function(net, der_ctrl=None, pq_area=None, qv_area=None, ow=None):

    # --- get pq_area and qv_area from inputs
    if pq_area is None and der_ctrl is not None:
        pq_area = getattr(getattr(der_ctrl, "pqv_area", dict()), "pq_area", None)
    if qv_area is None and der_ctrl is not None:
        qv_area = getattr(getattr(der_ctrl, "pqv_area", dict()), "qv_area", None)

    # --- initiate figure and axes
    n_cols = int(pq_area is not None) + int(qv_area is not None)
    if n_cols == 0:
        print("No plot made since no pq_area nor qv_area available.")
        return
    fig1, axs = plt.subplots(ncols=n_cols, figsize=(5.25*n_cols+1.5, 4.5))
    if n_cols == 2:
        (ax1, ax2) = axs
    elif pq_area is not None:
        ax1 = axs
    else:
        ax2 = axs

    # --- Plot P/Q diagram of allowed area and resulting points
    if pq_area is not None:
        verify_pq_area(pq_area, pq_area.name(), ax=ax1, circle_segment=28)
        S_N = net.sgen.sn_mva.at[0]
        xs = (net.res_sgen.q_mvar if ow is None else ow.output["res_sgen.q_mvar"][0]) / S_N
        ys = (net.res_sgen.p_mw if ow is None else ow.output["res_sgen.p_mw"][0]) / S_N
        for x, y in zip(xs, ys):
            ax1.scatter(x, y, label='Operation points', linestyle='-')
        ax1.set_ylim(0, max(np.max(ys.values), 1)+0.1)

    # --- Plot Q/V diagram of allowed area and resulting points
    if qv_area is not None:
        verify_qv_area(qv_area, qv_area.name(), ax=ax2)
        xs = net.res_bus.vm_pu.iloc[1:] if ow is None else ow.output['res_bus.vm_pu'][1]
        ys = (net.res_sgen.q_mvar if ow is None else ow.output["res_sgen.q_mvar"][0]) / S_N
        for x, y in zip(xs, ys):
            ax2.scatter(x, y, label='Operation points')

    # --- if two parts plotted, remove legends of both parts and add a common legend
    if n_cols == 2:
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
net = simple_test_net(p_vals)

# define PQV area, Q model and DERController
pqv_area = DERModels.PQVArea4120V2()
q_model = DERModels.QModelConstQ(q=0.5)
ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=q_model, pqv_area=pqv_area, saturate_sn_mva=False)

# run power flows and plot
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

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
# Let's repeat the last calculation with `saturate_sn_mva=True` and `q_prio=True` (which are both the default values of the DERController) to see the difference to the last results (again focus on left plot): There are no dots outside the apparent power circle due to reduced active powers.

# %%
# change saturate_sn_mva using a new controller
net = simple_test_net(p_vals)

# define PQV area, Q model and DERController
pqv_area = DERModels.PQVArea4120V2()
q_model = DERModels.QModelConstQ(q=0.5)

ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=q_model, pqv_area=pqv_area, saturate_sn_mva=True, q_prio=True)

# run timeseries and plot results
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

# %% [markdown]
# If we consider `q_prio=False`, we can see in the left plot, that, for some operation points, q is reduced to zero to hold apparent power limits:

# %%
# change q_prio using a new controller
net = simple_test_net(p_vals)

# define PQV area, Q model and DERController
ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=DERModels.PQVArea4120V2(), saturate_sn_mva=True, q_prio=False)

# run timeseries and plot results
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

# %% [markdown]
# ## Q(V) capability and voltage limitation function
#
# To visualize limiting the reactive power depending on the voltage of the RES connection point, we set the line length to 40 km. Then there is sufficient voltage rise to show that the controller reacts on over voltages via decreasing the inductive reactive power provision (overexcited behavior). That reduction is not used if the active power is low since then no overvoltages occur.
#
# Note that the apparent power limit is considered simultaneously.

# %%
net = simple_test_net(p_vals)
# change line length to cause voltage issues
net.line["length_km"] = 40

# define PQV area, Q model and DERController
ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=DERModels.PQVArea4120V2())

# run power flows and plot
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

# %% [markdown]
# ### Cosphi(P) curve
# ## "explination TODO"

# %%
net = simple_test_net(p_vals)

# define PQV area, Q model and DERController
q_model=DERModels.QModelCosphiPCurve(cosphi_p_curve={
        'p_points':(0, 0.5, 1), 'cosphi_points':(1, 1, 0.95)})

ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=q_model, pqv_area=DERModels.PQVArea4120V2())

# run power flows and plot
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

# %% [markdown]
# ### Cosphi(P)
# ## "explination TODO"

# %%
net = simple_test_net(p_vals)

# define PQV area, Q model and DERController
q_model = DERModels.QModelCosphiP(cosphi=0.95)

ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=q_model, pqv_area=DERModels.PQVArea4120V2())

# run power flows and plot
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

# %% [markdown]
# ### Q(V) curve
# ## "explination TODO"

# %%
net = simple_test_net(p_vals)

# define PQV area, Q model and DERController
q_model = DERModels.QModelQV(qv_curve=DERModels.QVCurve(
    v_points_pu=(0, 0.90, 1.00, 1.02, 2),
    q_points=(0.312, 0.312, 0, -0.312, -0.312)))

ctrl_sgen_new = pp.control.DERController(
    net, gid=net.sgen.index, q_model=q_model, pqv_area=DERModels.PQVArea4120V2())

# run power flows and plot
pp.runpp(net, run_control=True)
plot_function(net, ctrl_sgen_new)

# %% [markdown]
# Multiple active power injection states can also be analyzed performing a timeseries calculation of one controller sgen. This is applied here for demonstration reasons. Furthermore, negative active power values are assumed in the timeseries data to show the functionality of profile_sacle.

# %%
def simple_test_net_for_timeseries():
    # this is basically the same grid as simple_test_net() but with only one line to one bus
    # with one sgen
    net = pp.create_empty_network()
    pp.create_buses(net, 2, vn_kv=20)
    pp.create_ext_grid(net, 0)
    pp.create_sgen(net, 1, p_mw=1., sn_mva=1.)
    pp.create_line(net, 0, 1, length_km=5.0, std_type="NAYY 4x50 SE")

    # create outputwriter and define variables to log
    ow = pp.timeseries.OutputWriter(net)
    ow.log_variable("res_sgen", "p_mw")
    ow.log_variable("res_sgen", "q_mvar")

    return net, ow

ts_data = pd.DataFrame({"p_0": -p_vals})
data = pp.timeseries.DFData(ts_data)

# %% [markdown]
# ### Statcom capability
# Statcom capability does not depend on active power P, it can provide reactive power Q without active power.
# `min_q` and `max_q` are reactive power limits, adjustable and user defined parameters.

# %%
net, ow = simple_test_net_for_timeseries()

# define PQV area, Q model and DERController
pq_area = DERModels.PQAreaSTATCOM(min_q=-0.33, max_q=0.33)

ctrl_sgen_new = pp.control.DERController(
    net, gid=0, q_model=DERModels.QModelConstQ(q=0.5),
    pqv_area=pq_area,
    data_source=data, p_profile="p_0", profile_scale=-1)

# run power flows and plot
pp.timeseries.run_timeseries(net, time_steps=range(0, 25))
plot_function(net, pq_area=pq_area, ow=ow)

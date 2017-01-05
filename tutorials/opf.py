
# coding: utf-8

# # pandapower Optimal Power Flow
# This is an introduction into the usage of the pandapower optimal power flow. It shows how to set the constraints and the cost factors into the pandapower element tables.
# 
# ## Example Network
# 
# We use the following four bus example network for this tutorial:
# 
# <img src="pics/example_opf.png" width="50%">
# 
# We first create this network in pandapower:

# In[57]:

import pandapower as pp
net = pp.create_empty_network()

#create buses
bus1 = pp.create_bus(net, vn_kv=220.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
bus4 = pp.create_bus(net, vn_kv=110.)

#create 220/110 kV transformer
pp.create_transformer(net, bus1, bus2, std_type="100 MVA 220/110 kV")

#create 110 kV lines
pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')

#create loads
pp.create_load(net, bus2, p_kw=60e3)
pp.create_load(net, bus3, p_kw=70e3)
pp.create_load(net, bus4, p_kw=10e3)

#create generators
eg = pp.create_ext_grid(net, bus1)
g0 = pp.create_gen(net, bus3, p_kw=-80*1e3, min_p_kw=0, max_p_kw=-80e3,vm_pu=1.01, controllable=True)
g1 = pp.create_gen(net, bus4, p_kw=-100*1e3, min_p_kw=0, max_p_kw=-100e3, vm_pu=1.01, controllable=True)


# ## Loss Minimization
# 
# We run an OPF:

# In[58]:

pp.runopp(net, verbose=True)


# let's check the results:

# In[59]:

net.res_ext_grid


# In[60]:

net.res_gen


# Since no individiual generation costs were specified, the OPF minimizes overall power generation, which is equal to a loss minimization in the network. The loads at buses 3 and 4 are supplied by generators at the same bus, the load at Bus 2 is provided by a combination of the other generators so that the power transmission leads to minimal losses.

# ## Individual Generator Costs
# 
# Let's now assign individual costs to each generator.
# 
# We assign a cost of 10 ct/kW for the external grid, 15 ct/kw for the generator g0 and 12 ct/kw for generator g1:

# In[61]:

net.ext_grid.loc[eg, "cost_per_kw"] = 0.10
net.gen.loc[g0, "cost_per_kw"] = 0.15
net.gen.loc[g1, "cost_per_kw"] = 0.12


# And now run an OPF:

# In[62]:

pp.runopp(net, verbose=True)


# We can see that all active power is provided by the external grid. This makes sense, because the external grid has the lowest cost of all generators and we did not define any constraints.
# 
# We define a simple function that calculates the summed costs:

# In[63]:

def calc_costs(net):
    cost_gen = (-net.res_gen.p_kw * net.gen.cost_per_kw).sum()
    cost_eg = (-net.res_ext_grid.p_kw * net.ext_grid.cost_per_kw).sum()
    return (cost_gen + cost_eg) * 1e-3


# And calculate the dispatch costs:

# In[64]:

calc_costs(net)


# ### Transformer Constraint
# 
# Since all active power comes from the external grid and subsequently flows through the transformer, the transformer is overloaded with a loading of about 145%:

# In[65]:

net.res_trafo


# We now limit the transformer loading to 50%:

# In[66]:

net.trafo["max_loading_percent"] = 50


# (the max_loading_percent parameter can also be specified directly when creating the transformer)
# and run the OPF:

# In[67]:

pp.runopp(net)


# We can see that the transformer complies with the maximum loading:

# In[68]:

net.res_trafo


# And power generation is now split between the external grid and generator 1 (which is the second cheapest generation unit):

# In[69]:

net.res_ext_grid


# In[70]:

net.res_gen


# This comes of course with an increase in dispatch costs:

# In[71]:

calc_costs(net)


# ### Line Loading Constraints

# Wen now look at the line loadings:

# In[72]:

net.res_line


# and run the OPF with a 50% loading constraint:

# In[73]:

net.line["max_loading_percent"] = 50
pp.runopp(net, verbose=True)


# Now the line loading constraint is complied with:

# In[74]:

net.res_line


# And all generators are involved in supplying the loads:

# In[75]:

net.res_ext_grid


# In[76]:

net.res_gen


# This of course comes with a once again rising dispatch cost:

# In[77]:

calc_costs(net)


# ### Voltage Constraints
# 
# Finally, we have a look at the bus voltage:

# In[78]:

net.res_bus


# and constrain it:

# In[79]:

net.bus["min_vm_pu"] = 1.0
net.bus["max_vm_pu"] = 1.02
pp.runopp(net)


# We can see that all voltages are within the voltage band:

# In[80]:

net.res_bus


# And all generators are once again involved in supplying the loads:

# In[81]:

net.res_ext_grid


# In[82]:

net.res_gen


# This of course comes once again with rising dispatch costs:

# In[83]:

calc_costs(net)


# ## DC OPF
# 
# pandapower also provides the possibility of running a DC Optimal Power Flow:

# In[84]:

pp.rundcopp(net)


# Since voltage magnitudes are not included in the DC power flow formulation, voltage constraints canot be considered in the DC OPF:

# In[85]:

net.res_bus


# Line and transformer loading limits are however complied with:

# In[86]:

net.res_line


# In[87]:

net.res_trafo


# As are generator limits:

# In[88]:

net.gen


# In[89]:

net.res_gen


# The cost function is the same for the linearized OPF as for the non-linear one:

# In[91]:

calc_costs(net)


# ## Minimizing Active Power Curtailment
# 
# Now we assume that the generators are renewable energy sources, and we want to feed in as much of the energy they can provide as possible without violating any constraints.
# 
# We assign negative costs to the generators and costs of zero for the external grid.

# In[96]:

net.ext_grid.cost_per_kw = 0
net.gen.cost_per_kw = -1e-5
pp.runopp(net, verbose=True)


# Because of the negative costs, the OPF now maximizes power generation at the generators:

# In[97]:

net.res_gen


# In[98]:

net.res_ext_grid


# In[99]:

net.res_bus


# In[100]:

net.res_trafo


# In[101]:

net.res_line


# Obviously the voltage profile was the limiting factor for the generator feed-in. If we relax this constraint a little bit:

# In[102]:

net.bus.max_vm_pu = 1.05
pp.runopp(net)


# We see an increased feed-in of the generators:

# In[103]:

net.res_gen


# In[104]:

net.res_bus


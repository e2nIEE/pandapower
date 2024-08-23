import pandapower as pp
from pandapower.control.controller.shunt_control import DiscreteShuntController, ContinuousShuntController

net = pp.create_empty_network()
b = pp.create_buses(net, 2, 110)

pp.create_ext_grid(net, b[0])
pp.create_line_from_parameters(net, from_bus=b[0], to_bus=b[1], length_km=50, r_ohm_per_km=0.1021, x_ohm_per_km=0.1570796,
                               max_i_ka=0.461, c_nf_per_km=130)
pp.create_shunt(net, bus=b[1], q_mvar=-50, p_mw=0, step=1, max_step=5)

pp.runpp(net)

print(net.res_bus)


# DiscreteShuntController(net, shunt_index=0, bus_index=1, vm_set_pu=1.0, tol=1e-2)
ContinuousShuntController(net, shunt_index=0, bus_index=1, vm_set_pu=1.08, tol=1e-2)


pp.runpp(net, run_control=True)
print(net.res_bus)

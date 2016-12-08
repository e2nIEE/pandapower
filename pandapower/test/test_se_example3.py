from pandapower.estimation.state_estimation import state_estimation, create_measurement
import pandapower as pp
import numpy as np
import pytest


def test_3bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_ext_grid(net, 0)
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0, imax_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0, imax_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0, imax_ka=1)

    create_measurement(net, "pline_kw", 0, -0.0011e3, 0.01e3, line=0) # p12
    create_measurement(net, "qline_kvar", 0, 0.024e3, 0.01e3, line=0) # q12

    create_measurement(net, "pbus_kw", 2, 0.018e3, 0.01e3) # p3
    create_measurement(net, "qbus_kvar", 2, -0.1e3, 0.01e3) # q3

    create_measurement(net, "vbus_pu", 0, 1.08, 0.05) # u1
    create_measurement(net, "vbus_pu", 2, 1.015, 0.05) # u3

    v_start = np.array([1.0, 1.0, 1.0])
    delta_start = np.array([0., 0., 0.0])

    # 2. Do state estimation
    wls = state_estimation()
    wls.set_grid(net)
    success = wls.estimate(v_start, delta_start)
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    # 3. Print result
#    print("Result:")
#    print("V [p.u.]:")
#    print(v_result)
#    print(u"delta [°]:")
#    print(delta_result)

    target_v = np.array([ 1.0627,  1.0589,  1.0317])
    diff_v = target_v - v_result
    target_delta = np.array([ 0.,      0.8677, 3.1381])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)

if __name__ == '__main__':
    pytest.main(['-xs', __file__])

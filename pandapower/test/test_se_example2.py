from pandapower.estimation.state_estimation import state_estimation, create_measurement
import pandapower as pp
import numpy as np
import pytest


def test_2bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_ext_grid(net, 0)
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1,x_ohm_per_km=0.5, c_nf_per_km=0, imax_ka=1)

    create_measurement(net, "pline_kw", 0, 0.0111e3, 0.05e3, line=0)  # p12
    create_measurement(net, "qline_kvar", 0, 0.06e3, 0.05e3, line=0)  # q12

    create_measurement(net, "vbus_pu", 0, 1.019, 0.01)  # u1
    create_measurement(net, "vbus_pu", 1, 1.04, 0.01)   # u2

    v_start = np.array([1.0, 1.0])
    delta_start = np.array([0., 0.])

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

    target_v = np.array([[1.02083378, 1.03812899]])
    diff_v = target_v - v_result
    target_delta = np.array([[0.0, 3.11356604]])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-6)
    assert (np.nanmax(abs(diff_delta)) < 1e-6)

if __name__ == '__main__':
    pytest.main(['-xs', __file__])

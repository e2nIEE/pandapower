__author__ = 'menke'
from pandapower.estimation import estimate
import pandapower as pp
import numpy as np
import pytest
import pandapower.networks as nw


def test_2bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_ext_grid(net, 0)
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1,x_ohm_per_km=0.5, c_nf_per_km=0,
                                   imax_ka=1)

    pp.create_measurement(net, "p", "line", 0.0111e3, 0.05e3, 0, element=0)  # p12
    pp.create_measurement(net, "q", "line", 0.06e3, 0.05e3, 0, element=0)  # q12

    pp.create_measurement(net, "v", "bus", 1.019, 0.01, bus=0)  # u1
    pp.create_measurement(net, "v", "bus", 1.04, 0.01, bus=1)   # u2

    # 2. Do state estimation
    success = estimate(net, init='flat')

    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[1.02083378, 1.03812899]])
    diff_v = target_v - v_result
    target_delta = np.array([[0.0, 3.11356604]])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-6)
    assert (np.nanmax(abs(diff_delta)) < 1e-6)


def test_3bus():
    # Test case from book "Power System State Estimation", A. Abur, A. G. Exposito, p. 20ff.
    # S_ref = 1 MVA (PP standard)
    # V_ref = 1 kV
    # Z_ref = 1 Ohm

    # The example only had per unit values, but Pandapower expects kV, MVA, kW, kVar
    # Measurements should be in kW/kVar/A - Voltage in p.u.

    # 1. Create network
    net = pp.create_empty_network()
    pp.create_ext_grid(net, 0)
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_bus(net, name="bus4", vn_kv=1., in_service=0)  # out-of-service bus test
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                   imax_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                   imax_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                   imax_ka=1)

    pp.create_measurement(net, "v", "bus", 1.006, .004, bus=0)  # V at bus 1
    pp.create_measurement(net, "v", "bus", .968, .004, bus=1)   # V at bus 2

    pp.create_measurement(net, "p", "bus", -501, 10, 1)  # P at bus 2
    pp.create_measurement(net, "q", "bus", -286, 10, 1)  # Q at bus 2

    pp.create_measurement(net, "p", "line", 888, 8, 0, 0)   # Pline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "p", "line", 1173, 8, 0, 1)  # Pline (bus 1 -> bus 3) at bus 1
    pp.create_measurement(net, "q", "line", 568, 8, 0, 0)   # Qline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "q", "line", 663, 8, 0, 1)   # Qline (bus 1 -> bus 3) at bus 1

    # 2. Do state estimation
    success = estimate(net, init='flat')
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[0.9996, 0.9741, 0.9438, np.nan]])
    diff_v = target_v - v_result
    target_delta = np.array([[0., -1.2475, -2.7457, np.nan]])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_3bus_trafo():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_ext_grid(net, bus=3)
    pp.create_bus(net, name="bus1", vn_kv=10.)
    pp.create_bus(net, name="bus2", vn_kv=10.)
    pp.create_bus(net, name="bus3", vn_kv=10.)
    pp.create_bus(net, name="bus4", vn_kv=110.)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                   imax_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                   imax_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                   imax_ka=1)
    pp.create_transformer(net, 3, 0, std_type="25 MVA 110/10 kV")

    pp.create_measurement(net, "v", "bus", 1.006, .004, bus=0)  # V at bus 1
    pp.create_measurement(net, "v", "bus", .968, .004, bus=1)   # V at bus 2

    pp.create_measurement(net, "p", "bus", -501, 10, 1)  # P at bus 2
    pp.create_measurement(net, "q", "bus", -286, 10, 1)  # Q at bus 2

    pp.create_measurement(net, "p", "line", 888, 8, 0, 0)   # Pline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "p", "line", 1173, 8, 0, 1)  # Pline (bus 1 -> bus 3) at bus 1
    pp.create_measurement(net, "q", "line", 568, 8, 0, 0)   # Qline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "q", "line", 663, 8, 0, 1)   # Qline (bus 1 -> bus 3) at bus 1

    pp.create_measurement(net, "p", "transformer", 2067, 10, bus=3, element=0)  # transformer meas.
    pp.create_measurement(net, "q", "transformer", 1228, 10, bus=3, element=0)  # at hv side

    # 2. Do state estimation
    success = estimate(net, init='flat')
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([0.98712592, 0.98686807, 0.98654891, 0.99228971])
    diff_v = target_v - v_result
    target_delta = np.array([-0.47745512, -0.4899255, -0.50404252,  0.])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-6)
    assert (np.nanmax(abs(diff_delta)) < 1e-6)


def test_3bus_2():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_ext_grid(net, 0)
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0,
                                   imax_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0,
                                   imax_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0,
                                   imax_ka=1)

    pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0)  # p12
    pp.create_measurement(net, "q", "line", 0.024e3, 0.01e3, bus=0, element=0)    # q12

    pp.create_measurement(net, "p", "bus", 0.018e3, 0.01e3, bus=2)  # p3
    pp.create_measurement(net, "q", "bus", -0.1e3, 0.01e3, bus=2)   # q3

    pp.create_measurement(net, "v", "bus", 1.08, 0.05, 0)   # u1
    pp.create_measurement(net, "v", "bus", 1.015, 0.05, 2)  # u3

    # 2. Do state estimation
    success = estimate(net, init='flat')
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([ 1.0627,  1.0589,  1.0317])
    diff_v = target_v - v_result
    target_delta = np.array([ 0.,      0.8677, 3.1381])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_cigre_network():
    # 1. create network
    # test the mv ring network with all available voltage measurements and bus powers
    # test if switches and transformer will work correctly with the state estimation
    np.random.seed(123456)
    net = nw.create_cigre_network_mv(with_der=False)
    pp.runpp(net)

    for bus, row in net.res_bus.iterrows():
        pp.create_measurement(net, "v", "bus", row.vm_pu * r(0.01), 0.01, bus)
        # if np.random.randint(0, 4) == 0:
        #    continue
        pp.create_measurement(net, "p", "bus", -row.p_kw * r(), max(1.0, abs(0.03 * row.p_kw)),
                              bus)
        pp.create_measurement(net, "q", "bus", -row.q_kvar * r(), max(1.0, abs(0.03 * row.q_kvar)),
                              bus)

    # 2. Do state estimation
    success = estimate(net, init='slack')
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = net.res_bus.vm_pu.values
    diff_v = target_v - v_result
    target_delta = net.res_bus.va_degree.values
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 0.005)
    assert (np.nanmax(abs(diff_delta)) < 0.2)


def test_check_existing():
    net = pp.create_empty_network()
    pp.create_bus(net, 10.)
    pp.create_bus(net, 10.)
    pp.create_line(net, 0, 1, 0.5, std_type="149-AL1/24-ST1A 10.0")
    m1 = pp.create_measurement(net, "v", "bus", 1.006, .004, 0)  # V at bus 1
    m2 = pp.create_measurement(net, "v", "bus", 1.006, .004, 0)  # V at bus 1

    assert m1 == m2
    assert len(net.measurement) == 1
    m3 = pp.create_measurement(net, "v", "bus", 1.006, .004, 0, check_existing=False) # V at bus 1
    assert m3 != m2
    assert len(net.measurement) == 2

    m4 = pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0,
                               check_existing=True)
    m5 = pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0,
                               check_existing=True)
    assert m4 == m5

    m6 = pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0,
                               check_existing=False)
    assert m5 != m6


def r(v=0.03):
    return np.random.normal(1.0, v)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

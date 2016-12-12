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

    pp.create_measurement(net, "pline_kw", 0, 0.0111e3, 0.05e3, line=0)  # p12
    pp.create_measurement(net, "qline_kvar", 0, 0.06e3, 0.05e3, line=0)  # q12

    pp.create_measurement(net, "vbus_pu", 0, 1.019, 0.01)  # u1
    pp.create_measurement(net, "vbus_pu", 1, 1.04, 0.01)   # u2

    v_start = np.array([1.0, 1.0])
    delta_start = np.array([0., 0.])

    # 2. Do state estimation
    success = estimate(net, v_start, delta_start)

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

    pp.create_measurement(net, "vbus_pu", 0, 1.006, .004)  # V at bus 1
    pp.create_measurement(net, "vbus_pu", 1, .968, .004)   # V at bus 2

    pp.create_measurement(net, "pbus_kw", 1, -501, 10)    # P at bus 2
    pp.create_measurement(net, "qbus_kvar", 1, -286, 10)  # Q at bus 2

    pp.create_measurement(net, "pline_kw", 0, 888, 8, line=0)    # Pline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "pline_kw", 0, 1173, 8, line=1)   # Pline (bus 1 -> bus 3) at bus 1
    pp.create_measurement(net, "qline_kvar", 0, 568, 8, line=0)  # Qline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "qline_kvar", 0, 663, 8, line=1)  # Qline (bus 1 -> bus 3) at bus 1

    v_start = np.array([1.0, 1.0, 1.0, 0.])
    delta_start = np.array([0., 0., 0., 0.])

    # 2. Do state estimation
    success = estimate(net, v_start, delta_start)
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[0.9996, 0.9741, 0.9438, np.nan]])
    diff_v = target_v - v_result
    target_delta = np.array([[ 0., -1.2475, -2.7457, np.nan]])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


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

    pp.create_measurement(net, "pline_kw", 0, -0.0011e3, 0.01e3, line=0) # p12
    pp.create_measurement(net, "qline_kvar", 0, 0.024e3, 0.01e3, line=0) # q12

    pp.create_measurement(net, "pbus_kw", 2, 0.018e3, 0.01e3) # p3
    pp.create_measurement(net, "qbus_kvar", 2, -0.1e3, 0.01e3) # q3

    pp.create_measurement(net, "vbus_pu", 0, 1.08, 0.05) # u1
    pp.create_measurement(net, "vbus_pu", 2, 1.015, 0.05) # u3

    v_start = np.array([1.0, 1.0, 1.0])
    delta_start = np.array([0., 0., 0.0])

    # 2. Do state estimation
    success = estimate(net, v_start, delta_start)
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([ 1.0627,  1.0589,  1.0317])
    diff_v = target_v - v_result
    target_delta = np.array([ 0.,      0.8677, 3.1381])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_ring_network():
    # 1. create network
    # test the mv ring network with all available voltage measurements and bus powers
    # test if switches and transformer will work correctly with the state estimation
    np.random.seed(123456)
    net = nw.create_cigre_network_mv(with_der=False)
    pp.runpp(net)

    for bus, row in net.res_bus.iterrows():
        pp.create_measurement(net, "vbus_pu", bus, row.vm_pu * r(0.01), 0.01)
        # if np.random.randint(0, 4) == 0:
        #    continue
        pp.create_measurement(net, "pbus_kw", bus, -row.p_kw * r(), max(1.0, abs(0.03 * row.p_kw)))
        pp.create_measurement(net, "qbus_kvar", bus, -row.q_kvar * r(), max(1.0,
                                                                         abs(0.03 * row.q_kvar)))

    v_start = np.ones(net.bus.shape[0]) * 1.01
    delta_start = np.zeros_like(v_start)

    # 2. Do state estimation
    success = estimate(net, v_start, delta_start)
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
    m1 = pp.create_measurement(net, "vbus_pu", 0, 1.006, .004) # V at bus 1
    m2 = pp.create_measurement(net, "vbus_pu", 0, 1.006, .004) # V at bus 1

    assert m1 == m2
    assert len(net.measurement) == 1
    m3 = pp.create_measurement(net, "vbus_pu", 0, 1.006, .004, check_existing=False) # V at bus 1
    assert m3 != m2
    assert len(net.measurement) == 2

    m4 = pp.create_measurement(net, "pline_kw", 0, -0.0011e3, 0.01e3, line=0, check_existing=True) # p12
    m5 = pp.create_measurement(net, "pline_kw", 0, -0.0011e3, 0.01e3, line=0, check_existing=True) # p12
    assert m4 == m5

    m6 = pp.create_measurement(net, "pline_kw", 0, -0.0011e3, 0.01e3, line=0, check_existing=False) # p12
    assert m5 != m6


def r(v=0.03):
    return np.random.normal(1.0, v)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])


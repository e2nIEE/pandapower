from pandapower.estimation.state_estimation import state_estimation, create_measurement
import pandapower as pp
import numpy as np
import networks as nw
import pytest


def r(v=0.03):
    return np.random.normal(1.0, v)


def test_ring_network():
    # 1. create network
    # test the mv ring network with all available voltage measurements and bus powers
    # test if switches and transformer will work correctly with the state estimation
    np.random.seed(123456)
    net = nw.mv_network("ring")
    pp.runpp(net)

    for bus, row in net.res_bus.iterrows():
        create_measurement(net, "vbus_pu", bus, row.vm_pu * r(0.01), 0.01)
        # if np.random.randint(0, 4) == 0:
        #    continue
        create_measurement(net, "pbus_kw", bus, -row.p_kw * r(), max(1.0, abs(0.03 * row.p_kw)))
        create_measurement(net, "qbus_kvar", bus, -row.q_kvar * r(), max(1.0, abs(0.03 * row.q_kvar)))

    v_start = np.ones(net.bus.shape[0]) * 1.01
    delta_start = np.zeros_like(v_start)

    # 2. Do state estimation
    wls = state_estimation()
    wls.set_grid(net)
    success = wls.estimate(v_start, delta_start)
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    # 3. Print result
    print("Result:")
    print("V [p.u.]:")
    print(v_result)
    print(u"delta [°]:")
    print(delta_result)

    target_v = net.res_bus.vm_pu.values
    diff_v = target_v - v_result
    target_delta = net.res_bus.va_degree.values
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-2)
    assert (np.nanmax(abs(diff_delta)) < 0.45)

if __name__ == '__main__':
    pytest.main(['-xs', __file__])

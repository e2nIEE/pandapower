import pytest

import pandapower as pp
import numpy as np

@pytest.mark.xfail
def test_3point_pwl():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_mw=0.1, q_mvar=0, controllable=True, min_p_mw=0.1, max_p_mw=0.15,
                   max_q_mvar=0.05, min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # creating a pwl cost function that actually is realistic: The absolute value of the reactive power has costs.
    pp.create_pwl_cost(net, 0, "sgen", [[-50, 0, -1.5], [0, 50, 1.5]], power_type="q")

    pp.runopp(net)

    # The reactive power should be at zero to minimze the costs.
    assert np.isclose(net.res_sgen.q_mvar.values, 0, atol=1e-4)
    assert np.isclose(net.res_cost, abs(net.res_sgen.q_mvar.values)*1.5, atol=1e-4)
    #TODO costs seem to be assigned to ext_grid, not to sgen (net.res_ext_grid.q_mvar*1.5=net.res_cost)
    #     They are however correctly assigned in the gencost array, this seems to be a bug in PYPOWER

    net.sgen.min_q_mvar = 0.05
    net.sgen.max_q_mvar = 0.1
    pp.runopp(net)
    assert np.isclose(net.res_sgen.q_mvar.values, 0.05, atol=1e-4)
    assert np.isclose(net.res_cost, abs(net.res_sgen.q_mvar.values)*1.5, atol=1e-4)

if __name__ == "__main__":
    pytest.main(['-s', __file__])

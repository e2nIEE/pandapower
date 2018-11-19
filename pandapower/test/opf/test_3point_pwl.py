import pytest

import pandapower as pp


def test_3point_pwl():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_kw=-100, q_kvar=0, controllable=True, min_p_kw=100, max_p_kw=100.5, max_q_kvar=50,
                   min_q_kvar=0)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
#    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array(
#        [[-100, 1.5], [0, 0], [100, 1], ]), type="q")

    pp.create_pwl_cost(net, 0, "sgen", [(-50, 0, 1.5), (0, 50, 1.5)], power_type="q")

    # creating a pwl cost function that actually is realistic: The absolute value of the reactive power has costs.

    pp.runopp(net, verbose=False)

    # The reactive power should be at zero to minimze the costs.
    #TODO: it is actually not?
    assert abs(net.res_sgen.q_kvar.values ) < 1e-3

if __name__ == "__main__":
    pytest.main(['-s', __file__])

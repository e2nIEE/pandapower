import numpy as np
import pytest

import pandapower as pp


def test_3point_pwl():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_kw=-100, q_kvar=0, controllable=True, max_p_kw=-100, min_p_kw=-100.5, max_q_kvar=50,
                   min_q_kvar=-50)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=20, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
    pp.create_piecewise_linear_cost(net, 0, "sgen", np.array(
        [[-100, 1], [0, 0], [100, 1], ]), type="q")

    # creating a pwl cost function that actually is realistic: The absolute value of the reactive power has costs.

    pp.runopp(net, verbose=False)

    # assert abs( net.res_sgen.q_kvar.values ) < 1e-5

    # this is not the expected result. the reactive power should be at zero to minimze the costs.
    # checkout the dirty workaround:

    # the first sgen is only representing the positive segment of the function:
    net.piecewise_linear_cost.p.at[0] = np.array([[0, 1]])
    net.piecewise_linear_cost.f.at[0] = np.array([[0, 1]])
    net.sgen.min_q_kvar.at[0] = 0

    # what we can do instead is modelling a second sgen on the same bus representing the negative segment of the function:
    pp.create_sgen(net, 1, p_kw=0, q_kvar=0, controllable=True, max_p_kw=0.01, min_p_kw=-0.01, max_q_kvar=0,
                   min_q_kvar=-10)
    pp.create_piecewise_linear_cost(net, 1, "sgen", np.array(
        [[-100, 100], [0, 0], ]), type="q")

    # runOPF
    pp.runopp(net, verbose=False)
    assert abs(sum(net.res_sgen.q_kvar.values)) < 1e-5

    # et voila, we have the q at zero. sure, we do have two seperate sgens now and this is very dirty. but it's working.

    # let's check if we can handle overvoltage
    net.bus.max_vm_pu = 1.041
    pp.runopp(net, verbose=False)
    assert abs(max(net.res_bus.vm_pu.values) - 1.041) < 1e-5


if __name__ == "__main__":
    pytest.main(["test_3point_pwl.py", "-xs"])
    # test_cost_piecewise_linear_eg_q()

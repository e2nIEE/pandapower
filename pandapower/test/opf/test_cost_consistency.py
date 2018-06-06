import pandapower as pp
import pytest
from numpy import array

@pytest.fixture()
def base_net():
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=10)
    pp.create_bus(net, vn_kv=10)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_kw=200, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.runpp(net)
    return net

def test_contingency_sgen(base_net):

    net = base_net
    pp.create_sgen(net, 1, p_kw=-100, q_kvar =0, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                  min_q_kvar=-50)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    #-------------------------------------------
    #    p_min_kw      /|
    #                 / |
    #                /  |

    pp.create_piecewise_linear_cost(net, 0, "sgen", array([[net.sgen.min_p_kw.at[0], net.sgen.min_p_kw.at[0]], [0, 0]]))
    pp.runopp(net)


    assert abs(net.res_cost - net.res_sgen.p_kw.at[0]) < 1e-5
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    #-------------------------------------------
    #    p_min_kw       |\
    #                   | \
    #                   |  \
    net.piecewise_linear_cost.f.at[0] *= -1
    pp.runopp(net)

    assert abs(net.res_cost - net.res_sgen.p_kw.at[0]*-1) < 1e-5

    try:
        net.piecewise_linear_cost = net.piecewise_linear_cost.drop(index=0)
    except:
        net.piecewise_linear_cost = net.piecewise_linear_cost.drop(0)

    # first using a positive slope as in the case above
    pp.create_polynomial_cost(net, 0, "sgen", array([1, 0]))
    pp.runopp(net)
    assert abs(net.res_cost - net.res_sgen.p_kw.at[0]) < 1e-5

    # negative slope as in the case above
    net.polynomial_cost.c.at[0] *= -1
    pp.runopp(net)

    assert abs(net.res_cost - net.res_sgen.p_kw.at[0]*-1) < 1e-5


def test_contingency_load(base_net):
    net = base_net
    pp.create_load(net, 1, p_kw=-100, q_kvar=0, controllable=True, max_p_kw=150, min_p_kw=5,
                   max_q_kvar=50,
                   min_q_kvar=-50)
    # pwl costs
    # minimze the load by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    # -------------------------------------------
    #    p_min_kw      /|
    #                 / |
    #                /  |

    pp.create_piecewise_linear_cost(net, 1, "load", array(
        [[0, 0],[net.load.max_p_kw.at[1], net.load.max_p_kw.at[1]]]))
    pp.runopp(net)

    assert abs(net.res_cost - net.res_load.p_kw.at[1]) < 1e-5
    # maximize the load in by using a negative cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    # -------------------------------------------
    #    p_min_kw       |\
    #                   | \
    #                   |  \
    net.piecewise_linear_cost.f.at[0] *= -1
    pp.runopp(net)

    assert abs(net.res_cost - net.res_load.p_kw.at[1] * -1) < 1e-5

    # poly costs
    try:
        net.piecewise_linear_cost = net.piecewise_linear_cost.drop(index=0)
    except:
        # legacy fix
        net.piecewise_linear_cost = net.piecewise_linear_cost.drop(0)


    # first using a positive slope as in the case above
    pp.create_polynomial_cost(net, 1, "load", array([1, 0]))
    pp.runopp(net)
    assert abs(net.res_cost - net.res_load.p_kw.at[1]) < 1e-5

    # negative slope as in the case above
    net.polynomial_cost.c.at[0] *= -1
    pp.runopp(net)

    assert abs(net.res_cost - net.res_load.p_kw.at[1]*-1) < 1e-5


def test_contingency_gen(base_net):

    net = base_net
    pp.create_gen(net, 1, p_kw=-100, vm_pu = 1.05, controllable=True, max_p_kw=-5, min_p_kw=-150, max_q_kvar=50,
                  min_q_kvar=-50)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    #-------------------------------------------
    #    p_min_kw      /|
    #                 / |
    #                /  |

    pp.create_piecewise_linear_cost(net, 0, "gen", array([[net.gen.min_p_kw.at[0], net.gen.min_p_kw.at[0]], [0, 0]]))
    pp.runopp(net)


    assert abs(net.res_cost - net.res_gen.p_kw.at[0]) < 1e-5
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    #-------------------------------------------
    #    p_min_kw       |\
    #                   | \
    #                   |  \
    net.piecewise_linear_cost.f.at[0] *= -1
    pp.runopp(net)

    assert abs(net.res_cost - net.res_gen.p_kw.at[0]*-1) < 1e-5

    try:
        net.piecewise_linear_cost = net.piecewise_linear_cost.drop(index=0)
    except:
        # legacy fix
        net.piecewise_linear_cost = net.piecewise_linear_cost.drop(0)

    # first using a positive slope as in the case above
    pp.create_polynomial_cost(net, 0, "gen", array([1, 0]))
    pp.runopp(net)
    assert abs(net.res_cost - net.res_gen.p_kw.at[0]) < 1e-5

    # negative slope as in the case above
    net.polynomial_cost.c.at[0] *= -1
    pp.runopp(net)

    assert abs(net.res_cost - net.res_gen.p_kw.at[0]*-1) < 1e-5

if __name__ == "__main__":
    # net = base_net()
    # test_contingency_gen(net)
    pytest.main(['-s', __file__])

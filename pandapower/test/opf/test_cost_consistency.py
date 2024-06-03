import pandapower as pp
import pytest
from numpy import array, isclose
import pandas as pd

@pytest.fixture()
def base_net():
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=10)
    pp.create_bus(net, vn_kv=10)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.2, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.runpp(net)
    return net

def test_contingency_sgen(base_net):
    net = base_net
    pp.create_sgen(net, 1, p_mw=0.1, q_mvar=0, controllable=True, min_p_mw=0.005, max_p_mw=0.150,
                   max_q_mvar=0.05, min_q_mvar=-0.05)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    #-------------------------------------------
    #    p_min_mw      /|
    #                 / |
    #                /  |

    pwl = pp.create_pwl_cost(net, 0, "sgen", [[0, net.sgen.max_p_mw.at[0], 1]])
    pp.runopp(net)


    assert isclose(net.res_cost, net.res_sgen.p_mw.at[0], atol=1e-3)
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    #-------------------------------------------
    #    p_min_mw       |\
    #                   | \
    #                   |  \

    net.pwl_cost.points.loc[pwl] = [(0, net.sgen.max_p_mw.at[0], -1)]
    pp.runopp(net)

    assert isclose(net.res_cost, -net.res_sgen.p_mw.at[0], atol=1e-4)

    net.pwl_cost = net.pwl_cost.drop(0)

    # first using a positive slope as in the case above
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=1.)
    pp.runopp(net)
    assert isclose(net.res_cost, net.res_sgen.p_mw.at[0], atol=1e-3)

    # negative slope as in the case above
    net.poly_cost.cp1_eur_per_mw.at[0] *= -1
    pp.runopp(net)

    assert isclose(net.res_cost, -net.res_sgen.p_mw.at[0], atol=1e-4)


def test_contingency_load(base_net):
    net = base_net
    pp.create_gen(net, 1, p_mw=0.1, vm_pu = 1.05, controllable=True, min_p_mw=0.005, max_p_mw=0.150,
                  max_q_mvar=0.05, min_q_mvar=-0.05)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    #-------------------------------------------
    #    p_min_mw      /|
    #                 / |
    #                /  |

    pp.create_pwl_cost(net, 0, "gen",[[0, net.gen.max_p_mw.at[0], 1]])
    pp.runopp(net)


    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    #-------------------------------------------
    #    p_min_mw       |\
    #                   | \
    #                   |  \
    net.pwl_cost.points.iloc[0] = [(0, net.gen.max_p_mw.at[0], -1)]
    pp.runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)

    net.pwl_cost = net.pwl_cost.drop(0)

    # first using a positive slope as in the case above
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    pp.runopp(net)
    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)

    # negative slope as in the case above
    net.poly_cost.cp1_eur_per_mw.at[0] *= -1
    pp.runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)


def test_contingency_gen(base_net):
    net = base_net
    pp.create_gen(net, 1, p_mw=0.1, vm_pu = 1.05, controllable=True, min_p_mw=0.005, max_p_mw=0.150,
                  max_q_mvar=0.05, min_q_mvar=-0.05)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    #-------------------------------------------
    #    p_min_mw      /|
    #                 / |
    #                /  |

    pp.create_pwl_cost(net, 0, "gen", [[0, net.gen.max_p_mw.at[0], 1]])
    pp.runopp(net)


    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    #-------------------------------------------
    #    p_min_mw       |\
    #                   | \
    #                   |  \
    net.pwl_cost.points.iloc[0] =  [(0, net.gen.max_p_mw.at[0], -1)]
    pp.runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)

    net.pwl_cost = net.pwl_cost.drop(0)

    # first using a positive slope as in the case above
#    pp.create_pwl_cost(net, 0, "gen", array([1, 0]))
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    pp.runopp(net)
    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)

    # negative slope as in the case above
    net.poly_cost.cp1_eur_per_mw *= -1
    pp.runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)

if __name__ == "__main__":
    pytest.main(['-s', __file__])

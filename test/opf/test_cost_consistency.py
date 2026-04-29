import pytest
from numpy import isclose

from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line_from_parameters, \
    create_load, create_sgen, create_pwl_cost, create_poly_cost, create_gen
from pandapower.run import runpp, runopp


@pytest.fixture()
def base_net():
    net = create_empty_network()
    create_bus(net, vn_kv=10)
    create_bus(net, vn_kv=10)
    create_ext_grid(net, 0)
    create_load(net, 1, p_mw=0.2, controllable=False)
    create_line_from_parameters(net, 0, 1, 50, name="line", r_ohm_per_km=0.876,
                                c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                max_loading_percent=100 * 690)

    runpp(net)
    return net


def test_contingency_sgen(base_net):
    net = base_net
    create_sgen(net, 1, p_mw=0.1, q_mvar=0, controllable=True, min_p_mw=0.005, max_p_mw=0.150,
                max_q_mvar=0.05, min_q_mvar=-0.05)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    # -------------------------------------------
    #    p_min_mw      /|
    #                 / |
    #                /  |

    pwl = create_pwl_cost(net, 0, "sgen", [[0, net.sgen.max_p_mw.at[0], 1]])
    runopp(net)

    assert isclose(net.res_cost, net.res_sgen.p_mw.at[0], atol=1e-3)
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    # -------------------------------------------
    #    p_min_mw       |\
    #                   | \
    #                   |  \

    net.pwl_cost.at[pwl, 'points'] = [(0, net.sgen.max_p_mw.at[0], -1)]
    runopp(net)

    assert isclose(net.res_cost, -net.res_sgen.p_mw.at[0], atol=1e-4)

    net.pwl_cost = net.pwl_cost.drop(0)

    # first using a positive slope as in the case above
    create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=1.)
    runopp(net)
    assert isclose(net.res_cost, net.res_sgen.p_mw.at[0], atol=1e-3)

    # negative slope as in the case above
    net.poly_cost.at[0, "cp1_eur_per_mw"] *= -1
    runopp(net)

    assert isclose(net.res_cost, -net.res_sgen.p_mw.at[0], atol=1e-4)


def test_contingency_load(base_net):
    net = base_net
    create_gen(net, 1, p_mw=0.1, vm_pu=1.05, controllable=True, min_p_mw=0.005, max_p_mw=0.150,
               max_q_mvar=0.05, min_q_mvar=-0.05)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    # -------------------------------------------
    #    p_min_mw      /|
    #                 / |
    #                /  |

    create_pwl_cost(net, 0, "gen", [[0, net.gen.max_p_mw.at[0], 1]])
    runopp(net)

    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    # -------------------------------------------
    #    p_min_mw       |\
    #                   | \
    #                   |  \
    net.pwl_cost.at[0, 'points'] = [(0, net.gen.max_p_mw.at[0], -1)]
    runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)

    net.pwl_cost = net.pwl_cost.drop(0)

    # first using a positive slope as in the case above
    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    runopp(net)
    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)

    # negative slope as in the case above
    net.poly_cost.at[0, "cp1_eur_per_mw"] *= -1
    runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)


def test_contingency_gen(base_net):
    net = base_net
    create_gen(net, 1, p_mw=0.1, vm_pu=1.05, controllable=True, min_p_mw=0.005, max_p_mw=0.150,
               max_q_mvar=0.05, min_q_mvar=-0.05)
    # pwl costs
    # maximize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #                   |   /
    #                   |  /
    #                   | /
    #                   |/
    # -------------------------------------------
    #    p_min_mw      /|
    #                 / |
    #                /  |

    create_pwl_cost(net, 0, "gen", [[0, net.gen.max_p_mw.at[0], 1]])
    runopp(net)

    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)
    # minimize the sgen feed in by using a positive cost slope
    # using a slope of 1
    #               \   |
    #                \  |
    #                 \ |
    #                  \|
    # -------------------------------------------
    #    p_min_mw       |\
    #                   | \
    #                   |  \
    net.pwl_cost.at[0, 'points'] = [(0, net.gen.max_p_mw.at[0], -1)]
    runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)

    net.pwl_cost = net.pwl_cost.drop(0)

    # first using a positive slope as in the case above
    #    create_pwl_cost(net, 0, "gen", array([1, 0]))
    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    runopp(net)
    assert isclose(net.res_cost, net.res_gen.p_mw.at[0], atol=1e-3)

    # negative slope as in the case above
    net.poly_cost.cp1_eur_per_mw *= -1
    runopp(net)

    assert isclose(net.res_cost, -net.res_gen.p_mw.at[0], atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

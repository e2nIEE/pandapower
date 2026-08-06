from pandapower.create import create_bus, create_empty_network, create_ext_grid
from pandapower.run import runpp, set_user_pf_options
from math import isclose

from pandapower.test.toolbox.test_grid_modification import net


def _minimal_net():
    net = create_empty_network()
    bus = create_bus(net, vn_kv=20.0)
    create_ext_grid(net, bus=bus)
    return net


def test_runpp_accepts_solver_tolerance_alias():
    net = _minimal_net()

    runpp(net, solver_tolerance=1e-7, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-7)


def test_solver_tolerance_user_pf_option():
    net = _minimal_net()
    set_user_pf_options(net, solver_tolerance=1e-6)

    runpp(net, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-6)


def test_tolerance_mva_remains_backward_compatible():
    net = _minimal_net()

    runpp(net, tolerance_mva=1e-5, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-5)

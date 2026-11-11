from math import isclose

from pandapower.create import create_buses, create_ext_grid
from pandapower.network import pandapowerNet
from pandapower.run import runpp, set_user_pf_options


def _minimal_net():
    net = pandapowerNet(name="minimal_net")
    (bus,) = create_buses(net, 1, vn_kv=20.0)
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

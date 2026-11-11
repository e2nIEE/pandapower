from math import isclose

from pandapower.auxiliary import _internal_stored
from pandapower.create import create_buses, create_ext_grid, create_line_from_parameters, create_load
from pandapower.network import pandapowerNet
from pandapower.run import runpp, set_user_pf_options


def _minimal_net():
    net = pandapowerNet(name="minimal_net")
    (bus,) = create_buses(net, 1, vn_kv=20.0)
    create_ext_grid(net, bus=bus)
    return net


def _recycle_net():
    net = pandapowerNet(name="recycle_net")
    from_bus, to_bus = create_buses(net, 2, vn_kv=20.0)
    create_ext_grid(net, bus=from_bus)
    create_load(net, to_bus, p_mw=1.0, q_mvar=0.2)
    create_line_from_parameters(
        net,
        from_bus,
        to_bus,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
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


def test_solver_tolerance_applied_in_recycled_powerflow():
    net = _recycle_net()

    runpp(net, numba=False)
    assert _internal_stored(net)

    runpp(
        net,
        solver_tolerance=1e-7,
        numba=False,
        recycle={"bus_pq": True, "trafo": False, "gen": False},
    )

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-7)


def test_explicit_alias_overrides_stored_other_alias():
    net = _minimal_net()
    set_user_pf_options(net, solver_tolerance=1e-6)

    runpp(net, tolerance_mva=1e-2, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-2)

    net = _minimal_net()
    set_user_pf_options(net, tolerance_mva=1e-6)

    runpp(net, solver_tolerance=1e-2, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-2)


def test_new_tolerance_alias_replaces_stored_other_alias():
    net = _minimal_net()

    set_user_pf_options(net, tolerance_mva=1e-5)
    set_user_pf_options(net, solver_tolerance=1e-8)

    runpp(net, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-8)

    net = _minimal_net()

    set_user_pf_options(net, solver_tolerance=1e-8)
    set_user_pf_options(net, tolerance_mva=1e-5)

    runpp(net, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-5)


def test_tolerance_mva_remains_backward_compatible():
    net = _minimal_net()

    runpp(net, tolerance_mva=1e-5, numba=False)

    assert net.converged
    assert isclose(net._options["tolerance_mva"], 1e-5)

import logging

import pytest
from numpy import nan

from pandapower.create import (
    create_bus,
    create_buses,
    create_dcline,
    create_ext_grid,
    create_gen,
    create_line,
    create_load,
    create_sgen,
    create_transformer,
)
from pandapower.network import pandapowerNet
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters

logger = logging.getLogger(__name__)


def _opf_net():
    net = pandapowerNet(name="_opf_net")
    create_bus(net, 20, min_vm_pu=0.85, max_vm_pu=1.15)
    create_buses(net, 3, 0.4, min_vm_pu=0.85, max_vm_pu=1.15)
    create_transformer(net, 0, 1, "0.25 MVA 20/0.4 kV")
    create_line(net, 1, 2, 1, "48-AL1/8-ST1A 0.4")
    create_ext_grid(net, 0)
    create_dcline(
        net,
        1,
        3,
        -5,
        1,
        1,
        1,
        1.02,
        max_p_mw=0.01,
        min_q_from_mvar=-0.005,
        min_q_to_mvar=-0.005,
        max_q_from_mvar=0.005,
        max_q_to_mvar=0.005,
    )
    create_load(
        net, bus=2, p_mw=0.015, controllable=True, max_p_mw=0.025, min_p_mw=0, max_q_mvar=0.005, min_q_mvar=-0.005
    )
    create_gen(
        net,
        bus=1,
        p_mw=0.002,
        vm_pu=1.002,
        controllable=True,
        max_p_mw=0,
        min_p_mw=-0.025,
        max_q_mvar=0.025,
        min_q_mvar=-0.025,
    )
    create_sgen(
        net, bus=2, p_mw=0.001, controllable=True, max_p_mw=0.003, min_p_mw=0, max_q_mvar=0.001, min_q_mvar=-0.001
    )
    return net


def _run_check(net):
    try:
        _check_necessary_opf_parameters(net, logger)
        return True
    except KeyError:
        return False


def test_opf_data_check_basic():
    net = _opf_net()
    assert _run_check(net)


@pytest.mark.parametrize("par", ["min_vm_pu", "max_vm_pu"])
def test_opf_data_check_vm_lim_val(par):
    # no error due to missing voltage limits expected
    net = _opf_net()
    net.bus.at[0, par] = nan
    assert _run_check(net)


@pytest.mark.parametrize("par", ["min_vm_pu", "max_vm_pu"])
def test_opf_data_check_vm_lim_col(par):
    # no error due to missing voltage limits expected
    net = _opf_net()
    del net.bus[par]
    assert _run_check(net)


@pytest.mark.parametrize("par", ["max_p_mw", "min_q_from_mvar", "min_q_to_mvar", "max_q_from_mvar", "max_q_to_mvar"])
def test_opf_data_check_dcline_lim_val(par):
    # no error due to missing dcline constraint values expected
    net = _opf_net()
    net.dcline.at[0, par] = nan
    assert _run_check(net)


@pytest.mark.parametrize("par", ["max_p_mw", "min_q_from_mvar", "min_q_to_mvar", "max_q_from_mvar", "max_q_to_mvar"])
def test_opf_data_check_dcline_lim_col(par):
    # error due to missing dcline constraint columns expected
    net = _opf_net()
    del net.dcline[par]
    assert not _run_check(net)


@pytest.mark.parametrize(
    "elm, par",
    [(elm, par) for elm in ["load", "gen", "sgen"] for par in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]],
)
def test_opf_data_check_lim_val(elm, par):
    # no error due to missing (load, gen, sgen) constraint values expected
    net = _opf_net()
    net[elm].at[0, par] = nan
    assert _run_check(net)


@pytest.mark.parametrize(
    "elm, par",
    [(elm, par) for elm in ["load", "gen", "sgen"] for par in ["min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]],
)
def test_opf_data_check_lim_col(elm, par):
    # error due to missing (load, gen, sgen) constraint columns expected
    net = _opf_net()
    del net[elm][par]
    assert not _run_check(net)


@pytest.mark.parametrize("elm", ["line", "trafo", "trafo3w", "impedance"])
@pytest.mark.parametrize("par", ["min_angle_degree", "max_angle_degree"])
def test_opf_data_check_angle_limits_missing_column(elm, par, caplog):
    # missing angle limit columns must not raise or warn
    net = _opf_net()
    assert par not in net[elm].columns
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert _run_check(net)
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING and "angle" in rec.getMessage()]


@pytest.mark.parametrize("elm", ["line", "trafo"])
def test_opf_data_check_angle_limits_all_nan_logs_info(elm, caplog):
    net = _opf_net()
    net[elm]["min_angle_degree"] = nan
    net[elm]["max_angle_degree"] = nan
    with caplog.at_level(logging.INFO, logger=logger.name):
        assert _run_check(net)
    messages = [rec.getMessage() for rec in caplog.records]
    for par in ("min_angle_degree", "max_angle_degree"):
        assert any(f"{elm}.{par}" in msg and "unconstrained" in msg for msg in messages)
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING and "angle" in rec.getMessage()]


@pytest.mark.parametrize("elm", ["line", "trafo"])
@pytest.mark.parametrize("par", ["min_angle_degree", "max_angle_degree"])
def test_opf_data_check_angle_limits_zero_warns(elm, par, caplog):
    net = _opf_net()
    net[elm]["min_angle_degree"] = -30.0
    net[elm]["max_angle_degree"] = 30.0
    net[elm][par] = 0.0
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert _run_check(net)
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any(f"{elm}.{par} contains 0" in msg for msg in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

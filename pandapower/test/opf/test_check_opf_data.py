from numpy import nan
import pytest

import pandapower as pp
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _opf_net():
    net = pp.create_empty_network()
    pp.create_bus(net, 20, min_vm_pu=0.85, max_vm_pu=1.15)
    pp.create_buses(net, 3, 0.4, min_vm_pu=0.85, max_vm_pu=1.15)
    pp.create_transformer(net, 0, 1, "0.25 MVA 20/0.4 kV")
    pp.create_line(net, 1, 2, 1, "48-AL1/8-ST1A 0.4")
    pp.create_ext_grid(net, 0)
    pp.create_dcline(net, 1, 3, -5, 1, 1, 1, 1.02, max_p_mw=0.01, min_q_from_mvar=-.005,
                     min_q_to_mvar=-0.005, max_q_from_mvar=.005, max_q_to_mvar=.005)
    pp.create_load(net, bus=2, p_mw=0.015, controllable=True, max_p_mw=0.025, min_p_mw=0,
                   max_q_mvar=0.005, min_q_mvar=-0.005)
    pp.create_gen(net, bus=1, p_mw=0.002, vm_pu=1.002, controllable=True, max_p_mw=0, min_p_mw=-0.025,
                  max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_sgen(net, bus=2, p_mw=0.001, controllable=True, max_p_mw=0.003, min_p_mw=0,
                   max_q_mvar=.001, min_q_mvar=-.001)
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


def test_opf_data_check_vm_lim_val():
    # no error due to missing voltage limits expected
    for par in ["min_vm_pu", "max_vm_pu"]:
        net = _opf_net()
        net.bus[par].at[0] = nan
        assert _run_check(net)


def test_opf_data_check_vm_lim_col():
    # no error due to missing voltage limits expected
    for par in ["min_vm_pu", "max_vm_pu"]:
        net = _opf_net()
        del net.bus[par]
        assert _run_check(net)


def test_opf_data_check_dcline_lim_val():
    # no error due to missing dcline constraint values expected
    for par in ['max_p_mw', 'min_q_from_mvar', 'min_q_to_mvar', 'max_q_from_mvar', 'max_q_to_mvar']:
        net = _opf_net()
        net.dcline[par].at[0] = nan
        assert _run_check(net)


def test_opf_data_check_dcline_lim_col():
    # error due to missing dcline constraint columns expected
    for par in ['max_p_mw', 'min_q_from_mvar', 'min_q_to_mvar', 'max_q_from_mvar', 'max_q_to_mvar']:
        net = _opf_net()
        del net.dcline[par]
        assert not _run_check(net)


def test_opf_data_check_lim_val():
    # no error due to missing (load, gen, sgen) constraint values expected
    for elm in ['load', 'gen', 'sgen']:
        for par in ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']:
            net = _opf_net()
            net[elm][par].at[0] = nan
            assert _run_check(net)


def test_opf_data_check_lim_col():
    # error due to missing (load, gen, sgen) constraint columns expected
    for elm in ['load', 'gen', 'sgen']:
        for par in ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']:
            net = _opf_net()
            del net[elm][par]
            assert not _run_check(net)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

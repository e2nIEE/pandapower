from unittest.mock import patch, MagicMock

import pytest

import pandapower as pp
from pandapower.test.consistency_checks import runpp_pgm_with_consistency_checks


def test_minimal_net_pgm():
    # tests corner-case when the grid only has 1 bus and an ext-grid
    net = pp.create_empty_network()
    b = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b)
    runpp_pgm_with_consistency_checks(net)

    pp.create_load(net, b, p_mw=0.1)
    runpp_pgm_with_consistency_checks(net)

    b2 = pp.create_bus(net, 110)
    pp.create_switch(net, b, b2, 'b')
    pp.create_sgen(net, b2, p_mw=0.2)
    runpp_pgm_with_consistency_checks(net)


def test_runpp_pgm__asym():
    net = pp.create_empty_network()
    with pytest.raises(NotImplementedError, match="Asymmetric power flow by power-grid-model is not implemented yet"):
        pp.runpp_pgm(net, symmetric=False)


def test_runpp_pgm__non_convergence():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b1, vm_pu=1)
    b2 = pp.create_bus(net, 50)
    pp.create_line(net, b1, b2, 1, std_type="NAYY 4x50 SE")
    pp.runpp_pgm(net)
    ## TODO: Check error message
    assert net["converged"] is False


@patch("pandapower.run.logger")
def test_runpp_pgm__validation_fail(mock_logger: MagicMock):
    net = pp.create_empty_network()
    pp.create_bus(net, -110, index=123)
    pp.runpp_pgm(net, validate_input=True)

    err = mock_logger.error.call_args_list[0].args[0]
    assert "validation error" in err
    assert "bus-123" in err

    dbg = mock_logger.debug.call_args_list[0].args[0]
    assert "validation error" in dbg
    assert "bus-123" in dbg

    assert False


if __name__ == "__main__":
    pytest.main([__file__])

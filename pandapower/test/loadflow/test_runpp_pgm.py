import sys
from unittest.mock import patch, MagicMock

import pytest

import pandapower as pp
from pandapower.test.consistency_checks import runpp_pgm_with_consistency_checks, runpp_pgm_3ph_with_consistency_checks

try:
    import power_grid_model
    PGM_IMPORTED = True
except ImportError:
    PGM_IMPORTED = False


@pytest.mark.parametrize("consistency_fn" , [runpp_pgm_with_consistency_checks, runpp_pgm_3ph_with_consistency_checks])
@pytest.mark.skipif(not PGM_IMPORTED, reason="requires power_grid_model")
def test_minimal_net_pgm(consistency_fn):
    # tests corner-case when the grid only has 1 bus and an ext-grid
    net = pp.create_empty_network()
    b = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b)
    consistency_fn(net)

    pp.create_load(net, b, p_mw=0.1)
    consistency_fn(net)

    b2 = pp.create_bus(net, 110)
    pp.create_switch(net, b, b2, "b")
    pp.create_sgen(net, b2, p_mw=0.2, q_mvar=0.1)
    consistency_fn(net)


@pytest.mark.skipif(not PGM_IMPORTED, reason="requires power_grid_model")
def test_runpp_pgm__invalid_algorithm():
    net = pp.create_empty_network()
    with pytest.raises(
        KeyError,
        match="Invalid algorithm 'foo'",
    ):
        pp.runpp_pgm(net, algorithm="foo")


@patch("pandapower.run.logger")
@pytest.mark.skipif(not PGM_IMPORTED, reason="requires power_grid_model")
def test_runpp_pgm__internal_pgm_error(mock_logger: MagicMock):
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b1, vm_pu=1)
    b2 = pp.create_bus(net, 50)
    pp.create_line(net, b1, b2, 1, std_type="NAYY 4x50 SE")
    pp.runpp_pgm(net)

    assert net["converged"] is False

    if sys.version_info.major == 3 and sys.version_info.minor == 8:
        mock_logger.critical.assert_called_once_with("Internal PowerGridError occurred!")
    else:
        mock_logger.critical.assert_called_once_with("Internal ConflictVoltage occurred!")

    mock_logger.debug.assert_called_once()
    mock_logger.info.assert_called_once_with("Use validate_input=True to validate your input data.")


@patch("pandapower.run.logger")
@pytest.mark.skipif(not PGM_IMPORTED, reason="requires power_grid_model")
def test_runpp_pgm__validation_fail(mock_logger: MagicMock):
    net = pp.create_empty_network()
    pp.create_bus(net, -110, index=123)
    pp.runpp_pgm(net, validate_input=True)

    mock_logger.error.assert_called_once_with("1. Power Grid Model validation error: Check bus-123")
    mock_logger.debug.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])

# test_ext_grid.py

import itertools
import pandera as pa
import pytest

from pandapower import create_ext_grid
from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats,
    negativ_floats_plus_zero,
    not_ints_list,
    negativ_ints,
    all_allowed_floats,
)


class TestExtGridRequiredFields:
    """Tests for required ext_grid fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["vm_pu"], positiv_floats),
                itertools.product(["va_degree"], all_allowed_floats),
                itertools.product(["slack_weight"], all_allowed_floats),
                itertools.product(["in_service"], bools),
                itertools.product(["controllable"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0, in_service=True)

        net.ext_grid[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["vm_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["va_degree"], not_floats_list),
                itertools.product(["slack_weight"], not_floats_list),
                itertools.product(["in_service"], not_boolean_list),
                itertools.product(["controllable"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

        create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0, in_service=True)

        net.ext_grid[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestExtGridOptionalFields:
    """Tests for optional ext_grid fields, including group dependencies (opf, sc, 3ph)"""

    def test_all_optional_fields_valid(self):
        """Test: ext_grid with every optional field is valid and dependencies satisfied"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=100.0,
            min_p_mw=-100.0,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
            # SC group
            s_sc_max_mva=1000.0,
            s_sc_min_mva=500.0,
            # 3PH group (also part of SC group)
            rx_max=0.5,
            rx_min=0.1,
            r0x0_max=0.2,
            x0x_max=3.0,
            name="ext grid 1",
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: ext_grid with optional fields including nulls, with dependencies respected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Row 1: OPF present, SC/3PH absent
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.01,
            va_degree=0.0,
            in_service=True,
            max_p_mw=80.0,
            min_p_mw=-80.0,
            max_q_mvar=40.0,
            min_q_mvar=-40.0,
            name="alpha",
        )
        # Row 2: SC + 3PH present, OPF absent
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.03,
            va_degree=0.0,
            in_service=True,
            s_sc_max_mva=1100.0,
            s_sc_min_mva=600.0,
            rx_max=0.6,
            rx_min=0.2,
            r0x0_max=0.3,
            x0x_max=2.5,
            name=None,
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                itertools.product(["s_sc_max_mva"], positiv_floats),
                itertools.product(["s_sc_min_mva"], positiv_floats),
                itertools.product(["rx_max"], positiv_floats_plus_zero),
                itertools.product(["rx_min"], positiv_floats_plus_zero),
                itertools.product(["r0x0_max"], positiv_floats_plus_zero),
                itertools.product(["x0x_max"], positiv_floats_plus_zero),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=100.0,
            min_p_mw=-100.0,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
            # SC group
            s_sc_max_mva=1000.0,
            s_sc_min_mva=500.0,
            # 3PH group (also part of SC group)
            rx_max=0.5,
            rx_min=0.1,
            r0x0_max=0.2,
            x0x_max=3.0,
            name="ext grid 1",
        )
        net.ext_grid[parameter] = valid_value
        net.ext_grid["name"] = net.ext_grid["name"].astype("string")
        validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        """Test: OPF group must be complete if any OPF value is set"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, in_service=True)
        # Set only one OPF column -> should fail by group dependency
        net.ext_grid["max_p_mw"] = 100.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_sc_group_partial_missing_invalid(self):
        """Test: SC group must be complete if any SC value is set"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, in_service=True)
        # Set only s_sc_max_mva -> should fail by group dependency
        net.ext_grid["s_sc_max_mva"] = 1000.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_3ph_group_partial_missing_invalid(self):
        """Test: 3PH group must be complete if any 3PH value is set (and SC group too)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, in_service=True)
        # Set only rx_max -> should fail by group dependency
        net.ext_grid["rx_max"] = 0.4

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["s_sc_max_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["s_sc_min_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["rx_max"], [*negativ_floats, *not_floats_list]),
                itertools.product(["rx_min"], [*negativ_floats, *not_floats_list]),
                itertools.product(["r0x0_max"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x0x_max"], [*negativ_floats, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are not accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_ext_grid(
            net,
            bus=b0,
            vm_pu=1.02,
            va_degree=0.0,
            in_service=True,
            # OPF group
            max_p_mw=100.0,
            min_p_mw=-100.0,
            max_q_mvar=50.0,
            min_q_mvar=-50.0,
            # SC group
            s_sc_max_mva=1000.0,
            s_sc_min_mva=500.0,
            # 3PH group (also part of SC group)
            rx_max=0.5,
            rx_min=0.1,
            r0x0_max=0.2,
            x0x_max=3.0,
            name="ext grid 1",
        )
        net.ext_grid[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestExtGridForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, in_service=True)

        net.ext_grid["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestExtGridResults:
    """Tests for ext_grid results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_ext_grid_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_ext_grid_3ph_results(self):
        """Test: 3-phase results contain valid values per phase"""
        pass

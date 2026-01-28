# test_pandera_svc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_svc
from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    all_floats,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    negativ_ints,
    not_ints_list,
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    positiv_floats,
    negativ_floats,
    all_allowed_floats,
)

# Angle ranges per schema (inclusive)

valid_angle_range = [*[x for x in all_allowed_floats if 90 <= x <= 180], 90.0, 110.0, 180.0]
invalid_low_angle = [x for x in all_floats if x < 90]
invalid_high_angle = [x for x in all_floats if x > 180]
invalid_angle_range = invalid_low_angle + invalid_high_angle


class TestSvcRequiredFields:
    """Tests for required SVC fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["x_l_ohm"], positiv_floats_plus_zero),
                itertools.product(["x_cvar_ohm"], negativ_floats_plus_zero),
                itertools.product(["set_vm_pu"], all_allowed_floats),
                itertools.product(["thyristor_firing_angle_degree"], valid_angle_range),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_bus(net, 0.4, index=42)

        create_svc(
            net,
            bus=0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )

        net.svc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["x_l_ohm"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x_cvar_ohm"], [*positiv_floats, *not_floats_list]),
                itertools.product(["set_vm_pu"], not_floats_list),
                itertools.product(["thyristor_firing_angle_degree"], [*invalid_angle_range, *not_floats_list]),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_svc(
            net,
            bus=0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )

        net.svc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSvcOptionalFields:
    """Tests for optional SVC fields"""

    def test_all_optional_fields_valid(self):
        """Test: SVC with all optional fields is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.1,
            x_cvar_ohm=-0.2,
            set_vm_pu=1.02,
            thyristor_firing_angle_degree=120.0,
            controllable=True,
            in_service=False,
            name="SVC A",
            min_angle_degree=100.0,
            max_angle_degree=150.0,
        )
        # Ensure string dtype for name
        net.svc["name"] = net.svc["name"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Row 1: name set, angles NaN later
        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
            name="alpha",
        )
        # Row 2: min only initially (we'll null out max)
        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.1,
            x_cvar_ohm=-0.2,
            set_vm_pu=0.98,
            thyristor_firing_angle_degree=110.0,
            controllable=False,
            in_service=False,
            min_angle_degree=95.0,
        )
        # Row 3: max only initially (we'll null out min)
        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.2,
            x_cvar_ohm=-0.3,
            set_vm_pu=1.05,
            thyristor_firing_angle_degree=170.0,
            controllable=True,
            in_service=True,
            max_angle_degree=175.0,
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["min_angle_degree"], [90.0, 100.0, 150.0]),
                itertools.product(["max_angle_degree"], [180.0, 150.0, 100.0]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )

        if parameter == "name":
            net.svc[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.svc[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["min_angle_degree"], [*invalid_low_angle, *not_floats_list]),
                itertools.product(["max_angle_degree"], [*invalid_high_angle, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )
        net.svc[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_min_less_equal_max_check_passes(self):
        """Test: min_angle_degree <= max_angle_degree passes"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )
        net.svc["min_angle_degree"] = 100.0
        net.svc["max_angle_degree"] = 150.0

        validate_network(net)

    def test_min_greater_than_max_fails(self):
        """Test: min_angle_degree > max_angle_degree fails"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )
        net.svc["min_angle_degree"] = 160.0
        net.svc["max_angle_degree"] = 150.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSvcForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_svc(
            net,
            bus=b0,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_vm_pu=1.00,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )

        net.svc["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSvcResults:
    """Tests for svc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_svc_result_totals(self):
        """Test: reactive power and impedance results are consistent"""
        pass

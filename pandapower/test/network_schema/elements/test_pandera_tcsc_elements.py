# test_pandera_tcsc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_tcsc
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    all_floats,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints,
    positiv_ints_plus_zero,
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    all_ints,
    negativ_ints,
    not_ints_list,
    negativ_floats,
    positiv_floats,
    all_allowed_floats,
)

float_range = [x for x in all_allowed_floats if 90 <= x <= 180]
not_float_range = [x for x in all_floats if x < 90 or x > 180]
invalid_low_float_range = [x for x in all_floats if x < 90]
invalid_high_float_range = [x for x in all_floats if x > 180]


class TestTcscRequiredFields:
    """Tests for required TCSC fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], positiv_ints_plus_zero),
                itertools.product(["to_bus"], positiv_ints_plus_zero),
                itertools.product(["x_l_ohm"], positiv_floats_plus_zero),
                itertools.product(["x_cvar_ohm"], negativ_floats_plus_zero),
                itertools.product(["set_p_to_mw"], all_allowed_floats),
                itertools.product(["thyristor_firing_angle_degree"], float_range),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are rejected"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_bus(net, 0.4, index=42)
        create_tcsc(
            net,
            from_bus=0,
            to_bus=1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100,
            controllable=True,
            in_service=False,
        )
        net.tcsc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["to_bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["x_l_ohm"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["x_cvar_ohm"], [float(np.nan), pd.NA, *positiv_floats, *not_floats_list]),
                itertools.product(["set_p_to_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["thyristor_firing_angle_degree"], [float(np.nan), pd.NA, *not_float_range, *not_floats_list]),
                itertools.product(["controllable"], [float(np.nan), pd.NA, *not_boolean_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: Invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_tcsc(
            net,
            from_bus=0,
            to_bus=1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100,
            controllable=True,
            in_service=False,
        )
        net.tcsc[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTcscOptionalFields:
    """Tests for optional tcsc fields"""

    def test_empty_network_validation(self):
        """Test: tcsc with every optional fields is valid"""
        net = pandapowerNet(name="test_empty_network_validation")
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_tcsc(
            net,
            from_bus=0,
            to_bus=1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100,
            controllable=True,
            in_service=False,
            name="lorem ipsum",
            min_angle_degree=100.0,
            max_angle_degree=110.6,
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: TCSC with some optional fields (including nulls) is valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)  # ensure 42 exists for FK-positive tests
        # Row 1: name set, angles NaN
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
            name="tcsc1",
        )
        # Row 2: min_angle only
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
            min_angle_degree=100.0,
        )
        # Row 3: max_angle only
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
            max_angle_degree=150.0,
        )

        # Set nullable columns with mixed values
        net.tcsc["name"] = pd.Series(["tcsc1", pd.NA, pd.NA], dtype=pd.StringDtype())
        net.tcsc["min_angle_degree"] = [float(np.nan), 100.0, float(np.nan)]
        net.tcsc["max_angle_degree"] = [float(np.nan), float(np.nan), 150.0]

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                # name accepts strings and pd.NA
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["min_angle_degree"], [float(np.nan), 90.0, 100.0, 150.0]),
                itertools.product(["max_angle_degree"], [float(np.nan), 180.0, 150.0, 100.0]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)  # ensure 42 exists for FK-positive tests
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
        )
        if parameter == "name":
            net.tcsc[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.tcsc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
                itertools.product(["min_angle_degree"], [pd.NA, *invalid_low_float_range, *not_floats_list]),
                itertools.product(["max_angle_degree"], [pd.NA, *invalid_high_float_range, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)  # ensure 42 exists for FK-positive tests
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
        )
        net.tcsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_min_less_equal_max_check_passes(self):
        net = pandapowerNet(name="test_min_less_equal_max_check_passes")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)  # ensure 42 exists for FK-positive tests
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
        )
        net.tcsc["min_angle_degree"] = 100.0
        net.tcsc["max_angle_degree"] = 150.0
        validate_network(net)

    def test_min_equal_max_check_passes(self):
        """Test: min_angle_degree == max_angle_degree passes"""
        net = pandapowerNet(name="test_min_equal_max_check_passes")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
        )
        net.tcsc["min_angle_degree"] = 120.0
        net.tcsc["max_angle_degree"] = 120.0
        validate_network(net)

    def test_min_greater_than_max_fails(self):
        net = pandapowerNet(name="test_min_greater_than_max_fails")
        b0 = create_bus(net, 0.4)  # index 0
        b1 = create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)  # ensure 42 exists for FK-positive tests
        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=False,
        )
        net.tcsc["min_angle_degree"] = 160.0
        net.tcsc["max_angle_degree"] = 150.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

class TestTcscForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_from_bus_index(self):
        """Test: from_bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_from_bus_index")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )

        net.tcsc["from_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_to_bus_index(self):
        """Test: to_bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_to_bus_index")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_tcsc(
            net,
            from_bus=b0,
            to_bus=b1,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )

        net.tcsc["to_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FKs work with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_tcsc(
            net,
            from_bus=10,
            to_bus=42,
            x_l_ohm=0.0,
            x_cvar_ohm=-0.1,
            set_p_to_mw=0.0,
            thyristor_firing_angle_degree=100.0,
            controllable=True,
            in_service=True,
        )
        create_tcsc(
            net,
            from_bus=42,
            to_bus=100,
            x_l_ohm=0.1,
            x_cvar_ohm=-0.2,
            set_p_to_mw=1.0,
            thyristor_firing_angle_degree=110.0,
            controllable=False,
            in_service=False,
        )
        create_tcsc(
            net,
            from_bus=100,
            to_bus=10,
            x_l_ohm=0.2,
            x_cvar_ohm=-0.3,
            set_p_to_mw=2.0,
            thyristor_firing_angle_degree=120.0,
            controllable=True,
            in_service=True,
        )

        validate_network(net)


class TestTcscResults:
    """Tests for tcsc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_tcsc_voltage_results(self):
        """Test: Voltage results are within valid range"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_tcsc_power_results(self):
        """Test: Power results are consistent"""
        pass

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_tcsc
from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.network_schema.tools.helper import get_dtypes
from pandapower.network_schema.bus import bus_schema
from pandapower.test.pandera.elements.helper import (
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
    all_ints, negativ_ints, not_ints_list, negativ_floats, positiv_floats, all_allowed_floats
)

float_range = [x for x in all_allowed_floats if 90 <= x <= 180]
not_float_range = [x for x in all_floats if x < 90 or x > 180]
invalid_low_float_range = [x for x in all_floats if x < 90]
invalid_high_float_range = [x for x in all_floats if x > 180]

class TestTcscRequiredFields:
    """Tests for required TCSC fields"""

    # david fragen wegen chain
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
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_bus(net, 0.4, index=42)
        create_tcsc(net,from_bus=0, to_bus=1, x_l_ohm=0.0, x_cvar_ohm=-0.1, set_p_to_mw=0.0,
                    thyristor_firing_angle_degree=100, controllable=True, in_service=False)
        net.tcsc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["to_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["x_l_ohm"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x_cvar_ohm"], [*positiv_floats, *not_floats_list]),
                itertools.product(["set_p_to_mw"], not_floats_list),
                itertools.product(["thyristor_firing_angle_degree"], not_float_range),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: Invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_tcsc(net,from_bus=0, to_bus=1, x_l_ohm=0.0, x_cvar_ohm=-0.1, set_p_to_mw=0.0,
                    thyristor_firing_angle_degree=100, controllable=True, in_service=False)
        net.tcsc[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTcscOptionalFields:
    """Tests for optional tcsc fields"""

    def test_empty_network_validation(self):
        """Test: tcsc with every optional fields is valid"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_tcsc(net, from_bus=0, to_bus=1, x_l_ohm=0.0, x_cvar_ohm=-0.1, set_p_to_mw=0.0,
                    thyristor_firing_angle_degree=100, controllable=True, in_service=False,
                    name='lorem ipsum', min_angle_degree=100.0, max_angle_degree=110.6)
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: TCSC with some optional fields (including nulls) is valid"""
        net = create_empty_network()
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
        net.tcsc["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        # net.tcsc["min_angle_degree"] = float(np.nan)
        # net.tcsc["max_angle_degree"] = float(np.nan)
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                # name accepts strings and pd.NA
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["min_angle_degree"], [90.0, 100.0, 150.0]),
                itertools.product(["max_angle_degree"], [180.0, 150.0, 100.0]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = create_empty_network()
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
                itertools.product(["name"], not_strings_list),
                itertools.product(["min_angle_degree"], [*invalid_low_float_range, *not_floats_list]),
                itertools.product(["max_angle_degree"], [*invalid_high_float_range, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = create_empty_network()
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
        net = create_empty_network()
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

    def test_min_greater_than_max_fails(self):
        net = create_empty_network()
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

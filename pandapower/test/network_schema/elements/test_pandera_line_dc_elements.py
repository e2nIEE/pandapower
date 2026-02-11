# test_pandera_line_dc_elements.py

import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus_dc, create_line_dc_from_parameters, create_line_dc
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    positiv_ints,
    negativ_ints,
    negativ_ints_plus_zero,
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats,
    negativ_floats_plus_zero,
    all_allowed_floats,
    not_ints_list,
)

# df must be in [0, 1]

df_valid_range = [0.0, 0.5, 1.0]
df_invalid_range = [-0.1, 1.1]

STD_TYPE = "NAYY 4x50 SE"


class TestLineDcRequiredFields:
    """Tests for required line_dc fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus_dc"], positiv_ints_plus_zero),
                itertools.product(["to_bus_dc"], positiv_ints_plus_zero),
                itertools.product(["length_km"], positiv_floats),
                itertools.product(["r_ohm_per_km"], positiv_floats_plus_zero),
                itertools.product(["g_us_per_km"], positiv_floats_plus_zero),
                itertools.product(["max_i_ka"], positiv_floats_plus_zero),
                itertools.product(["parallel"], positiv_ints),
                itertools.product(["df"], df_valid_range),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus_dc(net, 0.4)  # index 0
        create_bus_dc(net, 0.4)  # index 1
        create_bus_dc(net, 0.4, index=42)

        create_line_dc_from_parameters(
            net,
            from_bus_dc=0,
            to_bus_dc=1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )
        net.line_dc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus_dc"], [*negativ_ints, *not_ints_list]),
                itertools.product(["to_bus_dc"], [*negativ_ints, *not_ints_list]),
                itertools.product(["length_km"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["r_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["g_us_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["max_i_ka"], [*negativ_floats, *not_floats_list]),
                itertools.product(["parallel"], [*negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["df"], [*df_invalid_range, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus_dc(net, 0.4)  # index 0
        create_bus_dc(net, 0.4)  # index 1

        create_line_dc_from_parameters(
            net,
            from_bus_dc=0,
            to_bus_dc=1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )
        net.line_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineDcOptionalFields:
    """Tests for optional line_dc fields and TDPF group dependencies"""

    def test_all_optional_fields_valid(self):
        """Test: line_dc with every optional field and complete TDPF group is valid"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)

        create_line_dc(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
            name="Line DC A",
            std_type="95-CU",
            type="ol",
            geo='{"type":"LineString","coordinates":[]}',
        )

        # Additional optional numeric fields
        net.line_dc["max_loading_percent"] = 100.0
        net.line_dc["alpha"] = 0.00393
        net.line_dc["temperature_degree_celsius"] = 25.0

        # TDPF group (must be complete if any is set)
        net.line_dc["tdpf"] = pd.Series([True], dtype="boolean")
        net.line_dc["wind_speed_m_per_s"] = 5.0
        net.line_dc["wind_angle_degree"] = 90.0
        net.line_dc["conductor_outer_diameter_m"] = 0.03
        net.line_dc["air_temperature_degree_celsius"] = 20.0
        net.line_dc["reference_temperature_degree_celsius"] = 20.0
        net.line_dc["solar_radiation_w_per_sq_m"] = 200.0
        net.line_dc["solar_absorptivity"] = 0.5
        net.line_dc["emissivity"] = 0.9
        net.line_dc["r_theta_kelvin_per_mw"] = 2.0
        net.line_dc["mc_joule_per_m_k"] = 3600.0

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid when TDPF group is not triggered"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)

        # Line 1: string fields
        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
            name="Line 1",
            type="ol",
        )
        # Line 2: max_loading_percent only
        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
            max_loading_percent=80.0,
        )

        # Line 3: alpha/temperature only
        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
            alpha=0.003,
            temperature_degree_celsius=30.0,
        )

        # Allow pd.NA in strings
        net.line_dc["std_type"] = pd.Series(data=pd.NA, dtype=pd.StringDtype)
        net.line_dc["type"] = pd.Series(data=pd.NA, dtype=pd.StringDtype)
        net.line_dc["geo"] = pd.Series(data=pd.NA, dtype=pd.StringDtype)

        validate_network(net)

    def test_tdpf_group_partial_missing_invalid(self):
        """Test: TDPF group must be complete if any TDPF value is set"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)

        # Case 1: tdpf flag only -> invalid
        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )
        net.line_dc["tdpf"] = pd.Series([True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 2: one tdpf param only -> invalid
        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )
        net.line_dc["wind_speed_m_per_s"] = 3.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 3: another tdpf param only -> invalid
        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )
        net.line_dc["reference_temperature_degree_celsius"] = 20.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["std_type"], strings),
                itertools.product(["type"], strings),
                itertools.product(["geo"], strings),
                itertools.product(["alpha"], all_allowed_floats),
                itertools.product(["temperature_degree_celsius"], all_allowed_floats),
                itertools.product(["max_loading_percent"], positiv_floats),
                itertools.product(["tdpf"], bools),
                itertools.product(["wind_speed_m_per_s"], all_allowed_floats),
                itertools.product(["wind_angle_degree"], all_allowed_floats),
                itertools.product(["conductor_outer_diameter_m"], all_allowed_floats),
                itertools.product(["air_temperature_degree_celsius"], all_allowed_floats),
                itertools.product(["reference_temperature_degree_celsius"], all_allowed_floats),
                itertools.product(["solar_radiation_w_per_sq_m"], all_allowed_floats),
                itertools.product(["solar_absorptivity"], all_allowed_floats),
                itertools.product(["emissivity"], all_allowed_floats),
                itertools.product(["r_theta_kelvin_per_mw"], all_allowed_floats),
                itertools.product(["mc_joule_per_m_k"], all_allowed_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted (TDPF group satisfied)"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)

        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )

        # Satisfy TDPF group
        net.line_dc["tdpf"] = pd.Series([True], dtype="boolean")
        net.line_dc["wind_speed_m_per_s"] = 2.0
        net.line_dc["wind_angle_degree"] = 90.0
        net.line_dc["conductor_outer_diameter_m"] = 0.03
        net.line_dc["air_temperature_degree_celsius"] = 25.0
        net.line_dc["reference_temperature_degree_celsius"] = 20.0
        net.line_dc["solar_radiation_w_per_sq_m"] = 150.0
        net.line_dc["solar_absorptivity"] = 0.5
        net.line_dc["emissivity"] = 0.8
        net.line_dc["r_theta_kelvin_per_mw"] = 2.0
        net.line_dc["mc_joule_per_m_k"] = 3600.0

        if parameter in {"name", "std_type", "type", "geo"}:
            net.line_dc[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter == "tdpf":
            net.line_dc[parameter] = pd.Series([valid_value], dtype="boolean")
        else:
            net.line_dc[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["std_type"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["geo"], not_strings_list),
                itertools.product(["alpha"], not_floats_list),
                itertools.product(["temperature_degree_celsius"], not_floats_list),
                itertools.product(["max_loading_percent"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["tdpf"], not_boolean_list),
                itertools.product(["wind_speed_m_per_s"], not_floats_list),
                itertools.product(["wind_angle_degree"], not_floats_list),
                itertools.product(["conductor_outer_diameter_m"], not_floats_list),
                itertools.product(["air_temperature_degree_celsius"], not_floats_list),
                itertools.product(["reference_temperature_degree_celsius"], not_floats_list),
                itertools.product(["solar_radiation_w_per_sq_m"], not_floats_list),
                itertools.product(["solar_absorptivity"], not_floats_list),
                itertools.product(["emissivity"], not_floats_list),
                itertools.product(["r_theta_kelvin_per_mw"], not_floats_list),
                itertools.product(["mc_joule_per_m_k"], not_floats_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected (TDPF group satisfied)"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)

        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )

        # Provide complete TDPF group so only the target parameter triggers failure
        net.line_dc["tdpf"] = pd.Series([True], dtype="boolean")
        net.line_dc["wind_speed_m_per_s"] = 2.0
        net.line_dc["wind_angle_degree"] = 90.0
        net.line_dc["conductor_outer_diameter_m"] = 0.03
        net.line_dc["air_temperature_degree_celsius"] = 25.0
        net.line_dc["reference_temperature_degree_celsius"] = 20.0
        net.line_dc["solar_radiation_w_per_sq_m"] = 150.0
        net.line_dc["solar_absorptivity"] = 0.5
        net.line_dc["emissivity"] = 0.8
        net.line_dc["r_theta_kelvin_per_mw"] = 2.0
        net.line_dc["mc_joule_per_m_k"] = 3600.0

        net.line_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineDcForeignKey:
    """Tests for foreign key constraints on dc bus indices"""

    def test_invalid_bus_index(self):
        """Test: from_bus_dc/to_bus_dc must reference existing bus_dc indices"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        b1 = create_bus_dc(net, 0.4)

        create_line_dc_from_parameters(
            net,
            from_bus_dc=b0,
            to_bus_dc=b1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            g_us_per_km=0.0,
            max_i_ka=0.2,
            parallel=1,
            df=0.5,
            in_service=True,
        )

        net.line_dc["from_bus_dc"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineDcResults:
    """Tests for line_dc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_line_dc_result_totals(self):
        """Test: aggregated p_mw results are consistent"""
        pass

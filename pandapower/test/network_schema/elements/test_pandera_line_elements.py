# test_pandera_line_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_line
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

# Additional ranges and helpers

df_valid_range = [0.0, 0.5, 1.0]
df_invalid_range = [-0.1, 1.1]
wind_angle_valid = [0.0, 90.0, 360.0]
wind_angle_invalid = [-0.1, 360.1]
endtemp_valid = [-200.0, 0.0, 40.0]
endtemp_invalid = [-1000.0, -274.0]
temp_valid = [-200.0, 0.0, 25.0]
temp_invalid = [-1000.0, -274.0]
ref_temp_valid = [-200.0, 20.0]
ref_temp_invalid = [-1000.0, -274.0]

STD_TYPE = "NAYY 4x50 SE"


class TestLineRequiredFields:
    """Tests for required line fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], positiv_ints_plus_zero),
                itertools.product(["to_bus"], positiv_ints_plus_zero),
                itertools.product(["length_km"], positiv_floats),
                itertools.product(["r_ohm_per_km"], positiv_floats_plus_zero),
                itertools.product(["x_ohm_per_km"], positiv_floats_plus_zero),
                itertools.product(["c_nf_per_km"], positiv_floats_plus_zero),
                itertools.product(["g_us_per_km"], positiv_floats_plus_zero),
                itertools.product(["max_i_ka"], positiv_floats),
                itertools.product(["parallel"], positiv_ints),
                itertools.product(["df"], df_valid_range),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1
        create_bus(net, 0.4, index=42)  # ensure FK-positive for 42

        create_line(net, from_bus=0, to_bus=1, length_km=1.0, in_service=True)

        net.line[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["to_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["length_km"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["r_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["c_nf_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["g_us_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["max_i_ka"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["parallel"], [*negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["df"], [*df_invalid_range, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1

        create_line(net, from_bus=0, to_bus=1, length_km=1.0, in_service=True)

        net.line[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineOptionalFields:
    """Tests for optional line fields and group dependencies (tdpf, opf)"""

    def test_all_optional_fields_valid(self):
        """Test: line with every optional field and tdpf group complete is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)

        # Optional text fields
        net.line["name"] = pd.Series(["Line A"], dtype="string")
        net.line["std_type"] = pd.Series(["custom_type"], dtype="string")
        net.line["type"] = pd.Series(["ol"], dtype="string")
        net.line["geo"] = pd.Series(['{"type":"LineString","coordinates":[]}'], dtype="string")

        # Zero-sequence params
        net.line["r0_ohm_per_km"] = 0.0
        net.line["x0_ohm_per_km"] = 0.0
        net.line["c0_nf_per_km"] = 0.0
        net.line["g0_us_per_km"] = 0.0

        # OPF group
        net.line["max_loading_percent"] = 100.0

        # Thermal params
        net.line["alpha"] = 0.00393
        net.line["temperature_degree_celsius"] = 25.0
        net.line["endtemp_degree"] = 40.0

        # TDPF group (complete)
        net.line["tdpf"] = pd.Series([True], dtype="boolean")
        net.line["wind_speed_m_per_s"] = 5.0
        net.line["wind_angle_degree"] = 90.0
        net.line["conductor_outer_diameter_m"] = 0.03
        net.line["air_temperature_degree_celsius"] = 20.0
        net.line["reference_temperature_degree_celsius"] = 20.0
        net.line["solar_radiation_w_per_sq_m"] = 200.0
        net.line["solar_absorptivity"] = 0.5
        net.line["emissivity"] = 0.9
        net.line["r_theta_kelvin_per_mw"] = 2.0
        net.line["mc_joule_per_m_k"] = 3600.0

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid when tdpf group is not triggered"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Line 1: name/type/alpha
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, name='test', alpha=0.0)
        # Line 2: max_loading_percent only (opf)
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, name='test', alpha=0.0, max_loading_percent=80.0)
        # Line 3: zero-sequence params only
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, name='test', alpha=0.0,
                    r0_ohm_per_km=0.1, x0_ohm_per_km=0.2, c0_nf_per_km=1.0, g0_us_per_km=0.0)

        net.line["name"].iat[0] = pd.NA
        net.line["std_type"].iat[1] = pd.NA
        net.line["geo"].iat[2] = pd.NA

        validate_network(net)

    def test_tdpf_group_partial_missing_invalid(self):
        """Test: tdpf group must be complete if any tdpf value is set"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Case 1: tdpf flag only -> invalid
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)
        net.line["tdpf"] = True
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 2: one tdpf param only -> invalid
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)
        net.line["wind_speed_m_per_s"] = 3.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Case 3: another tdpf param only -> invalid
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)
        net.line["reference_temperature_degree_celsius"] = 20.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)
        #TODO sc, 3ph not beeing checked in line.py

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["std_type"], strings),
                itertools.product(["type"], strings),
                itertools.product(["geo"], strings),
                itertools.product(["r0_ohm_per_km"], positiv_floats_plus_zero),
                itertools.product(["x0_ohm_per_km"], positiv_floats_plus_zero),
                itertools.product(["c0_nf_per_km"], positiv_floats_plus_zero),
                itertools.product(["g0_us_per_km"], positiv_floats_plus_zero),
                itertools.product(["max_loading_percent"], positiv_floats),
                itertools.product(["endtemp_degree"], endtemp_valid),
                itertools.product(["alpha"], all_allowed_floats),
                itertools.product(["temperature_degree_celsius"], temp_valid),
                itertools.product(["tdpf"], bools),
                itertools.product(["wind_speed_m_per_s"], positiv_floats_plus_zero),
                itertools.product(["wind_angle_degree"], wind_angle_valid),
                itertools.product(["conductor_outer_diameter_m"], positiv_floats_plus_zero),
                itertools.product(["air_temperature_degree_celsius"], all_allowed_floats),
                itertools.product(["reference_temperature_degree_celsius"], ref_temp_valid),
                itertools.product(["solar_radiation_w_per_sq_m"], positiv_floats_plus_zero),
                itertools.product(["solar_absorptivity"], all_allowed_floats),
                itertools.product(["emissivity"], all_allowed_floats),
                itertools.product(["r_theta_kelvin_per_mw"], all_allowed_floats),
                itertools.product(["mc_joule_per_m_k"], all_allowed_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted (tdpf group satisfied)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)

        # Satisfy tdpf group to avoid dependency failures when setting tdpf-related columns
        net.line["tdpf"] = pd.Series([True], dtype="boolean")
        net.line["wind_speed_m_per_s"] = 2.0
        net.line["wind_angle_degree"] = 90.0
        net.line["conductor_outer_diameter_m"] = 0.03
        net.line["air_temperature_degree_celsius"] = 25.0
        net.line["reference_temperature_degree_celsius"] = 20.0
        net.line["solar_radiation_w_per_sq_m"] = 150.0
        net.line["solar_absorptivity"] = 0.5
        net.line["emissivity"] = 0.8
        net.line["r_theta_kelvin_per_mw"] = 2.0
        net.line["mc_joule_per_m_k"] = 3600.0
        net.line["endtemp_degree"] = 40.0

        if parameter in {"name", "std_type", "type", "geo"}:
            net.line[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter == "tdpf":
            net.line[parameter] = pd.Series([valid_value], dtype="boolean")
        else:
            net.line[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["std_type"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["geo"], not_strings_list),
                itertools.product(["r0_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x0_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["c0_nf_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["g0_us_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["max_loading_percent"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["endtemp_degree"], [*endtemp_invalid, *not_floats_list]),
                itertools.product(["alpha"], not_floats_list),
                itertools.product(["temperature_degree_celsius"], [*temp_invalid, *not_floats_list]),
                itertools.product(["tdpf"], not_boolean_list),
                itertools.product(["wind_speed_m_per_s"], [*negativ_floats, *not_floats_list]),
                itertools.product(["wind_angle_degree"], [*wind_angle_invalid, *not_floats_list]),
                itertools.product(["conductor_outer_diameter_m"], not_floats_list),
                itertools.product(["air_temperature_degree_celsius"], not_floats_list),
                itertools.product(["reference_temperature_degree_celsius"], [*ref_temp_invalid, *not_floats_list]),
                itertools.product(["solar_radiation_w_per_sq_m"], not_floats_list),
                itertools.product(["solar_absorptivity"], not_floats_list),
                itertools.product(["emissivity"], not_floats_list),
                itertools.product(["r_theta_kelvin_per_mw"], not_floats_list),
                itertools.product(["mc_joule_per_m_k"], not_floats_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected (tdpf group satisfied)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)

        # Provide complete tdpf group so only the target parameter triggers failure
        net.line["tdpf"] = pd.Series([True], dtype="boolean")
        net.line["wind_speed_m_per_s"] = 2.0
        net.line["wind_angle_degree"] = 90.0
        net.line["conductor_outer_diameter_m"] = 0.03
        net.line["air_temperature_degree_celsius"] = 25.0
        net.line["reference_temperature_degree_celsius"] = 20.0
        net.line["solar_radiation_w_per_sq_m"] = 150.0
        net.line["solar_absorptivity"] = 0.5
        net.line["emissivity"] = 0.8
        net.line["r_theta_kelvin_per_mw"] = 2.0
        net.line["mc_joule_per_m_k"] = 3600.0
        net.line["endtemp_degree"] = 40.0

        net.line[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineForeignKey:
    """Tests for foreign key constraints on bus indices"""

    def test_invalid_bus_index(self):
        """Test: from_bus/to_bus must reference existing bus indices"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True)

        net.line["from_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineResults:
    """Tests for line results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_line_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_line_3ph_results(self):
        """Test: 3-phase results contain valid values per phase"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_line_sc_results(self):
        """Test: short-circuit results contain valid values"""
        pass

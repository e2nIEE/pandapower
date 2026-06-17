# test_pandera_line_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_line
from pandapower.network import pandapowerNet
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
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)  # ensure FK-positive for 42

        create_line(net, from_bus=0, to_bus=1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        net.line[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["from_bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["to_bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["length_km"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["r_ohm_per_km"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["x_ohm_per_km"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["c_nf_per_km"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["g_us_per_km"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["max_i_ka"], [float(np.nan), pd.NA, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["parallel"], [float(np.nan), pd.NA, *negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["df"], [float(np.nan), pd.NA, *df_invalid_range, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

        create_line(net, from_bus=0, to_bus=1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        net.line[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLineOptionalFields:
    """Tests for optional line fields and group dependencies (tdpf, opf)"""

    def test_all_optional_fields_valid(self):
        """Test: line with every optional field and tdpf group complete is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        # Optional text fields
        net.line["name"] = pd.Series(["Line A"], dtype=pd.StringDtype())
        net.line["type"] = pd.Series(["ol"], dtype=pd.StringDtype())
        net.line["geo"] = pd.Series(['{"type":"LineString","coordinates":[]}'], dtype=pd.StringDtype())

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
        net.line["tdpf"] = pd.Series([True], dtype=pd.BooleanDtype())
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

        # CIM columns
        net.line["origin_id"] = pd.Series(["cim_id_1"], dtype=pd.StringDtype())
        net.line["origin_class"] = pd.Series(["ACLineSegment"], dtype=pd.StringDtype())
        net.line["description"] = pd.Series(["Test line"], dtype=pd.StringDtype())
        net.line["terminal_to"] = pd.Series(["term_to_1"], dtype=pd.StringDtype())
        net.line["terminal_from"] = pd.Series(["term_from_1"], dtype=pd.StringDtype())
        net.line["EquipmentContainer_id"] = pd.Series(["container_1"], dtype=pd.StringDtype())

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid when tdpf group is not triggered"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Line 1: name/type/alpha
        create_line(
            net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, name="test", alpha=0.0, std_type=STD_TYPE
        )
        # Line 2: max_loading_percent only (opf)
        create_line(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1.0,
            in_service=True,
            name="test",
            alpha=0.0,
            max_loading_percent=80.0,
            std_type=STD_TYPE,
        )
        # Line 3: zero-sequence params only
        create_line(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1.0,
            in_service=True,
            name="test",
            alpha=0.0,
            r0_ohm_per_km=0.1,
            x0_ohm_per_km=0.2,
            c0_nf_per_km=1.0,
            g0_us_per_km=0.0,
            std_type=STD_TYPE,
        )

        net.line["name"].iat[0] = pd.NA
        net.line["std_type"].iat[1] = pd.NA
        net.line["geo"].iat[2] = pd.NA

        validate_network(net)

    def test_tdpf_group_partial_missing_invalid(self):
        """Test: tdpf group must be complete if any tdpf value is set"""

        # Case 1: tdpf flag only -> invalid
        net = pandapowerNet(name="test_tdpf_group_partial_missing_invalid0")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)
        net.line["tdpf"] = True
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "tdpf")

        # Case 2: one tdpf param only -> invalid
        net = pandapowerNet(name="test_tdpf_group_partial_missing_invalid1")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)
        net.line["wind_speed_m_per_s"] = 3.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "tdpf")

        # Case 3: another tdpf param only -> invalid
        net = pandapowerNet(name="test_tdpf_group_partial_missing_invalid2")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)
        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)
        net.line["reference_temperature_degree_celsius"] = 20.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net, "tdpf")
        # TODO sc, 3ph not being checked in line.py

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                # Nullable string columns - include pd.NA directly
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["std_type"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
                itertools.product(["geo"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["terminal_to"], [pd.NA, *strings]),
                itertools.product(["terminal_from"], [pd.NA, *strings]),
                itertools.product(["EquipmentContainer_id"], [pd.NA, *strings]),
                itertools.product(["amica_name"], [pd.NA, *strings]),
                # Nullable float columns (non-TDPF group) - include float(np.nan) directly
                itertools.product(["r0_ohm_per_km"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["x0_ohm_per_km"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["c0_nf_per_km"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["g0_us_per_km"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["max_loading_percent"], [float(np.nan), *positiv_floats]),
                itertools.product(["alpha"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["temperature_degree_celsius"], [float(np.nan), *temp_valid]),
                # TDPF group columns - test non-NA values only here (NA tested separately)
                itertools.product(["endtemp_degree"], endtemp_valid),
                itertools.product(["tdpf"], bools),
                itertools.product(["wind_speed_m_per_s"], positiv_floats_plus_zero),
                itertools.product(["wind_angle_degree"], wind_angle_valid),
                itertools.product(["conductor_outer_diameter_m"], all_allowed_floats),
                itertools.product(["air_temperature_degree_celsius"], all_allowed_floats),
                itertools.product(["reference_temperature_degree_celsius"], ref_temp_valid),
                itertools.product(["solar_radiation_w_per_sq_m"], all_allowed_floats),
                itertools.product(["solar_absorptivity"], all_allowed_floats),
                itertools.product(["emissivity"], all_allowed_floats),
                itertools.product(["r_theta_kelvin_per_mw"], all_allowed_floats),
                itertools.product(["mc_joule_per_m_k"], all_allowed_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted (tdpf group satisfied)"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

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

        # Handle dtype preservation for nullable columns
        nullable_string_columns = [
            "name", "std_type", "type", "geo", "origin_id", "origin_class",
            "description", "terminal_to", "terminal_from", "EquipmentContainer_id", "amica_name"
        ]

        if parameter in nullable_string_columns:
            net.line[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        elif parameter == "tdpf":
            net.line[parameter] = pd.Series([valid_value], dtype=pd.BooleanDtype())
        else:
            net.line[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                # String columns - invalid types
                itertools.product(["name"], not_strings_list),
                itertools.product(["std_type"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["geo"], not_strings_list),
                itertools.product(["origin_id"], not_strings_list),
                itertools.product(["origin_class"], not_strings_list),
                itertools.product(["description"], not_strings_list),
                itertools.product(["terminal_to"], not_strings_list),
                itertools.product(["terminal_from"], not_strings_list),
                itertools.product(["EquipmentContainer_id"], not_strings_list),
                itertools.product(["amica_name"], not_strings_list),
                # Zero-sequence columns - must be >= 0 if provided
                itertools.product(["r0_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["x0_ohm_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["c0_nf_per_km"], [*negativ_floats, *not_floats_list]),
                itertools.product(["g0_us_per_km"], [*negativ_floats, *not_floats_list]),
                # OPF column
                itertools.product(["max_loading_percent"], [*negativ_floats_plus_zero, *not_floats_list]),
                # Thermal columns
                itertools.product(["alpha"], not_floats_list),
                itertools.product(["temperature_degree_celsius"], [*temp_invalid, *not_floats_list]),
                # TDPF group columns - invalid types
                itertools.product(["endtemp_degree"], [*endtemp_invalid, *not_floats_list]),
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
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        # Provide complete tdpf group so only the target parameter triggers failure
        net.line["tdpf"] = pd.Series([True], dtype=pd.BooleanDtype())
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


class TestLineDependencyGroupNullValues:
    """Tests for nullable dependency group columns - all columns in group set to NA together"""

    def test_tdpf_group_all_null_valid(self):
        """Test: TDPF group columns can all be NA/NaN together"""
        net = pandapowerNet(name="test_tdpf_group_all_null_valid")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        # Set all TDPF columns to NA/NaN with correct dtypes
        net.line["tdpf"] = pd.Series([pd.NA], dtype=pd.BooleanDtype())
        net.line["wind_speed_m_per_s"] = float("nan")
        net.line["wind_angle_degree"] = float("nan")
        net.line["conductor_outer_diameter_m"] = float("nan")
        net.line["air_temperature_degree_celsius"] = float("nan")
        net.line["reference_temperature_degree_celsius"] = float("nan")
        net.line["solar_radiation_w_per_sq_m"] = float("nan")
        net.line["solar_absorptivity"] = float("nan")
        net.line["emissivity"] = float("nan")
        net.line["r_theta_kelvin_per_mw"] = float("nan")
        net.line["mc_joule_per_m_k"] = float("nan")
        net.line["endtemp_degree"] = float("nan")

        validate_network(net)

    @pytest.mark.parametrize(
        "column_name",
        [
            "r0_ohm_per_km", "x0_ohm_per_km", "c0_nf_per_km", "g0_us_per_km",
            "max_loading_percent", "alpha", "temperature_degree_celsius"
        ],
    )
    def test_individual_nullable_float_column_nan_valid(self, column_name):
        """Test: Each nullable float column (not in TDPF group) accepts NaN individually"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        net.line[column_name] = float("nan")
        validate_network(net)

    def test_mixed_null_and_valid_values_in_rows(self):
        """Test: Multiple rows with mixed NA and valid values"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        # Row 1: all optional fields filled, TDPF group complete
        create_line(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=1.0,
            in_service=True,
            std_type=STD_TYPE,
            name="Line A",
            alpha=0.003,
        )

        # Row 2: all optional fields NA/NaN
        create_line(
            net,
            from_bus=b0,
            to_bus=b1,
            length_km=2.0,
            in_service=False,
            std_type=STD_TYPE,
        )

        # Set nullable columns with mixed values
        net.line["name"] = pd.Series(["Line A", pd.NA], dtype=pd.StringDtype())
        net.line["std_type"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())
        net.line["type"] = pd.Series(["ol", pd.NA], dtype=pd.StringDtype())
        net.line["geo"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())
        net.line["origin_id"] = pd.Series(["cim_1", pd.NA], dtype=pd.StringDtype())
        net.line["origin_class"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())

        # Float columns with mixed NaN
        net.line["alpha"] = [0.003, float("nan")]
        net.line["temperature_degree_celsius"] = [float("nan"), float("nan")]
        net.line["max_loading_percent"] = [100.0, float("nan")]
        net.line["r0_ohm_per_km"] = [0.1, float("nan")]
        net.line["x0_ohm_per_km"] = [0.2, float("nan")]

        # TDPF group - Row 1 has values, Row 2 has all NaN
        net.line["tdpf"] = pd.Series([True, pd.NA], dtype=pd.BooleanDtype())
        net.line["wind_speed_m_per_s"] = [5.0, float("nan")]
        net.line["wind_angle_degree"] = [90.0, float("nan")]
        net.line["conductor_outer_diameter_m"] = [0.03, float("nan")]
        net.line["air_temperature_degree_celsius"] = [20.0, float("nan")]
        net.line["reference_temperature_degree_celsius"] = [20.0, float("nan")]
        net.line["solar_radiation_w_per_sq_m"] = [200.0, float("nan")]
        net.line["solar_absorptivity"] = [0.5, float("nan")]
        net.line["emissivity"] = [0.9, float("nan")]
        net.line["r_theta_kelvin_per_mw"] = [2.0, float("nan")]
        net.line["mc_joule_per_m_k"] = [3600.0, float("nan")]
        net.line["endtemp_degree"] = [40.0, float("nan")]

        validate_network(net)

    def test_cim_columns_all_na_valid(self):
        """Test: All CIM-related columns can be NA"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        # CIM columns from schema metadata
        cim_string_columns = [
            "name", "origin_id", "origin_class", "description",
            "terminal_to", "terminal_from", "EquipmentContainer_id", "geo"
        ]

        for col in cim_string_columns:
            net.line[col] = pd.Series([pd.NA], dtype=pd.StringDtype())

        validate_network(net)


class TestLineForeignKey:
    """Tests for foreign key constraints on bus indices"""

    def test_invalid_bus_index(self):
        """Test: from_bus/to_bus must reference existing bus indices"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        net.line["from_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_to_bus_index(self):
        """Test: to_bus must reference existing bus indices"""
        net = pandapowerNet(name="test_invalid_to_bus_index")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_line(net, from_bus=b0, to_bus=b1, length_km=1.0, in_service=True, std_type=STD_TYPE)

        net.line["to_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FKs work with non-sequential bus indices"""
        net = create_empty_network()
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_line(net, from_bus=10, to_bus=42, length_km=1.0, in_service=True, std_type=STD_TYPE)
        create_line(net, from_bus=42, to_bus=100, length_km=2.0, in_service=True, std_type=STD_TYPE)
        create_line(net, from_bus=100, to_bus=10, length_km=1.5, in_service=False, std_type=STD_TYPE)

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

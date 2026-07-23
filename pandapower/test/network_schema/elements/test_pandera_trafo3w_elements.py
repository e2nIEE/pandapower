# test_pandera_trafo3w_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus
from pandapower.network import pandapowerNet
from pandapower.create._utils import add_column_to_df
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    negativ_ints,
    not_ints_list,
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats,
    negativ_floats_plus_zero,
    all_allowed_floats,
)

# Allowed/invalid categorical values for tap-related columns
allowed_tap_side = ["hv", "mv", "lv"]
invalid_tap_side = [s for s in strings if s not in allowed_tap_side]
allowed_tap_changer_types = ["Ratio", "Symmetrical", "Ideal", "Tabular"]
invalid_tap_changer_types = [s for s in strings if s not in allowed_tap_changer_types]


def _create_valid_trafo3w_dataframe():
    """Helper to create a valid trafo3w DataFrame with all required fields."""
    return pd.DataFrame(
        [
            {
                "hv_bus": 0,
                "mv_bus": 1,
                "lv_bus": 2,
                "vn_hv_kv": 110.0,
                "vn_mv_kv": 20.0,
                "vn_lv_kv": 10.0,
                "sn_hv_mva": 63.0,
                "sn_mv_mva": 25.0,
                "sn_lv_mva": 25.0,
                "vk_hv_percent": 10.0,
                "vk_mv_percent": 8.0,
                "vk_lv_percent": 6.0,
                "vkr_hv_percent": 0.5,
                "vkr_mv_percent": 0.4,
                "vkr_lv_percent": 0.3,
                "pfe_kw": 30.0,
                "i0_percent": 0.1,
                "shift_mv_degree": 0.0,
                "shift_lv_degree": 0.0,
                "in_service": True,
            }
        ]
    )


def _create_net_with_buses():
    """Helper to create a network with HV, MV, and LV buses."""
    net = pandapowerNet(name="_create_net_with_buses")
    create_bus(net, 110.0)  # index 0 (HV)
    create_bus(net, 20.0)   # index 1 (MV)
    create_bus(net, 10.0)   # index 2 (LV)
    return net


class TestTrafo3wRequiredFields:
    """Tests for required trafo3w fields - these columns must be present and non-null"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["hv_bus", "mv_bus", "lv_bus"], positiv_ints_plus_zero),
                itertools.product(["vn_hv_kv", "vn_mv_kv", "vn_lv_kv"], positiv_floats),
                itertools.product(["sn_hv_mva", "sn_mv_mva", "sn_lv_mva"], positiv_floats),
                itertools.product(["vk_hv_percent", "vk_mv_percent", "vk_lv_percent"], positiv_floats),
                itertools.product(["vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent"], positiv_floats_plus_zero),
                itertools.product(["pfe_kw", "i0_percent", "shift_mv_degree", "shift_lv_degree"], all_allowed_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = _create_net_with_buses()
        create_bus(net, 0.4, index=42)

        net.trafo3w = _create_valid_trafo3w_dataframe()

        if parameter in {"vk_hv_percent", "vk_mv_percent", "vk_lv_percent"}:
            net.trafo3w["vkr_hv_percent"] = 0.0
            net.trafo3w["vkr_mv_percent"] = 0.0
            net.trafo3w["vkr_lv_percent"] = 0.0

        net.trafo3w[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(
                    ["hv_bus", "mv_bus", "lv_bus"],
                    [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list],
                ),
                itertools.product(
                    [
                        "vn_hv_kv", "vn_mv_kv", "vn_lv_kv",
                        "sn_hv_mva", "sn_mv_mva", "sn_lv_mva",
                        "vk_hv_percent", "vk_mv_percent", "vk_lv_percent",
                    ],
                    [float(np.nan), pd.NA, None, 0.0, *negativ_floats, *not_floats_list],
                ),
                itertools.product(
                    ["vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent"],
                    [float(np.nan), pd.NA, None, *negativ_floats, *not_floats_list],
                ),
                itertools.product(
                    ["pfe_kw", "i0_percent", "shift_mv_degree", "shift_lv_degree"],
                    [float(np.nan), pd.NA, None, *not_floats_list],
                ),
                itertools.product(["in_service"], [float(np.nan), pd.NA, None, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Neutralize vkr<vk to focus on target validation
        net.trafo3w["vkr_hv_percent"] = 0.0
        net.trafo3w["vkr_mv_percent"] = 0.0
        net.trafo3w["vkr_lv_percent"] = 0.0

        net.trafo3w[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_vkr_less_than_vk_checks_pass(self):
        """Test: vkr < vk for all sides passes validation"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()
        validate_network(net)

    def test_vkr_greater_than_vk_fails(self):
        """Test: vkr > vk for any side fails validation"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()
        net.trafo3w["vkr_hv_percent"] = 12.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafo3wOptionalFieldsNullable:
    """Tests for optional nullable fields - can be absent, null, or have valid values"""

    def test_all_optional_fields_valid(self):
        """Test: all optional fields with valid values are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Optional strings (nullable)
        net.trafo3w["name"] = pd.Series(["T1"], dtype="string")
        net.trafo3w["std_type"] = pd.Series(["Type-A"], dtype="string")
        net.trafo3w["vector_group"] = pd.Series(["YNd"], dtype="string")

        # Tap group (complete)
        net.trafo3w["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo3w["tap_pos"] = 1.0
        net.trafo3w["tap_neutral"] = 0.0
        net.trafo3w["tap_min"] = -5.0
        net.trafo3w["tap_max"] = 5.0
        net.trafo3w["tap_step_percent"] = 1.25
        net.trafo3w["tap_step_degree"] = 0.0
        net.trafo3w["tap_at_star_point"] = pd.Series([True], dtype="boolean")
        net.trafo3w["tap_changer_type"] = pd.Series(["Ideal"], dtype="string")

        # TDT group (complete)
        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        # OPF
        net.trafo3w["max_loading_percent"] = 85.0

        # Other optional floats
        net.trafo3w["vkr0_x"] = 0.1
        net.trafo3w["vk0_x"] = 3.0
        net.trafo3w["vk0_hv_percent"] = 5.0
        net.trafo3w["vk0_mv_percent"] = 4.0
        net.trafo3w["vk0_lv_percent"] = 3.0
        net.trafo3w["vkr0_hv_percent"] = 0.2
        net.trafo3w["vkr0_mv_percent"] = 0.15
        net.trafo3w["vkr0_lv_percent"] = 0.1
        net.trafo3w["CurrentLimit.value_hv"] = 100.0
        net.trafo3w["CurrentLimit.value_mv"] = 80.0
        net.trafo3w["CurrentLimit.value_lv"] = 60.0
        net.trafo3w["OperationalLimitType.acceptableDuration_hv"] = 100.0
        net.trafo3w["OperationalLimitType.acceptableDuration_mv"] = 100.0
        net.trafo3w["OperationalLimitType.acceptableDuration_lv"] = 100.0

        # Optional boolean
        net.trafo3w["power_station_unit"] = pd.Series([False], dtype="boolean")

        # CIM identifiers
        net.trafo3w["origin_id"] = pd.Series(["PT_001"], dtype="string")
        net.trafo3w["origin_class"] = pd.Series(["PowerTransformer"], dtype="string")
        net.trafo3w["terminal_hv"] = pd.Series(["TERM_HV1"], dtype="string")
        net.trafo3w["terminal_mv"] = pd.Series(["TERM_MV1"], dtype="string")
        net.trafo3w["terminal_lv"] = pd.Series(["TERM_LV1"], dtype="string")
        net.trafo3w["PowerTransformerEnd_id_hv"] = pd.Series(["PTE_HV1"], dtype="string")
        net.trafo3w["PowerTransformerEnd_id_mv"] = pd.Series(["PTE_MV1"], dtype="string")
        net.trafo3w["PowerTransformerEnd_id_lv"] = pd.Series(["PTE_LV1"], dtype="string")
        net.trafo3w["tapchanger_class"] = pd.Series(["RatioTapChanger"], dtype="string")
        net.trafo3w["tapchanger_id"] = pd.Series(["TC1"], dtype="string")
        net.trafo3w["description"] = pd.Series(["Test 3W Transformer"], dtype="string")
        net.trafo3w["OperationalLimitType.limitType_hv"] = pd.Series(["patl"], dtype="string")
        net.trafo3w["OperationalLimitType.limitType_mv"] = pd.Series(["patl"], dtype="string")
        net.trafo3w["OperationalLimitType.limitType_lv"] = pd.Series(["patl"], dtype="string")

        validate_network(net)

    def test_all_optional_nullable_fields_with_nulls(self):
        """Test: all nullable optional fields with null values are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Optional strings
        net.trafo3w["name"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["std_type"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["vector_group"] = pd.Series([pd.NA], dtype="string")

        # Tap group (group not triggered)
        net.trafo3w["tap_side"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["tap_pos"] = np.nan
        net.trafo3w["tap_neutral"] = np.nan
        net.trafo3w["tap_min"] = np.nan
        net.trafo3w["tap_max"] = np.nan
        net.trafo3w["tap_step_percent"] = np.nan
        net.trafo3w["tap_step_degree"] = np.nan
        net.trafo3w["tap_at_star_point"] = pd.Series([pd.NA], dtype="boolean")
        net.trafo3w["tap_changer_type"] = pd.Series([pd.NA], dtype="string")

        # TDT group
        net.trafo3w["tap_dependency_table"] = pd.Series([pd.NA], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([pd.NA], dtype="Int64")

        # OPF
        net.trafo3w["max_loading_percent"] = np.nan

        # Other optional floats
        net.trafo3w["vkr0_x"] = np.nan
        net.trafo3w["vk0_x"] = np.nan
        net.trafo3w["vk0_hv_percent"] = np.nan
        net.trafo3w["vk0_mv_percent"] = np.nan
        net.trafo3w["vk0_lv_percent"] = np.nan
        net.trafo3w["vkr0_hv_percent"] = np.nan
        net.trafo3w["vkr0_mv_percent"] = np.nan
        net.trafo3w["vkr0_lv_percent"] = np.nan
        net.trafo3w["CurrentLimit.value_hv"] = np.nan
        net.trafo3w["CurrentLimit.value_mv"] = np.nan
        net.trafo3w["CurrentLimit.value_lv"] = np.nan
        net.trafo3w["OperationalLimitType.acceptableDuration_hv"] = np.nan
        net.trafo3w["OperationalLimitType.acceptableDuration_mv"] = np.nan
        net.trafo3w["OperationalLimitType.acceptableDuration_lv"] = np.nan

        # Optional boolean
        net.trafo3w["power_station_unit"] = pd.Series([pd.NA], dtype="boolean")

        # CIM identifiers
        net.trafo3w["origin_id"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["origin_class"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["terminal_hv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["terminal_mv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["terminal_lv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["PowerTransformerEnd_id_hv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["PowerTransformerEnd_id_mv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["PowerTransformerEnd_id_lv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["tapchanger_class"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["tapchanger_id"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["description"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["OperationalLimitType.limitType_hv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["OperationalLimitType.limitType_mv"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["OperationalLimitType.limitType_lv"] = pd.Series([pd.NA], dtype="string")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["std_type"], [pd.NA, *strings]),
                itertools.product(["vector_group"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["terminal_hv"], [pd.NA, *strings]),
                itertools.product(["terminal_mv"], [pd.NA, *strings]),
                itertools.product(["terminal_lv"], [pd.NA, *strings]),
                itertools.product(["PowerTransformerEnd_id_hv"], [pd.NA, *strings]),
                itertools.product(["PowerTransformerEnd_id_mv"], [pd.NA, *strings]),
                itertools.product(["PowerTransformerEnd_id_lv"], [pd.NA, *strings]),
                itertools.product(["tapchanger_class"], [pd.NA, *strings]),
                itertools.product(["tapchanger_id"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["OperationalLimitType.limitType_hv"], [pd.NA, *strings]),
                itertools.product(["OperationalLimitType.limitType_mv"], [pd.NA, *strings]),
                itertools.product(["OperationalLimitType.limitType_lv"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_string_values(self, parameter, valid_value):
        """Test: valid optional string values (nullable) are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        add_column_to_df(net, 'trafo3w', parameter)
        net.trafo3w.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
                itertools.product(["std_type"], [float(np.nan), *not_strings_list]),
                itertools.product(["vector_group"], [float(np.nan), *not_strings_list]),
                itertools.product(["origin_id"], [float(np.nan), *not_strings_list]),
                itertools.product(["origin_class"], [float(np.nan), *not_strings_list]),
                itertools.product(["terminal_hv"], [float(np.nan), *not_strings_list]),
                itertools.product(["terminal_mv"], [float(np.nan), *not_strings_list]),
                itertools.product(["terminal_lv"], [float(np.nan), *not_strings_list]),
                itertools.product(["PowerTransformerEnd_id_hv"], [float(np.nan), *not_strings_list]),
                itertools.product(["PowerTransformerEnd_id_mv"], [float(np.nan), *not_strings_list]),
                itertools.product(["PowerTransformerEnd_id_lv"], [float(np.nan), *not_strings_list]),
                itertools.product(["tapchanger_class"], [float(np.nan), *not_strings_list]),
                itertools.product(["tapchanger_id"], [float(np.nan), *not_strings_list]),
                itertools.product(["description"], [float(np.nan), *not_strings_list]),
                itertools.product(["OperationalLimitType.limitType_hv"], [float(np.nan), *not_strings_list]),
                itertools.product(["OperationalLimitType.limitType_mv"], [float(np.nan), *not_strings_list]),
                itertools.product(["OperationalLimitType.limitType_lv"], [float(np.nan), *not_strings_list]),
            )
        ),
    )
    def test_invalid_optional_string_values(self, parameter, invalid_value):
        """Test: invalid optional string values are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        add_column_to_df(net, 'trafo3w', parameter)
        net.trafo3w[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["tap_side"], [*allowed_tap_side]), # pd.NA not included due to dependencies
                itertools.product(["tap_changer_type"], [pd.NA, *allowed_tap_changer_types]),
            )
        ),
    )
    def test_valid_categorical_string_values(self, parameter, valid_value):
        """Test: valid categorical string values are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy tap group dependencies
        self._setup_tap_group(net)

        add_column_to_df(net, 'trafo3w', parameter)
        net.trafo3w.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_side"], [*not_strings_list, *invalid_tap_side]),
                itertools.product(["tap_changer_type"], [float(np.nan), *not_strings_list, *invalid_tap_changer_types]),
            )
        ),
    )
    def test_invalid_categorical_string_values(self, parameter, invalid_value):
        """Test: invalid categorical string values are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy tap group dependencies
        self._setup_tap_group(net)

        net.trafo3w[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["tap_pos"], [*all_allowed_floats]),
                itertools.product(["tap_neutral"], [*all_allowed_floats]),
                itertools.product(["tap_min"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["tap_max"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["tap_step_degree"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vkr0_x"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vk0_x"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["max_loading_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vk0_hv_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vk0_mv_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vk0_lv_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vkr0_hv_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vkr0_mv_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["vkr0_lv_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["CurrentLimit.value_hv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["CurrentLimit.value_mv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["CurrentLimit.value_lv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["OperationalLimitType.acceptableDuration_hv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["OperationalLimitType.acceptableDuration_mv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["OperationalLimitType.acceptableDuration_lv"], [float(np.nan), *all_allowed_floats]),
            )
        ),
    )
    def test_valid_optional_float_no_range_values(self, parameter, valid_value):
        """Test: valid optional float values (nullable, no range) are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy tap group for tap group members
        if parameter in ("tap_pos", "tap_neutral"):
            self._setup_tap_group(net)

        add_column_to_df(net, 'trafo3w', parameter)
        net.trafo3w.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_pos"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_neutral"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_min"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_max"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_step_degree"], [pd.NA, *not_floats_list]),
                itertools.product(["vkr0_x"], [pd.NA, *not_floats_list]),
                itertools.product(["vk0_x"], [pd.NA, *not_floats_list]),
                itertools.product(["max_loading_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["vk0_hv_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["vk0_mv_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["vk0_lv_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["vkr0_hv_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["vkr0_mv_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["vkr0_lv_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["CurrentLimit.value_hv"], [pd.NA, *not_floats_list]),
                itertools.product(["CurrentLimit.value_mv"], [pd.NA, *not_floats_list]),
                itertools.product(["CurrentLimit.value_lv"], [pd.NA, *not_floats_list]),
                itertools.product(["OperationalLimitType.acceptableDuration_hv"], [pd.NA, *not_floats_list]),
                itertools.product(["OperationalLimitType.acceptableDuration_mv"], [pd.NA, *not_floats_list]),
                itertools.product(["OperationalLimitType.acceptableDuration_lv"], [pd.NA, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_float_no_range_values(self, parameter, invalid_value):
        """Test: invalid optional float values (wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy tap group for tap group members
        if parameter in ("tap_pos", "tap_neutral"):
            self._setup_tap_group(net)

        net.trafo3w[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["tap_step_percent"], [float(np.nan), *positiv_floats]),
            )
        ),
    )
    def test_valid_optional_float_gt_zero_values(self, parameter, valid_value):
        """Test: valid optional float values (nullable, > 0) are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        add_column_to_df(net, 'trafo3w', parameter)
        net.trafo3w.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_step_percent"], [pd.NA, 0.0, *negativ_floats, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_float_gt_zero_values(self, parameter, invalid_value):
        """Test: invalid optional float values (<= 0 or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["tap_at_star_point"], [pd.NA, *bools]),
                itertools.product(["power_station_unit"], [pd.NA, *bools]),
                itertools.product(["tap_dependency_table"], [*bools]),
            )
        ),
    )
    def test_valid_optional_boolean_values(self, parameter, valid_value):
        """Test: valid optional boolean values (nullable) are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy TDT group if testing tap_dependency_table
        if parameter == "tap_dependency_table":
            net.trafo3w["id_characteristic_table"] = pd.Series([0 if valid_value in bools else pd.NA], dtype="Int64")

        add_column_to_df(net, 'trafo3w', parameter)
        net.trafo3w.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_at_star_point"], [float(np.nan), *not_boolean_list]),
                itertools.product(["power_station_unit"], [float(np.nan), *not_boolean_list]),
                itertools.product(["tap_dependency_table"], [float(np.nan), *not_boolean_list]),
            )
        ),
    )
    def test_invalid_optional_boolean_values(self, parameter, invalid_value):
        """Test: invalid optional boolean values (wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy TDT group if testing tap_dependency_table
        if parameter == "tap_dependency_table":
            net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        net.trafo3w[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize("valid_value", [pd.NA, *positiv_ints_plus_zero])
    def test_valid_id_characteristic_table_values(self, valid_value):
        """Test: valid id_characteristic_table values (nullable, >= 0) are accepted"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy TDT group
        is_null = pd.isna(valid_value) if not isinstance(valid_value, int) else False
        net.trafo3w["tap_dependency_table"] = pd.Series([pd.NA if is_null else True], dtype="boolean")

        add_column_to_df(net, 'trafo3w', 'id_characteristic_table')
        net.trafo3w.at[0, 'id_characteristic_table'] = valid_value

        validate_network(net)

    @pytest.mark.parametrize("invalid_value", [float(np.nan), *negativ_ints, *not_ints_list])
    def test_invalid_id_characteristic_table_values(self, invalid_value):
        """Test: invalid id_characteristic_table values (< 0 or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Satisfy TDT group
        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def _setup_tap_group(self, net):
        """Setup complete tap group to satisfy dependencies"""
        net.trafo3w["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo3w["tap_pos"] = 1.0
        net.trafo3w["tap_neutral"] = 0.0


class TestTrafo3wDependencies:
    """Tests for group dependencies (tap, tdt)"""

    def test_tap_group_complete_valid(self):
        """Test: tap group with all columns present is valid"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["tap_side"] = pd.Series(["lv"], dtype="string")
        net.trafo3w["tap_pos"] = 2.0
        net.trafo3w["tap_neutral"] = 0.0

        validate_network(net)

    def test_tap_group_all_null_valid(self):
        """Test: tap group columns can all be NA/NaN together (group not triggered)"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["tap_side"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.trafo3w["tap_pos"] = float("nan")
        net.trafo3w["tap_neutral"] = float("nan")

        validate_network(net)

    @pytest.mark.parametrize("missing_column", ["tap_pos", "tap_neutral", "tap_side"])
    def test_tap_group_partial_missing_invalid(self, missing_column):
        """Test: tap group must be complete - partial values are invalid"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Set all tap columns
        net.trafo3w["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo3w["tap_pos"] = 1.0
        net.trafo3w["tap_neutral"] = 0.0

        # Set the target column to null
        if missing_column == "tap_side":
            net.trafo3w["tap_side"] = pd.Series([pd.NA], dtype="string")
        else:
            net.trafo3w[missing_column] = float("nan")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize("single_column", ["tap_pos", "tap_neutral", "tap_side"])
    def test_tap_group_only_one_set_invalid(self, single_column):
        """Test: tap group must be complete - only one column set is invalid"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Set only one tap column
        if single_column == "tap_side":
            net.trafo3w["tap_side"] = pd.Series(["hv"], dtype="string")
        else:
            net.trafo3w[single_column] = 1.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tap_group_row_consistency_valid(self):
        """Test: tap group - each row must have all values or all NA"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Add second row
        row2 = _create_valid_trafo3w_dataframe()
        net.trafo3w = pd.concat([net.trafo3w, row2], ignore_index=True)

        # Row 1: all tap values present
        # Row 2: all tap values NA
        net.trafo3w["tap_side"] = pd.Series(["hv", pd.NA], dtype=pd.StringDtype())
        net.trafo3w["tap_pos"] = [1.0, float("nan")]
        net.trafo3w["tap_neutral"] = [0.0, float("nan")]

        validate_network(net)

    def test_tap_group_row_partial_invalid(self):
        """Test: tap group - partial values in a row should fail"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Add second row
        row2 = _create_valid_trafo3w_dataframe()
        net.trafo3w = pd.concat([net.trafo3w, row2], ignore_index=True)

        # Row 1: all tap values present
        # Row 2: partial tap values (tap_side set, others NA)
        net.trafo3w["tap_side"] = pd.Series(["hv", "lv"], dtype=pd.StringDtype())
        net.trafo3w["tap_pos"] = [1.0, float("nan")]
        net.trafo3w["tap_neutral"] = [0.0, float("nan")]

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tdt_group_complete_valid(self):
        """Test: TDT group with all columns present is valid"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        validate_network(net)

    def test_tdt_group_all_na_valid(self):
        """Test: TDT group columns can all be NA together"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["tap_dependency_table"] = pd.Series([pd.NA], dtype=pd.BooleanDtype())
        net.trafo3w["id_characteristic_table"] = pd.Series([pd.NA], dtype="Int64")

        validate_network(net)

    def test_tdt_group_only_tap_dependency_table_set_invalid(self):
        """Test: TDT group - only tap_dependency_table set is invalid"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tdt_group_partial_na_invalid(self):
        """Test: TDT group - one value present, one NA is invalid"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([pd.NA], dtype="Int64")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tdt_group_row_consistency_valid(self):
        """Test: TDT group - each row must have all values or all NA"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Add second row
        row2 = _create_valid_trafo3w_dataframe()
        net.trafo3w = pd.concat([net.trafo3w, row2], ignore_index=True)

        # Row 1: complete, Row 2: all NA
        net.trafo3w["tap_dependency_table"] = pd.Series([True, pd.NA], dtype=pd.BooleanDtype())
        net.trafo3w["id_characteristic_table"] = pd.Series([0, pd.NA], dtype="Int64")

        validate_network(net)

    def test_tdt_group_row_partial_invalid(self):
        """Test: TDT group - partial values in a row should fail"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        # Add second row
        row2 = _create_valid_trafo3w_dataframe()
        net.trafo3w = pd.concat([net.trafo3w, row2], ignore_index=True)

        # Row 1: complete, Row 2: partial
        net.trafo3w["tap_dependency_table"] = pd.Series([True, True], dtype=pd.BooleanDtype())
        net.trafo3w["id_characteristic_table"] = pd.Series([0, pd.NA], dtype="Int64")

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafo3wForeignKeys:
    """Tests for foreign key constraints (hv_bus, mv_bus, lv_bus -> bus.index)"""

    def test_invalid_hv_bus_index(self):
        """Test: hv_bus FK must reference an existing bus index"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["hv_bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_mv_bus_index(self):
        """Test: mv_bus FK must reference an existing bus index"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["mv_bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_lv_bus_index(self):
        """Test: lv_bus FK must reference an existing bus index"""
        net = _create_net_with_buses()
        net.trafo3w = _create_valid_trafo3w_dataframe()

        net.trafo3w["lv_bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 110.0, index=10)
        create_bus(net, 20.0, index=20)
        create_bus(net, 10.0, index=42)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 10,
            "mv_bus": 20,
            "lv_bus": 42,
            "vn_hv_kv": 110.0,
            "vn_mv_kv": 20.0,
            "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0,
            "sn_mv_mva": 25.0,
            "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0,
            "vk_mv_percent": 8.0,
            "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5,
            "vkr_mv_percent": 0.4,
            "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0,
            "i0_percent": 0.1,
            "shift_mv_degree": 0.0,
            "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        validate_network(net)


class TestTrafo3wResults:
    """Tests for trafo3w results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented - requires power flow calculation")
    def test_trafo3w_power_flows(self):
        """Test: Power flow result fields have valid numeric values"""
        pass

    @pytest.mark.skip(reason="Not yet implemented - requires short-circuit calculation")
    def test_trafo3w_short_circuit_results(self):
        """Test: Short-circuit result fields contain expected ranges"""
        pass

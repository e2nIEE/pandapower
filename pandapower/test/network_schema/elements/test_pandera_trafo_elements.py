import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_transformer
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
    positiv_ints,
    negativ_ints,
    negativ_ints_plus_zero,
    not_ints_list,
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats,
    negativ_floats_plus_zero,
    all_allowed_floats,
)

# Allowed/invalid categorical values for tap-related columns
allowed_tap_side = ["hv", "lv"]
invalid_tap_side = [s for s in strings if s not in allowed_tap_side]
allowed_tap_changer_types = ["Ratio", "Symmetrical", "Ideal", "Tabular"]
invalid_tap_changer_types = [s for s in strings if s not in allowed_tap_changer_types]
allowed_tap2_changer_types = ["Ratio", "Symmetrical", "Ideal", "nan"]
invalid_tap2_changer_types = [s for s in strings if s not in allowed_tap2_changer_types]

# df must be in (0, 1] - exclusive min, inclusive max
df_valid_range = [0.001, 0.1, 0.5, 0.999, 1.0]
df_invalid_range = [0.0, -0.1, 1.1, 2.0]

# leakage ratios must be in [0, 1] - inclusive both ends
leakage_ratio_valid = [0.0, 0.25, 0.5, 0.75, 1.0]
leakage_ratio_invalid = [-0.1, -1.0, 1.1, 2.0]


def _create_valid_trafo_dataframe():
    """Helper to create a valid trafo DataFrame with all required fields."""
    return pd.DataFrame(
        [
            {
                "hv_bus": 0,
                "lv_bus": 1,
                "sn_mva": 25.0,
                "vn_hv_kv": 110.0,
                "vn_lv_kv": 10.0,
                "vk_percent": 10.0,
                "vkr_percent": 0.5,
                "pfe_kw": 30.0,
                "i0_percent": 0.1,
                "shift_degree": 0.0,
                "parallel": 1,
                "in_service": True,
            }
        ]
    )


def _create_net_with_buses():
    """Helper to create a network with HV and LV buses."""
    net = pandapowerNet(name="_create_net_with_buses")
    create_bus(net, 110.0)  # index 0 (HV)
    create_bus(net, 10.0)   # index 1 (LV)
    return net


class TestTrafoRequiredFields:
    """Tests for required trafo fields - these columns must be present and non-null"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["hv_bus", "lv_bus"], positiv_ints_plus_zero),
                itertools.product(["sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent"], positiv_floats),
                itertools.product(["vkr_percent", "pfe_kw", "i0_percent"], positiv_floats_plus_zero),
                itertools.product(["shift_degree"], all_allowed_floats),
                itertools.product(["parallel"], positiv_ints),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = _create_net_with_buses()
        create_bus(net, 0.4, index=42)  # Additional bus for testing higher indices

        net.trafo = _create_valid_trafo_dataframe()
        net.trafo[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product( ["hv_bus", "lv_bus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(
                    ["sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent"],
                    [float(np.nan), pd.NA, None, 0.0, *negativ_floats, *not_floats_list],
                ),
                itertools.product(
                    ["vkr_percent", "pfe_kw", "i0_percent"],
                    [float(np.nan), pd.NA, None, *negativ_floats, *not_floats_list],
                ),
                itertools.product(["shift_degree"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["parallel"], [float(np.nan), pd.NA, None, 0, *negativ_ints, *not_ints_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, None, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()
        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafoOptionalFieldsNullable:
    """Tests for optional nullable fields - can be absent, null, or have valid values"""

    def test_all_optional_fields_valid(self):
        """Test: all optional fields with valid values are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Optional strings (nullable)
        net.trafo["name"] = pd.Series(["T1"], dtype="string")
        net.trafo["std_type"] = pd.Series(["Type-A"], dtype="string")
        net.trafo["vector_group"] = pd.Series(["Dyn5"], dtype="string")

        # Tap group (complete)
        net.trafo["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo["tap_pos"] = 1.0
        net.trafo["tap_neutral"] = 0.0
        net.trafo["tap_min"] = -5.0
        net.trafo["tap_max"] = 5.0
        net.trafo["tap_step_percent"] = 1.25
        net.trafo["tap_step_degree"] = 0.0
        net.trafo["tap_changer_type"] = pd.Series(["Ratio"], dtype="string")
        net.trafo["tapchanger_id"] = pd.Series(["TC1"], dtype="string")
        net.trafo["tapchanger_class"] = pd.Series(["RatioTapChanger"], dtype="string")

        # Tap2 group (complete)
        net.trafo["tap2_side"] = pd.Series(["lv"], dtype="string")
        net.trafo["tap2_pos"] = 0.0
        net.trafo["tap2_neutral"] = 0.0
        net.trafo["tap2_min"] = -3.0
        net.trafo["tap2_max"] = 3.0
        net.trafo["tap2_step_percent"] = 1.0
        net.trafo["tap2_step_degree"] = 0.0
        net.trafo["tap2_changer_type"] = pd.Series(["Ideal"], dtype="string")
        net.trafo["tapchanger2_id"] = pd.Series(["TC2"], dtype="string")
        net.trafo["tapchanger2_class"] = pd.Series(["PhaseTapChanger"], dtype="string")

        # TDT group (complete)
        net.trafo["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        # OPF
        net.trafo["max_loading_percent"] = 100.0

        # Other optional
        net.trafo["df"] = 0.8
        net.trafo["oltc"] = pd.Series([True], dtype="boolean")
        net.trafo["power_station_unit"] = pd.Series([False], dtype="boolean")

        # Zero-sequence / SC parameters
        net.trafo["vk0_percent"] = 3.0
        net.trafo["vkr0_percent"] = 0.5
        net.trafo["mag0_percent"] = 60.0
        net.trafo["mag0_rx"] = 15.0
        net.trafo["si0_hv_partial"] = 0.5
        net.trafo["leakage_resistance_ratio_hv"] = 0.5
        net.trafo["leakage_reactance_ratio_hv"] = 0.5
        net.trafo["xn_ohm"] = 0.0
        net.trafo["rn_ohm"] = 10.0
        net.trafo["pt_percent"] = 100.0

        # CIM identifiers
        net.trafo["PowerTransformerEnd_id_hv"] = pd.Series(["PTE_HV1"], dtype="string")
        net.trafo["PowerTransformerEnd_id_lv"] = pd.Series(["PTE_LV1"], dtype="string")
        net.trafo["terminal_hv"] = pd.Series(["TERM_HV1"], dtype="string")
        net.trafo["terminal_lv"] = pd.Series(["TERM_LV1"], dtype="string")
        net.trafo["origin_class"] = pd.Series(["PowerTransformer"], dtype="string")
        net.trafo["origin_id"] = pd.Series(["PT_001"], dtype="string")

        # Description and operational limits
        net.trafo["description"] = pd.Series(["Test Transformer"], dtype="string")
        net.trafo["OperationalLimitType.limitType_hv"] = pd.Series(["patl"], dtype="string")
        net.trafo["OperationalLimitType.limitType_lv"] = pd.Series(["patl"], dtype="string")
        net.trafo["CurrentLimit.value_hv"] = 100.0
        net.trafo["CurrentLimit.value_lv"] = 100.0
        net.trafo["OperationalLimitType.acceptableDuration_hv"] = 100.0
        net.trafo["OperationalLimitType.acceptableDuration_lv"] = 100.0
        net.trafo["amica_name"] = pd.Series(["AMICA_T1"], dtype="string")

        validate_network(net)

    def test_all_optional_nullable_fields_with_nulls(self):
        """Test: all nullable optional fields with null values are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Optional strings -> all nulls
        net.trafo["name"] = pd.Series([pd.NA], dtype="string")
        net.trafo["std_type"] = pd.Series([pd.NA], dtype="string")
        net.trafo["vector_group"] = pd.Series([pd.NA], dtype="string")

        # Tap group -> all nulls (group not triggered)
        net.trafo["tap_side"] = pd.Series([pd.NA], dtype="string")
        net.trafo["tap_pos"] = np.nan
        net.trafo["tap_neutral"] = np.nan
        net.trafo["tap_min"] = np.nan
        net.trafo["tap_max"] = np.nan
        net.trafo["tap_step_percent"] = np.nan
        net.trafo["tap_step_degree"] = np.nan
        net.trafo["tap_changer_type"] = pd.Series([pd.NA], dtype="string")
        net.trafo["tapchanger_id"] = pd.Series([pd.NA], dtype="string")
        net.trafo["tapchanger_class"] = pd.Series([pd.NA], dtype="string")

        # Tap2 group -> all nulls
        net.trafo["tap2_side"] = pd.Series([pd.NA], dtype="string")
        net.trafo["tap2_pos"] = np.nan
        net.trafo["tap2_neutral"] = np.nan
        net.trafo["tap2_min"] = np.nan
        net.trafo["tap2_max"] = np.nan
        net.trafo["tap2_step_percent"] = np.nan
        net.trafo["tap2_step_degree"] = np.nan
        net.trafo["tap2_changer_type"] = pd.Series([pd.NA], dtype="string")
        net.trafo["tapchanger2_id"] = pd.Series([pd.NA], dtype="string")
        net.trafo["tapchanger2_class"] = pd.Series([pd.NA], dtype="string")

        # TDT group -> all nulls
        net.trafo["tap_dependency_table"] = pd.Series([pd.NA], dtype="boolean")
        net.trafo["id_characteristic_table"] = pd.Series([pd.NA], dtype="Int64")

        # OPF
        net.trafo["max_loading_percent"] = np.nan

        # Other optional nullables
        net.trafo["df"] = np.nan
        net.trafo["oltc"] = pd.Series([pd.NA], dtype="boolean")
        net.trafo["power_station_unit"] = pd.Series([pd.NA], dtype="boolean")

        # Zero-sequence / SC parameters -> all nulls
        net.trafo["vk0_percent"] = np.nan
        net.trafo["vkr0_percent"] = np.nan
        net.trafo["mag0_percent"] = np.nan
        net.trafo["mag0_rx"] = np.nan
        net.trafo["si0_hv_partial"] = np.nan
        net.trafo["leakage_resistance_ratio_hv"] = np.nan
        net.trafo["leakage_reactance_ratio_hv"] = np.nan

        # CIM identifiers -> all nulls
        net.trafo["PowerTransformerEnd_id_hv"] = pd.Series([pd.NA], dtype="string")
        net.trafo["PowerTransformerEnd_id_lv"] = pd.Series([pd.NA], dtype="string")
        net.trafo["terminal_hv"] = pd.Series([pd.NA], dtype="string")
        net.trafo["terminal_lv"] = pd.Series([pd.NA], dtype="string")
        net.trafo["origin_class"] = pd.Series([pd.NA], dtype="string")
        net.trafo["origin_id"] = pd.Series([pd.NA], dtype="string")

        # Description and operational limits -> all nulls
        net.trafo["description"] = pd.Series([pd.NA], dtype="string")
        net.trafo["OperationalLimitType.limitType_hv"] = pd.Series([pd.NA], dtype="string")
        net.trafo["OperationalLimitType.limitType_lv"] = pd.Series([pd.NA], dtype="string")
        net.trafo["CurrentLimit.value_hv"] = np.nan
        net.trafo["CurrentLimit.value_lv"] = np.nan
        net.trafo["OperationalLimitType.acceptableDuration_hv"] = np.nan
        net.trafo["OperationalLimitType.acceptableDuration_lv"] = np.nan
        net.trafo["amica_name"] = pd.Series([pd.NA], dtype="string")

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
                itertools.product(["terminal_lv"], [pd.NA, *strings]),
                itertools.product(["PowerTransformerEnd_id_hv"], [pd.NA, *strings]),
                itertools.product(["PowerTransformerEnd_id_lv"], [pd.NA, *strings]),
                itertools.product(["tapchanger_class"], [pd.NA, *strings]),
                itertools.product(["tapchanger_id"], [pd.NA, *strings]),
                itertools.product(["tapchanger2_class"], [pd.NA, *strings]),
                itertools.product(["tapchanger2_id"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["OperationalLimitType.limitType_hv"], [pd.NA, *strings]),
                itertools.product(["OperationalLimitType.limitType_lv"], [pd.NA, *strings]),
                itertools.product(["amica_name"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_string_values(self, parameter, valid_value):
        """Test: valid optional string values (nullable) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

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
                itertools.product(["terminal_lv"], [float(np.nan), *not_strings_list]),
                itertools.product(["PowerTransformerEnd_id_hv"], [float(np.nan), *not_strings_list]),
                itertools.product(["PowerTransformerEnd_id_lv"], [float(np.nan), *not_strings_list]),
                itertools.product(["tapchanger_class"], [float(np.nan), *not_strings_list]),
                itertools.product(["tapchanger_id"], [float(np.nan), *not_strings_list]),
                itertools.product(["tapchanger2_class"], [float(np.nan), *not_strings_list]),
                itertools.product(["tapchanger2_id"], [float(np.nan), *not_strings_list]),
                itertools.product(["description"], [float(np.nan), *not_strings_list]),
                itertools.product(["OperationalLimitType.limitType_hv"], [float(np.nan), *not_strings_list]),
                itertools.product(["OperationalLimitType.limitType_lv"], [float(np.nan), *not_strings_list]),
                itertools.product(["amica_name"], [float(np.nan), *not_strings_list]),
            )
        ),
    )
    def test_invalid_optional_string_values(self, parameter, invalid_value):
        """Test: invalid optional string values are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        add_column_to_df(net, 'trafo', parameter)
        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["tap_side"], [*allowed_tap_side]),
                itertools.product(["tap2_side"], [*allowed_tap_side]),
                itertools.product(["tap_changer_type"], [pd.NA, *allowed_tap_changer_types]),
                itertools.product(["tap2_changer_type"], [pd.NA, *allowed_tap2_changer_types]),
            )
        ),
    )
    def test_valid_categorical_string_values(self, parameter, valid_value):
        """Test: valid categorical string values are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_side"], [*not_strings_list, *invalid_tap_side]),
                itertools.product(["tap2_side"], [*not_strings_list, *invalid_tap_side]),
                itertools.product(["tap_changer_type"], [float(np.nan), *not_strings_list, *invalid_tap_changer_types]),
                itertools.product(["tap2_changer_type"], [float(np.nan), *not_strings_list, *invalid_tap2_changer_types]),
            )
        ),
    )
    def test_invalid_categorical_string_values(self, parameter, invalid_value):
        """Test: invalid categorical string values are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        net.trafo[parameter] = invalid_value

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
                itertools.product(["tap2_pos"], [*all_allowed_floats]),
                itertools.product(["tap2_neutral"], [*all_allowed_floats]),
                itertools.product(["tap2_min"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["tap2_max"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["mag0_rx"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["max_loading_percent"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["CurrentLimit.value_hv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["CurrentLimit.value_lv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["OperationalLimitType.acceptableDuration_hv"], [float(np.nan), *all_allowed_floats]),
                itertools.product(["OperationalLimitType.acceptableDuration_lv"], [float(np.nan), *all_allowed_floats]),
            )
        ),
    )
    def test_valid_optional_float_no_range_values(self, parameter, valid_value):
        """Test: valid optional float values (nullable, no range) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_pos"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_neutral"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_min"], [pd.NA, *not_floats_list]),
                itertools.product(["tap_max"], [pd.NA, *not_floats_list]),
                itertools.product(["tap2_pos"], [pd.NA, *not_floats_list]),
                itertools.product(["tap2_neutral"], [pd.NA, *not_floats_list]),
                itertools.product(["tap2_min"], [pd.NA, *not_floats_list]),
                itertools.product(["tap2_max"], [pd.NA, *not_floats_list]),
                itertools.product(["mag0_rx"], [pd.NA, *not_floats_list]),
                itertools.product(["max_loading_percent"], [pd.NA, *not_floats_list]),
                itertools.product(["CurrentLimit.value_hv"], [pd.NA, *not_floats_list]),
                itertools.product(["CurrentLimit.value_lv"], [pd.NA, *not_floats_list]),
                itertools.product(["OperationalLimitType.acceptableDuration_hv"], [pd.NA, *not_floats_list]),
                itertools.product(["OperationalLimitType.acceptableDuration_lv"], [pd.NA, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_float_no_range_values(self, parameter, invalid_value):
        """Test: invalid optional float values (wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["vk0_percent"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["vkr0_percent"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["mag0_percent"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["si0_hv_partial"], [float(np.nan), *positiv_floats_plus_zero]),
                itertools.product(["tap_step_degree"], [*positiv_floats_plus_zero]),
                itertools.product(["tap2_step_degree"], [*positiv_floats_plus_zero]),
            )
        ),
    )
    def test_valid_optional_float_ge_zero_values(self, parameter, valid_value):
        """Test: valid optional float values (nullable, >= 0) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["vk0_percent"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["vkr0_percent"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["mag0_percent"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["si0_hv_partial"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["tap_step_degree"], [pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["tap2_step_degree"], [pd.NA, *negativ_floats, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_float_ge_zero_values(self, parameter, invalid_value):
        """Test: invalid optional float values (< 0 or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["tap_step_percent"], [*positiv_floats]),
                itertools.product(["tap2_step_percent"], [*positiv_floats]),
            )
        ),
    )
    def test_valid_optional_float_gt_zero_values(self, parameter, valid_value):
        """Test: valid optional float values (nullable, > 0) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["tap_step_percent"], [pd.NA, 0.0, *negativ_floats, *not_floats_list]),
                itertools.product(["tap2_step_percent"], [pd.NA, 0.0, *negativ_floats, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_float_gt_zero_values(self, parameter, invalid_value):
        """Test: invalid optional float values (<= 0 or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy tap/tap2 group dependencies
        self._setup_tap_group(net)
        self._setup_tap2_group(net)

        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["leakage_resistance_ratio_hv"], [float(np.nan), *leakage_ratio_valid]),
                itertools.product(["leakage_reactance_ratio_hv"], [float(np.nan), *leakage_ratio_valid]),
            )
        ),
    )
    def test_valid_leakage_ratio_values(self, parameter, valid_value):
        """Test: valid leakage ratio values (nullable, [0, 1]) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["leakage_resistance_ratio_hv"], [pd.NA, *leakage_ratio_invalid, *not_floats_list]),
                itertools.product(["leakage_reactance_ratio_hv"], [pd.NA, *leakage_ratio_invalid, *not_floats_list]),
            )
        ),
    )
    def test_invalid_leakage_ratio_values(self, parameter, invalid_value):
        """Test: invalid leakage ratio values (outside [0, 1] or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    # df column (nullable, (0, 1])
    @pytest.mark.parametrize("valid_value", [float(np.nan), *df_valid_range])
    def test_valid_df_values(self, valid_value):
        """Test: valid df values (nullable, (0, 1]) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        add_column_to_df(net, 'trafo', 'df')
        net.trafo.at[0, 'df'] = valid_value

        validate_network(net)

    @pytest.mark.parametrize("invalid_value", [pd.NA, *df_invalid_range, *not_floats_list])
    def test_invalid_df_values(self, invalid_value):
        """Test: invalid df values (outside (0, 1] or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        net.trafo['df'] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["oltc"], [pd.NA, *bools]),
                itertools.product(["power_station_unit"], [pd.NA, *bools]),
                itertools.product(["tap_dependency_table"], [*bools]),
            )
        ),
    )
    def test_valid_optional_boolean_values(self, parameter, valid_value):
        """Test: valid optional boolean values (nullable) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy TDT group if testing tap_dependency_table
        if parameter == "tap_dependency_table":
            net.trafo["id_characteristic_table"] = pd.Series([0 if valid_value in bools else pd.NA], dtype="Int64")

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["oltc"], [float(np.nan), *not_boolean_list]),
                itertools.product(["power_station_unit"], [float(np.nan), *not_boolean_list]),
                itertools.product(["tap_dependency_table"], [float(np.nan), *not_boolean_list]),
            )
        ),
    )
    def test_invalid_optional_boolean_values(self, parameter, invalid_value):
        """Test: invalid optional boolean values (wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy TDT group if testing tap_dependency_table
        if parameter == "tap_dependency_table":
            net.trafo["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize("valid_value", [pd.NA, *positiv_ints_plus_zero])
    def test_valid_id_characteristic_table_values(self, valid_value):
        """Test: valid id_characteristic_table values (nullable, >= 0) are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy TDT group
        is_null = pd.isna(valid_value) if not isinstance(valid_value, int) else False
        net.trafo["tap_dependency_table"] = pd.Series([pd.NA if is_null else True], dtype="boolean")

        add_column_to_df(net, 'trafo', 'id_characteristic_table')
        net.trafo.at[0, 'id_characteristic_table'] = valid_value

        validate_network(net)

    @pytest.mark.parametrize("invalid_value", [float(np.nan), *negativ_ints, *not_ints_list])
    def test_invalid_id_characteristic_table_values(self, invalid_value):
        """Test: invalid id_characteristic_table values (< 0 or wrong type) are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        # Satisfy TDT group
        net.trafo["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo["id_characteristic_table"] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def _setup_tap_group(self, net):
        """Setup complete tap group to satisfy dependencies"""
        net.trafo["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo["tap_pos"] = 1.0
        net.trafo["tap_neutral"] = 0.0
        net.trafo["tap_step_percent"] = 1.25
        net.trafo["tap_step_degree"] = 0.0

    def _setup_tap2_group(self, net):
        """Setup complete tap2 group to satisfy dependencies"""
        net.trafo["tap2_side"] = pd.Series(["lv"], dtype="string")
        net.trafo["tap2_pos"] = 0.0
        net.trafo["tap2_neutral"] = 0.0
        net.trafo["tap2_step_percent"] = 1.0
        net.trafo["tap2_step_degree"] = 0.0


class TestTrafoOptionalFieldsNotNullable:
    """Tests for optional fields that are NOT nullable (required=False, nullable not set)"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["xn_ohm"], all_allowed_floats),
                itertools.product(["pt_percent"], all_allowed_floats),
                itertools.product(["xn_ohm", "rn_ohm", "pt_percent"], all_allowed_floats),
            )
        ),
    )
    def test_valid_optional_not_nullable_values(self, parameter, valid_value):
        """Test: valid values for optional non-nullable fields are accepted"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        add_column_to_df(net, 'trafo', parameter)
        net.trafo.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["xn_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["pt_percent"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["xn_ohm", "rn_ohm", "pt_percent"], [float(np.nan), pd.NA, None, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_not_nullable_values(self, parameter, invalid_value):
        """Test: null values for optional non-nullable fields are rejected"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        net.trafo[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafoForeignKeys:
    """Tests for foreign key constraints (hv_bus, lv_bus -> bus.index)"""

    def test_invalid_hv_bus_index(self):
        """Test: hv_bus FK must reference an existing bus index"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        net.trafo["hv_bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_lv_bus_index(self):
        """Test: lv_bus FK must reference an existing bus index"""
        net = _create_net_with_buses()
        net.trafo = _create_valid_trafo_dataframe()

        net.trafo["lv_bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")

        create_bus(net, 110.0, index=10)
        create_bus(net, 10.0, index=42)

        net.trafo = pd.DataFrame([{
            "hv_bus": 10,
            "lv_bus": 42,
            "sn_mva": 25.0,
            "vn_hv_kv": 110.0,
            "vn_lv_kv": 10.0,
            "vk_percent": 10.0,
            "vkr_percent": 0.5,
            "pfe_kw": 30.0,
            "i0_percent": 0.1,
            "shift_degree": 0.0,
            "parallel": 1,
            "in_service": True,
        }])

        validate_network(net)


class TestTrafoResults:
    """Tests for trafo results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented - requires power flow calculation")
    def test_trafo_power_flows(self):
        """Test: Power flow result fields have valid numeric values"""
        pass

    @pytest.mark.skip(reason="Not yet implemented - requires short-circuit calculation")
    def test_trafo_short_circuit_results(self):
        """Test: Short-circuit result fields contain expected ranges"""
        pass

    @pytest.mark.skip(reason="Not yet implemented - requires 3-phase calculation")
    def test_trafo_3ph_results(self):
        """Test: 3-phase result fields contain expected values"""
        pass

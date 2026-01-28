# test_pandera_trafo3w_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus
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


class TestTrafo3wRequiredFields:
    """Tests for required trafo3w fields"""

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
        net = create_empty_network()
        create_bus(net, 110.0)        # index 0 (HV)
        create_bus(net, 20.0)         # index 1 (MV)
        create_bus(net, 10.0)         # index 2 (LV)
        create_bus(net, 0.4, index=42)

        net.trafo3w = pd.DataFrame([{
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
        }])
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
                itertools.product(["hv_bus", "mv_bus", "lv_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(
                    ["vn_hv_kv", "vn_mv_kv", "vn_lv_kv", "sn_hv_mva", "sn_mv_mva", "sn_lv_mva", "vk_hv_percent", "vk_mv_percent", "vk_lv_percent"],
                    [*negativ_floats_plus_zero, *not_floats_list],
                ),
                itertools.product(["vkr_hv_percent", "vkr_mv_percent", "vkr_lv_percent"], [*negativ_floats, *not_floats_list]),
                itertools.product(["pfe_kw", "i0_percent", "shift_mv_degree", "shift_lv_degree"], not_floats_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 110.0)        # index 0 (HV)
        create_bus(net, 20.0)         # index 1 (MV)
        create_bus(net, 10.0)         # index 2 (LV)

        net.trafo3w = pd.DataFrame([{
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
        }])

        # Neutralize vkr<vk to focus on target validation
        net.trafo3w["vkr_hv_percent"] = 0.0
        net.trafo3w["vkr_mv_percent"] = 0.0
        net.trafo3w["vkr_lv_percent"] = 0.0

        net.trafo3w[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_vkr_less_than_vk_checks_pass(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 1.0, "vkr_mv_percent": 0.5, "vkr_lv_percent": 0.1,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])
        validate_network(net)

    def test_vkr_greater_than_vk_fails(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 12.0, "vkr_mv_percent": 0.5, "vkr_lv_percent": 0.1,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafo3wOptionalFields:
    """Tests for optional trafo3w fields"""

    def test_all_optional_fields_valid(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        # Optional strings
        net.trafo3w["name"] = pd.Series(["T1"], dtype="string")
        net.trafo3w["std_type"] = pd.Series(["Type-A"], dtype="string")
        net.trafo3w["vector_group"] = pd.Series(["YNd"], dtype="string")

        # Tap group (complete)
        net.trafo3w["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo3w["tap_pos"] = 1.0
        net.trafo3w["tap_neutral"] = 0.0
        net.trafo3w["tap_step_percent"] = 1.25
        net.trafo3w["tap_step_degree"] = 0.0
        net.trafo3w["tap_at_star_point"] = pd.Series([True], dtype="boolean")
        net.trafo3w["tap_changer_type"] = pd.Series(["Ideal"], dtype="string")

        # TDT group (complete)
        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        # Other optional
        net.trafo3w["max_loading_percent"] = 85.0
        net.trafo3w["vkr0_x"] = 0.1
        net.trafo3w["vk0_x"] = 3.0

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        # Tap group -> all nulls, so group not triggered
        net.trafo3w["tap_side"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["tap_pos"] = np.nan
        net.trafo3w["tap_neutral"] = np.nan
        net.trafo3w["tap_changer_type"] = pd.Series([pd.NA], dtype="string")

        # TDT group left null
        net.trafo3w["tap_dependency_table"] = pd.Series([pd.NA], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([pd.NA], dtype="Int64")

        # Other nullables
        net.trafo3w["name"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["std_type"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["vector_group"] = pd.Series([pd.NA], dtype="string")
        net.trafo3w["vkr0_x"] = np.nan
        net.trafo3w["vk0_x"] = np.nan
        net.trafo3w["max_loading_percent"] = np.nan

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["std_type"], strings),
                itertools.product(["vector_group"], strings),
                itertools.product(["tap_side"], allowed_tap_side),
                itertools.product(["tap_changer_type"], allowed_tap_changer_types),
                itertools.product(["tap_step_percent"], positiv_floats),
                itertools.product(["tap_step_degree"], all_allowed_floats),
                itertools.product(["tap_at_star_point"], bools),
                itertools.product(["vkr0_x"], all_allowed_floats),
                itertools.product(["vk0_x"], all_allowed_floats),
                itertools.product(["max_loading_percent"], all_allowed_floats),
                itertools.product(["id_characteristic_table"], positiv_ints_plus_zero),
                itertools.product(["tap_dependency_table"], bools),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)
        create_bus(net, 0.4, index=42)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        # For TDT: ensure the counterpart exists when one is set
        if parameter == "tap_dependency_table":
            net.trafo3w["tap_dependency_table"] = pd.Series([valid_value], dtype="boolean")
            net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")
        elif parameter == "id_characteristic_table":
            net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
            net.trafo3w["id_characteristic_table"] = pd.Series([valid_value], dtype="Int64")
        else:
            net.trafo3w[parameter] = valid_value

        # Satisfy tap group when tap_side is used
        if parameter == "tap_side":
            net.trafo3w["tap_side"] = pd.Series([valid_value], dtype="string")
            net.trafo3w["tap_pos"] = 1.0
            net.trafo3w["tap_neutral"] = 0.0

        # Keep expected dtypes
        if parameter in {"name", "std_type", "vector_group", "tap_side", "tap_changer_type"}:
            net.trafo3w[parameter] = net.trafo3w[parameter].astype("string")
        if parameter in {"tap_dependency_table", "tap_at_star_point"}:
            net.trafo3w[parameter] = net.trafo3w[parameter].astype("boolean")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["std_type"], not_strings_list),
                itertools.product(["vector_group"], not_strings_list),
                itertools.product(["tap_side"], invalid_tap_side),
                itertools.product(["tap_changer_type"], invalid_tap_changer_types),
                itertools.product(["tap_step_percent"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["tap_step_degree"], not_floats_list),
                itertools.product(["tap_at_star_point"], not_boolean_list),
                itertools.product(["max_loading_percent"], not_floats_list),
                itertools.product(["tap_dependency_table"], not_boolean_list),
                itertools.product(["id_characteristic_table"], [*negativ_ints, *not_ints_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        # Satisfy groups so only target parameter triggers failure
        net.trafo3w["tap_pos"] = 1.0
        net.trafo3w["tap_neutral"] = 0.0
        net.trafo3w["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        net.trafo3w[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafo3wDependencies:
    """Tests for group dependencies and FK"""

    def test_tap_group_partial_missing_invalid(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        net.trafo3w["tap_side"] = pd.Series(["mv"], dtype="string")
        net.trafo3w["tap_neutral"] = 0.0
        net.trafo3w["tap_pos"] = pd.NA  # missing -> fail

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tap_group_complete_valid(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        net.trafo3w["tap_side"] = pd.Series(["lv"], dtype="string")
        net.trafo3w["tap_pos"] = 2.0
        net.trafo3w["tap_neutral"] = 0.0
        validate_network(net)

    def test_tdt_group_partial_missing_invalid(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([pd.NA], dtype="Int64")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tdt_group_complete_valid(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        net.trafo3w["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo3w["id_characteristic_table"] = pd.Series([0], dtype="Int64")
        validate_network(net)

    def test_invalid_bus_fk(self):
        net = create_empty_network()
        create_bus(net, 110.0)
        create_bus(net, 20.0)
        create_bus(net, 10.0)

        net.trafo3w = pd.DataFrame([{
            "hv_bus": 0, "mv_bus": 1, "lv_bus": 2,
            "vn_hv_kv": 110.0, "vn_mv_kv": 20.0, "vn_lv_kv": 10.0,
            "sn_hv_mva": 63.0, "sn_mv_mva": 25.0, "sn_lv_mva": 25.0,
            "vk_hv_percent": 10.0, "vk_mv_percent": 8.0, "vk_lv_percent": 6.0,
            "vkr_hv_percent": 0.5, "vkr_mv_percent": 0.4, "vkr_lv_percent": 0.3,
            "pfe_kw": 30.0, "i0_percent": 0.1, "shift_mv_degree": 0.0, "shift_lv_degree": 0.0,
            "in_service": True,
        }])

        net.trafo3w["hv_bus"] = 9999  # FK violation
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafo3wResults:
    """Tests for trafo3w results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_trafo3w_power_flows(self):
        """Test: Power flow result fields have valid numeric values"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_trafo3w_short_circuit_results(self):
        """Test: Short-circuit result fields contain expected ranges"""
        pass
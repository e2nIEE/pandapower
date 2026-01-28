# test_pandera_trafo_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_transformer
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
    negativ_floats_plus_zero,
    all_allowed_floats,
    negativ_floats,
)

# Common std_type available in pandapower

STD_TYPE = "25 MVA 110/10 kV"


class TestTrafoRequiredFields:
    """Tests for required trafo fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["hv_bus"], positiv_ints_plus_zero),
                itertools.product(["lv_bus"], positiv_ints_plus_zero),
                itertools.product(["sn_mva"], positiv_floats),
                itertools.product(["vn_hv_kv"], positiv_floats),
                itertools.product(["vn_lv_kv"], positiv_floats),
                itertools.product(["vk_percent"], positiv_floats),
                itertools.product(["vkr_percent"], positiv_floats_plus_zero),
                itertools.product(["pfe_kw"], positiv_floats_plus_zero),
                itertools.product(["i0_percent"], positiv_floats_plus_zero),
                itertools.product(["parallel"], positiv_ints),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = create_empty_network()
        create_bus(net, 110)  # index 0 (HV)
        create_bus(net, 10)  # index 1 (LV)
        create_bus(net, 0.4, index=42)

        create_transformer(net, hv_bus=0, lv_bus=1, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["hv_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["lv_bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vn_hv_kv"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vn_lv_kv"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vk_percent"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vkr_percent"], [*negativ_floats, *not_floats_list]),
                itertools.product(["pfe_kw"], [*negativ_floats, *not_floats_list]),
                itertools.product(["i0_percent"], [*negativ_floats, *not_floats_list]),
                itertools.product(["parallel"], [*negativ_ints_plus_zero, *not_ints_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = create_empty_network()
        create_bus(net, 110)
        create_bus(net, 10)

        create_transformer(net, hv_bus=0, lv_bus=1, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafoOptionalFields:
    """Tests for optional trafo fields and group dependencies (tap, tap2, tdt, opf)"""

    def test_all_optional_fields_valid(self):
        """All optional fields set; tap/tap2/tdt groups complete; OPF present"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        # Text optionals
        net.trafo["name"] = pd.Series(["Trafo A"], dtype="string")
        net.trafo["std_type"] = pd.Series(["custom_type"], dtype="string")
        net.trafo["vector_group"] = pd.Series(["Dyn5"], dtype="string")
        net.trafo["tap_changer_type"] = pd.Series(["Ratio"], dtype="string")
        net.trafo["tap2_changer_type"] = pd.Series(["Ideal"], dtype="string")

        # OPF single-column group
        net.trafo["max_loading_percent"] = 100

        # Other numerics
        net.trafo["shift_degree"] = 0.0
        net.trafo["df"] = 0.8

        # Zero sequence / SC optionals (accepted individually)
        net.trafo["vk0_percent"] = 3.0
        net.trafo["vkr0_percent"] = 0.5
        net.trafo["mag0_percent"] = 60.0
        net.trafo["mag0_rx"] = 15.0
        net.trafo["si0_hv_partial"] = 0.5

        # Tap group (complete)
        net.trafo["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo["tap_neutral"] = 0.0
        net.trafo["tap_min"] = -5.0
        net.trafo["tap_max"] = 5.0
        net.trafo["tap_step_percent"] = 2.5
        net.trafo["tap_step_degree"] = 0.0
        net.trafo["tap_pos"] = 0.0

        # Tap2 group (complete)
        net.trafo["tap2_side"] = pd.Series(["lv"], dtype="string")
        net.trafo["tap2_neutral"] = 0.0
        net.trafo["tap2_min"] = -3.0
        net.trafo["tap2_max"] = 3.0
        net.trafo["tap2_step_percent"] = 1.0
        net.trafo["tap2_step_degree"] = 0.0
        net.trafo["tap2_pos"] = 0.0

        # TDT group (complete)
        net.trafo["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        # Leakage ratios and SC extras
        net.trafo["leakage_resistance_ratio_hv"] = 0.5
        net.trafo["leakage_reactance_ratio_hv"] = 0.5
        net.trafo["xn_ohm"] = 0.0
        net.trafo["pt_percent"] = 100.0

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optionals with nulls; ensure groups not partially triggered"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        # Row 1: name/vector_group only
        create_transformer(
            net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1, max_loading_percent=80.0
        )
        net.trafo["name"] = pd.Series(["T1"], dtype="string")
        net.trafo["vector_group"] = pd.Series(["Dyn5"], dtype="string")

        # Row 2: OPF present
        create_transformer(
            net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1, max_loading_percent=80
        )

        # Row 3: tdt complete, others NA
        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=False, parallel=1)
        net.trafo["tap_dependency_table"] = pd.Series([pd.NA, True, pd.NA], dtype="boolean")
        net.trafo["id_characteristic_table"] = pd.Series([pd.NA, 1, pd.NA], dtype="Int64")
        net.trafo["std_type"] = pd.Series([pd.NA, pd.NA, pd.NA], dtype="string")
        # net.trafo["max_loading_percent"] = 80 #TODO no error buut with create there is
        validate_network(net)

    def test_tap_group_partial_missing_invalid(self):
        """Any tap column set -> all tap columns must be present"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo["tap_pos"] = 0.0  # partial -> invalid
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Another partial case
        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)
        net.trafo["tap_side"] = pd.Series(["hv"], dtype="string")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tap2_group_partial_missing_invalid(self):
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo["tap2_pos"] = 0.0  # partial -> invalid
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)
        net.trafo["tap2_side"] = pd.Series(["lv"], dtype="string")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_tdt_group_partial_missing_invalid(self):
        """Tap dependency table group must be complete"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)
        net.trafo["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)
        net.trafo["id_characteristic_table"] = pd.Series([pd.NA, 1], dtype="Int64")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["std_type"], strings),
                itertools.product(["shift_degree"], all_allowed_floats),
                itertools.product(["df"], [0.1, 0.5, 1.0]),
                itertools.product(["vector_group"], strings),
                itertools.product(["vk0_percent"], positiv_floats_plus_zero),
                itertools.product(["vkr0_percent"], positiv_floats_plus_zero),
                itertools.product(["mag0_percent"], positiv_floats_plus_zero),
                itertools.product(["mag0_rx"], all_allowed_floats),
                itertools.product(["si0_hv_partial"], positiv_floats_plus_zero),
                itertools.product(["tap_changer_type"], ["Ratio", "Symmetrical", "Ideal", "Tabular"]),
                itertools.product(["tap2_changer_type"], ["Ratio", "Symmetrical", "Ideal", "nan"]),
                itertools.product(["leakage_resistance_ratio_hv"], [0.0, 0.5, 1.0]),
                itertools.product(["leakage_reactance_ratio_hv"], [0.0, 0.5, 1.0]),
                itertools.product(["xn_ohm"], all_allowed_floats),
                itertools.product(["pt_percent"], all_allowed_floats),
                itertools.product(["max_loading_percent"], [0, 50, 100]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Valid optional values accepted (groups satisfied)"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        # Satisfy tap/tap2/tdt groups to avoid dependency failures
        net.trafo["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo["tap_neutral"] = 0.0
        net.trafo["tap_min"] = -5.0
        net.trafo["tap_max"] = 5.0
        net.trafo["tap_step_percent"] = 2.5
        net.trafo["tap_step_degree"] = 0.0
        net.trafo["tap_pos"] = 0.0

        net.trafo["tap2_side"] = pd.Series(["lv"], dtype="string")
        net.trafo["tap2_neutral"] = 0.0
        net.trafo["tap2_min"] = -3.0
        net.trafo["tap2_max"] = 3.0
        net.trafo["tap2_step_percent"] = 1.0
        net.trafo["tap2_step_degree"] = 0.0
        net.trafo["tap2_pos"] = 0.0

        net.trafo["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        # Set target parameter
        if parameter in {"name", "std_type", "vector_group", "tap_changer_type", "tap2_changer_type"}:
            net.trafo[parameter] = pd.Series([valid_value], dtype="string")
        else:
            net.trafo[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["std_type"], not_strings_list),
                itertools.product(["shift_degree"], not_floats_list),
                itertools.product(["df"], [0.0, -0.1, 1.1, *not_floats_list]),
                itertools.product(["vector_group"], not_strings_list),
                itertools.product(["vk0_percent"], [*negativ_floats, *not_floats_list]),
                itertools.product(["vkr0_percent"], [*negativ_floats, *not_floats_list]),
                itertools.product(["mag0_percent"], [*negativ_floats, *not_floats_list]),
                itertools.product(["mag0_rx"], not_floats_list),
                itertools.product(["si0_hv_partial"], [*negativ_floats, *not_floats_list]),
                itertools.product(["tap_changer_type"], ["bad_type", *not_strings_list]),
                itertools.product(["tap2_changer_type"], ["bad_type", *not_strings_list]),
                itertools.product(["leakage_resistance_ratio_hv"], [*negativ_floats, 1.1, *not_floats_list]),
                itertools.product(["leakage_reactance_ratio_hv"], [*negativ_floats, 1.1, *not_floats_list]),
                itertools.product(["xn_ohm"], not_floats_list),
                itertools.product(["pt_percent"], not_floats_list),
                itertools.product(["max_loading_percent"], [*not_ints_list]),  # TODO int?
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Invalid optional values rejected (groups satisfied)"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)

        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        # Satisfy tap/tap2/tdt groups
        net.trafo["tap_side"] = pd.Series(["hv"], dtype="string")
        net.trafo["tap_neutral"] = 0.0
        net.trafo["tap_min"] = -5.0
        net.trafo["tap_max"] = 5.0
        net.trafo["tap_step_percent"] = 2.5
        net.trafo["tap_step_degree"] = 0.0
        net.trafo["tap_pos"] = 0.0

        net.trafo["tap2_side"] = pd.Series(["lv"], dtype="string")
        net.trafo["tap2_neutral"] = 0.0
        net.trafo["tap2_min"] = -3.0
        net.trafo["tap2_max"] = 3.0
        net.trafo["tap2_step_percent"] = 1.0
        net.trafo["tap2_step_degree"] = 0.0
        net.trafo["tap2_pos"] = 0.0

        net.trafo["tap_dependency_table"] = pd.Series([True], dtype="boolean")
        net.trafo["id_characteristic_table"] = pd.Series([0], dtype="Int64")

        net.trafo[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_min_less_equal_max_check_passes(self):
        """Schema check: min_angle_degree <= max_angle_degree"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)
        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo["min_angle_degree"] = 0.0
        net.trafo["max_angle_degree"] = 10.0
        validate_network(net)

    def test_min_greater_than_max_fails(self):
        """Schema check: min_angle_degree > max_angle_degree -> fail"""
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)
        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo["min_angle_degree"] = 20.0
        net.trafo["max_angle_degree"] = 10.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafoForeignKey:
    """Tests for FK constraints on hv_bus/lv_bus"""

    def test_invalid_hv_bus_index(self):
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)
        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo["hv_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_lv_bus_index(self):
        net = create_empty_network()
        b_hv = create_bus(net, 110)
        b_lv = create_bus(net, 10)
        create_transformer(net, hv_bus=b_hv, lv_bus=b_lv, std_type=STD_TYPE, in_service=True, parallel=1)

        net.trafo["lv_bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestTrafoResults:
    """Tests for trafo results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_trafo_result_totals(self):
        """Aggregated p/q and loading results are consistent"""
        pass

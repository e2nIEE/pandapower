# test_pandera_sgen_elements.py

import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus, create_sgen
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    positiv_floats,
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    negativ_floats,
    not_ints_list,
    negativ_ints,
    all_allowed_floats,
    all_allowed_ints,
)


class TestSgenRequiredFields:
    """Tests for required sgen fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], all_allowed_floats),
                itertools.product(["q_mvar"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = create_empty_network()
        create_bus(net, 0.4)          # 0
        create_bus(net, 0.4)          # 1
        create_bus(net, 0.4, index=42)

        create_sgen(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], not_floats_list),
                itertools.product(["q_mvar"], not_floats_list),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_sgen(net, bus=0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenOptionalFields:
    """Tests for optional sgen fields and group dependencies (opf, qcc)"""

    def test_all_optional_fields_valid(self):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Create base sgen
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.2, scaling=1.0, in_service=True)

        # String/boolean optionals
        net.sgen["name"] = pd.Series(["SGen A"], dtype="string")
        net.sgen["type"] = pd.Series(["PV"], dtype="string")
        net.sgen["controllable"] = pd.Series([True], dtype="boolean")

        # OPF group (complete)
        net.sgen["max_p_mw"] = 2.0
        net.sgen["min_p_mw"] = -2.0
        net.sgen["max_q_mvar"] = 1.0
        net.sgen["min_q_mvar"] = -1.0

        # QCC group (complete)
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")

        # SC-related optionals (not enforced by group dependency in schema)
        net.sgen["sn_mva"] = 1.0
        net.sgen["k"] = 0.0
        net.sgen["rx"] = 0.0
        net.sgen["current_source"] = pd.Series([False], dtype="boolean")
        net.sgen["generator_type"] = pd.Series(["async"], dtype="string")
        net.sgen["lrc_pu"] = 5.0
        net.sgen["max_ik_ka"] = 10.0
        net.sgen["kappa"] = 1.8

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields incl. nulls; groups satisfied when present"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        # Row 1: OPF group complete
        create_sgen(net, bus=b0, p_mw=0.8, q_mvar=0.1, scaling=1.0, in_service=True,
                    max_p_mw=1.5,
                    min_p_mw=-1.0,
                    max_q_mvar=0.8,
                    min_q_mvar=-0.6,
                    name='alpha')
        # Row 2: QCC group complete
        create_sgen(net, bus=b0, p_mw=1.2, q_mvar=0.0, scaling=0.9, in_service=True,
                    id_q_capability_characteristic=1,
                    curve_style='constantYValue',
                    reactive_capability_curve=True,
                    type='wye')

        # Row 3: other optionals without triggering groups
        create_sgen(net, bus=b0, p_mw=2.0, q_mvar=-0.2, scaling=1.1, in_service=False,
                    sn_mva=2.0)
        net.sgen["controllable"] = pd.Series([pd.NA, True, False], dtype="boolean")
        net.sgen["name"] = pd.Series(["alpha", pd.NA, "gamma"], dtype="string")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["type"], strings),
                itertools.product(["sn_mva"], positiv_floats),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["k"], positiv_floats_plus_zero),
                itertools.product(["rx"], positiv_floats_plus_zero),
                itertools.product(["current_source"], bools),
                itertools.product(["generator_type"], ["current_source", "async", "async_doubly_fed"]),
                itertools.product(["lrc_pu"], all_allowed_floats),
                itertools.product(["max_ik_ka"], all_allowed_floats),
                itertools.product(["kappa"], all_allowed_floats),
                itertools.product(["id_q_capability_characteristic"], all_allowed_ints),
                itertools.product(["curve_style"], ["straightLineYValues", "constantYValue"]),
                itertools.product(["reactive_capability_curve"], bools),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Satisfy OPF + QCC groups so target column won't fail on dependency
        net.sgen["max_p_mw"] = 2.0
        net.sgen["min_p_mw"] = -2.0
        net.sgen["max_q_mvar"] = 1.0
        net.sgen["min_q_mvar"] = -1.0
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")

        if parameter in {"name", "type", "curve_style", "generator_type"}:
            net.sgen[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter in {"controllable", "current_source", "reactive_capability_curve"}:
            net.sgen[parameter] = pd.Series([valid_value], dtype="boolean")
        elif parameter == "id_q_capability_characteristic":
            net.sgen[parameter] = pd.Series([valid_value], dtype="Int64")
        else:
            net.sgen[parameter] = valid_value

        validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Set only one OPF column -> should fail
        net.sgen["max_p_mw"] = 100.0
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_qcc_group_partial_missing_invalid(self):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Only id_q_capability_characteristic
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Only curve_style
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Only reactive_capability_curve
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["k"], [*negativ_floats, *not_floats_list]),
                itertools.product(["rx"], [*negativ_floats, *not_floats_list]),
                itertools.product(["current_source"], not_boolean_list),
                itertools.product(["generator_type"], not_strings_list),
                itertools.product(["lrc_pu"], not_floats_list),
                itertools.product(["max_ik_ka"], not_floats_list),
                itertools.product(["kappa"], not_floats_list),
                itertools.product(["id_q_capability_characteristic"], not_ints_list),
                itertools.product(["curve_style"], not_strings_list),
                itertools.product(["reactive_capability_curve"], not_boolean_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        # Provide complete OPF + QCC groups
        net.sgen["max_p_mw"] = 2.0
        net.sgen["min_p_mw"] = -2.0
        net.sgen["max_q_mvar"] = 1.0
        net.sgen["min_q_mvar"] = -1.0
        net.sgen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        net.sgen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        net.sgen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")

        net.sgen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_sgen(net, bus=b0, p_mw=1.0, q_mvar=0.0, scaling=1.0, in_service=True)

        net.sgen["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSgenResults:
    """Tests for sgen results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_sgen_result_totals(self):
        """Aggregated p_mw / q_mvar results are consistent"""
        pass
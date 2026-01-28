# test_pandera_ward_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_ward
from pandapower.create import create_empty_network, create_bus
from pandapower.network_schema.tools.validation.network_validation import validate_network

from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    positiv_ints_plus_zero,
    not_ints_list,
    negativ_ints,
    all_allowed_floats,
)


class TestWardRequiredFields:
    """Tests for required ward fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["ps_mw"], all_allowed_floats),
                itertools.product(["qs_mvar"], all_allowed_floats),
                itertools.product(["pz_mw"], all_allowed_floats),
                itertools.product(["qz_mvar"], all_allowed_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)           # index 0
        create_bus(net, 0.4)           # index 1
        create_bus(net, 0.4, index=42) # ensure 42 exists for FK-positive tests

        create_ward(net, bus=0, ps_mw=1.0, qs_mvar=0.5, pz_mw=0.1, qz_mvar=0.05, in_service=True, name="w1")

        net.ward[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["ps_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["qs_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["pz_mw"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["qz_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1

        create_ward(net, bus=0, ps_mw=1.0, qs_mvar=0.5, pz_mw=0.1, qz_mvar=0.05, in_service=True)

        net.ward[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestWardOptionalFields:
    """Tests for optional ward fields"""

    def test_all_optional_fields_valid(self):
        """Test: ward with optional 'name' set is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ward(net, bus=b0, ps_mw=1.0, qs_mvar=0.2, pz_mw=0.1, qz_mvar=0.05, in_service=True, name="Ward A")
        # Ensure string dtype as per schema
        net.ward["name"] = net.ward["name"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: ward with optional 'name' including nulls is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_ward(net, bus=b0, ps_mw=0.8, qs_mvar=0.1, pz_mw=0.0, qz_mvar=0.0, in_service=True, name="alpha")
        create_ward(net, bus=b1, ps_mw=1.1, qs_mvar=0.3, pz_mw=0.2, qz_mvar=0.1, in_service=False, name=None)

        net.ward["name"] = pd.Series(["w1", pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(itertools.product(["name"], [pd.NA, *strings])),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ward(net, bus=b0, ps_mw=1.0, qs_mvar=0.2, pz_mw=0.1, qz_mvar=0.05, in_service=True)
        net.ward[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.product(["name"], not_strings_list)),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ward(net, bus=b0, ps_mw=1.0, qs_mvar=0.2, pz_mw=0.1, qz_mvar=0.05, in_service=True)
        net.ward[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestWardForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_ward(net, bus=b0, ps_mw=1.0, qs_mvar=0.2, pz_mw=0.1, qz_mvar=0.05, in_service=True)
        net.ward["bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestWardResults:
    """Tests for ward results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_ward_result_pq(self):
        """Test: p_mw / q_mvar results are present and numeric"""
        pass
# test_pandera_load_dc_elements.py

import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus_dc, create_load_dc
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
    all_allowed_floats,
    negativ_floats,
    positiv_floats_plus_zero,
)


class TestLoadDcRequiredFields:
    """Tests for required load_dc fields"""

    @pytest.mark.parametrize(
        "parameter, valid_value",
        list(
            itertools.chain(
                itertools.product(["bus_dc"], positiv_ints_plus_zero),
                itertools.product(["p_dc_mw"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus_dc(net, 0.4)          # index 0
        create_bus_dc(net, 0.4)          # index 1
        create_bus_dc(net, 0.4, index=42)

        create_load_dc(net, bus_dc=0, p_dc_mw=1.0, scaling=1.0, in_service=True)
        net.load_dc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus_dc"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_dc_mw"], not_floats_list),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus_dc(net, 0.4)          # index 0
        create_bus_dc(net, 0.4)          # index 1

        create_load_dc(net, bus_dc=0, p_dc_mw=1.0, scaling=1.0, in_service=True)
        net.load_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadDcOptionalFields:
    """Tests for optional load_dc fields"""

    def test_all_optional_fields_valid(self):
        """Test: load_dc with all optional fields is valid"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)

        create_load_dc(
            net,
            bus_dc=b0,
            p_dc_mw=1.2,
            scaling=1.0,
            in_service=True,
            name="LoadDC A",
            type="consumer",
            zone="area-1",
            controllable=True,
        )

        # enforce extension dtypes where needed
        net.load_dc["name"] = net.load_dc["name"].astype("string")
        net.load_dc["type"] = net.load_dc["type"].astype("string")
        net.load_dc["zone"] = net.load_dc["zone"].astype("string")
        net.load_dc["controllable"] = net.load_dc["controllable"].astype("boolean")

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)

        # Row 1: strings None, controllable NA
        create_load_dc(net, bus_dc=b0, p_dc_mw=0.5, scaling=1.0, in_service=True, name=None)
        # Row 2: set strings and controllable True
        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=0.8, in_service=True, type="prosumer", zone="Z2")
        # Row 3: controllable False
        create_load_dc(net, bus_dc=b0, p_dc_mw=2.0, scaling=1.1, in_service=False)

        # Cast to required extension dtypes and set nulls
        net.load_dc["name"] = pd.Series([pd.NA, "L2", "L3"], dtype="string")
        net.load_dc["type"] = pd.Series([pd.NA, "prosumer", pd.NA], dtype="string")
        net.load_dc["zone"] = pd.Series(["Z1", "Z2", pd.NA], dtype="string")
        net.load_dc["controllable"] = pd.Series([pd.NA, True, False], dtype="boolean")

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["type"], strings),
                itertools.product(["zone"], strings),
                itertools.product(["controllable"], bools),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=1.0, in_service=True)

        if parameter in {"name", "type", "zone"}:
            net.load_dc[parameter] = pd.Series([valid_value], dtype="string")
        elif parameter == "controllable":
            net.load_dc[parameter] = pd.Series([valid_value], dtype="boolean")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter, invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["zone"], not_strings_list),
                itertools.product(["controllable"], not_boolean_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        create_load_dc(
            net,
            bus_dc=b0,
            p_dc_mw=1.0,
            scaling=1.0,
            in_service=True,
            # set valid defaults so only target param fails
            name="ok",
            type="ok",
            zone="ok",
            controllable=True,
        )

        net.load_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadDcForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus_dc must reference an existing bus_dc index"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        create_load_dc(net, bus_dc=b0, p_dc_mw=1.0, scaling=1.0, in_service=True)

        net.load_dc["bus_dc"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestLoadDcResults:
    """Tests for load_dc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_load_dc_result_totals(self):
        """Test: aggregated p_dc_mw results are consistent"""
        pass
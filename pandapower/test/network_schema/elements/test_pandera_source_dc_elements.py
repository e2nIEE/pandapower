# test_pandera_source_dc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_empty_network, create_bus_dc, create_source_dc
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
)


class TestSourceDcRequiredFields:
    """Tests for required source_dc fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus_dc"], positiv_ints_plus_zero),
                itertools.product(["vm_pu"], all_allowed_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Valid required values are accepted"""
        net = create_empty_network()
        create_bus_dc(net, 0.4)  # index 0
        create_bus_dc(net, 0.4)  # index 1
        create_bus_dc(net, 0.4, index=42)

        create_source_dc(net, bus_dc=0, vm_pu=1.0, in_service=True)
        net.source_dc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus_dc"], [*negativ_ints, *not_ints_list]),
                itertools.product(["vm_pu"], not_floats_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Invalid required values are rejected"""
        net = create_empty_network()
        create_bus_dc(net, 0.4)  # 0
        create_bus_dc(net, 0.4)  # 1

        create_source_dc(net, bus_dc=0, vm_pu=1.0, in_service=True)
        net.source_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize("parameter", ["vm_pu", "in_service"])
    def test_required_fields_nan_invalid(self, parameter):
        """NaN in required columns is invalid"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        create_source_dc(net, bus_dc=b0, vm_pu=1.0, in_service=True)

        net.source_dc[parameter] = float(np.nan)
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSourceDcOptionalFields:
    """Tests for optional source_dc fields"""

    def test_all_optional_fields_valid(self):
        """All optional fields set"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)

        create_source_dc(net, bus_dc=b0, vm_pu=1.02, in_service=True, name="SRC A", type="voltage_source")
        # ensure string dtypes
        net.source_dc["name"] = net.source_dc["name"].astype("string")
        net.source_dc["type"] = net.source_dc["type"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Optional fields including nulls are valid"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)

        # Row 1: name None
        create_source_dc(net, bus_dc=b0, vm_pu=1.0, in_service=True, name=None)
        # Row 2: type set, name NA
        create_source_dc(net, bus_dc=b0, vm_pu=0.98, in_service=False, type="converter")

        net.source_dc["name"] = pd.Series([pd.NA, "SRC-B"], dtype="string")
        net.source_dc["type"] = pd.Series([pd.NA, "converter"], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)

        create_source_dc(net, bus_dc=b0, vm_pu=1.01, in_service=True, name="ok", type="ok")
        net.source_dc[parameter] = pd.Series([valid_value], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)

        create_source_dc(net, bus_dc=b0, vm_pu=1.0, in_service=True, name="ok", type="ok")
        net.source_dc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSourceDcForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """bus_dc must reference an existing bus_dc index"""
        net = create_empty_network()
        b0 = create_bus_dc(net, 0.4)
        create_source_dc(net, bus_dc=b0, vm_pu=1.0, in_service=True)

        net.source_dc["bus_dc"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSourceDcResults:
    """Tests for source_dc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_source_dc_result_totals(self):
        """Aggregated p_dc_mw results are consistent"""
        pass

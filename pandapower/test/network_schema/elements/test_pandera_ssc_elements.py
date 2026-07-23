# test_pandera_ssc_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_ssc
from pandapower.network import pandapowerNet
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
    positiv_floats_plus_zero,
    negativ_floats_plus_zero,
    positiv_floats,
    negativ_floats,
    all_allowed_floats,
)


class TestSscRequiredFields:
    """Tests for required SSC fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["r_ohm"], positiv_floats_plus_zero),
                itertools.product(["x_ohm"], negativ_floats_plus_zero),
                itertools.product(["set_vm_pu"], all_allowed_floats),
                itertools.product(["vm_internal_pu"], all_allowed_floats),
                itertools.product(["va_internal_degree"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1
        create_bus(net, 0.4, index=42)

        create_ssc(
            net,
            bus=0,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
        )
        net.ssc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, *negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["x_ohm"], [float(np.nan), pd.NA, *positiv_floats, *not_floats_list]),
                itertools.product(["set_vm_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["vm_internal_pu"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["va_internal_degree"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["controllable"], [float(np.nan), pd.NA, *not_boolean_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_ssc(
            net,
            bus=0,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
        )
        net.ssc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSscOptionalFields:
    """Tests for optional SSC fields"""

    def test_all_optional_fields_valid(self):
        """Test: SSC with all optional fields is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_ssc(
            net,
            bus=b0,
            r_ohm=0.1,
            x_ohm=-0.2,
            set_vm_pu=1.02,
            vm_internal_pu=1.03,
            va_internal_degree=3.0,
            controllable=True,
            in_service=False,
            name="SSC A",
        )
        net.ssc["name"] = net.ssc["name"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)

        create_ssc(
            net,
            bus=b0,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
            name=None,
        )
        create_ssc(
            net,
            bus=b0,
            r_ohm=0.2,
            x_ohm=-0.3,
            set_vm_pu=0.98,
            vm_internal_pu=0.99,
            va_internal_degree=-2.0,
            controllable=False,
            in_service=False,
            name="SSC B",
        )

        net.ssc["name"] = pd.Series([pd.NA, "SSC B"], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        create_ssc(
            net,
            bus=b0,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
            name="ok",
        )
        net.ssc[parameter] = pd.Series([valid_value], dtype="string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(itertools.chain(itertools.product(["name"], [float(np.nan), *not_strings_list]))),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)

        create_ssc(
            net,
            bus=b0,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
            name="ok",
        )
        net.ssc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestSscForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)

        create_ssc(
            net,
            bus=b0,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
        )
        net.ssc["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_ssc(
            net,
            bus=10,
            r_ohm=0.0,
            x_ohm=-0.1,
            set_vm_pu=1.00,
            vm_internal_pu=1.01,
            va_internal_degree=0.0,
            controllable=True,
            in_service=True,
        )
        create_ssc(
            net,
            bus=42,
            r_ohm=0.1,
            x_ohm=-0.2,
            set_vm_pu=0.98,
            vm_internal_pu=0.99,
            va_internal_degree=-1.0,
            controllable=False,
            in_service=False,
        )
        create_ssc(
            net,
            bus=100,
            r_ohm=0.2,
            x_ohm=-0.3,
            set_vm_pu=1.02,
            vm_internal_pu=1.03,
            va_internal_degree=2.0,
            controllable=True,
            in_service=True,
        )

        validate_network(net)


class TestSscResults:
    """Tests for ssc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_ssc_result_totals(self):
        """Test: aggregated reactive power results are consistent"""
        pass

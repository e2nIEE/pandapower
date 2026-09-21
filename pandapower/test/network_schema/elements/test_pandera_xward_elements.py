# test_pandera_xward_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_xward
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
    positiv_floats,
    negativ_floats_plus_zero,
    all_allowed_floats,
)


class TestXWardRequiredFields:
    """Tests for required xward fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["ps_mw"], all_allowed_floats),
                itertools.product(["qs_mvar"], all_allowed_floats),
                itertools.product(["pz_mw"], all_allowed_floats),
                itertools.product(["qz_mvar"], all_allowed_floats),
                itertools.product(["r_ohm"], positiv_floats),  # gt(0)
                itertools.product(["x_ohm"], positiv_floats),  # gt(0)
                itertools.product(["vm_pu"], positiv_floats),  # gt(0)
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_xward(
            net,
            bus=0,
            ps_mw=1.0,
            qs_mvar=0.5,
            pz_mw=0.1,
            qz_mvar=0.05,
            r_ohm=0.01,
            x_ohm=0.02,
            vm_pu=1.0,
            in_service=True,
            name="xw1",
            slack_weight=1.0,
        )

        net.xward[parameter] = valid_value
        if parameter == "name":
            net.xward["name"] = net.xward["name"].astype("string")
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["ps_mw"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["qs_mvar"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["pz_mw"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["qz_mvar"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["r_ohm"], [float(np.nan), pd.NA, None, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["x_ohm"], [float(np.nan), pd.NA, None, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vm_pu"], [float(np.nan), pd.NA, None, *negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, None, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # 0
        create_bus(net, 0.4)  # 1

        create_xward(
            net,
            bus=0,
            ps_mw=1.0,
            qs_mvar=0.5,
            pz_mw=0.1,
            qz_mvar=0.05,
            r_ohm=0.01,
            x_ohm=0.02,
            vm_pu=1.0,
            in_service=True,
        )

        net.xward[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestXWardOptionalFieldsNullable:
    """Tests for optional nullable fields - can be absent, null, or have valid values"""

    def test_all_optional_fields_valid(self):
        """Test: xward with optional fields set is valid"""
        net = pandapowerNet(name="test_all_optional_fields_valid")
        b0 = create_bus(net, 0.4)

        create_xward(
            net,
            bus=b0,
            ps_mw=0.8,
            qs_mvar=0.2,
            pz_mw=0.1,
            qz_mvar=0.05,
            r_ohm=0.02,
            x_ohm=0.03,
            vm_pu=1.01,
            slack_weight=0.5,
            in_service=True,
            name="XWard A",
        )

        # CIM columns
        net.xward["origin_id"] = pd.Series(["cim_id_1"], dtype=pd.StringDtype())
        net.xward["origin_class"] = pd.Series(["EquivalentInjection"], dtype=pd.StringDtype())
        net.xward["terminal"] = pd.Series(["term_1"], dtype=pd.StringDtype())
        net.xward["description"] = pd.Series(["Test xward"], dtype=pd.StringDtype())

        validate_network(net)

    def test_all_optional_nullable_fields_with_nulls(self):
        """Test: all nullable optional fields with null values are accepted"""
        net = pandapowerNet(name="test_all_optional_nullable_fields_with_nulls")
        b0 = create_bus(net, 0.4)

        create_xward(
            net,
            bus=b0,
            ps_mw=1.0,
            qs_mvar=0.3,
            pz_mw=0.2,
            qz_mvar=0.1,
            r_ohm=0.05,
            x_ohm=0.07,
            vm_pu=1.0,
            in_service=True,
        )

        # All nullable string columns -> NA
        net.xward["name"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.xward["origin_id"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.xward["origin_class"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.xward["terminal"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        net.xward["description"] = pd.Series([pd.NA], dtype=pd.StringDtype())

        # Nullable float column -> NaN
        net.xward["slack_weight"] = float(np.nan)

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
        b0 = create_bus(net, 0.4)
        b1 = create_bus(net, 0.4)

        create_xward(
            net,
            bus=b0,
            ps_mw=1.0,
            qs_mvar=0.3,
            pz_mw=0.2,
            qz_mvar=0.1,
            r_ohm=0.05,
            x_ohm=0.07,
            vm_pu=1.0,
            in_service=True,
            name="alpha",
            slack_weight=0.5,
        )
        create_xward(
            net,
            bus=b1,
            ps_mw=0.5,
            qs_mvar=0.1,
            pz_mw=0.0,
            qz_mvar=0.0,
            r_ohm=0.02,
            x_ohm=0.03,
            vm_pu=1.02,
            in_service=False,
        )

        # Set nullable columns with mixed values
        net.xward["name"] = pd.Series(["alpha", pd.NA], dtype=pd.StringDtype())
        net.xward["slack_weight"] = [0.5, float(np.nan)]
        net.xward["origin_id"] = pd.Series(["cim_1", pd.NA], dtype=pd.StringDtype())
        net.xward["origin_class"] = pd.Series([pd.NA, "EquivalentInjection"], dtype=pd.StringDtype())
        net.xward["terminal"] = pd.Series([pd.NA, pd.NA], dtype=pd.StringDtype())
        net.xward["description"] = pd.Series(["Desc 1", pd.NA], dtype=pd.StringDtype())

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["origin_id"], [pd.NA, *strings]),
                itertools.product(["origin_class"], [pd.NA, *strings]),
                itertools.product(["terminal"], [pd.NA, *strings]),
                itertools.product(["description"], [pd.NA, *strings]),
                itertools.product(["slack_weight"], [float(np.nan), *all_allowed_floats]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        b0 = create_bus(net, 0.4)

        create_xward(
            net,
            bus=b0,
            ps_mw=1.0,
            qs_mvar=0.3,
            pz_mw=0.2,
            qz_mvar=0.1,
            r_ohm=0.02,
            x_ohm=0.03,
            vm_pu=1.0,
            in_service=True,
        )

        if parameter in ["name", "origin_id", "origin_class", "terminal", "description"]:
            net.xward[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.xward[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
                itertools.product(["origin_id"], [float(np.nan), *not_strings_list]),
                itertools.product(["origin_class"], [float(np.nan), *not_strings_list]),
                itertools.product(["terminal"], [float(np.nan), *not_strings_list]),
                itertools.product(["description"], [float(np.nan), *not_strings_list]),
                itertools.product(["slack_weight"], [pd.NA, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)

        create_xward(
            net,
            bus=b0,
            ps_mw=1.0,
            qs_mvar=0.3,
            pz_mw=0.2,
            qz_mvar=0.1,
            r_ohm=0.02,
            x_ohm=0.03,
            vm_pu=1.0,
            in_service=True,
        )

        net.xward[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestXWardForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        net = pandapowerNet(name="test_invalid_bus_index")
        b0 = create_bus(net, 0.4)

        create_xward(
            net,
            bus=b0,
            ps_mw=1.0,
            qs_mvar=0.3,
            pz_mw=0.2,
            qz_mvar=0.1,
            r_ohm=0.02,
            x_ohm=0.03,
            vm_pu=1.0,
            in_service=True,
        )

        net.xward["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_xward(
            net,
            bus=10,
            ps_mw=1.0,
            qs_mvar=0.3,
            pz_mw=0.2,
            qz_mvar=0.1,
            r_ohm=0.05,
            x_ohm=0.07,
            vm_pu=1.0,
            in_service=True,
        )
        create_xward(
            net,
            bus=42,
            ps_mw=0.5,
            qs_mvar=0.1,
            pz_mw=0.0,
            qz_mvar=0.0,
            r_ohm=0.02,
            x_ohm=0.03,
            vm_pu=1.02,
            in_service=True,
        )
        create_xward(
            net,
            bus=100,
            ps_mw=0.8,
            qs_mvar=0.2,
            pz_mw=0.1,
            qz_mvar=0.05,
            r_ohm=0.03,
            x_ohm=0.04,
            vm_pu=0.99,
            in_service=False,
        )

        validate_network(net)


class TestXWardResults:
    """Tests for xward results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_xward_result_pq(self):
        """Test: p_mw / q_mvar results are present and numeric"""
        pass

# test_pandera_xward_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_xward
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
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1
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
        # ensure string dtype for name when touched
        if parameter == "name":
            net.xward["name"] = net.xward["name"].astype("string")
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
                itertools.product(["r_ohm"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["x_ohm"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["vm_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
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


class TestXWardOptionalFields:
    """Tests for optional xward fields"""

    def test_all_optional_fields_valid(self):
        """Test: xward with optional fields set is valid"""
        net = create_empty_network()
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
        net.xward["name"] = net.xward["name"].astype("string")
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: optional fields including nulls are valid"""
        net = create_empty_network()
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
            slack_weight=None,
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
            name=None,
            slack_weight=None,
        )

        net.xward["name"] = pd.Series(["x1", pd.NA], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["slack_weight"], [*all_allowed_floats, float(np.nan)]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
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

        if parameter == "name":
            net.xward[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.xward[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["slack_weight"], not_floats_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
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
        net = create_empty_network()
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


class TestXWardResults:
    """Tests for xward results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_xward_result_pq(self):
        """Test: p_mw / q_mvar results are present and numeric"""
        pass
# test_pandera_b2b_vsc_elements.py

import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_b2b_vsc
from pandapower.create import create_empty_network, create_bus, create_bus_dc
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


class TestB2BVSCRequiredFields:
    """Tests for required b2b_vsc fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["bus_dc_plus"], positiv_ints_plus_zero),
                itertools.product(["bus_dc_minus"], positiv_ints_plus_zero),
                itertools.product(["r_ohm"], all_allowed_floats),
                itertools.product(["x_ohm"], all_allowed_floats),
                itertools.product(["r_dc_ohm"], all_allowed_floats),
                itertools.product(["pl_dc_mw"], all_allowed_floats),
                itertools.product(["control_mode_ac"], ["vm_pu", "q_mvar", "slack"]),
                itertools.product(["control_value_ac"], all_allowed_floats),
                itertools.product(["control_mode_dc"], ["vm_pu", "p_mw"]),
                itertools.product(["control_value_dc"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, vn_kv=110.0)  # index 0
        create_bus(net, vn_kv=110.0)  # index 1
        create_bus(net, vn_kv=110.0, index=42)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0, index=42)

        create_b2b_vsc(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="test",
        )

        net.b2b_vsc[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_plus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_minus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], not_floats_list),
                itertools.product(["x_ohm"], not_floats_list),
                itertools.product(["r_dc_ohm"], not_floats_list),
                itertools.product(["pl_dc_mw"], not_floats_list),
                itertools.product(["control_mode_ac"], [*strings, *not_strings_list]),
                itertools.product(["control_value_ac"], not_floats_list),
                itertools.product(["control_mode_dc"], [*strings, *not_strings_list]),
                itertools.product(["control_value_dc"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["in_service"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 110.0)  # index 0
        create_bus(net, 110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0, index=42)
        create_b2b_vsc(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.b2b_vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestB2BVSCOptionalFields:
    """Tests for optional b2b_vsc fields"""

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
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_b2b_vsc(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
            name="initial",
        )

        net.b2b_vsc[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_b2b_vsc(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.b2b_vsc[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestB2BVSCSchemaForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_b2b_vsc(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.b2b_vsc["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_plus_index(self):
        """Test: bus_dc_plus FK must reference an existing dc bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_b2b_vsc(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.b2b_vsc["bus_dc_plus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_minus_index(self):
        """Test: bus_dc_minus FK must reference an existing dc bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 110.0)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_b2b_vsc(
            net,
            bus=b0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.2,
            r_dc_ohm=0.05,
            pl_dc_mw=0.0,
            control_mode_ac="vm_pu",
            control_value_ac=1.0,
            control_mode_dc="p_mw",
            control_value_dc=0.0,
            controllable=True,
            in_service=True,
        )

        net.b2b_vsc["bus_dc_minus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestB2BVSCResults:
    """Tests for b2b_vsc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_b2b_vsc_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_b2b_vsc_internal_results(self):
        """Test: internal vm/va and dc quantities are within expected ranges"""
        pass


import itertools
import pandas as pd
import pandera as pa
import pytest
import numpy as np

from pandapower import create_asymmetric_sgen
from pandapower.create import create_empty_network, create_bus, create_gen
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
    negativ_floats,
    negativ_floats_plus_zero,
    not_ints_list,
    negativ_ints,
    all_allowed_floats, zero_float, all_allowed_ints,
)


class TestGenRequiredFields:
    """Tests for required asymmetric_sgen fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_mw"], all_allowed_floats),
                itertools.product(["vm_pu"], positiv_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
                itertools.product(["slack"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1
        create_bus(net, 0.4, index=42)

        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
        )
        net.gen[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_mw"], not_floats_list),
                itertools.product(["vm_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
                itertools.product(["slack"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are not accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus(net, 0.4, index=42)

        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
        )
        net.gen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestGenOptionalFields:
    """Tests for optional gen fields, including group dependencies"""
    def test_all_optional_fields_valid(self):
        """Test: gen with every optional field is valid and dependencies satisfied"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
            # optional
            name='test',
            type='sync',
            sn_mva=1.0,
            max_q_mvar=1.1,
            min_q_mvar=1.0,
            max_p_mw=1.1,
            min_p_mw=1.0,
            vn_kv=1.0,
            xdss_pu=1.0,
            rdss_ohm=1.0,
            cos_phi=1.0,
            power_station_trafo=0,
            id_q_capability_characteristic=0,
            curve_style='straightLineYValues',
            reactive_capability_curve=True,
            slack_weight=1.0,
            controllable=True,
            pg_percent=1.0,
            min_vm_pu=1.1,
            max_vm_pu=1.0,
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: gen with optional fields including nulls, with dependencies respected"""
        net = create_empty_network()
        create_bus(net, 0.4)

        # Row 1: opf + q_lim_enforced present
        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
            # optional
            max_q_mvar=1.1,
            min_q_mvar=1.0,
            max_p_mw=1.1,
            min_p_mw=1.0,
            controllable=True,
            min_vm_pu=1.0,
            max_vm_pu=1.1,
        )
        # Row 2: qcc present
        # TODO: min_vm_pu in create_gen is set to 0.0 - schema demands greater than zero
        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
            # optional
            id_q_capability_characteristic=0,
            curve_style='straightLineYValues',
            reactive_capability_curve=True,
            slack_weight=1.0,
            controllable=True,
        )
        # Row 3: sc present
        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
            # optional
            vn_kv=1.0,
            xdss_pu=1.0,
            rdss_ohm=1.0,
            cos_phi=1.0,
            power_station_trafo=0,
            pg_percent=1.0,
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], strings),
                itertools.product(["type"], strings),
                itertools.product(["sn_mva"], positiv_floats),
                itertools.product(["max_q_mvar"], all_allowed_floats),
                itertools.product(["min_q_mvar"], all_allowed_floats),
                itertools.product(["max_p_mw"], all_allowed_floats),
                itertools.product(["min_p_mw"], all_allowed_floats),
                itertools.product(["vn_kv"], all_allowed_floats),
                itertools.product(["xdss_pu"], positiv_floats),
                itertools.product(["rdss_ohm"], positiv_floats),
                itertools.product(["cos_phi"], [*zero_float, 0.4]),
                itertools.product(["in_service"], bools),
                itertools.product(["id_q_capability_characteristic"], all_allowed_ints),
                itertools.product(["curve_style"], strings),
                itertools.product(["reactive_capability_curve"], bools),
                itertools.product(["slack_weight"], all_allowed_floats),
                itertools.product(["controllable"], bools),
                itertools.product(["pg_percent"], all_allowed_floats),
                itertools.product(["min_vm_pu"], positiv_floats),
                itertools.product(["max_vm_pu"], positiv_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        #TODO: bool or pd.BooleanDtype
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_gen(
            net,
            bus=0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
            # optional
            name='test',
            type='sync',
            sn_mva=1.0,
            max_q_mvar=1.1,
            min_q_mvar=1.0,
            max_p_mw=1.1,
            min_p_mw=1.0,
            vn_kv=1.0,
            xdss_pu=1.0,
            rdss_ohm=1.0,
            cos_phi=1.0,
            power_station_trafo=0,
            id_q_capability_characteristic=0,
            curve_style='straightLineYValues',
            reactive_capability_curve=True,
            slack_weight=1.0,
            controllable=True,
            pg_percent=1.0,
            min_vm_pu=1.1,
            max_vm_pu=1.0,
        )
        net.gen[parameter] = valid_value
        net.gen["name"] = net.gen["name"].astype("string")
        net.gen["type"] = net.gen["type"].astype("string")
        net.gen["curve_style"] = net.gen["curve_style"].astype("string")

        validate_network(net)

    def test_opf_group_partial_missing_invalid(self):
        """Test: OPF group must be complete if any OPF value is set (gen)"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_gen(net, bus=b0, p_mw=-1.0, vm_pu=0.5, scaling=1.0, in_service=True, slack=True)

        # Set only one OPF column -> should fail due to group dependency
        # Choose max_p_mw; other OPF columns are present as NaN in net.gen by default
        net.gen["max_p_mw"] = 100.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_q_lim_enforced_group_partial_missing_invalid(self):
        """Test: q_lim_enforced group (max_q_mvar/min_q_mvar) must be complete"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_gen(net, bus=b0, p_mw=-1.0, vm_pu=0.5, scaling=1.0, in_service=True, slack=True)

        # Set only max_q_mvar -> should fail because min_q_mvar is missing/NaN
        net.gen["max_q_mvar"] = 1.0

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_qcc_group_partial_missing_invalid(self):
        """Test: QCC group must be complete if any value is set"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_gen(net, bus=b0, p_mw=-1.0, vm_pu=0.5, scaling=1.0, in_service=True, slack=True)

        # Set only one QCC column at a time -> each should fail
        # id_q_capability_characteristic only
        net.gen["id_q_capability_characteristic"] = pd.Series([0], dtype="Int64")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Reset and set only curve_style
        create_gen(net, bus=b0, p_mw=-1.0, vm_pu=0.5, scaling=1.0, in_service=True, slack=True)
        net.gen["curve_style"] = pd.Series(["straightLineYValues"], dtype="string")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

        # Reset and set only reactive_capability_curve
        create_gen(net, bus=b0, p_mw=-1.0, vm_pu=0.5, scaling=1.0, in_service=True, slack=True)
        net.gen["reactive_capability_curve"] = pd.Series([True], dtype="boolean")
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["max_p_mw"], not_floats_list),
                itertools.product(["min_p_mw"], not_floats_list),
                itertools.product(["max_q_mvar"], not_floats_list),
                itertools.product(["min_q_mvar"], not_floats_list),
                itertools.product(["vn_kv"], not_floats_list),
                itertools.product(["xdss_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["rdss_ohm"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["cos_phi"], [*negativ_floats, 1.1, *not_floats_list]),
                itertools.product(["id_q_capability_characteristic"], not_ints_list),
                itertools.product(["power_station_trafo"], not_ints_list),
                itertools.product(["curve_style"], [*strings, *not_strings_list]),
                itertools.product(["reactive_capability_curve"], not_boolean_list),
                itertools.product(["slack_weight"], not_floats_list),
                itertools.product(["controllable"], not_boolean_list),
                itertools.product(["pg_percent"], not_floats_list),
                itertools.product(["min_vm_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["max_vm_pu"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are not accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_gen(
            net,
            bus=b0,
            p_mw=-1.0,
            vm_pu=0.5,
            scaling=1.0,
            in_service=True,
            slack=True,
            # provide complete groups so only the target parameter triggers failure
            max_q_mvar=1.1,
            min_q_mvar=1.0,
            max_p_mw=1.1,
            min_p_mw=1.0,
            controllable=True,
            min_vm_pu=1.0,
            max_vm_pu=1.1,
            id_q_capability_characteristic=0,
            curve_style="straightLineYValues",
            reactive_capability_curve=True,
        )
        net.gen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

class TestGenForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)
        create_gen(net, bus=b0, p_mw=-1.0, vm_pu=0.5, scaling=1.0, in_service=True, slack=True)

        net.gen["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

class TestGenResults:
    """Tests for gen results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_gen_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass
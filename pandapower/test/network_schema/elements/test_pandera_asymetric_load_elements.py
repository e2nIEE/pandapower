
# test_asymmetric_load.py

import itertools
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_asymmetric_load
from pandapower.create import create_empty_network, create_bus
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
    all_allowed_floats,
)


class TestAsymmetricLoadRequiredFields:
    """Tests for required asymmetric_load fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_a_mw"], positiv_floats_plus_zero),
                itertools.product(["p_b_mw"], positiv_floats_plus_zero),
                itertools.product(["p_c_mw"], positiv_floats_plus_zero),
                itertools.product(["q_a_mvar"], all_allowed_floats),
                itertools.product(["q_b_mvar"], all_allowed_floats),
                itertools.product(["q_c_mvar"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
                itertools.product(["type"], ["wye", "delta"]),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)
        create_bus(net, 0.4, index=42)

        create_asymmetric_load(
            net,
            bus=0,
            p_a_mw=1.0,
            p_b_mw=1.0,
            p_c_mw=1.0,
            q_a_mvar=0.5,
            q_b_mvar=0.5,
            q_c_mvar=0.5,
            scaling=1.0,
            in_service=True,
            type="wye",
            name="test",
            sn_mva=10.0,
        )

        net.asymmetric_load[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_a_mw"], [*negativ_floats, *not_floats_list]),
                itertools.product(["p_b_mw"], [*negativ_floats, *not_floats_list]),
                itertools.product(["p_c_mw"], [*negativ_floats, *not_floats_list]),
                itertools.product(["q_a_mvar"], not_floats_list),
                itertools.product(["q_b_mvar"], not_floats_list),
                itertools.product(["q_c_mvar"], not_floats_list),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
                itertools.product(["type"], [*strings, *not_strings_list]),  # invalid strings + non-strings
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        create_asymmetric_load(
            net,
            bus=0,
            p_a_mw=1.0,
            p_b_mw=1.0,
            p_c_mw=1.0,
            q_a_mvar=0.5,
            q_b_mvar=0.5,
            q_c_mvar=0.5,
            scaling=1.0,
            in_service=True,
            type="wye",
            name="test",
            sn_mva=10.0,
        )

        net.asymmetric_load[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricLoadOptionalFields:
    """Tests for optional asymmetric_load fields"""

    def test_all_optional_fields_valid(self):
        """Test: asymmetric_load with every optional field is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_load(
            net,
            bus=b0,
            p_a_mw=10.0,
            p_b_mw=12.0,
            p_c_mw=11.0,
            q_a_mvar=5.0,
            q_b_mvar=6.0,
            q_c_mvar=4.0,
            scaling=1.0,
            in_service=True,
            type="wye",
            name="lorem ipsum",
            sn_mva=25.0,
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: asymmetric_load with optional fields including nulls is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_load(
            net,
            bus=b0,
            p_a_mw=10.0,
            p_b_mw=10.0,
            p_c_mw=10.0,
            q_a_mvar=5.0,
            q_b_mvar=5.0,
            q_c_mvar=5.0,
            scaling=1.0,
            in_service=True,
            type="delta",
            name="lorem ipsum",
        )
        create_asymmetric_load(
            net,
            bus=b0,
            p_a_mw=8.0,
            p_b_mw=9.0,
            p_c_mw=7.5,
            q_a_mvar=3.0,
            q_b_mvar=3.5,
            q_c_mvar=2.5,
            scaling=1.0,
            in_service=False,
            type="wye",
            sn_mva=15.0,
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["sn_mva"], positiv_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_load(
            net,
            bus=b0,
            p_a_mw=10.0,
            p_b_mw=10.0,
            p_c_mw=10.0,
            q_a_mvar=5.0,
            q_b_mvar=5.0,
            q_c_mvar=5.0,
            scaling=1.0,
            in_service=True,
            type="wye",
            name="initial",
            sn_mva=20.0,
        )

        if parameter == "name":
            net.asymmetric_load[parameter] = pd.Series([valid_value], dtype=pd.StringDtype())
        else:
            net.asymmetric_load[parameter] = valid_value
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_load(
            net,
            bus=b0,
            p_a_mw=1.0,
            p_b_mw=1.0,
            p_c_mw=1.0,
            q_a_mvar=0.2,
            q_b_mvar=0.2,
            q_c_mvar=0.2,
            scaling=1.0,
            in_service=True,
            type="wye",
        )

        net.asymmetric_load[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricLoadForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_load(
            net,
            bus=b0,
            p_a_mw=1.0,
            p_b_mw=1.0,
            p_c_mw=1.0,
            q_a_mvar=0.5,
            q_b_mvar=0.5,
            q_c_mvar=0.5,
            scaling=1.0,
            in_service=True,
            type="delta",
        )

        # Set to a non-existent bus index
        net.asymmetric_load["bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricLoadResults:
    """Tests for asymmetric_load results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_asymmetric_load_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_asymmetric_load_3ph_results(self):
        """Test: 3-phase results contain valid values per phase"""
        pass
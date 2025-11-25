
# test_asymmetric_sgen.py

import itertools
import pandas as pd
import pandera as pa
import pytest
import numpy as np

from pandapower import create_asymmetric_sgen
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


class TestAsymmetricSgenRequiredFields:
    """Tests for required asymmetric_sgen fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], positiv_ints_plus_zero),
                itertools.product(["p_a_mw"], negativ_floats_plus_zero),
                itertools.product(["p_b_mw"], negativ_floats_plus_zero),
                itertools.product(["p_c_mw"], negativ_floats_plus_zero),
                itertools.product(["q_a_mvar"], all_allowed_floats),
                itertools.product(["q_b_mvar"], all_allowed_floats),
                itertools.product(["q_c_mvar"], all_allowed_floats),
                itertools.product(["scaling"], positiv_floats_plus_zero),
                itertools.product(["in_service"], bools),
                itertools.product(["current_source"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1
        create_bus(net, 0.4, index=42)

        create_asymmetric_sgen(
            net,
            bus=0,
            p_a_mw=-1.0,
            q_a_mvar=0.5,
            p_b_mw=-1.0,
            q_b_mvar=0.5,
            p_c_mw=-1.0,
            q_c_mvar=0.5,
            scaling=1.0,
            in_service=True,
            current_source=False,
            type="PV",
            name="test",
            sn_mva=10.0,
        )
        net.asymmetric_sgen[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [*negativ_ints, *not_ints_list]),
                itertools.product(["p_a_mw"], [*positiv_floats, *not_floats_list]),
                itertools.product(["p_b_mw"], [*positiv_floats, *not_floats_list]),
                itertools.product(["p_c_mw"], [*positiv_floats, *not_floats_list]),
                itertools.product(["q_a_mvar"], not_floats_list),
                itertools.product(["q_b_mvar"], not_floats_list),
                itertools.product(["q_c_mvar"], not_floats_list),
                itertools.product(["scaling"], [*negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], not_boolean_list),
                itertools.product(["current_source"], not_boolean_list),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = create_empty_network()
        create_bus(net, 0.4)          # index 0
        create_bus(net, 0.4)          # index 1

        create_asymmetric_sgen(
            net,
            bus=0,
            p_a_mw=-1.0,
            q_a_mvar=0.5,
            p_b_mw=-1.0,
            q_b_mvar=0.5,
            p_c_mw=-1.0,
            q_c_mvar=0.5,
            scaling=1.0,
            in_service=True,
            current_source=False,
            type="PV",
            name="test",
            sn_mva=10.0,
        )
        net.asymmetric_sgen[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricSgenOptionalFields:
    """Tests for optional asymmetric_sgen fields"""

    def test_all_optional_fields_valid(self):
        """Test: asymmetric_sgen with every optional field is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-10.0,
            q_a_mvar=5.0,
            p_b_mw=-12.0,
            q_b_mvar=6.0,
            p_c_mw=-11.0,
            q_c_mvar=4.0,
            scaling=1.0,
            in_service=True,
            current_source=True,
            type="WP",
            name="lorem ipsum",
            sn_mva=25.0,
        )
        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: asymmetric_sgen with optional fields including nulls is valid"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-10.0,
            q_a_mvar=5.0,
            p_b_mw=-10.0,
            q_b_mvar=5.0,
            p_c_mw=-10.0,
            q_c_mvar=5.0,
            scaling=1.0,
            in_service=True,
            current_source=False,
            name="lorem ipsum",
        )
        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-8.0,
            q_a_mvar=3.0,
            p_b_mw=-9.0,
            q_b_mvar=3.5,
            p_c_mw=-7.5,
            q_c_mvar=2.5,
            scaling=1.0,
            in_service=False,
            current_source=True,
            type="CHP",
        )
        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-8.0,
            q_a_mvar=3.0,
            p_b_mw=-9.0,
            q_b_mvar=3.5,
            p_c_mw=-7.5,
            q_c_mvar=2.5,
            scaling=1.0,
            in_service=False,
            current_source=True,
            sn_mva=15.0,
        )
        net.asymmetric_sgen['type'].at[0] = None
        net.asymmetric_sgen['type'].at[2] = None

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(['name'], strings),
                #itertools.product(["type"], [pd.NA, "PV", "WP", "CHP"]),
                itertools.product(["sn_mva"], positiv_floats),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-10.0,
            q_a_mvar=5.0,
            p_b_mw=-10.0,
            q_b_mvar=5.0,
            p_c_mw=-10.0,
            q_c_mvar=5.0,
            scaling=1.0,
            in_service=True,
            current_source=False,
            **{parameter: valid_value}
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], [*strings, *not_strings_list]),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-1.0,
            q_a_mvar=0.2,
            p_b_mw=-1.0,
            q_b_mvar=0.2,
            p_c_mw=-1.0,
            q_c_mvar=0.2,
            scaling=1.0,
            in_service=True,
            current_source=True,
        )
        net.asymmetric_sgen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricSgenForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        b0 = create_bus(net, 0.4)

        create_asymmetric_sgen(
            net,
            bus=b0,
            p_a_mw=-1.0,
            q_a_mvar=0.5,
            p_b_mw=-1.0,
            q_b_mvar=0.5,
            p_c_mw=-1.0,
            q_c_mvar=0.5,
            scaling=1.0,
            in_service=True,
            current_source=False,
            type="PV",
        )

        net.asymmetric_sgen["bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricSgenResults:
    """Tests for asymmetric_sgen results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_asymmetric_sgen_result_totals(self):
        """Test: aggregated p_mw / q_mvar results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_asymmetric_sgen_3ph_results(self):
        """Test: 3-phase results contain valid values per phase"""
        pass
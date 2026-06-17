# test_pandera_asymmetric_sgen_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus, create_asymmetric_sgen
from pandapower.network import pandapowerNet
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
                # TODO: missing in docu and create function
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: valid required values are accepted"""
        net = pandapowerNet(name="test_valid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
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
            # current_source=False,
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
                itertools.product(["p_a_mw"], [float(np.nan), pd.NA, *positiv_floats, *not_floats_list]),
                itertools.product(["p_b_mw"], [float(np.nan), pd.NA, *positiv_floats, *not_floats_list]),
                itertools.product(["p_c_mw"], [float(np.nan), pd.NA, *positiv_floats, *not_floats_list]),
                itertools.product(["q_a_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["q_b_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["q_c_mvar"], [float(np.nan), pd.NA, *not_floats_list]),
                itertools.product(["scaling"], [float(np.nan), pd.NA, *negativ_floats, *not_floats_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1

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
            # current_source=False, # See todo above, this is treated as custom currently
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
        net = pandapowerNet(name="test_all_optional_fields_valid")
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
            # current_source=True,
            type="WP",
            name="lorem ipsum",
            sn_mva=25.0,
        )

        # Add phase-specific sn values
        net.asymmetric_sgen["sn_a_mva"] = 8.0
        net.asymmetric_sgen["sn_b_mva"] = 9.0
        net.asymmetric_sgen["sn_c_mva"] = 7.0

        validate_network(net)

    def test_optional_fields_with_nulls(self):
        """Test: asymmetric_sgen with optional fields including nulls is valid"""
        net = pandapowerNet(name="test_optional_fields_with_nulls")
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
            # current_source=False,
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
            # current_source=True,
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
            # current_source=True,
            sn_mva=15.0,
        )
        net.asymmetric_sgen["sn_a_mva"] = [float(np.nan), 5.0, float(np.nan)]
        net.asymmetric_sgen["sn_b_mva"] = [6.0, float(np.nan), float(np.nan)]
        net.asymmetric_sgen["sn_c_mva"] = [float(np.nan), float(np.nan), 4.0]

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["type"], [pd.NA, *strings]),
                itertools.product(["sn_mva"], [float(np.nan), *positiv_floats]),
                itertools.product(["sn_a_mva"], [float(np.nan), *positiv_floats]),
                itertools.product(["sn_b_mva"], [float(np.nan), *positiv_floats]),
                itertools.product(["sn_c_mva"], [float(np.nan), *positiv_floats]),
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
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
            # current_source=False,
            **{parameter: valid_value},
        )
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], not_strings_list),
                itertools.product(["type"], not_strings_list),
                itertools.product(["sn_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["sn_a_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["sn_b_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
                itertools.product(["sn_c_mva"], [*negativ_floats_plus_zero, *not_floats_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
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
            # current_source=True,
        )
        net.asymmetric_sgen[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestAsymmetricSgenForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = pandapowerNet(name="test_invalid_bus_index")
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
            # current_source=False,
            type="PV",
        )

        net.asymmetric_sgen["bus"] = 9999

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_valid_bus_index_non_sequential")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_asymmetric_sgen(
            net, bus=10, p_a_mw=-1.0, q_a_mvar=0.5,
            p_b_mw=-1.0, q_b_mvar=0.5, p_c_mw=-1.0, q_c_mvar=0.5,
            scaling=1.0, in_service=True,
        )
        create_asymmetric_sgen(
            net, bus=42, p_a_mw=-2.0, q_a_mvar=0.3,
            p_b_mw=-2.0, q_b_mvar=0.3, p_c_mw=-2.0, q_c_mvar=0.3,
            scaling=0.9, in_service=True,
        )
        create_asymmetric_sgen(
            net, bus=100, p_a_mw=-0.5, q_a_mvar=0.1,
            p_b_mw=-0.5, q_b_mvar=0.1, p_c_mw=-0.5, q_c_mvar=0.1,
            scaling=1.0, in_service=False,
        )

        validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = create_empty_network()
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus(net, 0.4, index=100)

        create_asymmetric_sgen(
            net, bus=10, p_a_mw=-1.0, q_a_mvar=0.5,
            p_b_mw=-1.0, q_b_mvar=0.5, p_c_mw=-1.0, q_c_mvar=0.5,
            scaling=1.0, in_service=True,
        )
        create_asymmetric_sgen(
            net, bus=42, p_a_mw=-2.0, q_a_mvar=0.3,
            p_b_mw=-2.0, q_b_mvar=0.3, p_c_mw=-2.0, q_c_mvar=0.3,
            scaling=0.9, in_service=True,
        )
        create_asymmetric_sgen(
            net, bus=100, p_a_mw=-0.5, q_a_mvar=0.1,
            p_b_mw=-0.5, q_b_mvar=0.1, p_c_mw=-0.5, q_c_mvar=0.1,
            scaling=1.0, in_service=False,
        )

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

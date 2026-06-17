# test_pandera_vsc_bipolar_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create._utils import add_column_to_df
from pandapower.create import create_bus, create_bus_dc, create_vsc_bipolar
from pandapower.network import pandapowerNet
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


def _create_valid_vsc_bipolar_row(required=True, bus=0, bus_dc_plus=0, bus_dc_minus=1):
    df = {
        "bus": bus,
        "bus_dc_plus": bus_dc_plus,
        "bus_dc_minus": bus_dc_minus,
        "r_ohm": 0.1,
        "x_ohm": 0.05,
        "r_dc_ohm": 0.02,
        "pl_dc_mw": 0.5,
        "control_mode": "Vac_phi",
        "control_value_1": 1.0,
        "control_value_2": 10.0,
        "controllable": True,
        "in_service": True,
    }
    if not required:
        df["name"] = "test"
    return df


def _create_net_with_buses():
    """Helper to create a network with AC and DC buses."""
    net = pandapowerNet(name="_create_net_with_buses")
    create_bus(net, 0.4)  # AC index 0
    create_bus(net, 0.4)  # AC index 1
    create_bus_dc(net, vn_kv=110.0)  # DC index 0
    create_bus_dc(net, vn_kv=110.0)  # DC index 1
    return net


class TestVscBipolarRequiredFields:
    """Tests for required vsc_bipolar fields"""

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
                itertools.product(["control_mode"], ["Vac_phi", "Vdc_phi", "Vdc_Q", "Pac_Vac", "Pac_Qac", "Vdc_Vac"]),
                itertools.product(["control_value_1"], all_allowed_floats),
                itertools.product(["control_value_2"], all_allowed_floats),
                itertools.product(["controllable"], bools),
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
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1
        create_bus_dc(net, vn_kv=110.0, index=42)

        # Create a valid vsc_bipolar element
        row = _create_valid_vsc_bipolar_row(bus=0, bus_dc_plus=0, bus_dc_minus=1)
        create_vsc_bipolar(net, **row)

        net.vsc_bipolar[parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["bus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_plus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["bus_dc_minus"], [float(np.nan), pd.NA, None, *negativ_ints, *not_ints_list]),
                itertools.product(["r_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["x_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["r_dc_ohm"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["pl_dc_mw"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(
                    ["control_mode"], [float(np.nan), pd.NA, None, *strings, *not_strings_list]
                ),
                itertools.product(["control_value_1"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["control_value_2"], [float(np.nan), pd.NA, None, *not_floats_list]),
                itertools.product(["controllable"], [float(np.nan), pd.NA, None, *not_boolean_list]),
                itertools.product(["in_service"], [float(np.nan), pd.NA, None, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)  # index 0
        create_bus(net, 0.4)  # index 1
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        row = _create_valid_vsc_bipolar_row(bus=0, bus_dc_plus=0, bus_dc_minus=1)
        create_vsc_bipolar(net, **row)

        net.vsc_bipolar[parameter] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVscBipolarOptionalFields:
    """Tests for optional vsc_bipolar fields"""

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
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        row = _create_valid_vsc_bipolar_row(required=False, bus=b0, bus_dc_plus=0, bus_dc_minus=1)
        create_vsc_bipolar(net=net, **row)

        add_column_to_df(net, "vsc_bipolar", parameter)
        net.vsc_bipolar.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [float(np.nan), *not_strings_list]),
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        b0 = create_bus(net, 0.4)
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        row = _create_valid_vsc_bipolar_row(required=False, bus=b0, bus_dc_plus=0, bus_dc_minus=1)
        create_vsc_bipolar(net=net, **row)
        net.vsc_bipolar[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestVscBipolarForeignKey:
    """Tests for foreign key constraints"""

    def test_invalid_bus_index(self):
        """Test: bus FK must reference an existing bus index"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_bipolar(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.05,
            r_dc_ohm=0.02,
            pl_dc_mw=0.5,
            control_mode="Vac_phi",
            control_value_1=1.0,
            control_value_2=10.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_bipolar["bus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_plus_index(self):
        """Test: bus_dc_plus FK must reference an existing bus_dc index"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_bipolar(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.05,
            r_dc_ohm=0.02,
            pl_dc_mw=0.5,
            control_mode="Vac_phi",
            control_value_1=1.0,
            control_value_2=10.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_bipolar["bus_dc_plus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_invalid_bus_dc_minus_index(self):
        """Test: bus_dc_minus FK must reference an existing bus_dc index"""
        net = create_empty_network()
        create_bus(net, 0.4)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 0
        create_bus_dc(net, vn_kv=110.0)  # index 1

        create_vsc_bipolar(
            net,
            bus=0,
            bus_dc_plus=0,
            bus_dc_minus=1,
            r_ohm=0.1,
            x_ohm=0.05,
            r_dc_ohm=0.02,
            pl_dc_mw=0.5,
            control_mode="Vac_phi",
            control_value_1=1.0,
            control_value_2=10.0,
            controllable=True,
            in_service=True,
        )

        net.vsc_bipolar["bus_dc_minus"] = 9999
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_valid_bus_index_non_sequential(self):
        """Test: bus FK works with non-sequential bus indices"""
        net = pandapowerNet(name="test_invalid_fk_index")
        create_bus(net, 0.4, index=10)
        create_bus(net, 0.4, index=42)
        create_bus_dc(net, vn_kv=110.0, index=10)
        create_bus_dc(net, vn_kv=110.0, index=42)

        create_vsc_bipolar(
            net=net,
            bus=10,
            bus_dc_plus=10,
            bus_dc_minus=42,
            r_ohm=0.1,
            x_ohm=0.05,
            r_dc_ohm=0.02,
            pl_dc_mw=0.5,
            control_mode="Vac_phi",
            control_value_1=1.0,
            control_value_2=10.0,
            controllable=True,
            in_service=True,
        )

        validate_network(net)


class TestVscBipolarResults:
    """Tests for vsc_bipolar results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_vsc_bipolar_result_totals(self):
        """Test: aggregated p_mw / q_mvar / dc results are consistent"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_vsc_bipolar_ac_dc_results(self):
        """Test: AC and DC side results contain valid values"""
        pass

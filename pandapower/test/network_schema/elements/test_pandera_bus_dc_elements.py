import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower import create_bus_dc
from pandapower.create import create_empty_network
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_boolean_list,
    negativ_floats,
    positiv_floats,
    not_allowed_floats, zero_float,
)


class TestBusDCRequiredFields:
    """Tests for required dc bus fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["vn_kv"], positiv_floats),
                itertools.product(["in_service"], bools),
            )
        )
    )
    def test_valid_required_values(self, parameter, valid_value):
        net = create_empty_network()

        # A minimal valid bus_dc
        create_bus_dc(net, vn_kv=1.0, in_service=True)

        # Modify the tested parameter
        net.bus_dc.at[0, parameter] = valid_value

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(
                    ["vn_kv"],
                    [*not_floats_list, *negativ_floats, *zero_float]
                ),
                itertools.product(
                    ["in_service"],
                    [*not_boolean_list]
                ),
            )
        )
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        net = create_empty_network()
        create_bus_dc(net, vn_kv=1.0, in_service=True)

        net.bus_dc[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusDCOptionalFields:
    """Tests for optional dc bus fields"""

    def test_bus_dc_with_optional_fields(self):
        net = create_empty_network()
        create_bus_dc(
            net,
            vn_kv=1.0,
            in_service=True,
            name="my_dc_bus",
            type="n",
            zone="europe",
            geo="POINT(0 0)",
            max_vm_pu=1.1,
            min_vm_pu=0.9,
        )
        validate_network(net)

    def test_bus_dc_with_optional_fields_including_nulls(self):
        net = create_empty_network()
        create_bus_dc(net, vn_kv=1.0, in_service=True, name='bye world')
        create_bus_dc(net, vn_kv=1.0, in_service=True, type='b')
        create_bus_dc(net, vn_kv=1.0, in_service=True, zone="somewhere")
        create_bus_dc(net, vn_kv=1.0, in_service=True, geo=pd.NA)
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name", "type", "zone", "geo"], [pd.NA, *strings]),
                itertools.product(["min_vm_pu", "max_vm_pu"], [float(np.nan), np.nan, *positiv_floats]),
            )
        )
    )
    def test_valid_optional_values(self, parameter, valid_value):
        net = create_empty_network()
        create_bus_dc(net, vn_kv=1.0, in_service=True, **{parameter: valid_value})
        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name", "type", "zone", "geo"], [*not_strings_list, np.nan]),
                itertools.product(["min_vm_pu", "max_vm_pu"], [*not_floats_list, *not_allowed_floats]),
            )
        )
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        net = create_empty_network()
        create_bus_dc(net, vn_kv=1.0, in_service=True)
        net.bus_dc[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusDCResults:
    """Tests for bus_dc results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bus_dc_voltage_results(self):
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bus_dc_power_results(self):
        pass

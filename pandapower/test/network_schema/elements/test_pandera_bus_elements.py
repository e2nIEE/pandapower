# test_pandera_bus_elements.py

import itertools
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from pandapower.create import create_bus
from pandapower.create._utils import add_tag_group_to_df
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network
from pandapower.test.network_schema.elements.helper import (
    strings,
    bools,
    not_strings_list,
    not_floats_list,
    not_allowed_floats,
    not_boolean_list,
    negativ_floats,
    negativ_floats_plus_zero,
    positiv_floats,
    positiv_floats_plus_zero,
    zero_float,
)


class TestBusRequiredFields:
    """Tests for required bus fields"""

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [pd.NA, *strings]),
                itertools.product(["vn_kv"], positiv_floats),
                itertools.product(["in_service"], bools),
            )
        ),
    )
    def test_valid_required_values(self, parameter, valid_value):
        """Test: Invalid required values are rejected"""
        net = pandapowerNet(name="test_valid_required_values")
        kwargs = {parameter: valid_value}
        vn_kv = kwargs.pop("vn_kv", 0.4)
        create_bus(net, vn_kv, **kwargs)

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["name"], [np.nan, *not_strings_list]),
                itertools.product(["vn_kv"], [np.nan, pd.NA, *not_floats_list, *negativ_floats, *zero_float]),
                itertools.product(["in_service"], [np.nan, pd.NA, *not_boolean_list]),
            )
        ),
    )
    def test_invalid_required_values(self, parameter, invalid_value):
        """Test: Invalid required values are rejected"""
        net = pandapowerNet(name="test_invalid_required_values")
        create_bus(net, 0.4)
        net.bus[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusOptionalFields:
    """Tests for optional bus fields"""

    def test_bus_with_optional_fields(self):
        """Test: Bus with every optional fields is valid"""
        net = pandapowerNet(name="test_bus_with_optional_fields")
        create_bus(net, vn_kv=0.4, zone="everywhere", max_vm_pu=1.1, min_vm_pu=0.9, geodata=(0, 0), type="b")
        validate_network(net)

    def test_buses_with_optional_fields_including_nullvalues(self):
        """Test: Buses with some optional fields is valid"""
        net = pandapowerNet(name="test_buses_with_optional_fields_including_nullvalues")
        create_bus(net, 0.4, zone="nowhere")
        create_bus(net, 0.4, max_vm_pu=1.1, min_vm_pu=0.9)
        create_bus(net, 0.4, geodata=(1, 2))
        create_bus(net, 0.4, type="x")

        validate_network(net)

    def test_valid_type_values(self):
        """Test: Valid 'type' values are accepted"""
        net = pandapowerNet(name="test_valid_type_values")
        create_bus(net, 0.4)
        create_bus(net, 0.4)

        net.bus["type"].at[0] = "x"
        net.bus["type"].at[1] = pd.NA

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,valid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu", "max_vm_pu"], [np.nan, *positiv_floats]),
                itertools.product(["type", "zone", "geo"], [pd.NA, *strings]),
                itertools.product(
                    ["origin_id", "origin_class", "origin_profile", "cim_topnode",
                     "ConnectivityNodeContainer_id", "Substation_id", "description",
                     "Busbar_id", "Busbar_name", "GeographicalRegion_id", "GeographicalRegion_name",
                     "SubGeographicalRegion_id", "SubGeographicalRegion_name", "ucte_country"],
                    [pd.NA, *strings]
                )
            )
        ),
    )
    def test_valid_optional_values(self, parameter, valid_value):
        """Test: valid optional values are accepted"""
        net = pandapowerNet(name="test_valid_optional_values")
        create_bus(net, 0.4, **{parameter: valid_value})

        validate_network(net)

    @pytest.mark.parametrize(
        "parameter,invalid_value",
        list(
            itertools.chain(
                itertools.product(["min_vm_pu"], [*negativ_floats, *not_floats_list, *not_allowed_floats]),
                itertools.product(["max_vm_pu"], [*negativ_floats_plus_zero, *not_floats_list, *not_allowed_floats]),
                itertools.product(["type", "zone", "geo"], [np.nan, float(np.nan), *not_strings_list]),
                itertools.product(
                    ["origin_id", "origin_class", "origin_profile", "cim_topnode",
                     "ConnectivityNodeContainer_id", "Substation_id", "description",
                     "Busbar_id", "Busbar_name", "GeographicalRegion_id", "GeographicalRegion_name",
                     "SubGeographicalRegion_id", "SubGeographicalRegion_name", "ucte_country"],
                    [np.nan, float(np.nan), *not_strings_list]
                )
            )
        ),
    )
    def test_invalid_optional_values(self, parameter, invalid_value):
        """Test: Invalid optional values are rejected"""
        net = pandapowerNet(name="test_invalid_optional_values")
        create_bus(net, 0.4)

        # for OPF columns, add group dependency so only target parameter triggers failure
        #  otherwise the "min < max" check will fail.
        if parameter in ["min_vm_pu", "max_vm_pu"]:
            add_tag_group_to_df(net, "bus", "opf")

        net.bus[parameter] = invalid_value

        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusCrossFieldConstraints:
    """Tests for cross-field constraints"""

    def test_min_vm_pu_greater_than_max_vm_pu_rejected(self):
        """Test: min_vm_pu must be <= max_vm_pu"""
        net = pandapowerNet(name="test_min_max_constraint")
        create_bus(net, 0.4, min_vm_pu=1.5, max_vm_pu=1.0)
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)

    def test_max_vm_pu_valid_upper_bound(self):
        """Test: max_vm_pu = 2 is valid (boundary)"""
        net = pandapowerNet(name="test_max_vm_pu_valid")
        create_bus(net, 0.4, min_vm_pu=0.0, max_vm_pu=2.0)
        validate_network(net)

    def test_min_vm_pu_valid_zero(self):
        """Test: min_vm_pu = 0 is valid (boundary)"""
        net = pandapowerNet(name="test_min_vm_pu_valid")
        create_bus(net, 0.4, min_vm_pu=0.0, max_vm_pu=1.0)
        validate_network(net)

    @pytest.mark.parametrize("invalid_value", [2.1, 3.0, 100.0])
    def test_max_vm_pu_upper_bound(self, invalid_value):
        """Test: max_vm_pu must be <= 2"""
        net = pandapowerNet(name="test_max_vm_pu_upper")
        create_bus(net, 0.4, min_vm_pu=0.0)
        net.bus["max_vm_pu"] = invalid_value
        with pytest.raises(pa.errors.SchemaError):
            validate_network(net)


class TestBusCimUcteFields:
    """Tests for CIM and UCTE fields"""

    def test_cim_fields_valid(self):
        """Test: CIM fields with valid string values are accepted"""
        net = pandapowerNet(name="test_cim_fields")
        create_bus(net, 0.4)
        net.bus["origin_id"] = pd.Series(["cim_123"], dtype=pd.StringDtype())
        net.bus["origin_class"] = pd.Series(["BusbarSection"], dtype=pd.StringDtype())
        net.bus["origin_profile"] = pd.Series(["CIM"], dtype=pd.StringDtype())
        net.bus["cim_topnode"] = pd.Series(["top_1"], dtype=pd.StringDtype())
        net.bus["ConnectivityNodeContainer_id"] = pd.Series(["cnc_1"], dtype=pd.StringDtype())
        net.bus["Substation_id"] = pd.Series(["sub_1"], dtype=pd.StringDtype())
        net.bus["description"] = pd.Series(["Test bus"], dtype=pd.StringDtype())
        net.bus["Busbar_id"] = pd.Series(["bb_1"], dtype=pd.StringDtype())
        net.bus["Busbar_name"] = pd.Series(["Busbar 1"], dtype=pd.StringDtype())
        net.bus["GeographicalRegion_id"] = pd.Series(["gr_1"], dtype=pd.StringDtype())
        net.bus["GeographicalRegion_name"] = pd.Series(["Region 1"], dtype=pd.StringDtype())
        net.bus["SubGeographicalRegion_id"] = pd.Series(["sgr_1"], dtype=pd.StringDtype())
        net.bus["SubGeographicalRegion_name"] = pd.Series(["SubRegion 1"], dtype=pd.StringDtype())
        net.bus["ucte_country"] = pd.Series(["DE"], dtype=pd.StringDtype())
        validate_network(net)

    def test_cim_fields_nullable(self):
        """Test: CIM fields can be null"""
        net = pandapowerNet(name="test_cim_null")
        create_bus(net, 0.4)
        net.bus["origin_id"] = pd.Series([pd.NA], dtype=pd.StringDtype())
        validate_network(net)


class TestBusResults:
    """Tests for bus results after calculations"""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bus_voltage_results(self):
        """Test: Voltage results are within valid range"""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_bus_power_results(self):
        """Test: Power results are consistent"""
        pass
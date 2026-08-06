# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandera as pa
import pytest

from pandapower.create import (
    create_bus, create_ext_grid, create_line_from_parameters,
    create_load_from_cosphi, create_shunt_as_capacitor, create_sgen_from_cosphi,
    create_series_reactor_as_impedance, create_transformer_from_parameters, create_load
)
from pandapower.run import runpp
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_convenience_create_functions():
    net = pandapowerNet(name="test_convenience_create_functions")
    b1 = create_bus(net, 110.0)
    b2 = create_bus(net, 110.0)
    b3 = create_bus(net, 20)
    create_ext_grid(net, b1)
    create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=20.0,
        r_ohm_per_km=0.0487,
        x_ohm_per_km=0.1382301,
        c_nf_per_km=160.0,
        max_i_ka=0.664,
    )

    l0 = create_load_from_cosphi(
        net, b2, 10, 0.95, "underexcited", name="load", test_kwargs="dummy_string"
    )
    runpp(net, init="flat")

    assert np.isclose(net.load.p_mw.at[l0], 9.5)
    assert net.load.q_mvar.at[l0] > 0
    assert np.sqrt(net.load.p_mw.at[l0] ** 2 + net.load.q_mvar.at[l0] ** 2) == 10
    assert np.isclose(net.res_bus.vm_pu.at[b2], 0.99990833838)
    assert net.load.name.at[l0] == "load"
    assert net.load.test_kwargs.at[l0] == "dummy_string"

    sh0 = create_shunt_as_capacitor(
        net, b2, 10, loss_factor=0.01, name="shunt", test_kwargs="dummy_string"
    )
    runpp(net, init="flat")
    assert np.isclose(net.res_shunt.q_mvar.at[sh0], -10.043934174)
    assert np.isclose(net.res_shunt.p_mw.at[sh0], 0.10043933665)
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0021942964)
    assert net.shunt.name.at[sh0] == "shunt"
    assert net.shunt.test_kwargs.at[sh0] == "dummy_string"

    sg0 = create_sgen_from_cosphi(
        net, b2, 5, 0.95, "overexcited", name="sgen", test_kwargs="dummy_string"
    )
    runpp(net, init="flat")
    assert np.sqrt(net.sgen.p_mw.at[sg0] ** 2 + net.sgen.q_mvar.at[sg0] ** 2) == 5
    assert np.isclose(net.sgen.p_mw.at[sg0], 4.75)
    assert net.sgen.q_mvar.at[sg0] > 0
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0029376578)
    assert net.sgen.name.at[sg0] == "sgen"
    assert net.sgen.test_kwargs.at[sg0] == "dummy_string"

    tol = 1e-6
    base_z = 110 ** 2 / 100
    sind = create_series_reactor_as_impedance(
        net, b1, b2, r_ohm=100, x_ohm=200, sn_mva=100, test_kwargs="dummy_string"
    )
    assert net.impedance.at[sind, "rft_pu"] - 100 / base_z < tol
    assert net.impedance.at[sind, "xft_pu"] - 200 / base_z < tol
    assert net.impedance.test_kwargs.at[sind] == "dummy_string"

    tid = create_transformer_from_parameters(
        net,
        hv_bus=b2,
        lv_bus=b3,
        sn_mva=0.1,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=5,
        vk_percent=20,
        pfe_kw=1,
        i0_percent=1,
        test_kwargs="dummy_string",
    )

    validate_network(net)

    create_load(net, b3, 0.1)
    assert net.trafo.at[tid, "df"] == 1
    runpp(net)
    validate_network(net)

    tr_l = net.res_trafo.at[tid, "loading_percent"]
    net.trafo.at[tid, "df"] = 2
    runpp(net)
    tr_l_2 = net.res_trafo.at[tid, "loading_percent"]
    assert tr_l == tr_l_2 * 2
    net.trafo.at[tid, "df"] = 0
    with pytest.raises(UserWarning):
        runpp(net)
    assert net.trafo.test_kwargs.at[tid] == "dummy_string"

    with pytest.raises(pa.errors.SchemaError):
        validate_network(net)

def test_const_percent_values_deprecated_handling():
    # This test checks that passing const_z_percent and const_i_percent to create_load
    # sets all four percent columns and triggers the deprecation warning.
    net = pandapowerNet(name="test_const_percent_values_deprecated_handling")
    b1 = create_bus(net, 20)
    with pytest.warns(DeprecationWarning, match="const_z_percent and const_i_percent will be deprecated"):
        idx = create_load(
            net, b1, p_mw=1.0, q_mvar=0.5,
            const_z_percent=11, const_i_percent=22
        )
    # Check that the values are set correctly
    load_idx = net.load.loc[idx]
    assert load_idx.const_z_p_percent == 11
    assert load_idx.const_z_q_percent == 11
    assert load_idx.const_i_p_percent == 22
    assert load_idx.const_i_q_percent == 22

    validate_network(net)

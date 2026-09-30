# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
from copy import deepcopy

from pandapower import pandapowerNet, pp_dir
from pandapower.auxiliary import get_free_id
from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_gen,
    create_lines,
    create_load,
    create_sgen,
    create_transformer_from_parameters,
)
from pandapower.file_io import from_pickle
from pandapower.toolbox.comparison import nets_equal


def assert_net_equal(net1, net2, **kwargs):
    """
    Raises AssertionError if grids are not equal.
    """
    assert nets_equal(net1, net2, **kwargs)


def assert_res_equal(net1, net2, **kwargs):
    """
    Raises AssertionError if results are not equal.
    """
    if "check_only_results" in kwargs:
        if not kwargs["check_only_results"]:
            raise ValueError("'check_only_results' cannot be False in assert_res_equal().")
        kwargs = deepcopy(kwargs)
        del kwargs["check_only_results"]
    assert nets_equal(net1, net2, check_only_results=True, **kwargs)


def create_test_network():
    """
    Creates a simple pandapower test network
    """
    net = pandapowerNet(name='test_network')
    (b1,) = create_buses(net, 1, name="bus1", vn_kv=10.0)
    create_ext_grid(net, b1)
    (b2,) = create_buses(net, 1, name="bus2", geodata=(1.0, 2.0), vn_kv=0.4)
    (b3,) = create_buses(net, 1, name="bus3", geodata=(1.0, 3.0), vn_kv=0.4, index=7)
    (b4,) = create_buses(net, 1, name="bus4", vn_kv=10.0)
    create_transformer_from_parameters(
        net,
        b4,
        b2,
        vk_percent=3.75,
        tap_max=2,
        vn_lv_kv=0.4,
        shift_degree=150,
        tap_neutral=0,
        vn_hv_kv=10.0,
        vkr_percent=2.8125,
        tap_pos=0,
        tap_side="hv",
        tap_min=-2,
        tap_step_percent=2.5,
        tap_step_degree=0,
        i0_percent=0.68751,
        sn_mva=0.016,
        pfe_kw=0.11,
        name=None,
        in_service=True,
        index=None,
        tap_changer_type="Ratio",
    )
    # 0.016 MVA 10/0.4 kV ET 16/23  SGB

    create_lines(
        net,
        b2,
        b3,
        1,
        {"r_ohm_per_km": 0.2067, "x_ohm_per_km": 0.1897522, "c_nf_per_km": 720.0, "max_i_ka": 0.328},
        name="line1",
        ices=0.389985,
        geodata=[(1.0, 2.0), (3.0, 4.0)],
    )
    # NAYY 1x150RM 0.6/1kV ir
    create_lines(
        net,
        b1,
        b4,
        1,
        {"r_ohm_per_km": 0.876, "x_ohm_per_km": 0.1159876, "c_nf_per_km": 260.0, "max_i_ka": 0.123},
        name="line2",
    )

    # NAYSEY 3x35rm/16 6/10kV

    create_load(net, b2, p_mw=0.010, q_mvar=0, name="load1")
    create_load(net, b3, p_mw=0.040, q_mvar=0.002, name="load2")
    create_gen(net, b4, p_mw=0.200, vm_pu=1.0)
    create_sgen(net, b3, p_mw=0.050, sn_mva=0.1)

    return net


def create_test_network2():
    """Creates a simple pandapower test network
    """
    return from_pickle(os.path.join(pp_dir, "test", "loadflow", "testgrid.p"))


def add_grid_connection(net, vn_kv=20., zone=None):
    """Creates a new grid connection for create_result_test_network()
    """
    (b1,) = create_buses(net, 1, vn_kv=vn_kv, zone=zone)
    create_ext_grid(net, b1, vm_pu=1.01)
    b2 = get_free_id(net.bus) + 2  # shake up the indices so that non-consecutive indices are tested
    (b2,) = create_buses(net, 1, vn_kv=vn_kv, zone=zone, index=b2)
    l1 = create_test_line(net, b1, b2)
    return b1, b2, l1


def create_test_line(net, b1, b2, in_service=True):
    return create_lines(
        net,
        b1,
        b2,
        12.2,
        {
            "r_ohm_per_km": 0.08,
            "x_ohm_per_km": 0.12,
            "c_nf_per_km": 300,
            "max_i_ka": 0.2,
        },
        df=0.8,
        in_service=in_service,
        index=get_free_id(net.line) + 1,
    )[0]

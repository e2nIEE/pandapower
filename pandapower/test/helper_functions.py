# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
from copy import deepcopy

from pandapower import pp_dir
from pandapower.auxiliary import get_free_id
from pandapower.create import create_bus, create_empty_network, create_ext_grid, create_transformer_from_parameters, \
    create_line_from_parameters, create_load, create_gen, create_sgen
from pandapower.file_io import from_pickle
from pandapower.toolbox import nets_equal


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
    net = create_empty_network(name='test_network')
    b1 = create_bus(net, name="bus1", vn_kv=10.)
    create_ext_grid(net, b1)
    b2 = create_bus(net, name="bus2", geodata=(1., 2.), vn_kv=.4)
    b3 = create_bus(net, name="bus3", geodata=(1., 3.), vn_kv=.4, index=7)
    b4 = create_bus(net, name="bus4", vn_kv=10.)
    create_transformer_from_parameters(net, b4, b2, vk_percent=3.75,
                                       tap_max=2, vn_lv_kv=0.4,
                                       shift_degree=150, tap_neutral=0,
                                       vn_hv_kv=10.0, vkr_percent=2.8125,
                                       tap_pos=0, tap_side="hv", tap_min=-2,
                                       tap_step_percent=2.5, i0_percent=0.68751,
                                       sn_mva=0.016, pfe_kw=0.11, name=None,
                                       in_service=True, index=None)
    # 0.016 MVA 10/0.4 kV ET 16/23  SGB

    create_line_from_parameters(net, b2, b3, 1, name="line1", r_ohm_per_km=0.2067,
                                ices=0.389985, c_nf_per_km=720.0, max_i_ka=0.328,
                                x_ohm_per_km=0.1897522, geodata=[[1., 2.], [3., 4.]])
    # NAYY 1x150RM 0.6/1kV ir
    create_line_from_parameters(net, b1, b4, 1, name="line2", r_ohm_per_km=0.876,
                                c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876)

    # NAYSEY 3x35rm/16 6/10kV

    create_load(net, b2, p_mw=0.010, q_mvar=0, name="load1")
    create_load(net, b3, p_mw=0.040, q_mvar=0.002, name="load2")
    create_gen(net, b4, p_mw=0.200, vm_pu=1.0)
    create_sgen(net, b3, p_mw=0.050, sn_mva=0.1)

    return net


def create_test_network2():
    """Creates a simple pandapower test network
    """
    net = from_pickle(os.path.join(pp_dir, "test", "loadflow", "testgrid.p"))
    #    net = pp.file_io.from_pickle("testgrid.p")

    return net


def add_grid_connection(net, vn_kv=20., zone=None):
    """Creates a new grid connection for create_result_test_network()
    """
    b1 = create_bus(net, vn_kv=vn_kv, zone=zone)
    create_ext_grid(net, b1, vm_pu=1.01)
    b2 = get_free_id(net.bus) + 2  # shake up the indices so that non-consecutive indices are tested
    b2 = create_bus(net, vn_kv=vn_kv, zone=zone, index=b2)
    l1 = create_test_line(net, b1, b2)
    return b1, b2, l1


def create_test_line(net, b1, b2, in_service=True):
    return create_line_from_parameters(
        net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12, c_nf_per_km=300, max_i_ka=.2, df=.8,
        in_service=in_service, index=get_free_id(net.line) + 1
    )

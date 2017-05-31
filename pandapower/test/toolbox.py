# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


import os
from math import isnan

import numpy as np
import pandas.util.testing as pdt
import pytest

import pandapower as pp
import pandapower.test

try:
    import pplog as logging
except:
    import logging


def run_all_tests():
    """ function executing all tests
    """
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    pytest.main([os.path.abspath(os.path.dirname(pandapower.test.__file__)), "-s"])
    logger.setLevel(logging.INFO)


def assert_mpc_equal(mpc1, mpc2):
    for name in ['bus', 'gen', 'branch', 'baseMVA']:
        try:
            assert np.allclose(mpc1[name], mpc2[name])
        except AssertionError:
            print(
                "Conversion from pandapower net to Matpower case failed creating %s table" % name)
            raise
    try:
        assert np.array_equal(mpc1['version'], mpc2['version'])
    except AssertionError:
        print("Pypower version changed from {} to {}".format(
            mpc2['version'], mpc1['version']))


def assert_net_equal(a_net, b_net, reindex=False):
    """Returns True if the given pandapower networks are equal.
    Raises AssertionError if grids are not equal.
    """
    status = True
    namelist = ['bus', 'bus_geodata', 'load', 'sgen', 'ext_grid', 'line', 'shunt', 'line_geodata',
                'trafo', 'switch', 'trafo3w', 'gen', 'ext_grid', 'res_line', 'res_bus', 'res_sgen',
                'res_gen', 'res_shunt', 'res_load', 'res_ext_grid', 'res_trafo']
    for name in namelist:
        if name in a_net or name in b_net:
            if not (a_net[name] is None and b_net[name] is None):
                try:
                    pdt.assert_frame_equal(
                        a_net[name], b_net[name], check_dtype=True, check_like=reindex)
                except AssertionError:
                    pytest.fail("Tables are not equal: %s" % name)
                    status = False

    return status


def assert_res_equal(a, b):
    """Returns True if the result tables of the given pandapower networks are equal.
    Raises AssertionError if results are not equal.
    """
    namelist = ['res_line', 'res_bus', 'res_gen', 'res_sgen',
                'res_load', 'res_ext_grid', 'res_trafo', 'res_trafo3w']

    for name in namelist:
        if name in a or name in b:
            if not (a[name] is None and b[name] is None):
                try:
                    pdt.assert_frame_equal(a[name], b[name])
                except AssertionError:
                    pytest.fail("Result tables are not equal: %s" % name)
                    raise


def assert_res_out_of_service(net, idx, name):
    """Returns True if the result tables of the given pandapower network contain NaNs resp. 0
    Raises AssertionError if datatype is not according to specifications.

    Specifications are:

        res_bus["vm_pu"]  		     nan
        res_bus.va_degree  	         nan
        res_bus["p_kw"]       	     0
        res_bus["q_kvar"]        	  0

        res_line.p_from_kw  		  0
        res_line.q_from_kvar		  0
        res_line.p_to_kw		     0
        res_line.q_to_kvar		     0
        res_line.i_ka			     0
        res_line["loading_percent"] 0

        res_trafo			        all 0

        res_load			        all 0

        res_ext_grid		        all nan

        res_gen["p_kw"]		         0
        res_gen-q_kvar 		         0
        res_gen_va_degree	         nan

        res_sgen 		           all  0

      Future: Elements out of service will not appear in result table!

"""

    status = True
    try:
        if name == 'bus':
            assert isnan(net["res_bus"]["vm_pu"].at[idx])
            assert isnan(net["res_bus"].va_degree.at[idx])
            assert net["res_bus"]["p_kw"].at[idx] == 0
            assert net["res_bus"]["q_kvar"].at[idx] == 0
        elif name == 'gen':
            if net.gen.in_service.any():
                assert net["res_gen"]["p_kw"].at[idx] == 0
                assert net["res_gen"]["q_kvar"].at[idx] == 0
                assert isnan(net["res_gen"].va_degree.at[idx])
            else:
                assert net.res_gen is None
        elif name in ('load', 'trafo', 'sgen', 'line'):
            assert (net['res_' + name].loc[idx] == 0).all()
        elif name == 'ext_grid':
            assert idx not in net.res_ext_grid.index
        else:
            print("Element res_{} does not exist!".format(name, name))
            status = False

    except AssertionError:
        pytest.fail(
            "res_{} table is not according to specifications if {} is out of service".format(name, name))
        status = False
        raise

    return status


# TODO Future res_out_of_service:
#    try:
#        assert eval('net.res_'+name+'.loc['+str(idx)+'].isnull().all()')
#    except AssertionError:
#        print ("res_{} table is not according to specifications if {} is out of service".format(name, name))

# def assertItemsEqual(expected_seq, actual_seq, msg = None):
#        """An unordered sequence specific comparison. It asserts that
#        actual_seq and expected_seq have the same element counts.
#        Equivalent to: :
#
#            self.assertEqual(Counter(iter(actual_seq)),
#                             Counter(iter(expected_seq)))
#
#        Asserts that each element has the same count in both sequences.
#        Example:
#            - [0, 1, 1] and [1, 0, 1] compare equal.
#            - [0, 0, 1] and [0, 1] compare unequal.
#        """
#        import warnings
#        import sys
#        import collections
#        import unittest
#
#        first_seq, second_seq = list(expected_seq), list(actual_seq)
#        with warnings.catch_warnings():
#            if sys.py3kwarning:
#                # Silence Py3k warning raised during the sorting
#                for _msg in ["(code|dict|type) inequality comparisons",
#                             "builtin_function_or_method order comparisons",
#                             "comparing unequal types"]:
#                    warnings.filterwarnings("ignore", _msg, DeprecationWarning)
#            try:
#                first = collections.Counter(first_seq)
#                second = collections.Counter(second_seq)
#            except TypeError:
#                # Handle case with unhashable elements
#                differences = unittest.case._count_diff_all_purpose(first_seq, second_seq)
#            else:
#                if first == second:
#                    return
#                differences = unittest.case._count_diff_hashable(first_seq, second_seq)
#
#        if differences:
#            raise AssertionError


def create_test_network():
    """Creates a simple pandapower test network
    """
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, name="bus1", vn_kv=10.)
    pp.create_ext_grid(net, b1)
    b2 = pp.create_bus(net, name="bus2", geodata=(1, 2), vn_kv=.4)
    b3 = pp.create_bus(net, name="bus3", geodata=(1, 3), vn_kv=.4)
    b4 = pp.create_bus(net, name="bus4", vn_kv=10.)
    pp.create_transformer_from_parameters(net, b4, b2, vsc_percent=3.75,
                                          tp_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tp_mid=0,
                                          vn_hv_kv=10.0, vscr_percent=2.8125,
                                          tp_pos=0, tp_side="hv", tp_min=-2,
                                          tp_st_percent=2.5, i0_percent=0.68751,
                                          sn_kva=16.0, pfe_kw=0.11, name=None,
                                          in_service=True, index=None)
    # 0.016 MVA 10/0.4 kV ET 16/23  SGB

    pp.create_line_from_parameters(net, b2, b3, 1, name="line1", r_ohm_per_km=0.2067,
                                   ices=0.389985, c_nf_per_km=720.0, max_i_ka=0.328,
                                   x_ohm_per_km=0.1897522, geodata=np.array([[1, 2], [3, 4]]))
    # NAYY 1x150RM 0.6/1kV ir
    pp.create_line_from_parameters(net, b1, b4, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876)

    # NAYSEY 3x35rm/16 6/10kV

    pp.create_load(net, b2, p_kw=10, q_kvar=0, name="load1")
    pp.create_load(net, b3, p_kw=40, q_kvar=2, name="load2")
    pp.create_gen(net, 3, p_kw=-200., vm_pu=1.0)
    pp.create_sgen(net, 2, p_kw=-50, sn_kva=100)

    return net


def create_test_network2():
    """Creates a simple pandapower test network
    """
    folder = os.path.abspath(os.path.dirname(pandapower.test.__file__))
    net = pp.from_pickle(os.path.join(folder, "loadflow", "testgrid.p"))
    #    net = pp.file_io.from_pickle("testgrid.p")

    return net


def add_grid_connection(net, vn_kv=20., zone=None):
    """Creates a new grid connection for create_result_test_network()
    """
    b1 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    b2 = pp.get_free_id(net.bus) + 2  # shake up the indices so that non-consecutive indices are tested
    b2 = pp.create_bus(net, vn_kv=vn_kv, zone=zone, index=b2)
    l1 = create_test_line(net, b1, b2)
    return b1, b2, l1


def create_test_line(net, b1, b2, in_service=True):
    return pp.create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                          c_nf_per_km=300, max_i_ka=.2, df=.8,
                                          in_service=in_service, index=pp.get_free_id(net.line) + 1)


if __name__ == "__main__":
    run_all_tests()

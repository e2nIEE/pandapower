# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapower.converter as cv
from pandapower.run import LoadflowNotConverged, reset_results

from pypower.runpf import runpf
from pypower.ppoption import ppoption
# from pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT
from pypower.idx_bus import VM, BUS_I, VA
# from pypower.idx_gen import PG, QG, GEN_BUS


def test_to_ppc():
    # pypower cases to validate
    functions = ['case4gs', 'case6ww', 'case14', 'case30', 'case30pwl', 'case30Q',
    'case24_ieee_rts', 'case39']
    for fn in functions:
        # get pypower results
        pypower_module = __import__('pypower.' + fn)
        pypower_submodule = getattr(pypower_module, fn)
        pypower_function = getattr(pypower_submodule, fn)
        ppc_net = pypower_function()

        # get net from pandapower
        res_pypower, status_pypower = runpf(ppc_net, ppopt=ppoption(VERBOSE=0, OUT_ALL=0))

        pandapower_module = __import__('pandapower', fromlist=['networks'])
        pandapower_function = getattr(pandapower_module.networks, fn)
        net = pandapower_function()
        reset_results(net)

        # convert to ppc
        ppc = cv.to_ppc(net)
        # runpf from converted ppc
        res_converted_pp, status_converted_pp = runpf(ppc, ppopt=ppoption(VERBOSE=0, OUT_ALL=0))

        if status_converted_pp and status_pypower:
            # get lookup pp2ppc
            bus_lookup = net['_bus_lookup']
            # check for equality in bus voltages
            pp_buses = bus_lookup[res_converted_pp['bus'][:, BUS_I].astype(int)]
            assert np.allclose(res_converted_pp['bus'][pp_buses, VM:VA + 1],
                               res_pypower['bus'][:, VM:VA + 1])
            # ToDo: check equality of branch and gen values
            # pp_gen = bus_lookup[res_converted_pp['bus'][:, BUS_I].astype(int)]
            # assert np.allclose(res_pypower['gen'][res_pypower['order']['gen']['e2i'], PG:QG+1]
            #                    , res_converted_pp['gen'][:, PG:QG+1])
        else:
            raise LoadflowNotConverged("Loadflow did not converge!")


if __name__ == "__main__":
    pytest.main(["test_to_ppc.py", "-s"])

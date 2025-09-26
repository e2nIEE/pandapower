# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

from pandapower.converter import to_ppc, to_mpc
from pandapower.networks import case4gs, case6ww, case30, case39
from pandapower.pf.runpf_pypower import _runpf_pypower
from pandapower.powerflow import LoadflowNotConverged
from pandapower.pypower.idx_bus import VM, BUS_I, VA
from pandapower.results import reset_results
from pandapower.run import runpp
from pandapower.test.converter.test_from_ppc import get_testgrids


def test_to_ppc_and_mpc():
    # pypower cases to validate
    case_functions = [case4gs, case6ww, case30, case39]
    case_names = ["case4gs", "case6ww", "case30", "case39"]
    for pandapower_function, case_name in zip(case_functions, case_names):
        # get pypower grids with results
        ppc_net = get_testgrids("pypower_cases", f"{case_name}.json")

        # get pandapower grids
        net = pandapower_function()
        reset_results(net)

        # This should be reviewed
        runpp(net)

        # convert pandapower grids to ppc
        ppc = to_ppc(net)
        # convert pandapower grids to mpc (no result validation)
        mpc = to_mpc(net)

        # validate voltage results of pandapower-to-ppc-converted grids vs. original pypower results
        net["_options"]["ac"] = True
        net["_options"]["numba"] = True
        net["_options"]["tolerance_mva"] = 1e-8
        net["_options"]["algorithm"] = "fdbx"
        net["_options"]["max_iteration"] = 30
        net["_options"]["enforce_q_lims"] = False
        net["_options"]["calculate_voltage_angles"] = True
        res_converted_pp, status_converted_pp = _runpf_pypower(ppc, net["_options"])

        if status_converted_pp:
            # get lookup pp2ppc
            bus_lookup = net["_pd2ppc_lookups"]["bus"]
            # check for equality in bus voltages
            pp_buses = bus_lookup[res_converted_pp["bus"][:, BUS_I].astype(np.int64)]
            res1 = res_converted_pp["bus"][pp_buses, VM : VA + 1]
            res2 = ppc_net["bus"][:, VM : VA + 1]
            assert np.allclose(res1, res2)
        else:
            raise LoadflowNotConverged("Loadflow did not converge!")


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

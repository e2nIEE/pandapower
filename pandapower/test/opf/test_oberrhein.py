# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandapower as pp
import pandapower.networks as nw
import pytest
try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_opf_oberrhein():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """
    # create net
    net = nw.mv_oberrhein()

    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9
    net.line["max_loading_percent"] = 200
    net.trafo["max_loading_percent"] = 100
    net.sgen["min_p_kw"] = -net.sgen.sn_kva
    net.sgen["max_p_kw"] = 0
    net.sgen["max_q_kvar"] = 1
    net.sgen["min_q_kvar"] = -1
    net.sgen["controllable"] = 1
    net.load["controllable"] = 0
    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

if __name__ == "__main__":
    pytest.main(["test_oberrhein.py", "-xs"])

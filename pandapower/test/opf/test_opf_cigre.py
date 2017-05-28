# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


import pandapower as pp
import pytest
import pandapower.networks as nw
try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


def test_opf_oberrhein():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """
    # create net
    net = nw.create_cigre_network_mv(with_der="pv_wind")

    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9
    net.line["max_loading_percent"] = 200
    net.trafo["max_loading_percent"] = 100
    net.sgen["min_p_kw"] = -net.sgen.sn_kva
    net.sgen["max_p_kw"] = 0
    net.sgen["max_q_kvar"] = 10
    net.sgen["min_q_kvar"] = -10
    net.sgen["controllable"] = 1
    net.load["controllable"] = 0
    net.sgen.in_service[net.sgen.bus==4]=False
    net.sgen.in_service[net.sgen.bus == 6] = False
    net.sgen.in_service[net.sgen.bus == 8] = False
    net.sgen.in_service[net.sgen.bus == 9] = False

    # run OPF
    pp.runopp(net, verbose=False)
    assert net["OPF_converged"]

if __name__ == "__main__":
    pytest.main(["test_opf_cigre.py", "-xs"])

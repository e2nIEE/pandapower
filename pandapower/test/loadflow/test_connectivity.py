# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pytest
import pandapower.networks as pn


def test_connectivity():
    net = pn.create_cigre_network_mv(with_der=False)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")

    isolated_branch =  pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                                             std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                                             name="IsolatedLine")

    pp.runpp(net, verbose=False, check_connectivity=True)

    assert net['converged']




if __name__ == "__main__":
    pytest.main(["test_connectivity.py", "-xs"])
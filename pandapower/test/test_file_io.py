# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
from pandapower.test.toolbox import assert_net_equal, create_test_network
import os
import pytest

def test_file_io():
    net_in = create_test_network()
    pp.file_io.to_pickle(net_in, "testfile.p")
    net_out = pp.file_io.from_pickle("testfile.p")
    assert_net_equal(net_in, net_out)
    os.remove('testfile.p')


if __name__ == "__main__":
    pytest.main(["test_file_io.py", "-xs"])


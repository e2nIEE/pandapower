# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
from pandapower.test.toolbox import assert_net_equal, create_test_network
import os
import pytest

def test_pickle():
    net_in = create_test_network()
    pp.to_pickle(net_in, "testfile.p")
    net_out = pp.from_pickle("testfile.p")
    assert_net_equal(net_in, net_out)
    os.remove('testfile.p')

def test_excel():
    net_in = create_test_network()
    net_in.line_geodata.drop(net_in.line_geodata.index, inplace=True) #TODO: line geodata does not work! 
    pp.to_excel(net_in, "testfile.xlsx")
    net_out = pp.from_excel("testfile.xlsx")
    assert_net_equal(net_in, net_out)
    os.remove('testfile.xlsx')
    
if __name__ == "__main__":
    pytest.main(["test_file_io.py", "-xs"])


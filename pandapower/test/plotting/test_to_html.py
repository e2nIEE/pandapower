# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import os
import pytest

import pandapower.plotting as plot
from pandapower.test.toolbox import assert_net_equal, create_test_network, tempdir, net_in

def test_html(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.html")
    plot.to_html(net_in, filename)

if __name__ == "__main__":
    pytest.main(["test_to_html.py"])

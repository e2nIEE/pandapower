# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os

import pytest

from pandapower.plotting.to_html import to_html
from pandapower.test.helper_functions import create_test_network


def test_html(tmp_path):
    net = create_test_network()
    filename = os.path.abspath(str(tmp_path)) + "testfile.html"
    to_html(net, filename)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

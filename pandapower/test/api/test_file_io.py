# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import os

import pytest

import pandapower as pp
from pandapower.test.toolbox import assert_net_equal, create_test_network


def test_pickle():
    net_in = create_test_network()
    pp.to_pickle(net_in, "testfile.p")
    net_out = pp.from_pickle("testfile.p")
    assert_net_equal(net_in, net_out, True)
    os.remove('testfile.p')


def test_excel():
    net_in = create_test_network()
    net_in.line_geodata.loc[0, "coords"] = [(1.1, 2.2), (3.3, 4.4)]
    net_in.line_geodata.loc[1, "coords"] = [(5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]
    pp.to_excel(net_in, "testfile.xlsx")
    net_out = pp.from_excel("testfile.xlsx")
    assert_net_equal(net_in, net_out)
    os.remove('testfile.xlsx')


def test_json():
    net_in = create_test_network()
    net_in.line_geodata.loc[0, "coords"] = [(1.1, 2.2), (3.3, 4.4)]
    net_in.line_geodata.loc[1, "coords"] = [(5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]
    pp.to_json(net_in, "testfile.json")
    net_out = pp.from_json("testfile.json")
    assert_net_equal(net_in, net_out, reindex=True)
    os.remove('testfile.json')


def test_html():
    net_in = create_test_network()
    pp.to_html(net_in, "testfile.html")
    os.remove('testfile.html')


def test_convert_format():
    folder = os.path.abspath(os.path.dirname(pp.__file__))
    net =  pp.from_pickle(os.path.join(folder, "test", "api", "old_net.p"))
    pp.runpp(net)
    assert net.converged


if __name__ == "__main__":
    pytest.main(["test_file_io.py"])
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import os
import shutil
import pytest
import tempfile

import networks
import pandapower as pp
from pandapower.test.toolbox import assert_net_equal, create_test_network


@pytest.yield_fixture(scope="module")
def tempdir():
    # we create a temporary folder to store all test files and remove it afterwards
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture(scope="module", params=[1])  # TODO
def net_in(request):
    if request.param == 1:
        net = create_test_network()
        net.line_geodata.loc[0, "coords"] = [(1.1, 2.2), (3.3, 4.4)]
        net.line_geodata.loc[11, "coords"] = [(5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]
        return net
    if request.param == 2:
        return networks.case145()


def test_pickle(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.p")
    pp.to_pickle(net_in, filename)
    net_out = pp.from_pickle(filename)
    assert_net_equal(net_in, net_out)


def test_excel(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.xlsx")
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)


def test_json(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    pp.to_json(net_in, filename)
    net_out = pp.from_json(filename)
    assert_net_equal(net_in, net_out)


def test_sqlite(net_in, tempdir):
    from pandapower.sql_io import to_sqlite, from_sqlite
    filename = os.path.join(tempdir, "testfile.db")
    to_sqlite(net_in, filename)
    net_out = from_sqlite(filename)
    assert_net_equal(net_in, net_out)


def test_html(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.html")
    pp.to_html(net_in, filename)


def test_convert_format():  # TODO what is this thing testing ?
    folder = os.path.abspath(os.path.dirname(pp.__file__))
    net =  pp.from_pickle(os.path.join(folder, "test", "api", "old_net.p"))
    pp.runpp(net)
    assert net.converged


if __name__ == "__main__":
    pytest.main(["test_file_io.py"])

# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import os


def _get_networks_path():
    return os.path.abspath(os.path.dirname(pp.networks.__file__))


def case4gs(generator_nodes_as_pv=False):
    """
    Calls the pickle file case4gs.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.

    RETURN:

         **net** - Returns the required ieee network case4gs

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case4gs()
    """
    case4gs = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case4gs.p"))
    return case4gs


def case6ww(generator_nodes_as_pv=False):
    """
    Calls the pickle file case6ww.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.

    RETURN:

         **net** - Returns the required ieee network case6ww

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case6ww()
    """
    case6ww = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case6ww.p"))
    return case6ww


def case9(generator_nodes_as_pv=False):
    """
    Calls the pickle file case9.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network was published in Anderson and Fouad's book 'Power System Control and Stability' for the first time in 1980.

    RETURN:

         **net** - Returns the required ieee network case9

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case9()
    """
    case9 = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case9.p"))
    return case9


def case9Q():
    """
    Calls the pickle file case9Q.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network is highly correlated to case9.

    RETURN:

         **net** - Returns the required ieee network case9Q

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case9Q()
    """
    case9Q = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case9Q.p"))
    return case9Q


def case30():
    """
    Calls the pickle file case30.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Washington <http://www2.ee.washington.edu/research/pstca/pf30/pg_tca30bus.htm>`_ and `Illinois University <http://icseg.iti.illinois.edu/ieee-30-bus-system/>`_.

    RETURN:

         **net** - Returns the required ieee network case30

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case30()
    """
    case30 = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case30.p"))
    return case30


def case30pwl():
    """
    Calls the pickle file case30pwl.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network is highly correlated to case30.

    RETURN:

         **net** - Returns the required ieee network case30pwl

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case30pwl()
    """
    case30pwl = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles",
                                            "case30pwl.p"))
    return case30pwl


def case30Q():
    """
    Calls the pickle file case30Q.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network is highly correlated to case30.

    RETURN:

         **net** - Returns the required ieee network case30Q

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case30Q()
    """
    case30Q = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case30Q.p"))
    return case30Q

if __name__ == "__main__":
    net = pp.networks.case9()

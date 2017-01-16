# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import os


def _get_networks_path():
    return os.path.abspath(os.path.dirname(pp.networks.__file__))


def case4gs():
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


def case6ww():
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


def case9():
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


def case14():
    """
    Calls the pickle file case14.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network was converted from IEEE Common Data Format (ieee14cdf.txt) on 20-Sep-2004 by
    cdf2matp, rev. 1.11, to matpower format and finally converted to pandapower format by
    pandapower.converter.from_ppc. The vn_kv was adapted considering the proposed voltage levels in
    `Washington <http://www2.ee.washington.edu/research/pstca/pf14/ieee14cdf.txt>`_

    RETURN:

         **net** - Returns the required ieee network case14

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case14()
    """
    case14 = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case14.p"))
    return case14


def case24_ieee_rts():
    """
    Calls the pickle file case24_ieee_rts.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University <http://icseg.iti.illinois.edu/ieee-24-bus-system/>`_.

    RETURN:

         **net** - Returns the required ieee network case24

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case24_ieee_rts()
    """
    case24 = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles",
                                         "case24_ieee_rts.p"))
    return case24


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


def case39():
    """
    Calls the pickle file case39.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University <http://icseg.iti.illinois.edu/ieee-39-bus-system/>`_.
    Because the Pypower data origin proposes vn_kv=345 for all nodes the transformers connect node of the same voltage level.

    RETURN:

         **net** - Returns the required ieee network case39

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case39()
    """
    case39 = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case39.p"))
    return case39


def case57(vn_kv_area1=115, vn_kv_area2=500, vn_kv_area3=138, vn_kv_area4=345, vn_kv_area5=230,
           vn_kv_area6=161):
    """
    This function provides the ieee case57 network with the data origin `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University <http://icseg.iti.illinois.edu/ieee-57-bus-system/>`_.
    Because the Pypower data origin proposes no vn_kv some assumption must be made. There are six areas with coinciding voltage level. These are:

    - area 1 with coinciding voltage level comprises node 1-17
    - area 2 with coinciding voltage level comprises node 18-20
    - area 3 with coinciding voltage level comprises node 21-24 + 34-40 + 44-51
    - area 4 with coinciding voltage level comprises node 25 + 30-33
    - area 5 with coinciding voltage level comprises node 41-43 + 56-57
    - area 6 with coinciding voltage level comprises node 52-55 + 26-29

    RETURN:

         **net** - Returns the required ieee network case57

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.case57()
    """
    case57 = pp.from_pickle(os.path.join(_get_networks_path(), "ieee_case_pickles", "case57.p"))
    Idx_area1 = case57.bus[case57.bus.vn_kv == 110].index  # default 115
    Idx_area2 = case57.bus[case57.bus.vn_kv == 120].index  # default 500
    Idx_area3 = case57.bus[case57.bus.vn_kv == 125].index  # default 138
    Idx_area4 = case57.bus[case57.bus.vn_kv == 130].index  # default 345
    Idx_area5 = case57.bus[case57.bus.vn_kv == 140].index  # default 230
    Idx_area6 = case57.bus[case57.bus.vn_kv == 150].index  # default 161
    case57.bus.vn_kv.loc[Idx_area1] = vn_kv_area1
    case57.bus.vn_kv.loc[Idx_area2] = vn_kv_area2
    case57.bus.vn_kv.loc[Idx_area3] = vn_kv_area3
    case57.bus.vn_kv.loc[Idx_area4] = vn_kv_area4
    case57.bus.vn_kv.loc[Idx_area5] = vn_kv_area5
    case57.bus.vn_kv.loc[Idx_area6] = vn_kv_area6
    return case57


if __name__ == "__main__":
    import pandapower.networks as pn
    net = pn.networks.case9()
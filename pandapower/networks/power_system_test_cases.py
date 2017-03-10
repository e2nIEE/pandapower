# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import os


def _get_networks_path():
    return os.path.abspath(os.path.dirname(pp.networks.__file__))


def _get_cases_path():
    return os.path.join(_get_networks_path(), "power_system_test_case_pickles")


def case4gs():
    """
    Calls the pickle file case4gs.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.

    OUTPUT:
         **net** - Returns the required ieee network case4gs

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case4gs()
    """
    case4gs = pp.from_pickle(os.path.join(_get_cases_path(), "case4gs.p"))
    return case4gs


def case6ww():
    """
    Calls the pickle file case6ww.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.

    OUTPUT:
         **net** - Returns the required ieee network case6ww

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case6ww()
    """
    case6ww = pp.from_pickle(os.path.join(_get_cases_path(), "case6ww.p"))
    return case6ww


def case9():
    """
    Calls the pickle file case9.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network was published in Anderson and Fouad's book 'Power System Control and Stability' for the first time in 1980.

    OUTPUT:
         **net** - Returns the required ieee network case9

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case9()
    """
    case9 = pp.from_pickle(os.path.join(_get_cases_path(), "case9.p"))
    return case9


def case14():
    """
    Calls the pickle file case14.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network was converted from IEEE Common Data Format (ieee14cdf.txt) on 20-Sep-2004 by
    cdf2matp, rev. 1.11, to matpower format and finally converted to pandapower format by
    pandapower.converter.from_ppc. The vn_kv was adapted considering the proposed voltage levels in
    `Washington case 14 <http://www2.ee.washington.edu/research/pstca/pf14/ieee14cdf.txt>`_

    OUTPUT:
         **net** - Returns the required ieee network case14

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case14()
    """
    case14 = pp.from_pickle(os.path.join(_get_cases_path(), "case14.p"))
    return case14


def case24_ieee_rts():
    """
    Calls the pickle file case24_ieee_rts.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University case 24 <http://icseg.iti.illinois.edu/ieee-24-bus-system/>`_.

    OUTPUT:
         **net** - Returns the required ieee network case24

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case24_ieee_rts()
    """
    case24 = pp.from_pickle(os.path.join(_get_cases_path(),
                                         "case24_ieee_rts.p"))
    return case24


def case30():
    """
    Calls the pickle file case30.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Washington case 30 <http://www2.ee.washington.edu/research/pstca/pf30/pg_tca30bus.htm>`_ and `Illinois University case 30 <http://icseg.iti.illinois.edu/ieee-30-bus-system/>`_.

    OUTPUT:
         **net** - Returns the required ieee network case30

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case30()
    """
    case30 = pp.from_pickle(os.path.join(_get_cases_path(), "case30.p"))
    return case30


def case33bw():
    """
    Calls the pickle file case33bw.p which data is provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `M. Baran, F. Wu, Network reconfiguration in distribution systems for loss reduction and load balancing <http://ieeexplore.ieee.org/document/25627/>`_ IEEE Transactions on Power Delivery, 1989.

    OUTPUT:
         **net** - Returns the required ieee network case33bw

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case33bw()
    """
    case33bw = pp.from_pickle(os.path.join(_get_cases_path(), "case33bw.p"))
    return case33bw


def case39():
    """
    Calls the pickle file case39.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University case 39 <http://icseg.iti.illinois.edu/ieee-39-bus-system/>`_.
    Because the Pypower data origin proposes vn_kv=345 for all nodes the transformers connect node of the same voltage level.

    OUTPUT:
         **net** - Returns the required ieee network case39

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case39()
    """
    case39 = pp.from_pickle(os.path.join(_get_cases_path(), "case39.p"))
    return case39


def case57(vn_kv_area1=115, vn_kv_area2=500, vn_kv_area3=138, vn_kv_area4=345, vn_kv_area5=230,
           vn_kv_area6=161):
    """
    This function provides the ieee case57 network with the data origin `PYPOWER case 57 <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University case 57 <http://icseg.iti.illinois.edu/ieee-57-bus-system/>`_.
    Because the Pypower data origin proposes no vn_kv some assumption must be made. There are six areas with coinciding voltage level. These are:

    - area 1 with coinciding voltage level comprises node 1-17
    - area 2 with coinciding voltage level comprises node 18-20
    - area 3 with coinciding voltage level comprises node 21-24 + 34-40 + 44-51
    - area 4 with coinciding voltage level comprises node 25 + 30-33
    - area 5 with coinciding voltage level comprises node 41-43 + 56-57
    - area 6 with coinciding voltage level comprises node 52-55 + 26-29

    OUTPUT:
         **net** - Returns the required ieee network case57

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case57()
    """
    case57 = pp.from_pickle(os.path.join(_get_cases_path(), "case57.p"))
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


def case118():
    """
    Calls the pickle file case118.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Washington case 118 <http://www2.ee.washington.edu/research/pstca/pf118/pg_tca118bus.htm>`_ and `Illinois University case 118 <http://icseg.iti.illinois.edu/ieee-118-bus-system/>`_.

    OUTPUT:
         **net** - Returns the required ieee network case118

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case118()
    """
    case118 = pp.from_pickle(os.path.join(_get_cases_path(), "case118.p"))
    return case118


def case300():
    """
    Calls the pickle file case300.p which data origin is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Washington case 300 <http://www2.ee.washington.edu/research/pstca/pf300/pg_tca300bus.htm>`_ and `Illinois University case 300 <http://icseg.iti.illinois.edu/ieee-300-bus-system/>`_.

    OUTPUT:
         **net** - Returns the required ieee network case300

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case300()
    """
    case300 = pp.from_pickle(os.path.join(_get_cases_path(), "case300.p"))
    return case300


def case1354pegase():
    """
    Calls the pickle file case1354pegase.p which data is provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE <https://arxiv.org/abs/1603.01533>`_, 2016.

    OUTPUT:
         **net** - Returns the required ieee network case1354pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case1354pegase()
    """
    case1354pegase = pp.from_pickle(os.path.join(_get_cases_path(),
                                                 "case1354pegase.p"))
    return case1354pegase


def case2869pegase():
    """
    Calls the pickle file case33bw.p which data is provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE <https://arxiv.org/abs/1603.01533>`_, 2016.

    OUTPUT:
         **net** - Returns the required ieee network case2869pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case300()
    """
    case2869pegase = pp.from_pickle(os.path.join(_get_cases_path(),
                                                 "case2869pegase.p"))
    return case2869pegase


def case9241pegase():
    """
    Calls the pickle file case33bw.p which data is provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE <https://arxiv.org/abs/1603.01533>`_, 2016.

    OUTPUT:
         **net** - Returns the required ieee network case9241pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case9241pegase()
    """
    case9241pegase = pp.from_pickle(os.path.join(_get_cases_path(),
                                                 "case9241pegase.p"))
    return case9241pegase


def GBreducednetwork():
    """
    Calls the pickle file GBreducednetwork.p which data is provided by `W. A. Bukhsh, Ken McKinnon, Network data of real transmission networks, April 2013  <http://www.maths.ed.ac.uk/optenergy/NetworkData/reducedGB/>`_.
    This data is a representative model of electricity transmission network in Great Britain (GB). It was originally developed at the University of Strathclyde in 2010.

    OUTPUT:
         **net** - Returns the required ieee network GBreducednetwork

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.GBreducednetwork()
    """
    GBreducednetwork = pp.from_pickle(os.path.join(_get_cases_path(),
                                                   "GBreducednetwork.p"))
    return GBreducednetwork


def GBnetwork():
    """
    Calls the pickle file GBnetwork.p which data is provided by `W. A. Bukhsh, Ken McKinnon, Network data of real transmission networks, April 2013  <http://www.maths.ed.ac.uk/optenergy/NetworkData/fullGB/>`_.
    This data represents detailed model of electricity transmission network of Great Britian (GB). It consists of 2224 nodes, 3207 branches and 394 generators. This data is obtained from publically available data on National grid website. The data was originally pointing out by Manolis Belivanis, University of Strathclyde.

    OUTPUT:
         **net** - Returns the required ieee network GBreducednetwork

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.GBnetwork()
    """
    GBnetwork = pp.from_pickle(os.path.join(_get_cases_path(),
                                            "GBnetwork.p"))
    return GBnetwork


def iceland():
    """
    Calls the pickle file iceland.p which data is provided by `W. A. Bukhsh, Ken McKinnon, Network data of real transmission networks, April 2013  <http://www.maths.ed.ac.uk/optenergy/NetworkData/iceland/>`_.
    This data represents electricity transmission network of Iceland. It consists of 118 nodes, 206 branches and 35 generators. It was originally developed in PSAT format by Patrick McNabb, Durham University in January 2011.

    OUTPUT:
         **net** - Returns the required ieee network iceland

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.iceland()
    """
    iceland = pp.from_pickle(os.path.join(_get_cases_path(),
                                          "iceland.p"))
    return iceland

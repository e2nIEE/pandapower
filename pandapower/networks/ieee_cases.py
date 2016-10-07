# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 14:05:47 2016

@author: smeinecke
"""

import pandapower as pp
import os


def _get_networks_path():
    return os.path.abspath(os.path.dirname(pp.networks.__file__))


def case4gs(generator_nodes_as_pv=False):
    """
    Calls the pickle file case4gs.p which data origin is pypower.

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
    Calls the pickle file case6ww.p which data origin is pypower.

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
    Calls the pickle file case9.p which data origin is pypower.

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
    Calls the pickle file case9Q.p which data origin is pypower.

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
    Calls the pickle file case30.p which data origin is pypower.

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
    Calls the pickle file case30pwl.p which data origin is pypower.

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
    Calls the pickle file case30Q.p which data origin is pypower.

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

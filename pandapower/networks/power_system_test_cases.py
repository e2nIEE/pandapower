# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import os

import pandapower as pp


def _get_networks_path():
    return os.path.abspath(os.path.dirname(pp.networks.__file__))


def _get_cases_path():
    return os.path.join(_get_networks_path(), "power_system_test_case_pickles")


def _change_ref_bus(net, ref_bus_idx, ext_grid_p=0):
    """
    This function changes the current reference bus / buses, declared by net.ext_grid.bus towards \
    the given 'ref_bus_idx'
    If ext_grid_p is a list, it must be in the same order as net.ext_grid.index.
    """
    # cast ref_bus_idx and ext_grid_p as list
    if isinstance(ref_bus_idx, list):
        ref_bus_idx = [ref_bus_idx]
    if isinstance(ext_grid_p, list):
        ext_grid_p = [ext_grid_p]
    for i in ref_bus_idx:
        if i not in net.gen.bus.values and i not in net.ext_grid.bus.values:
            raise ValueError("Index %i is not in net.gen.bus or net.ext_grid.bus." % i)
    # determine indeces of ext_grid and gen connected to ref_bus_idx
    gen_idx = net.gen.loc[net.gen.bus.isin(ref_bus_idx)].index
    ext_grid_idx = net.ext_grid.loc[~net.ext_grid.bus.isin(ref_bus_idx)].index
    # old ext_grid -> gen
    j = 0
    for i in ext_grid_idx:
        ext_grid_data = net.ext_grid.loc[i]
        net.ext_grid.drop(i, inplace=True)
        pp.create_gen(net, ext_grid_data.bus, ext_grid_p[j],
                      vm_pu=ext_grid_data.vm_pu, controllable=True,
                      min_q_kvar=ext_grid_data.min_q_kvar, max_q_kvar=ext_grid_data.max_q_kvar,
                      min_p_kw=ext_grid_data.min_p_kw, max_p_kw=ext_grid_data.max_p_kw)
        j += 1
    # store gen data
    gen_data = net.gen.loc[gen_idx]
    net.gen.drop(net.gen.index[gen_idx], inplace=True)
    # old gen at ref_bus -> ext_grid (and sgen)
    for i in gen_idx:
        gen_i_data = gen_data.loc[i]
        if gen_i_data.bus not in net.ext_grid.bus.values:
            pp.create_ext_grid(net, gen_i_data.bus, vm_pu=gen_i_data.vm_pu, va_degree=0.,
                               min_q_kvar=gen_i_data.min_q_kvar, max_q_kvar=gen_i_data.max_q_kvar,
                               min_p_kw=gen_i_data.min_p_kw, max_p_kw=gen_i_data.max_p_kw)
        else:
            pp.create_sgen(net, gen_i_data.bus, p_kw=gen_i_data.p_kw,
                           min_q_kvar=gen_i_data.min_q_kvar, max_q_kvar=gen_i_data.max_q_kvar,
                           min_p_kw=gen_i_data.min_p_kw, max_p_kw=gen_i_data.max_p_kw)


def case4gs():
    """
    This is the 4 bus example from J. J. Grainger and W. D. Stevenson, Power system analysis. \
    McGraw-Hill, 1994. pp. 337-338. Its data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.

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
    Calls the pickle file case6ww.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_. It represents the 6 bus example from pp. \
    104, 112, 119, 123-124, 549 from A. J. Wood and B. F. Wollenberg, Power generation, operation, \
    and control. John Wiley & Sons, 2012..

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
    Calls the pickle file case9.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network was published in Anderson and Fouad's book 'Power System Control and Stability' \
    for the first time in 1980.

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
    Calls the pickle file case14.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
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
    The IEEE 24-bus reliability test system was developed by the IEEE reliability subcommittee \
    and published in 1979.
    Some more information about this network are given by `Illinois University case 24 \
    <http://icseg.iti.illinois.edu/ieee-24-bus-system/>`_.
    The data origin for this network data is `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.

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
    This function calls the pickle file case30.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Washington case 30 \
    <http://www2.ee.washington.edu/research/pstca/pf30/pg_tca30bus.htm>`_ and `Illinois University case 30 <http://icseg.iti.illinois.edu/ieee-30-bus-system/>`_.

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
    Calls the pickle file case33bw.p which data is provided by \
    `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `M. Baran, F. Wu, Network reconfiguration in distribution systems \
    for loss reduction and load balancing \
    <http://ieeexplore.ieee.org/document/25627/>`_ IEEE Transactions on Power Delivery, 1989.

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
    Calls the pickle file case39.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    This network was published the first time in G. Bills et al., On-line stability analysis \
    study, RP 90-1, E. P. R. I. North American Rockwell Corporation, Edison Electric Institute, \
    Ed. IEEE Press, Oct. 1970,. Some more information about this network are given by \
    `Illinois University case 39 <http://icseg.iti.illinois.edu/ieee-39-bus-system/>`_.
    Because the Pypower data origin proposes vn_kv=345 for all nodes the transformers connect \
    node of the same voltage level.

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
    This function provides the ieee case57 network with the data origin `PYPOWER case 57 \
    <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Illinois University case 57 \
    <http://icseg.iti.illinois.edu/ieee-57-bus-system/>`_.
    Because the Pypower data origin proposes no vn_kv some assumption must be made. There are six \
    areas with coinciding voltage level. These are:

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
    Idx_area1 = case57.bus[case57.bus.vn_kv == 110].index
    Idx_area2 = case57.bus[case57.bus.vn_kv == 120].index
    Idx_area3 = case57.bus[case57.bus.vn_kv == 125].index
    Idx_area4 = case57.bus[case57.bus.vn_kv == 130].index
    Idx_area5 = case57.bus[case57.bus.vn_kv == 140].index
    Idx_area6 = case57.bus[case57.bus.vn_kv == 150].index
    case57.bus.vn_kv.loc[Idx_area1] = vn_kv_area1  # default 115
    case57.bus.vn_kv.loc[Idx_area2] = vn_kv_area2  # default 500
    case57.bus.vn_kv.loc[Idx_area3] = vn_kv_area3  # default 138
    case57.bus.vn_kv.loc[Idx_area4] = vn_kv_area4  # default 345
    case57.bus.vn_kv.loc[Idx_area5] = vn_kv_area5  # default 230
    case57.bus.vn_kv.loc[Idx_area6] = vn_kv_area6  # default 161
    return case57


def case89pegase():
    """
    Calls the pickle file case89pegase.p which data is provided by \
    `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin are the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and \
    PEGASE <https://arxiv.org/abs/1603.01533>`_, 2016 and S. Fliscounakis, P. Panciatici, \
    F. Capitanescu, and L. Wehenkel, Contingency ranking with respect to overloads in very large \
    power systems taking into account uncertainty, preventive, and corrective actions, \
    IEEE Transactions on Power Systems, vol. 28, no. 4, pp. 4909-4917, Nov 2013..

    OUTPUT:
         **net** - Returns the required ieee network case89pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case89pegase()
    """
    case89pegase = pp.from_pickle(os.path.join(_get_cases_path(), "case89pegase.p"))
    return case89pegase


def case118():
    """
    Calls the pickle file case118.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by `Washington case 118 \
    <http://www2.ee.washington.edu/research/pstca/pf118/pg_tca118bus.htm>`_ and \
    `Illinois University case 118 <http://icseg.iti.illinois.edu/ieee-118-bus-system/>`_.

    OUTPUT:
         **net** - Returns the required ieee network case118

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case118()
    """
    case118 = pp.from_pickle(os.path.join(_get_cases_path(), "case118.p"))
    return case118


def case145():
    """
    Calls the pickle file case145.p which data origin is \
    `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    This data is converted by MATPOWER 5.1 using CDF2MPC on 18-May-2016 from 'dd50cdf.txt'.

    OUTPUT:
         **net** - Returns the required ieee network case145

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case145()
    """
    case145 = pp.from_pickle(os.path.join(_get_cases_path(), "case145.p"))
    return case145


def case300():
    """
    Calls the pickle file case300.p which data origin is \
    `PYPOWER <https:/pypi.python.org/pypi/PYPOWER>`_.
    Some more information about this network are given by \
    `Washington case 300 <http://www2.ee.washington.edu/research/pstca/pf300/pg_tca300bus.htm>`_ \
    and `Illinois University case 300 <http://icseg.iti.illinois.edu/ieee-300-bus-system/>`_.

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
    This grid represents a part of the European high voltage transmission network. The data is \
    provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin are the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016 and S. Fliscounakis, P. Panciatici, F. Capitanescu, \
    and L. Wehenkel, Contingency ranking with respect to overloads in very large power systems \
    taking into account uncertainty, preventive, and corrective actions, IEEE Transactions on \
    Power Systems, vol. 28, no. 4, pp. 4909-4917, Nov 2013..

    OUTPUT:
         **net** - Returns the required ieee network case1354pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case1354pegase()
    """
    case1354pegase = pp.from_pickle(os.path.join(_get_cases_path(), "case1354pegase.p"))
    return case1354pegase


def case1888rte(ref_bus_idx=1246):
    """
    This case accurately represents the size and complexity of French very high voltage and high \
    voltage transmission network. The data is provided by `MATPOWER \
    <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016.

    OPTIONAL:

        **ref_bus_idx** - Since the MATPOWER case provides a reference bus without connected \
            generator, because a distributed slack is assumed, to convert the data to pandapower, \
            another bus has been assumed as reference bus. Via 'ref_bus_idx' the User can choose a \
            reference bus, which should have a generator connected to. Please be aware that by \
            changing the reference bus to another bus than the proposed default value, maybe a \
            powerflow does not converge anymore!

    OUTPUT:
         **net** - Returns the required ieee network case1888rte

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case1888rte()
    """
    case1888rte = pp.from_pickle(os.path.join(_get_cases_path(), "case1888rte.p"))
    case1888rte.ext_grid.loc[0, ['min_p_kw',  'max_p_kw',  'min_q_kvar', 'max_q_kvar']] = 2 * \
        case1888rte.ext_grid.loc[0, ['min_p_kw',  'max_p_kw',  'min_q_kvar', 'max_q_kvar']]

    if ref_bus_idx != 1246:  # change reference bus
        _change_ref_bus(case1888rte, ref_bus_idx, ext_grid_p=[89.5e3])
    return case1888rte


def case2848rte(ref_bus_idx=271):
    """
    This case accurately represents the size and complexity of French very high voltage and high \
    voltage transmission network. The data is provided by \
    `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016.

    OPTIONAL:

        **ref_bus_idx** - Since the MATPOWER case provides a reference bus without connected \
            generator, because a distributed slack is assumed, to convert the data to pandapower, \
            another bus has been assumed as reference bus. Via 'ref_bus_idx' the User can choose a \
            reference bus, which should have a generator connected to. Please be aware that by \
            changing the reference bus to another bus than the proposed default value, maybe a \
            powerflow does not converge anymore!

    OUTPUT:
         **net** - Returns the required ieee network case2848rte

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case2848rte()
    """
    case2848rte = pp.from_pickle(os.path.join(_get_cases_path(), "case2848rte.p"))
    if ref_bus_idx != 271:  # change reference bus
        _change_ref_bus(case2848rte, ref_bus_idx, ext_grid_p=[-44.01e3])
    return case2848rte


def case2869pegase():
    """
    This grid represents a part of the European high voltage transmission network. The data is \
    provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin i the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016 and S. Fliscounakis, P. Panciatici, F. Capitanescu, \
    and L. Wehenkel, Contingency ranking with respect to overloads in very large power systems \
    taking into account uncertainty, preventive, and corrective actions, IEEE Transactions on \
    Power Systems, vol. 28, no. 4, pp. 4909-4917, Nov 2013..

    OUTPUT:
         **net** - Returns the required ieee network case2869pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case2869pegase()
    """
    case2869pegase = pp.from_pickle(os.path.join(_get_cases_path(), "case2869pegase.p"))
    return case2869pegase


def case3120sp():
    """
    This case represents the Polish 400, 220 and 110 kV networks during summer 2008 morning peak \
    conditions. The data was provided by Roman Korab <roman.korab@polsl.pl> and to pandapower \
    converted from `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.

    OUTPUT:
         **net** - Returns the required ieee network case3120sp

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case3120sp()
    """
    case3120sp = pp.from_pickle(os.path.join(_get_cases_path(), "case3120sp.p"))
    return case3120sp


def case6470rte(ref_bus_idx=5988):
    """
    This case accurately represents the size and complexity of French very high voltage and high \
    voltage transmission network. The data is provided by \
    `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016.

    OPTIONAL:

        **ref_bus_idx** - Since the MATPOWER case provides a reference bus without connected \
            generator, because a distributed slack is assumed, to convert the data to pandapower, \
            another bus has been assumed as reference bus. Via 'ref_bus_idx' the User can choose a \
            reference bus, which should have a generator connected to. Please be aware that by \
            changing the reference bus to another bus than the proposed default value, maybe a \
            powerflow does not converge anymore!

    OUTPUT:
         **net** - Returns the required ieee network case6470rte

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case6470rte()
    """
    case6470rte = pp.from_pickle(os.path.join(_get_cases_path(), "case6470rte.p"))
    case6470rte.ext_grid.loc[0, ['min_p_kw',  'max_p_kw',  'min_q_kvar', 'max_q_kvar']] = 2 * \
        case6470rte.ext_grid.loc[0, ['min_p_kw',  'max_p_kw',  'min_q_kvar', 'max_q_kvar']]
    if ref_bus_idx != 5988:  # change reference bus
        _change_ref_bus(case6470rte, ref_bus_idx, ext_grid_p=[169.41e3])
    return case6470rte


def case6495rte(ref_bus_idx=[6077, 6161, 6305, 6306, 6307, 6308]):
    """
    This case accurately represents the size and complexity of French very high voltage and high \
    voltage transmission network. The data is provided by \
    `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016.

    OPTIONAL:

        **ref_bus_idx** - Since the MATPOWER case provides a reference bus without connected \
            generator, because a distributed slack is assumed, to convert the data to pandapower, \
            another bus has been assumed as reference bus. Via 'ref_bus_idx' the User can choose a \
            reference bus, which should have a generator connected to. Please be aware that by \
            changing the reference bus to another bus than the proposed default value, maybe a \
            powerflow does not converge anymore!

    OUTPUT:
         **net** - Returns the required ieee network case6495rte

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case6495rte()
    """
    case6495rte = pp.from_pickle(os.path.join(_get_cases_path(), "case6495rte.p"))
    if ref_bus_idx != [6077, 6161, 6305, 6306, 6307, 6308]:  # change reference bus
        _change_ref_bus(case6495rte, ref_bus_idx, ext_grid_p=[-1382.35e3, -2894.13e3, -1498.32e3,
                                                              -1498.32e3, -1493.11e3, -1493.12e3])
    return case6495rte


def case6515rte(ref_bus_idx=6171):
    """
    This case accurately represents the size and complexity of French very high voltage and high \
    voltage transmission network. The data is provided by `MATPOWER \
    <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin is the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016.

    OPTIONAL:

        **ref_bus_idx** - Since the MATPOWER case provides a reference bus without connected \
            generator, because a distributed slack is assumed, to convert the data to pandapower, \
            another bus has been assumed as reference bus. Via 'ref_bus_idx' the User can choose a \
            reference bus, which should have a generator connected to. Please be aware that by \
            changing the reference bus to another bus than the proposed default value, maybe a \
            powerflow does not converge anymore!

    OUTPUT:
         **net** - Returns the required ieee network case6515rte

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case6515rte()
    """
    case6515rte = pp.from_pickle(os.path.join(_get_cases_path(), "case6515rte.p"))
    if ref_bus_idx != 6171:  # change reference bus
        _change_ref_bus(case6515rte, ref_bus_idx, ext_grid_p=-2850.78e3)
    return case6515rte


def case9241pegase():
    """
    This grid represents a part of the European high voltage transmission network. The data is \
    provided by `MATPOWER <http://www.pserc.cornell.edu/matpower/>`_.
    The data origin are the paper `C. Josz, S. Fliscounakis, J. Maenght, P. Panciatici, AC power \
    flow data in MATPOWER and QCQP format: iTesla, RTE snapshots, and PEGASE \
    <https://arxiv.org/abs/1603.01533>`_, 2016 and S. Fliscounakis, P. Panciatici, F. Capitanescu, \
    and L. Wehenkel, Contingency ranking with respect to overloads in very large power systems \
    taking into account uncertainty, preventive, and corrective actions, IEEE Transactions on \
    Power Systems, vol. 28, no. 4, pp. 4909-4917, Nov 2013..

    OUTPUT:
         **net** - Returns the required ieee network case9241pegase

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.case9241pegase()
    """
    case9241pegase = pp.from_pickle(os.path.join(_get_cases_path(), "case9241pegase.p"))
    return case9241pegase


def GBreducednetwork():
    """
    Calls the pickle file GBreducednetwork.p which data is provided by `W. A. Bukhsh, Ken \
    McKinnon, Network data of real transmission networks, April 2013  \
    <http://www.maths.ed.ac.uk/optenergy/NetworkData/reducedGB/>`_.
    This data is a representative model of electricity transmission network in Great Britain (GB). \
    It was originally developed at the University of Strathclyde in 2010.

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
    Calls the pickle file GBnetwork.p which data is provided by `W. A. Bukhsh, Ken McKinnon, \
    Network data of real transmission networks, April 2013  \
    <http://www.maths.ed.ac.uk/optenergy/NetworkData/fullGB/>`_.
    This data represents detailed model of electricity transmission network of Great Britian (GB). \
    It consists of 2224 nodes, 3207 branches and 394 generators. This data is obtained from \
    publically available data on National grid website. The data was originally pointing out by \
    Manolis Belivanis, University of Strathclyde.

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
    Calls the pickle file iceland.p which data is provided by `W. A. Bukhsh, Ken McKinnon, Network \
    data of real transmission networks, April 2013  \
    <http://www.maths.ed.ac.uk/optenergy/NetworkData/iceland/>`_.
    This data represents electricity transmission network of Iceland. It consists of 118 nodes, \
    206 branches and 35 generators. It was originally developed in PSAT format by Patrick McNabb, \
    Durham University in January 2011.

    OUTPUT:
         **net** - Returns the required ieee network iceland

    EXAMPLE:
         import pandapower.networks as pn

         net = pn.iceland()
    """
    iceland = pp.from_pickle(os.path.join(_get_cases_path(),
                                          "iceland.p"))
    return iceland

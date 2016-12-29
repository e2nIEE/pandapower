# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy.core.numeric as ncn
import numpy as np

from pypower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_STATUS, GEN_BUS, PG, VG, QG
from pypower.idx_bus import PV, REF, VA, VM, BUS_TYPE, NONE, VMAX, VMIN, PQ
from numpy import array,  zeros, isnan


def _build_gen_ppc(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    # get in service elements
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']

    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    xw_end = gen_end + len(net["xward"])

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.

    # initialize generator matrix
    ppc["gen"] = np.zeros(shape=(xw_end, 21), dtype=float)
    ppc["gen"][:] = np.array([0, 0, 0, q_lim_default, -q_lim_default, 1.,
                              1., 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # add ext grid / slack data
    ppc["gen"][:eg_end, GEN_BUS] = bus_lookup[eg_is["bus"].values]
    ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    eg_buses = bus_lookup[eg_is["bus"].values]
    if calculate_voltage_angles:
        ppc["bus"][eg_buses, VA] = eg_is["va_degree"].values
    ppc["bus"][eg_buses, BUS_TYPE] = REF

    # add generator / pv data
    if gen_end > eg_end:
        ppc["gen"][eg_end:gen_end, GEN_BUS] = bus_lookup[gen_is["bus"].values]
        ppc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        ppc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[gen_is["bus"].values]
        ppc["bus"][gen_buses, BUS_TYPE] = PV
        ppc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

        if enforce_q_lims:
            ppc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3
            ppc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3

            qmax = ppc["gen"][eg_end:gen_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=np.isnan(qmax))
            ppc["gen"][eg_end:gen_end, [QMIN]] = qmax

            qmin = ppc["gen"][eg_end:gen_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=np.isnan(qmin))
            ppc["gen"][eg_end:gen_end, [QMAX]] = qmin

    # add extended ward pv node data
    if xw_end > gen_end:
        xw = net["xward"]
        xw_is = is_elems['xward']
        ppc["gen"][gen_end:xw_end, GEN_BUS] = bus_lookup[xw["ad_bus"].values]
        ppc["gen"][gen_end:xw_end, VG] = xw["vm_pu"].values
        ppc["gen"][gen_end:xw_end, GEN_STATUS] = xw_is
        ppc["gen"][gen_end:xw_end, QMIN] = -q_lim_default
        ppc["gen"][gen_end:xw_end, QMAX] = q_lim_default

        xward_buses = bus_lookup[net["xward"]["ad_bus"].values]
        ppc["bus"][xward_buses[xw_is], BUS_TYPE] = PV
        ppc["bus"][xward_buses[~xw_is], BUS_TYPE] = NONE
        ppc["bus"][xward_buses, VM] = net["xward"]["vm_pu"].values


def _update_gen_ppc(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles):
    '''
    Takes the ppc network and updates the gen values from the values in net.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    # get in service elements
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']

    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    xw_end = gen_end + len(net["xward"])

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.

    # add ext grid / slack data
    ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    if calculate_voltage_angles:
        eg_buses = bus_lookup[eg_is["bus"].values]
        ppc["bus"][eg_buses, VA] = eg_is["va_degree"].values

    # add generator / pv data
    if gen_end > eg_end:
        ppc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        ppc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[gen_is["bus"].values]
        ppc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

        if enforce_q_lims:
            ppc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3
            ppc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3

            qmax = ppc["gen"][eg_end:gen_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=np.isnan(qmax))
            ppc["gen"][eg_end:gen_end, [QMIN]] = qmax

            qmin = ppc["gen"][eg_end:gen_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=np.isnan(qmin))
            ppc["gen"][eg_end:gen_end, [QMAX]] = qmin

    # add extended ward pv node data
    if xw_end > gen_end:
        xw = net["xward"]
        xw_is = is_elems["xward"]
        ppc["gen"][gen_end:xw_end, VG] = xw["vm_pu"].values
        ppc["gen"][gen_end:xw_end, GEN_STATUS] = xw_is
        ppc["gen"][gen_end:xw_end, QMIN] = -q_lim_default
        ppc["gen"][gen_end:xw_end, QMAX] = q_lim_default

        xward_buses = bus_lookup[net["xward"]["ad_bus"].values]
        ppc["bus"][xward_buses[xw_is], BUS_TYPE] = PV
        ppc["bus"][xward_buses[~xw_is], BUS_TYPE] = NONE
        ppc["bus"][xward_buses, VM] = net["xward"]["vm_pu"].values
        
def _build_gen_opf(net, ppc, gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is,
                   delta=1e-10):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    sg_end = gen_end + len(sg_is)

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
    p_lim_default = 1e9

    # initialize generator matrix
    ppc["gen"] = zeros(shape=(sg_end, 21), dtype=float)
    ppc["gen"][:] = array([0, 0, 0, q_lim_default, -q_lim_default, 1., 1., 1, p_lim_default,
                              -p_lim_default, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # add sgens first so pv bus types won't be overwritten
    if sg_end > gen_end:
        ppc["gen"][gen_end:sg_end, GEN_BUS] = bus_lookup[sg_is["bus"].values]
        ppc["gen"][gen_end:sg_end, PG] = - sg_is["p_kw"].values * 1e-3 * sg_is["scaling"].values
        ppc["gen"][gen_end:sg_end, QG] = sg_is["q_kvar"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[sg_is["bus"].values]
        ppc["bus"][gen_buses, BUS_TYPE] = PQ

        # set constraints for PV generators
        if "min_q_kvar" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, QMAX] = - (sg_is["min_q_kvar"].values * 1e-3 - delta)
            qmax = ppc["gen"][gen_end:sg_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=isnan(qmax))
            ppc["gen"][gen_end:sg_end, [QMIN]] = qmax

        if "max_q_kvar" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, QMIN] = - (sg_is["max_q_kvar"].values * 1e-3 + delta)
            qmin = ppc["gen"][gen_end:sg_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=isnan(qmin))
            ppc["gen"][gen_end:sg_end, [QMAX]] = qmin - 1e-10

        if "min_p_kw" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, PMIN] = - (sg_is["min_p_kw"].values * 1e-3 - delta)
            pmax = ppc["gen"][gen_end:sg_end, [PMIN]]
            ncn.copyto(pmax, -p_lim_default, where=isnan(pmax))
            ppc["gen"][gen_end:sg_end, [PMIN]] = pmax

        if "max_p_kw" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, PMAX] = - (sg_is["max_p_kw"].values * 1e-3 + delta)
            min_p_kw = ppc["gen"][gen_end:sg_end, [PMAX]]
            ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
            ppc["gen"][gen_end:sg_end, [PMAX]] = min_p_kw

    # add ext grid / slack data
    ppc["gen"][:eg_end, GEN_BUS] = bus_lookup[eg_is["bus"].values]
    ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    eg_buses = bus_lookup[eg_is["bus"].values]
    if calculate_voltage_angles:
        ppc["bus"][eg_buses, VA] = eg_is["va_degree"].values
    ppc["bus"][eg_buses, BUS_TYPE] = REF
    ppc["bus"][eg_buses, VM] = eg_is["vm_pu"].values

    # REF busses don't have flexible voltages by definition:
    ppc["bus"][eg_buses, VMAX] = ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, VM]
    ppc["bus"][eg_buses, VMIN] = ppc["bus"][ppc["bus"][:, BUS_TYPE] == REF, VM]

    # add generator / pv data
    if gen_end > eg_end:
        ppc["gen"][eg_end:gen_end, GEN_BUS] = bus_lookup[gen_is["bus"].values]
        ppc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        ppc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = bus_lookup[gen_is["bus"].values]
        ppc["bus"][gen_buses, BUS_TYPE] = PV
        ppc["bus"][gen_buses, VM] = gen_is["vm_pu"].values

        # set constraints for PV generators
        ppc["gen"][eg_end:gen_end, QMIN] = - (gen_is["max_q_kvar"].values * 1e-3 + delta)
        ppc["gen"][eg_end:gen_end, QMAX] = - (gen_is["min_q_kvar"].values * 1e-3 - delta)
        ppc["gen"][eg_end:gen_end, PMIN] = - (gen_is["min_p_kw"].values * 1e-3 - delta)
        ppc["gen"][eg_end:gen_end, PMAX] = - (gen_is["max_p_kw"].values * 1e-3 + delta)

        qmin = ppc["gen"][eg_end:gen_end, [QMIN]]
        ncn.copyto(qmin, -q_lim_default, where=isnan(qmin))
        ppc["gen"][eg_end:gen_end, [QMIN]] = qmin

        qmax = ppc["gen"][eg_end:gen_end, [QMAX]]
        ncn.copyto(qmax, q_lim_default, where=isnan(qmax))
        ppc["gen"][eg_end:gen_end, [QMAX]] = qmax

        min_p_kw = ppc["gen"][eg_end:gen_end, [PMIN]]
        ncn.copyto(min_p_kw, -p_lim_default, where=isnan(min_p_kw))
        ppc["gen"][eg_end:gen_end, [PMIN]] = min_p_kw

        pmax = ppc["gen"][eg_end:gen_end, [PMAX]]
        ncn.copyto(pmax, p_lim_default, where=isnan(pmax))
        ppc["gen"][eg_end:gen_end, [PMAX]] = pmax

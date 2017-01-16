# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy.core.numeric as ncn
import numpy as np

from pandas import DataFrame
from pypower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_STATUS, GEN_BUS, PG, VG, QG
from pypower.idx_bus import PV, REF, VA, VM, BUS_TYPE, NONE, VMAX, VMIN, PQ
from numpy import array,  zeros, isnan

from pandapower.create import create_gen

def _build_gen_ppc(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles,
                   copy_constraints_to_ppc=False, opf=False):
    '''
    wrapper function to call either the PF or the OPF version
    '''

    if opf:
        _build_gen_opf(net, ppc, is_elems, bus_lookup, calculate_voltage_angles, delta=1e-10)
    else:
        _build_gen_pf(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles,
                   copy_constraints_to_ppc)


def _build_gen_pf(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles,
                   copy_constraints_to_ppc):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    if len(net.dcline) > 0:
        add_dcline_gens(net, copy_constraints_to_ppc)
        is_elems["gen"] = net.gen[net.gen.in_service==True]
    # get in service elements
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']

    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    xw_end = gen_end + len(net["xward"])

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
    p_lim_default = 1e9

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

        if enforce_q_lims or copy_constraints_to_ppc:
            _copy_q_limits_to_ppc(ppc, eg_end, gen_end, gen_is)
            _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default)

        if copy_constraints_to_ppc:
            _copy_p_limits_to_ppc(ppc, eg_end, gen_end, gen_is)
            _replace_nans_with_default_p_limits_in_ppc(ppc, eg_end, gen_end, p_lim_default)

    # add extended ward pv node data
    if xw_end > gen_end:
        _copy_xward_values_to_ppc(net, ppc, is_elems, gen_end, xw_end, bus_lookup, q_lim_default)


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
            _copy_q_limits_to_ppc(ppc, eg_end, gen_end, gen_is)
            _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default)

    # add extended ward pv node data
    if xw_end > gen_end:
        _copy_xward_values_to_ppc(net, ppc, is_elems, gen_end, xw_end, bus_lookup, q_lim_default, update_lookup=False)


def _build_gen_opf(net, ppc, is_elems, bus_lookup, calculate_voltage_angles, delta=1e-10):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    if len(net.dcline) > 0:
        add_dcline_gens(net, copy_constraints_to_ppc=True)
        ppc["dcline"] = net.dcline[["loss_kw", "loss_percent"]].values
        is_elems["gen"] = net.gen[net.gen.in_service==True]
    # get in service elements
    eg_is = is_elems['ext_grid']
    gen_is = is_elems['gen']
    sg_is = net.sgen[(net.sgen.in_service & net.sgen.controllable) == True] \
        if "controllable" in net.sgen.columns else DataFrame()

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
            max_q_kvar = ppc["gen"][gen_end:sg_end, [QMIN]]
            ncn.copyto(max_q_kvar, -q_lim_default, where=isnan(max_q_kvar))
            ppc["gen"][gen_end:sg_end, [QMIN]] = max_q_kvar

        if "max_q_kvar" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, QMIN] = - (sg_is["max_q_kvar"].values * 1e-3 + delta)
            min_q_kvar = ppc["gen"][gen_end:sg_end, [QMAX]]
            ncn.copyto(min_q_kvar, q_lim_default, where=isnan(min_q_kvar))
            ppc["gen"][gen_end:sg_end, [QMAX]] = min_q_kvar - 1e-10

        if "max_p_kw" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, PMIN] = - (sg_is["max_p_kw"].values * 1e-3 - delta)
            max_p_kw = ppc["gen"][gen_end:sg_end, [PMIN]]
            ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
            ppc["gen"][gen_end:sg_end, [PMIN]] = max_p_kw

        if "min_p_kw" in sg_is.columns:
            ppc["gen"][gen_end:sg_end, PMAX] = - (sg_is["min_p_kw"].values * 1e-3 + delta)
            min_p_kw = ppc["gen"][gen_end:sg_end, [PMAX]]
            ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
            ppc["gen"][gen_end:sg_end, [PMAX]] = min_p_kw

    # add ext grid / slack data
    ppc["gen"][:eg_end, GEN_BUS] = bus_lookup[eg_is["bus"].values]
    ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values
    if "max_p_kw" in eg_is.columns:
        ppc["gen"][:eg_end, PMIN] = - (eg_is["max_p_kw"].values * 1e-3 - delta)
        max_p_kw = ppc["gen"][:eg_end, [PMIN]]
        ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
        ppc["gen"][:eg_end, [PMIN]] = max_p_kw

    if "min_p_kw" in eg_is.columns:
        ppc["gen"][:eg_end, PMAX] = - (eg_is["min_p_kw"].values * 1e-3 + delta)
        min_p_kw = ppc["gen"][:eg_end, [PMAX]]
        ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
        ppc["gen"][:eg_end, [PMAX]] = min_p_kw

    if "min_q_kvar" in eg_is.columns:
        ppc["gen"][:eg_end, QMAX] = - (eg_is["min_q_kvar"].values * 1e-3 - delta)
        max_q_kvar = ppc["gen"][:eg_end, [QMIN]]
        ncn.copyto(max_q_kvar, -q_lim_default, where=isnan(max_q_kvar))
        ppc["gen"][:eg_end, [QMIN]] = max_q_kvar

    if "max_q_kvar" in eg_is.columns:
        ppc["gen"][:eg_end, QMIN] = - (eg_is["max_q_kvar"].values * 1e-3 + delta)
        min_q_kvar = ppc["gen"][:eg_end, [QMAX]]
        ncn.copyto(min_q_kvar, q_lim_default, where=isnan(min_q_kvar))
        ppc["gen"][:eg_end, [QMAX]] = min_q_kvar - 1e-10

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
        _copy_q_limits_to_ppc(ppc, eg_end, gen_end, gen_is, delta)
        _copy_p_limits_to_ppc(ppc, eg_end, gen_end, gen_is, delta)

        _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default)
        _replace_nans_with_default_p_limits_in_ppc(ppc, eg_end, gen_end, p_lim_default)


def _copy_q_limits_to_ppc(ppc, eg_end, gen_end, gen_is, delta=0.0):
    # Note: Pypower has generator reference system, pandapower uses load reference system (max <-> min)
    ppc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3 - delta
    ppc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3 + delta


def _copy_p_limits_to_ppc(ppc, eg_end, gen_end, gen_is, delta=0.0):
    ppc["gen"][eg_end:gen_end, PMIN] = -gen_is["max_p_kw"].values * 1e-3 + delta
    ppc["gen"][eg_end:gen_end, PMAX] = -gen_is["min_p_kw"].values * 1e-3 - delta


def _replace_nans_with_default_q_limits_in_ppc(ppc, eg_end, gen_end, q_lim_default):
    # Note: Pypower has generator reference system, pandapower uses load reference system (max <-> min)
    max_q_kvar = ppc["gen"][eg_end:gen_end, [QMIN]]
    ncn.copyto(max_q_kvar, -q_lim_default, where=np.isnan(max_q_kvar))
    ppc["gen"][eg_end:gen_end, [QMIN]] = max_q_kvar

    min_q_kvar = ppc["gen"][eg_end:gen_end, [QMAX]]
    ncn.copyto(min_q_kvar, q_lim_default, where=np.isnan(min_q_kvar))
    ppc["gen"][eg_end:gen_end, [QMAX]] = min_q_kvar


def _replace_nans_with_default_p_limits_in_ppc(ppc, eg_end, gen_end, p_lim_default):
    # Note: Pypower has generator reference system, pandapower uses load reference system (max <-> min)
    max_p_kw = ppc["gen"][eg_end:gen_end, [PMIN]]
    ncn.copyto(max_p_kw, -p_lim_default, where=isnan(max_p_kw))
    ppc["gen"][eg_end:gen_end, [PMIN]] = max_p_kw

    min_p_kw = ppc["gen"][eg_end:gen_end, [PMAX]]
    ncn.copyto(min_p_kw, p_lim_default, where=isnan(min_p_kw))
    ppc["gen"][eg_end:gen_end, [PMAX]] = min_p_kw


def _copy_xward_values_to_ppc(net, ppc, is_elems, gen_end, xw_end, bus_lookup, q_lim_default, update_lookup=True):
    xw = net["xward"]
    xw_is = is_elems['xward']
    if update_lookup:
        ppc["gen"][gen_end:xw_end, GEN_BUS] = bus_lookup[xw["ad_bus"].values]
    ppc["gen"][gen_end:xw_end, VG] = xw["vm_pu"].values
    ppc["gen"][gen_end:xw_end, GEN_STATUS] = xw_is
    ppc["gen"][gen_end:xw_end, QMIN] = -q_lim_default
    ppc["gen"][gen_end:xw_end, QMAX] = q_lim_default

    xward_buses = bus_lookup[net["xward"]["ad_bus"].values]
    ppc["bus"][xward_buses[xw_is], BUS_TYPE] = PV
    ppc["bus"][xward_buses[~xw_is], BUS_TYPE] = NONE
    ppc["bus"][xward_buses, VM] = net["xward"]["vm_pu"].values

def add_dcline_gens(net, copy_constraints_to_ppc):
    for _, dctab in net.dcline.iterrows():
        pfrom = dctab.p_kw
        pto = - (pfrom* (1 - dctab.loss_percent / 100) - dctab.loss_kw)
        pmax = dctab.max_p_kw if copy_constraints_to_ppc else np.nan
        create_gen(net, bus=dctab.to_bus, p_kw=pto, vm_pu=dctab.vm_to_pu, 
                   min_p_kw=-pmax, max_p_kw=0., 
                   max_q_kvar=dctab.max_q_to_kvar, min_q_kvar=dctab.min_q_to_kvar,
                   in_service=dctab.in_service, cost_per_kw=0.)
        create_gen(net, bus=dctab.from_bus, p_kw=pfrom, vm_pu=dctab.vm_from_pu, 
                   min_p_kw=0, max_p_kw=pmax, 
                   max_q_kvar=dctab.max_q_from_kvar, min_q_kvar=dctab.min_q_from_kvar, 
                   in_service=dctab.in_service, cost_per_kw=-dctab.cost_per_kw)
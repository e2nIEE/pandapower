# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy.core.numeric as ncn
import numpy as np

from pypower.idx_gen import QMIN, QMAX, GEN_STATUS, GEN_BUS, PG, VG
from pypower.idx_bus import PV, REF, VA, VM, BUS_TYPE, NONE

from pandapower.auxiliary import get_indices


def _build_gen_ppc(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles):
    '''
    Takes the empty ppc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **ppc** - The PYPOWER format network to fill in values
    '''
    # get in service elements
    eg_is = is_elems['eg']
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
    ppc["gen"][:eg_end, GEN_BUS] = get_indices(eg_is["bus"].values, bus_lookup)
    ppc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    ppc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid buses
    eg_buses = get_indices(eg_is["bus"].values, bus_lookup)
    if calculate_voltage_angles:
        ppc["bus"][eg_buses, VA] = eg_is["va_degree"].values
    ppc["bus"][eg_buses, BUS_TYPE] = REF

    # add generator / pv data
    if gen_end > eg_end:
        ppc["gen"][eg_end:gen_end, GEN_BUS] = get_indices(gen_is["bus"].values, bus_lookup)
        ppc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        ppc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator buses
        gen_buses = get_indices(gen_is["bus"].values, bus_lookup)
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
        bus_is = is_elems['bus']
        xw_is = np.in1d(xw.bus.values, bus_is.index) \
            & xw.in_service.values.astype(bool)
        ppc["gen"][gen_end:xw_end, GEN_BUS] = get_indices(xw["ad_bus"].values, bus_lookup)
        ppc["gen"][gen_end:xw_end, VG] = xw["vm_pu"].values
        ppc["gen"][gen_end:xw_end, GEN_STATUS] = xw_is
        ppc["gen"][gen_end:xw_end, QMIN] = -q_lim_default
        ppc["gen"][gen_end:xw_end, QMAX] = q_lim_default

        xward_buses = get_indices(net["xward"]["ad_bus"].values, bus_lookup)
        ppc["bus"][xward_buses[xw_is], BUS_TYPE] = PV
        ppc["bus"][xward_buses[~xw_is], BUS_TYPE] = NONE
        ppc["bus"][xward_buses, VM] = net["xward"]["vm_pu"].values

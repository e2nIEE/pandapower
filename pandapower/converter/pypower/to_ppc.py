# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandapower.run import _pd2ppc, _select_is_elements
import pplog

logger = pplog.getLogger(__name__)


def pp2ppc(net):
    """
    This function converts a pandapower net to a pypower case files.

    INPUT:

        **net** - The pandapower net.

    OPTIONAL:

        **...** - ...

    OUTPUT:

        **ppc**

    EXAMPLE:

        import pandapower.converter as pc

        import pandapower.networks as pn

        net = pn.case9()

        ppc = pc.pp2ppc(net)

    """
    is_elems = _select_is_elements(net, dict(is_elems=False, ppc=False, Ybus=False))
    ppc, ppci, bus_lookup = _pd2ppc(net, is_elems, trafo_model="pi", copy_voltage_boundaries=True)

    return ppc

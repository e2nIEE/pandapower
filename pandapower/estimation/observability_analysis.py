# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.stats import chi2
from scipy.sparse import csr_matrix
import scipy

from pandapower.estimation.util import set_bb_switch_impedance, reset_bb_switch_impedance
from pandapower.estimation.ppc_conversion import pp2eppci, _initialize_voltage
from pandapower.estimation.results import eppci2pp
from pandapower.estimation.algorithm.base import (WLSAlgorithm, 
                                                  WLSZeroInjectionConstraintsAlgorithm,
                                                  IRWLSAlgorithm)
from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.algorithm.lp import LPAlgorithm
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra, BaseAlgebraDecoupled


try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)

def observability_analysis(net, zero_injection='auto', fuse_buses_with_bb_switch='all'):
    # change the configuration of the pp net to avoid auto fusing of buses connected
    # through bb switch with elements on each bus if this feature enabled
    bus_to_be_fused = None
    if fuse_buses_with_bb_switch != 'all' and not net.switch.empty:
        if isinstance(fuse_buses_with_bb_switch, str):
            raise UserWarning("fuse_buses_with_bb_switch parameter is not correctly initialized")
        elif hasattr(fuse_buses_with_bb_switch, '__iter__'):
            bus_to_be_fused = fuse_buses_with_bb_switch    
        set_bb_switch_impedance(net, bus_to_be_fused)

    net, ppc, eppci = pp2eppci(net, zero_injection=zero_injection,
                               v_start=None, delta_start=None, calculate_voltage_angles=True)
    
    
    r_inv = csr_matrix(np.diagflat(1/eppci.r_cov ** 2))
    E = eppci.E

    sem = BaseAlgebraDecoupled(eppci)
    # jacobian matrix H
    H = csr_matrix(sem.create_hx_jacobian(E))
    Ha = H.toarray()

    # gain matrix G_m
    # G_m = H^t * R^-1 * H
    G_m = H.T * H
    Ga = G_m.toarray()
    

    if np.isfinite(np.linalg.cond(Ga)):
        B = np.linalg.inv(Ga)
    else:
        raise np.linalg.LinAlgError
    
#    scipy.sparse.linalg.inv(G_m)
#    np.linalg.inv(Ga)
   
    print(124)


if __name__ == "__main__":
    from pandapower.estimation.util import add_virtual_meas_from_loadflow
    import pandapower.networks as nw
    import pandapower as pp
    from pandapower.estimation import estimate

    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    net.measurement = net.measurement.iloc[:35, :]
    observability_analysis(net)
    estimate(net)
    
    



# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from numpy import roots, conj, r_
from numpy.core.umath import exp


def _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6):
    if npv:
        dVa[pv] = dx[j1:j2]
    if npq:
        dVa[pq] = dx[j3:j4]
        dVm[pq] = dx[j5:j6]
    dV = dVm * exp(1j * dVa)

    iwa_multiplier = _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pq, pv)

    Vm += iwa_multiplier * dVm
    Va += iwa_multiplier * dVa
    return Vm, Va


def _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pq, pv):
    """
    Calculates the iwamato multiplier to increase convergence
    """

    c0=-F                               # c0 = ys-y(x)= -F
    c1=-J * dx                          # c1 = -Jdx
    c2=-_evaluate_Yx(Ybus, dV, pv, pq)  # c2 = -y(dx)

    g0 = c0.dot(c1)
    g1 = c1.dot(c1) + 2 * c0.dot(c2)
    g2 = 3.0 * c1.dot(c2)
    g3 = 2.0 * c2.dot(c2)

    np_roots = roots([g3, g2, g1, g0])[2].real
    print("iwamoto muliplier:", np_roots)
    #print(g0,g1,g2,g3,np_roots)
    #print(g0+ g1*a + g2*a**2 + g3*a**3)

    return np_roots


def _evaluate_Yx(Ybus, V, pv, pq):
    ## evaluate y(x)
    Yx = V * conj(Ybus * V)
    F = r_[Yx[pv].real,
           Yx[pq].real,
           Yx[pq].imag]
    return F

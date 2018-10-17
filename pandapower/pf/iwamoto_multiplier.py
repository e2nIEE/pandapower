from numpy import roots, conj, r_
from numpy.core.umath import exp


def _iwamoto_step(Ybus, J, F, dx, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4,
                  j5, j6, Sbus):
    if npv:
        dVa[pv] = dx[j1:j2]
    if npq:
        dVa[pq] = dx[j3:j4]
        dVm[pq] = dx[j5:j6]
    dV = dVm * exp(1j * dVa)

    iwa_multiplier = _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pvpq, pq, createJ, pvpq_lookup, npv, npq,
                                                 numba, Sbus, pv)

    Vm += iwa_multiplier * dVm
    Va += iwa_multiplier * dVa
    return Vm, Va


def _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba, Sbus, pv):
    """
    Calculates the iwamato multiplier to increase convergence
    """

    c0=F
    c1=J * dx
    c2=_evaluate_Fx2(Ybus, dV, Sbus, pv, pq)

    g0 = c0.dot(c1)
    g1 = c1.dot(c1) + 2 * c0.dot(c2)
    g2 = 3.0 * c1.dot(c2)
    g3 = 2.0 * c2.dot(c2)

    np_roots = roots([g3, g2, g1, g0])[2].real
    print("iwamoto muliplier:", np_roots)
    #print(g0,g1,g2,g3,np_roots)
    #print(g0+ g1*a + g2*a**2 + g3*a**3)

    return np_roots


def _evaluate_Fx2(Ybus, V, Sbus, pv, pq):
    ## evalute F(x)
    mis = V * conj(Ybus * V)
    F = r_[mis[pv].real,
           mis[pq].real,
           mis[pq].imag]
    return F

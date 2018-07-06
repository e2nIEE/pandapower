from numpy import roots
from numpy.core.umath import exp

from pandapower.pf.create_jacobian import create_jacobian_matrix


def _iwamoto_step(Ybus, J, F, dx, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4,
                  j5, j6):
    if npv:
        dVa[pv] = dx[j1:j2]
    if npq:
        dVa[pq] = dx[j3:j4]
        dVm[pq] = dx[j5:j6]
    dV = dVm * exp(1j * dVa)
    if not (dV == 0.0).any():
        iwa_multiplier = _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pvpq, pq, createJ, pvpq_lookup, npv, npq,
                                                 numba)
    else:
        iwa_multiplier = 1.0
    Vm += iwa_multiplier * dVm
    Va += iwa_multiplier * dVa
    return Vm, Va


def _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba):
    """
    Calculates the iwamato multiplier to increase convergence
    """
    J_iwa = create_jacobian_matrix(Ybus, dV, pvpq, pq, createJ, pvpq_lookup, npv, npq, numba)
    J_dx = J * dx
    J_iwa_dx = 0.5 * dx * J_iwa * dx

    g0 = -F.dot(J_dx)
    g1 = J_dx.dot(J_dx) + 2 * F.dot(J_iwa_dx)
    g2 = -3.0 * J_dx.dot(J_iwa_dx)
    g3 = 2.0 * J_iwa_dx.dot(J_iwa_dx)

    np_roots = roots([g3, g2, g1, g0])[2].real
    return np_roots

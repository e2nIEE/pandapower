# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from numpy import roots, conj, r_, exp, all, isfinite, abs, argmin


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


def _select_multiplier(g3, g2, g1, g0):
    """Pick the physical Iwamoto step multiplier from the cubic

        g3*a^3 + g2*a^2 + g1*a + g0 = 0.

    The cubic minimises ||F(x + a*dx)||^2 along the Newton direction, so the root we want
    is the one closest to a = 1 (the undamped Newton step), which is what the method
    falls back to when no damping is needed.

    Selection rules, in order:
      * discard roots with a non-negligible imaginary part (relative to the root scale),
      * discard non-positive roots -- a <= 0 would step backwards along the Newton
        direction or not at all,
      * among the rest take the one nearest 1.0.
    If nothing survives, return 1.0 (plain Newton), which is always a safe fallback.

    This replaces an unguarded ``roots([...])[2].real``. numpy orders roots by DESCENDING
    magnitude, so index 2 is the smallest-magnitude root of a cubic -- usually, but not
    always, the physical one. It breaks when:
      * the cubic has three real roots and the physical one is not the smallest
        (e.g. roots 0.22 / 1.0 / 2.28 -> [2] picks 0.22 instead of 1.0),
      * the smallest-magnitude root is one of a complex pair, in which case ``.real``
        silently returns a meaningless number (a near-zero multiplier stalls Newton),
      * g3 (or more) vanishes near convergence, so ``roots`` returns FEWER than three
        values and index 2 raises IndexError.
    """
    coeffs = [g3, g2, g1, g0]
    if not all(isfinite(coeffs)):
        return 1.0

    # np.roots strips leading zeros itself, but an all-zero cubic (exactly converged:
    # dx == 0, so every g vanishes) yields an empty array -- handle it before indexing.
    r = roots(coeffs)
    if r.size == 0:
        return 1.0

    scale = max(1.0, float(abs(r).max()))
    real = r[abs(r.imag) <= 1e-9 * scale].real
    real = real[real > 1e-12]
    if real.size == 0:
        return 1.0

    return float(real[argmin(abs(real - 1.0))])


def _get_iwamoto_multiplier(Ybus, J, F, dV, dx, pq, pv):
    """
    Calculates the iwamoto multiplier to increase convergence
    """

    c0=-F                               # c0 = ys-y(x)= -F
    c1=-J * dx                          # c1 = -Jdx
    c2=-_evaluate_Yx(Ybus, dV, pv, pq)  # c2 = -y(dx)

    g0 = c0.dot(c1)
    g1 = c1.dot(c1) + 2 * c0.dot(c2)
    g2 = 3.0 * c1.dot(c2)
    g3 = 2.0 * c2.dot(c2)

    return _select_multiplier(g3, g2, g1, g0)


def _evaluate_Yx(Ybus, V, pv, pq):
    ## evaluate y(x)
    Yx = V * conj(Ybus * V)
    F = r_[Yx[pv].real,
           Yx[pq].real,
           Yx[pq].imag]
    return F

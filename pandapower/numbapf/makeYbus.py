from pypower.idx_bus import BUS_I, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_STATUS, SHIFT, TAP

import numba as nb
import numpy as np


@nb.jit(nb.types.UniTuple(nb.complex128[:,:], 3)(nb.f4, nb.f8[:,:], nb.complex128[:,:]), nopython=True, cache=True)
def makeYbus(baseMVA, bus, branch):
    """Builds the bus admittance matrix and branch admittance matrices.

    See pypower's makeYbus for description.

    @author Jan-Hendrik Menke (numba translation)
    """

    tap = np.ones(branch.shape[0], dtype=np.complex128)
    Yf = np.zeros((branch.shape[0], bus.shape[0]), dtype=np.complex128)
    Yt = np.zeros_like(Yf)
    Ybus = np.zeros((bus.shape[0], bus.shape[0]), dtype=np.complex128)

    # error check is omitted for now
    # check that bus numbers are equal to indices to bus (one set of bus nums)
    if np.any(bus[:, BUS_I] != np.arange(bus.shape[0])):
        raise Exception('buses must appear in order by bus number\n')

    for l in range(branch.shape[0]):
        if branch[l, TAP] != 0.:
            tap[l] = branch[l, TAP]
        tap[l] *= np.exp(1j * np.pi / 180 * branch[l, SHIFT])
        ytt = branch[l, BR_STATUS] / (branch[l, BR_R] + 1j * branch[l, BR_X]) + 1j * branch[l, BR_STATUS] * branch[l, BR_B] * 0.5
        from_bus = int(branch[l, F_BUS].real)
        to_bus = int(branch[l, T_BUS].real)
        Yf[l, from_bus] = ytt / (tap[l] * np.conj(tap[l]))
        Yf[l, to_bus] = - branch[l, BR_STATUS] / (branch[l, BR_R] + 1j * branch[l, BR_X]) / np.conj(tap[l])
        Yt[l, from_bus] = - branch[l, BR_STATUS] / (branch[l, BR_R] + 1j * branch[l, BR_X]) / tap[l]
        Yt[l, to_bus] = ytt
        for j in range(bus.shape[0]):
            Ybus[from_bus, j] += Yf[l, j]
            Ybus[to_bus, j] += Yt[l, j]

    for i in range(bus.shape[0]):
        Ybus[i, i] += (bus[i, GS] + 1j * bus[i, BS]) / baseMVA

    return Ybus, Yf, Yt

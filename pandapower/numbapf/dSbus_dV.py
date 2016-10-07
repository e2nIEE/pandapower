import numpy as np
import numba as nb


@nb.jit(nb.types.UniTuple(nb.complex128[:,:], 2)(nb.complex128[:,:], nb.complex128[:]), nopython=True, cache=True)
def dSbus_dV(Ybus, V):
    """
    Computes partial derivatives of power injection w.r.t. voltage.

    See pypower's dSbus_dV for description.

    @author Jan-Hendrik Menke (numba translation)
    """

    # dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dS_dVm = np.zeros(Ybus.shape, dtype=np.complex128)
    buffer = np.zeros(Ybus.shape[1], dtype=np.complex128)
    for i in range(Ybus.shape[0]):
        for j in range(Ybus.shape[1]):
            buffer[i] += Ybus[i, j] * V[j]
            dS_dVm[i, j] += V[j] * np.conjugate(Ybus[i, j] * V[i] / abs(V[i]))
        dS_dVm[i, i] += np.conjugate(buffer[i]) * V[i] / abs(V[i])

    # dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    dS_dVa = np.zeros(Ybus.shape, dtype=np.complex128)
    buffer = np.zeros(Ybus.shape[1], dtype=np.complex128)
    for i in range(Ybus.shape[0]):
        for j in range(Ybus.shape[1]):
            buffer[i] += Ybus[i, j] * V[j]
            dS_dVa[i, j] -= Ybus[i, j] * V[i]
        dS_dVa[i, i] += buffer[i]
        for j in range(Ybus.shape[1]):
            dS_dVa[i, j] = 1j * V[j] * np.conjugate(dS_dVa[i, j])

    return dS_dVm, dS_dVa

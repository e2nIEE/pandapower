# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes partial derivatives of branch currents w.r.t. voltage.
"""

from numpy import diag, asmatrix, asarray
from scipy.sparse import issparse, csr_matrix as sparse


def dIbr_dV(branch, Yf, Yt, V):
    """Computes partial derivatives of branch currents w.r.t. voltage.

    Returns four matrices containing partial derivatives of the complex
    branch currents at "from" and "to" ends of each branch w.r.t voltage
    magnitude and voltage angle respectively (for all buses). If C{Yf} is a
    sparse matrix, the partial derivative matrices will be as well. Optionally
    returns vectors containing the currents themselves. The following
    explains the expressions used to form the matrices::

        If = Yf * V

    Partials of V, Vf & If w.r.t. voltage angles::
        dV/dVa  = j * diag(V)
        dVf/dVa = sparse(range(nl), f, j*V(f)) = j * sparse(range(nl), f, V(f))
        dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)

    Partials of V, Vf & If w.r.t. voltage magnitudes::
        dV/dVm  = diag(V / abs(V))
        dVf/dVm = sparse(range(nl), f, V(f) / abs(V(f))
        dIf/dVm = Yf * dV/dVm = Yf * diag(V / abs(V))

    Derivations for "to" bus are similar.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    i = range(len(V))

    Vnorm = V / abs(V)

    if issparse(Yf):
        diagV = sparse((V, (i, i)))
        diagVnorm = sparse((Vnorm, (i, i)))
    else:
        diagV       = asmatrix( diag(V) )
        diagVnorm   = asmatrix( diag(Vnorm) )

    dIf_dVa = Yf * 1j * diagV
    dIf_dVm = Yf * diagVnorm
    dIt_dVa = Yt * 1j * diagV
    dIt_dVm = Yt * diagVnorm

    # Compute currents.
    if issparse(Yf):
        If = Yf * V
        It = Yt * V
    else:
        If = asarray( Yf * asmatrix(V).T ).flatten()
        It = asarray( Yt * asmatrix(V).T ).flatten()

    return dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, If, It

# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes partial derivatives of power flows w.r.t. voltage.
"""

from numpy import conj, arange, diag, zeros, asmatrix, asarray, asscalar
from scipy.sparse import issparse, csr_matrix as sparse

from pypower.idx_brch import F_BUS, T_BUS

def dSbr_dV(branch, Yf, Yt, V):
    """Computes partial derivatives of power flows w.r.t. voltage.

    returns four matrices containing partial derivatives of the complex
    branch power flows at "from" and "to" ends of each branch w.r.t voltage
    magnitude and voltage angle respectively (for all buses). If C{Yf} is a
    sparse matrix, the partial derivative matrices will be as well. Optionally
    returns vectors containing the power flows themselves. The following
    explains the expressions used to form the matrices::

        If = Yf * V;
        Sf = diag(Vf) * conj(If) = diag(conj(If)) * Vf

    Partials of V, Vf & If w.r.t. voltage angles::
        dV/dVa  = j * diag(V)
        dVf/dVa = sparse(range(nl), f, j*V(f)) = j * sparse(range(nl), f, V(f))
        dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)

    Partials of V, Vf & If w.r.t. voltage magnitudes::
        dV/dVm  = diag(V / abs(V))
        dVf/dVm = sparse(range(nl), f, V(f) / abs(V(f))
        dIf/dVm = Yf * dV/dVm = Yf * diag(V / abs(V))

    Partials of Sf w.r.t. voltage angles::
        dSf/dVa = diag(Vf) * conj(dIf/dVa)
                        + diag(conj(If)) * dVf/dVa
                = diag(Vf) * conj(Yf * j * diag(V))
                        + conj(diag(If)) * j * sparse(range(nl), f, V(f))
                = -j * diag(Vf) * conj(Yf * diag(V))
                        + j * conj(diag(If)) * sparse(range(nl), f, V(f))
                = j * (conj(diag(If)) * sparse(range(nl), f, V(f))
                        - diag(Vf) * conj(Yf * diag(V)))

    Partials of Sf w.r.t. voltage magnitudes::
        dSf/dVm = diag(Vf) * conj(dIf/dVm)
                        + diag(conj(If)) * dVf/dVm
                = diag(Vf) * conj(Yf * diag(V / abs(V)))
                        + conj(diag(If)) * sparse(range(nl), f, V(f)/abs(V(f)))

    Derivations for "to" bus are similar.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, "AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation", MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ## define
    f = branch[:, F_BUS].astype(int)       ## list of "from" buses
    t = branch[:, T_BUS].astype(int)       ## list of "to" buses
    nl = len(f)
    nb = len(V)
    il = arange(nl)
    ib = arange(nb)

    Vnorm = V / abs(V)

    if issparse(Yf):
        ## compute currents
        If = Yf * V
        It = Yt * V

        diagVf = sparse((V[f], (il, il)))
        diagIf = sparse((If, (il, il)))
        diagVt = sparse((V[t], (il, il)))
        diagIt = sparse((It, (il, il)))
        diagV  = sparse((V, (ib, ib)))
        diagVnorm = sparse((Vnorm, (ib, ib)))

        shape = (nl, nb)
        # Partial derivative of S w.r.t voltage phase angle.
        dSf_dVa = 1j * (conj(diagIf) *
            sparse((V[f], (il, f)), shape) - diagVf * conj(Yf * diagV))

        dSt_dVa = 1j * (conj(diagIt) *
            sparse((V[t], (il, t)), shape) - diagVt * conj(Yt * diagV))

        # Partial derivative of S w.r.t. voltage amplitude.
        dSf_dVm = diagVf * conj(Yf * diagVnorm) + conj(diagIf) * \
            sparse((Vnorm[f], (il, f)), shape)

        dSt_dVm = diagVt * conj(Yt * diagVnorm) + conj(diagIt) * \
            sparse((Vnorm[t], (il, t)), shape)
    else:  ## dense version
        ## compute currents
        If = asarray( Yf * asmatrix(V).T ).flatten()
        It = asarray( Yt * asmatrix(V).T ).flatten()

        diagVf      = asmatrix( diag(V[f]) )
        diagIf      = asmatrix( diag(If) )
        diagVt      = asmatrix( diag(V[t]) )
        diagIt      = asmatrix( diag(It) )
        diagV       = asmatrix( diag(V) )
        diagVnorm   = asmatrix( diag(Vnorm) )
        temp1       = asmatrix( zeros((nl, nb), complex) )
        temp2       = asmatrix( zeros((nl, nb), complex) )
        temp3       = asmatrix( zeros((nl, nb), complex) )
        temp4       = asmatrix( zeros((nl, nb), complex) )
        for i in range(nl):
            fi, ti = f[i], t[i]
            temp1[i, fi] = asscalar(V[fi])
            temp2[i, fi] = asscalar(Vnorm[fi])
            temp3[i, ti] = asscalar(V[ti])
            temp4[i, ti] = asscalar(Vnorm[ti])

        dSf_dVa = 1j * (conj(diagIf) * temp1 - diagVf * conj(Yf * diagV))
        dSf_dVm = diagVf * conj(Yf * diagVnorm) + conj(diagIf) * temp2
        dSt_dVa = 1j * (conj(diagIt) * temp3 - diagVt * conj(Yt * diagV))
        dSt_dVm = diagVt * conj(Yt * diagVnorm) + conj(diagIt) * temp4

    # Compute power flow vectors.
    Sf = V[f] * conj(If)
    St = V[t] * conj(It)

    return dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St

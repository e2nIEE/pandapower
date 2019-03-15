# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes 2nd derivatives of complex power flow w.r.t. voltage.
"""

from numpy import ones, conj
from scipy.sparse import csr_matrix


def d2Sbr_dV2(Cbr, Ybr, V, lam):
    """Computes 2nd derivatives of complex power flow w.r.t. voltage.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage angle
    and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the complex branch power flows. Takes sparse connection
    matrix C{Cbr}, sparse branch admittance matrix C{Ybr}, voltage vector C{V}
    and C{nl x 1} vector of multipliers C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nb = len(V)
    nl = len(lam)
    ib = range(nb)
    il = range(nl)

    diaglam = csr_matrix((lam, (il, il)))
    diagV = csr_matrix((V, (ib, ib)))

    A = Ybr.H * diaglam * Cbr
    B = conj(diagV) * A * diagV
    D = csr_matrix( ((A * V) * conj(V), (ib, ib)) )
    E = csr_matrix( ((A.T * conj(V) * V), (ib, ib)) )
    F = B + B.T
    G = csr_matrix((ones(nb) / abs(V), (ib, ib)))

    Haa = F - D - E
    Hva = 1j * G * (B - B.T - D + E)
    Hav = Hva.T
    Hvv = G * F * G

    return Haa, Hav, Hva, Hvv

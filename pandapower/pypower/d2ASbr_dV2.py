# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes 2nd derivatives of |complex power flow|**2 w.r.t. V.
"""

from scipy.sparse import csr_matrix

from pypower.d2Sbr_dV2 import d2Sbr_dV2


def d2ASbr_dV2(dSbr_dVa, dSbr_dVm, Sbr, Cbr, Ybr, V, lam):
    """Computes 2nd derivatives of |complex power flow|**2 w.r.t. V.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage
    angle and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the square of the magnitude of branch complex power flows.
    Takes sparse first derivative matrices of complex flow, complex flow
    vector, sparse connection matrix C{Cbr}, sparse branch admittance matrix
    C{Ybr}, voltage vector C{V} and C{nl x 1} vector of multipliers C{lam}.
    Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @see: L{dSbr_dV}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    il = range(len(lam))

    diaglam = csr_matrix((lam, (il, il)))
    diagSbr_conj = csr_matrix((Sbr.conj(), (il, il)))

    Saa, Sav, Sva, Svv = d2Sbr_dV2(Cbr, Ybr, V, diagSbr_conj * lam)

    Haa = 2 * ( Saa + dSbr_dVa.T * diaglam * dSbr_dVa.conj() ).real
    Hva = 2 * ( Sva + dSbr_dVm.T * diaglam * dSbr_dVa.conj() ).real
    Hav = 2 * ( Sav + dSbr_dVa.T * diaglam * dSbr_dVm.conj() ).real
    Hvv = 2 * ( Svv + dSbr_dVm.T * diaglam * dSbr_dVm.conj() ).real

    return Haa, Hav, Hva, Hvv

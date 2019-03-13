# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Computes 2nd derivatives of |complex current|**2 w.r.t. V.
"""

from scipy.sparse import csr_matrix as sparse

from pypower.d2Ibr_dV2 import d2Ibr_dV2


def d2AIbr_dV2(dIbr_dVa, dIbr_dVm, Ibr, Ybr, V, lam):
    """Computes 2nd derivatives of |complex current|**2 w.r.t. V.

    Returns 4 matrices containing the partial derivatives w.r.t. voltage
    angle and magnitude of the product of a vector C{lam} with the 1st partial
    derivatives of the square of the magnitude of the branch currents.
    Takes sparse first derivative matrices of complex flow, complex flow
    vector, sparse branch admittance matrix C{Ybr}, voltage vector C{V} and
    C{nl x 1} vector of multipliers C{lam}. Output matrices are sparse.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @see: L{dIbr_dV}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    # define
    il = range(len(lam))

    diaglam = sparse((lam, (il, il)))
    diagIbr_conj = sparse((Ibr.conj(), (il, il)))

    Iaa, Iav, Iva, Ivv = d2Ibr_dV2(Ybr, V, diagIbr_conj * lam)

    Haa = 2 * ( Iaa + dIbr_dVa.T * diaglam * dIbr_dVa.conj() ).real
    Hva = 2 * ( Iva + dIbr_dVm.T * diaglam * dIbr_dVa.conj() ).real
    Hav = 2 * ( Iav + dIbr_dVa.T * diaglam * dIbr_dVm.conj() ).real
    Hvv = 2 * ( Ivv + dIbr_dVm.T * diaglam * dIbr_dVm.conj() ).real

    return Haa, Hav, Hva, Hvv

# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Partial derivatives of squared flow magnitudes w.r.t voltage.
"""

from scipy.sparse import csr_matrix


def dAbr_dV(dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St):
    """Partial derivatives of squared flow magnitudes w.r.t voltage.

    Returns four matrices containing partial derivatives of the square of
    the branch flow magnitudes at "from" & "to" ends of each branch w.r.t
    voltage magnitude and voltage angle respectively (for all buses), given
    the flows and flow sensitivities. Flows could be complex current or
    complex or real power. Notation below is based on complex power. The
    following explains the expressions used to form the matrices:

    Let Af refer to the square of the apparent power at the "from" end of
    each branch::

        Af = abs(Sf)**2
           = Sf .* conj(Sf)
           = Pf**2 + Qf**2

    then ...

    Partial w.r.t real power::
        dAf/dPf = 2 * diag(Pf)

    Partial w.r.t reactive power::
        dAf/dQf = 2 * diag(Qf)

    Partial w.r.t Vm & Va::
        dAf/dVm = dAf/dPf * dPf/dVm + dAf/dQf * dQf/dVm
        dAf/dVa = dAf/dPf * dPf/dVa + dAf/dQf * dQf/dVa

    Derivations for "to" bus are similar.

    For more details on the derivations behind the derivative code used
    in PYPOWER information, see:

    [TN2]  R. D. Zimmerman, I{"AC Power Flows, Generalized OPF Costs and
    their Derivatives using Complex Matrix Notation"}, MATPOWER
    Technical Note 2, February 2010.
    U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}

    @return: The partial derivatives of the squared flow magnitudes w.r.t
             voltage magnitude and voltage angle given the flows and flow
             sensitivities. Flows could be complex current or complex or
             real power.
    @see: L{dIbr_dV}, L{dSbr_dV}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    il = range(len(Sf))

    dAf_dPf = csr_matrix((2 * Sf.real, (il, il)))
    dAf_dQf = csr_matrix((2 * Sf.imag, (il, il)))
    dAt_dPt = csr_matrix((2 * St.real, (il, il)))
    dAt_dQt = csr_matrix((2 * St.imag, (il, il)))

    # Partial derivative of apparent power magnitude w.r.t voltage
    # phase angle.
    dAf_dVa = dAf_dPf * dSf_dVa.real + dAf_dQf * dSf_dVa.imag
    dAt_dVa = dAt_dPt * dSt_dVa.real + dAt_dQt * dSt_dVa.imag
    # Partial derivative of apparent power magnitude w.r.t. voltage
    # amplitude.
    dAf_dVm = dAf_dPf * dSf_dVm.real + dAf_dQf * dSf_dVm.imag
    dAt_dVm = dAt_dPt * dSt_dVm.real + dAt_dQt * dSt_dVm.imag

    return dAf_dVa, dAf_dVm, dAt_dVa, dAt_dVm

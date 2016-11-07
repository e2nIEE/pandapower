# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

"""Runs a power flow.
"""

from os.path import dirname, join

from time import time

from numpy import r_, zeros, pi, ones, exp, argmax
from numpy import flatnonzero as find

from pypower.loadcase import loadcase
from pypower.ppoption import ppoption
from pypower.makeBdc import makeBdc
from pypower.makeSbus import makeSbus
from pypower.fdpf import fdpf
from pypower.gausspf import gausspf
from pypower.makeB import makeB
from pypower.pfsoln import pfsoln
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS

from pandapower.pypower_extensions.newtonpf import newtonpf
from pandapower.pypower_extensions.dcpf import dcpf
from pandapower.pypower_extensions.bustypes import bustypes

def _runpf(casedata=None, init='flat', ac=True, Numba=True, ppopt=None):
    """Runs a power flow.

    Similar to runpf() from pypower. See Pypower documentation for more information.

    Changes by University of Kassel (Florian Schaefer):
        Numba can be used for pf calculations.
        Changes in structure (AC as well as DC PF can be calculated)
    """

    ## default arguments
    if casedata is None:
        casedata = join(dirname(__file__), 'case9') 
    ppopt = ppoption(ppopt)

    ## options
    verbose = ppopt["VERBOSE"]

    ## read data
    ppci = loadcase(casedata)

    # get data for calc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)

    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int)  ## what buses are they at?

    ##-----  run the power flow  -----
    t0 = time()

    if not ac or (ac and init == 'dc'):  # DC formulation
        if verbose:
            print(' -- DC Power Flow\n')

        ## initial state
        Va0 = bus[:, VA] * (pi / 180)

        ## build B matrices and phase shift injections
        B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)

        ## compute complex bus power injections [generation - load]
        ## adjusted for phase shifters and real shunts
        Pbus = makeSbus(baseMVA, bus, gen) - Pbusinj - bus[:, GS] / baseMVA

        ## "run" the power flow
        Va = dcpf(B, Pbus, Va0, ref, pv, pq)

        ## update data matrices with solution
        branch[:, [QF, QT]] = zeros((branch.shape[0], 2))
        branch[:, PF] = (Bf * Va + Pfinj) * baseMVA
        branch[:, PT] = -branch[:, PF]
        bus[:, VM] = ones(bus.shape[0])
        bus[:, VA] = Va * (180 / pi)
        ## update Pg for slack generator (1st gen at ref bus)
        ## (note: other gens at ref bus are accounted for in Pbus)
        ##      Pg = Pinj + Pload + Gs
        ##      newPg = oldPg + newPinj - oldPinj

        refgen = zeros(len(ref), dtype=int)
        for k in range(len(ref)):
            temp = find(gbus == ref[k])
            refgen[k] = on[temp[0]]
        gen[refgen, PG] = gen[refgen, PG] + (B[ref, :] * Va - Pbus[ref]) * baseMVA
        success = 1

        if ac and init=='dc':
            # get results from DC powerflow for AC powerflow
            ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch

    if ac:  ## AC formulation
        # options
        qlim = ppopt["ENFORCE_Q_LIMS"]  ## enforce Q limits on gens?

        ## check if numba is available and the corresponding flag
        try:
            from numba import _version as nb_version

            # get Numba Version (in order to use it it must be > 0.25)
            nbVersion = float(nb_version.version_version[:4])

            if nbVersion < 0.25:
                print('Warning: Numba version too old -> Upgrade to a version > 0.25. Numba is disabled\n')
                Numba = False

        except ImportError:
            # raise UserWarning('Numba cannot be imported. Call runpp() with Numba=False!')
            print('Warning: Numba cannot be imported. Numba is disabled. Call runpp() with Numba=False!\n')
            Numba = False

        if Numba:
            from pandapower.pypower_extensions.makeYbus import makeYbus
        else:
            from pypower.makeYbus import makeYbus

        alg = ppopt['PF_ALG']
        if verbose > 0:
            if alg == 1:
                solver = 'Newton'
            elif alg == 2:
                solver = 'fast-decoupled, XB'
            elif alg == 3:
                solver = 'fast-decoupled, BX'
            elif alg == 4:
                solver = 'Gauss-Seidel'
            else:
                solver = 'unknown'
            print(' -- AC Power Flow (%s)\n' % solver)

        ## initial state
        # V0    = ones(bus.shape[0])            ## flat start
        V0 = bus[:, VM] * exp(1j * pi / 180 * bus[:, VA])
        V0[gbus] = gen[on, VG] / abs(V0[gbus]) * V0[gbus]

        if qlim:
            ref0 = ref  ## save index and angle of
            Varef0 = bus[ref0, VA]  ##   original reference bus(es)
            limited = []  ## list of indices of gens @ Q lims
            fixedQg = zeros(gen.shape[0])  ## Qg of gens at Q limits

        repeat = True
        while repeat:
            ## build admittance matrices
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

            ## compute complex bus power injections [generation - load]
            Sbus = makeSbus(baseMVA, bus, gen)

            ## run the power flow
            alg = ppopt["PF_ALG"]
            if alg == 1:
                V, success, _ = newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppopt, Numba)
            elif alg == 2 or alg == 3:
                Bp, Bpp = makeB(baseMVA, bus, branch, alg)
                V, success, _ = fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt)
            elif alg == 4:
                V, success, _ = gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
            else:
                raise ValueError('Only Newton''s method, fast-decoupled, and '
                             'Gauss-Seidel power flow algorithms currently '
                             'implemented.\n')

            ## update data matrices with solution
            bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, pv, pq)

            if qlim:  ## enforce generator Q limits
                ## find gens with violated Q constraints
                gen_status = gen[:, GEN_STATUS] > 0
                qg_max_lim = gen[:, QG] > gen[:, QMAX]
                qg_min_lim = gen[:, QG] < gen[:, QMIN]

                mx = find(gen_status & qg_max_lim)
                mn = find(gen_status & qg_min_lim)

                if len(mx) > 0 or len(mn) > 0:  ## we have some Q limit violations
                    # No PV generators
                    if len(pv) == 0:
                        if verbose:
                            if len(mx) > 0:
                                print('Gen %d [only one left] exceeds upper Q limit : INFEASIBLE PROBLEM\n' % mx + 1)
                            else:
                                print('Gen %d [only one left] exceeds lower Q limit : INFEASIBLE PROBLEM\n' % mn + 1)

                        success = 0
                        break

                    ## one at a time?
                    if qlim == 2:  ## fix largest violation, ignore the rest
                        k = argmax(r_[gen[mx, QG] - gen[mx, QMAX],
                                      gen[mn, QMIN] - gen[mn, QG]])
                        if k > len(mx):
                            mn = mn[k - len(mx)]
                            mx = []
                        else:
                            mx = mx[k]
                            mn = []

                    if verbose and len(mx) > 0:
                        for i in range(len(mx)):
                            print('Gen ' + str(mx[i] + 1) + ' at upper Q limit, converting to PQ bus\n')

                    if verbose and len(mn) > 0:
                        for i in range(len(mn)):
                            print('Gen ' + str(mn[i] + 1) + ' at lower Q limit, converting to PQ bus\n')

                    ## save corresponding limit values
                    fixedQg[mx] = gen[mx, QMAX]
                    fixedQg[mn] = gen[mn, QMIN]
                    mx = r_[mx, mn].astype(int)

                    ## convert to PQ bus
                    gen[mx, QG] = fixedQg[mx]  ## set Qg to binding
                    for i in range(len(mx)):  ## [one at a time, since they may be at same bus]
                        gen[mx[i], GEN_STATUS] = 0  ## temporarily turn off gen,
                        bi = gen[mx[i], GEN_BUS]  ## adjust load accordingly,
                        bus[bi, [PD, QD]] = (bus[bi, [PD, QD]] - gen[mx[i], [PG, QG]])

                    if len(ref) > 1 and any(bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] == REF):
                        raise ValueError('Sorry, PYPOWER cannot enforce Q '
                                         'limits for slack buses in systems '
                                         'with multiple slacks.')

                    bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] = PQ  ## & set bus type to PQ

                    ## update bus index lists of each type of bus
                    ref_temp = ref
                    ref, pv, pq = bustypes(bus, gen)
                    if verbose and ref != ref_temp:
                        print('Bus %d is new slack bus\n' % ref)

                    limited = r_[limited, mx].astype(int)
                else:
                    repeat = 0  ## no more generator Q limits violated
            else:
                repeat = 0  ## don't enforce generator Q limits, once is enough

        if qlim and len(limited) > 0:
            ## restore injections from limited gens [those at Q limits]
            gen[limited, QG] = fixedQg[limited]  ## restore Qg value,
            for i in range(len(limited)):  ## [one at a time, since they may be at same bus]
                bi = gen[limited[i], GEN_BUS]  ## re-adjust load,
                bus[bi, [PD, QD]] = bus[bi, [PD, QD]] + gen[limited[i], [PG, QG]]
                gen[limited[i], GEN_STATUS] = 1  ## and turn gen back on

                #            if ref != ref0:
                #                ## adjust voltage angles to make original ref bus correct
                #                bus[:, VA] = bus[:, VA] - bus[ref0, VA] + Varef0


    ppci["et"] = time() - t0
    ppci["success"] = success

    ##-----  output results  -----
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    results = ppci

    return results, success

if __name__ == '__main__':
    _runpf()

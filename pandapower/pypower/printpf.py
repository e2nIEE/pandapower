# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Prints power flow results."""

from sys import stdout

from numpy import any, arange, argmax, argmin, diff, exp, imag, int64, logical_or, min, ones, pi, r_, real, sort, zeros
from numpy import flatnonzero as find

from pandapower.pypower.idx_brch import (
    BR_B,
    BR_G,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    MU_SF,
    MU_ST,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    SHIFT,
    T_BUS,
    TAP,
)
from pandapower.pypower.idx_bus import (
    BS,
    BUS_AREA,
    BUS_I,
    BUS_TYPE,
    GS,
    LAM_P,
    LAM_Q,
    MU_VMAX,
    MU_VMIN,
    PD,
    QD,
    REF,
    VA,
    VM,
    VMAX,
    VMIN,
)
from pandapower.pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    MU_PMAX,
    MU_PMIN,
    MU_QMAX,
    MU_QMIN,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
)
from pandapower.pypower.isload import isload
from pandapower.pypower.ppoption import ppoption
from pandapower.pypower.run_userfcn import run_userfcn


def printpf(
    baseMVA, bus=None, gen=None, branch=None, f=None, success=None, et=None, fd=None, ppopt=None
):  # pragma: no cover
    """Prints power flow results.

    Prints power flow and optimal power flow results to C{fd} (a file
    descriptor which defaults to C{stdout}), with the details of what
    gets printed controlled by the optional C{ppopt} argument, which is a
    PYPOWER options vector (see L{ppoption} for details).

    The data can either be supplied in a single C{results} dict, or
    in the individual arguments: C{baseMVA}, C{bus}, C{gen}, C{branch}, C{f},
    C{success} and C{et}, where C{f} is the OPF objective function value,
    C{success} is C{True} if the solution converged and C{False} otherwise,
    and C{et} is the elapsed time for the computation in seconds. If C{f} is
    given, it is assumed that the output is from an OPF run, otherwise it is
    assumed to be a simple power flow run.

    Examples::
        ppopt = ppoptions(OUT_GEN=1, OUT_BUS=0, OUT_BRANCH=0)
        fd = open(fname, 'w+b')
        results = runopf(ppc)
        printpf(results)
        printpf(results, fd)
        printpf(results, fd, ppopt)
        printpf(baseMVA, bus, gen, branch, f, success, et)
        printpf(baseMVA, bus, gen, branch, f, success, et, fd)
        printpf(baseMVA, bus, gen, branch, f, success, et, fd, ppopt)
        fd.close()

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##----- initialization -----
    ## default arguments
    if isinstance(baseMVA, dict):
        have_results_struct = 1
        results = baseMVA
        if gen is None:
            ppopt = ppoption()  ## use default options
        else:
            ppopt = gen
        if ppopt["OUT_ALL"] == 0:
            return  ## nothin' to see here, bail out now
        if bus is None:
            fd = stdout  ## print to stdout by default
        else:
            fd = bus
        baseMVA, bus, gen, branch, success, et = (
            results["baseMVA"],
            results["bus"],
            results["gen"],
            results["branch"],
            results["success"],
            results["et"],
        )
        if "f" in results:
            f = results["f"]
        else:
            f = None
    else:
        have_results_struct = 0
        if ppopt is None:
            ppopt = ppoption()  ## use default options
            if fd is None:
                fd = stdout  ## print to stdout by default
        if ppopt["OUT_ALL"] == 0:
            return  ## nothin' to see here, bail out now

    isOPF = f is not None  ## FALSE -> only simple PF data, TRUE -> OPF data

    ## options
    isDC = ppopt["PF_DC"]  ## use DC formulation?
    OUT_ALL = ppopt["OUT_ALL"]
    OUT_ANY = OUT_ALL == 1  ## set to true if any pretty output is to be generated
    OUT_SYS_SUM = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt["OUT_SYS_SUM"])
    OUT_AREA_SUM = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt["OUT_AREA_SUM"])
    OUT_BUS = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt["OUT_BUS"])
    OUT_BRANCH = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt["OUT_BRANCH"])
    OUT_GEN = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt["OUT_GEN"])
    OUT_ANY = OUT_ANY | ((OUT_ALL == -1) and (OUT_SYS_SUM or OUT_AREA_SUM or OUT_BUS or OUT_BRANCH or OUT_GEN))

    if OUT_ALL == -1:
        OUT_ALL_LIM = ppopt["OUT_ALL_LIM"]
    elif OUT_ALL == 1:
        OUT_ALL_LIM = 2
    else:
        OUT_ALL_LIM = 0

    OUT_ANY = OUT_ANY or (OUT_ALL_LIM >= 1)
    if OUT_ALL_LIM == -1:
        OUT_V_LIM = ppopt["OUT_V_LIM"]
        OUT_LINE_LIM = ppopt["OUT_LINE_LIM"]
        OUT_PG_LIM = ppopt["OUT_PG_LIM"]
        OUT_QG_LIM = ppopt["OUT_QG_LIM"]
    else:
        OUT_V_LIM = OUT_ALL_LIM
        OUT_LINE_LIM = OUT_ALL_LIM
        OUT_PG_LIM = OUT_ALL_LIM
        OUT_QG_LIM = OUT_ALL_LIM

    OUT_ANY = OUT_ANY or ((OUT_ALL_LIM == -1) and (OUT_V_LIM or OUT_LINE_LIM or OUT_PG_LIM or OUT_QG_LIM))
    ptol = 1e-4  ## tolerance for displaying shadow prices

    ## create map of external bus numbers to bus indices
    i2e = bus[:, BUS_I].astype(int64)
    e2i = zeros(max(i2e) + 1, int64)
    e2i[i2e] = arange(bus.shape[0])

    ## sizes of things
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of branches
    ng = gen.shape[0]  ## number of generators

    ## zero out some data to make printout consistent for DC case
    if isDC:
        bus[:, r_[QD, BS]] = zeros((nb, 2))
        gen[:, r_[QG, QMAX, QMIN]] = zeros((ng, 3))
        branch[:, r_[BR_R, BR_B, BR_G]] = zeros((nl, 3))

    ## parameters
    ties = find(
        bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA]
        != bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA]
    )
    ## area inter-ties
    tap = ones(nl)  ## default tap ratio = 1 for lines
    xfmr = find(branch[:, TAP]).real  ## indices of transformers
    tap[xfmr] = branch[xfmr, TAP].real  ## include transformer tap ratios
    tap = tap * exp(-1j * pi / 180 * branch[:, SHIFT])  ## add phase shifters
    nzld = find((bus[:, PD] != 0.0) | (bus[:, QD] != 0.0))
    sorted_areas = sort(bus[:, BUS_AREA])
    ## area numbers
    s_areas = sorted_areas[r_[1, find(diff(sorted_areas)) + 1]]
    nzsh = find((bus[:, GS] != 0.0) | (bus[:, BS] != 0.0))
    allg = find(~isload(gen))
    ong = find((gen[:, GEN_STATUS] > 0) & ~isload(gen))
    onld = find((gen[:, GEN_STATUS] > 0) & isload(gen))
    V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA])
    out = find(branch[:, BR_STATUS] == 0)  ## out-of-service branches
    nout = len(out)
    if isDC:
        loss = zeros(nl)
    else:
        loss = (
            baseMVA
            * abs(V[e2i[branch[:, F_BUS].real.astype(int64)]] / tap - V[e2i[branch[:, T_BUS].real.astype(int64)]]) ** 2
            / (branch[:, BR_R] - 1j * branch[:, BR_X])
        )

    fchg = abs(V[e2i[branch[:, F_BUS].real.astype(int64)]] / tap) ** 2 * branch[:, BR_G] * baseMVA / 2
    tchg = abs(V[e2i[branch[:, T_BUS].real.astype(int64)]]) ** 2 * branch[:, BR_G] * baseMVA / 2
    loss[out] = zeros(nout)
    fchg[out] = zeros(nout)
    tchg[out] = zeros(nout)

    ##----- print the stuff -----
    if OUT_ANY:
        ## convergence & elapsed time
        if success:
            fd.write(f"\nConverged in {et:.2f} seconds")
        else:
            fd.write(f"\nDid not converge ({et:.2f} seconds)\n")

        ## objective function value
        if isOPF:
            fd.write(f"\nObjective Function Value = {f:.2f} $/hr")

    if OUT_SYS_SUM:
        fd.write(f"\n{'':=^80}")
        fd.write("\n| PyPower (ppci) System Summary - these are not valid for pandapower DataFrames|")
        fd.write(f"\n{'':=^80}")
        fd.write("\n\nHow many?                How much?              P (MW)            Q (MVAr)")
        fd.write("\n---------------------    -------------------  -------------  -----------------")
        fd.write(
            f"\n{'Buses':<14}{nb:6d}  {'Total Gen Capacity':^24}{sum(gen[allg, PMAX]):7.1f}       "
            f"{sum(gen[allg, QMIN]):7.1f} to {sum(gen[allg, QMAX]):.1f}"
        )
        fd.write(
            f"\nGenerators     {len(allg):5d}     On-line Capacity     {sum(gen[ong, PMAX]):7.1f}       "
            f"{sum(gen[ong, QMIN]):7.1f} to {sum(gen[ong, QMAX]):.1f}"
        )
        fd.write(
            f"\nCommitted Gens {len(ong):5d}     Generation (actual)  {sum(gen[ong, PG]):7.1f}           "
            f"{sum(gen[ong, QG]):7.1f}"
        )
        fd.write(
            f"\nLoads          {len(nzld) + len(onld):5d}     Load                 "
            f"{sum(bus[nzld, PD]) - sum(gen[onld, PG]):7.1f}           {sum(bus[nzld, QD]) - sum(gen[onld, QG]):7.1f}"
        )
        fd.write(
            f"\n  Fixed        {len(nzld):5d}       Fixed              {sum(bus[nzld, PD]):7.1f}           "
            f"{sum(bus[nzld, QD]):7.1f}"
        )
        fd.write(
            f"\n  Dispatchable {len(onld):5d}       Dispatchable       {-sum(gen[onld, PG]):7.1f} of "
            f"{-sum(gen[onld, PMIN]):7.1f}{-sum(gen[onld, QG]):7.1f}"
        )
        fd.write(
            f"\nShunts         {len(nzsh):5d}     Shunt (inj)          {-sum(bus[nzsh, VM] ** 2 * bus[nzsh, GS]):7.1f}"
            f"           {sum(bus[nzsh, VM] ** 2 * bus[nzsh, BS]):7.1f}"
        )
        fd.write(
            f"\nBranches       {nl:5d}     Losses (I^2 * Z)     {sum(loss.real):8.2f}          {sum(loss.imag):8.2f}"
        )
        fd.write(
            f"\nTransformers   {len(xfmr):5d}     Branch Charging (inj)     -            {sum(fchg) + sum(tchg):7.1f}"
        )
        fd.write(
            f"\nInter-ties     {len(ties):5d}     Total Inter-tie Flow "
            f"{sum(abs(branch[ties, PF] - branch[ties, PT])) / 2:7.1f}           "
            f"{sum(abs(branch[ties, QF] - branch[ties, QT])) / 2:7.1f}"
        )
        fd.write(f"\nAreas          {len(s_areas):5d}")
        fd.write("\n")
        fd.write(f"\n{'':<17}{'Minimum':^25}  {'Maximum':^32}")
        fd.write(f"\n{'':<17}{'':-^25}  {'':-^32}")
        minv = min(bus[:, VM])
        mini = argmin(bus[:, VM])
        maxv = max(bus[:, VM])
        maxi = argmax(bus[:, VM])
        fd.write(
            f"\nVoltage Magnitude {minv:7.3f} p.u. @ bus {bus[mini, BUS_I]:4d}     "
            f"{maxv:7.3f} p.u. @ bus {bus[maxi, BUS_I]:4d}"
        )
        minv = min(bus[:, VA])
        mini = argmin(bus[:, VA])
        maxv = max(bus[:, VA])
        maxi = argmax(bus[:, VA])
        fd.write(
            f"\nVoltage Angle   {minv:8.2f} deg   @ bus {bus[mini, BUS_I]:4d}   "
            f"{maxv:8.2f} deg   @ bus {bus[maxi, BUS_I]:4d}"
        )
        if not isDC:
            maxv = max(loss.real)
            maxi = argmax(loss.real)
            fd.write(
                f"\nP Losses (I^2*R){'-':^27} {maxv:8.2f} MW    @ line {branch[maxi, F_BUS].real:d}-"
                f"{branch[maxi, T_BUS].real:d}"
            )
            maxv = max(loss.imag)
            maxi = argmax(loss.imag)
            fd.write(
                f"\nQ Losses (I^2*X){'-':^27} {maxv:8.2f} MVAr  @ line {branch[maxi, F_BUS].real:d}-"
                f"{branch[maxi, T_BUS].real:d}"
            )
        if isOPF:
            minv = min(bus[:, LAM_P])
            mini = argmin(bus[:, LAM_P])
            maxv = max(bus[:, LAM_P])
            maxi = argmax(bus[:, LAM_P])
            fd.write(
                f"\nLambda P        {minv:8.2f} $/MWh @ bus {bus[mini, BUS_I]:4d}   {maxv:8.2f} $/MWh @ bus "
                f"{bus[maxi, BUS_I]:4d}"
            )
            minv = min(bus[:, LAM_Q])
            mini = argmin(bus[:, LAM_Q])
            maxv = max(bus[:, LAM_Q])
            maxi = argmax(bus[:, LAM_Q])
            fd.write(
                f"\nLambda Q        {minv:8.2f} $/MWh @ bus {bus[mini, BUS_I]:4d}   {maxv:8.2f} $/MWh @ bus "
                f"{bus[maxi, BUS_I]:4d}"
            )
        fd.write("\n")

    if OUT_AREA_SUM:
        fd.write(f"\n{'':=^80}")
        fd.write(f"\n|     {'Area Summary':<73}|")
        fd.write(f"\n{'':=^80}")
        fd.write("\nArea  # of      # of Gens        # of Loads         # of    # of   # of   # of")
        fd.write("\n Num  Buses   Total  Online   Total  Fixed  Disp    Shunt   Brchs  Xfmrs   Ties")
        fd.write("\n----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----")
        for i in range(len(s_areas)):
            a = s_areas[i]
            ib = find(bus[:, BUS_AREA] == a)
            ig = find((bus[e2i[gen[:, GEN_BUS].astype(int64)], BUS_AREA] == a) & ~isload(gen))
            igon = find(
                (bus[e2i[gen[:, GEN_BUS].astype(int64)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & ~isload(gen)
            )
            ildon = find(
                (bus[e2i[gen[:, GEN_BUS].astype(int64)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & isload(gen)
            )
            inzld = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, PD], bus[:, QD]))
            inzsh = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, GS], bus[:, BS]))
            ibrch = find(
                (bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA] == a)
                & (bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA] == a)
            )
            in_tie = find(
                (bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA] == a)
                & (bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA] != a)
            )
            out_tie = find(
                (bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA] != a)
                & (bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA] == a)
            )
            if not any(xfmr + 1):
                nxfmr = 0
            else:
                nxfmr = len(
                    find(
                        (bus[e2i[branch[xfmr, F_BUS].real.astype(int64)], BUS_AREA] == a)
                        & (bus[e2i[branch[xfmr, T_BUS].real.astype(int64)], BUS_AREA] == a)
                    )
                )
            fd.write(
                f"\n{a:3d}  {len(ib):6d}   {len(ig):5d}  {len(igon):5d}   {len(inzld) + len(ildon):5d}  {len(inzld):5d}"
                f"  {len(ildon):5d}   {len(inzsh):5d}   {len(ibrch):5d}  {nxfmr:5d}  {len(in_tie) + len(out_tie):5d}"
            )

        fd.write("\n----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----")
        fd.write(
            f"\nTot: {nb:6d}   {len(allg):5d}  {len(ong):5d}   {len(nzld) + len(onld):5d}  {len(nzld):5d}  "
            f"{len(onld):5d}   {len(nzsh):5d}   {nl:5d}  {len(xfmr):5d}  {len(ties):5d}"
        )
        fd.write("\n")
        fd.write("\nArea      Total Gen Capacity           On-line Gen Capacity         Generation")
        fd.write("\n Num     MW           MVAr            MW           MVAr             MW    MVAr")
        fd.write("\n----   ------  ------------------   ------  ------------------    ------  ------")
        for i in range(len(s_areas)):
            a = s_areas[i]
            ig = find((bus[e2i[gen[:, GEN_BUS].astype(int64)], BUS_AREA] == a) & ~isload(gen))
            igon = find(
                (bus[e2i[gen[:, GEN_BUS].astype(int64)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & ~isload(gen)
            )
            fd.write(
                f"\n{a:3d}   {sum(gen[ig, PMAX]):7.1f}  {sum(gen[ig, QMIN]):7.1f} to {sum(gen[ig, QMAX]):.1f}  "
                f"{sum(gen[igon, PMAX]):7.1f}  {sum(gen[igon, QMIN]):7.1f} to {sum(gen[igon, QMAX]):7.1f}   "
                f"{sum(gen[igon, PG]):7.1f} {sum(gen[igon, QG]):7.1f}"
            )

        fd.write("\n----   ------  ------------------   ------  ------------------    ------  ------")
        fd.write("\n")
        fd.write("\nArea    Disp Load Cap       Disp Load         Fixed Load        Total Load")
        fd.write("\n Num      MW     MVAr       MW     MVAr       MW     MVAr       MW     MVAr")
        fd.write("\n----    ------  ------    ------  ------    ------  ------    ------  ------")
        Qlim = (gen[:, QMIN] == 0) * gen[:, QMAX] + (gen[:, QMAX] == 0) * gen[:, QMIN]
        for i in range(len(s_areas)):
            a = s_areas[i]
            ildon = find(
                (bus[e2i[gen[:, GEN_BUS].astype(int64)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & isload(gen)
            )
            inzld = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, PD], bus[:, QD]))
            fd.write(
                f"\n{a:3d}    {-sum(gen[ildon, PMIN]):7.1f} {-sum(Qlim[ildon]):7.1f}   {-sum(gen[ildon, PG]):7.1f} "
                f"{-sum(gen[ildon, QG]):7.1f}   {sum(bus[inzld, PD]):7.1f} {sum(bus[inzld, QD]):7.1f}   "
                f"{-sum(gen[ildon, PG]) + sum(bus[inzld, PD]):7.1f} {-sum(gen[ildon, QG]) + sum(bus[inzld, QD]):7.1f}"
            )

        fd.write("\n----    ------  ------    ------  ------    ------  ------    ------  ------")
        fd.write(
            f"\nTot:   {-sum(gen[onld, PMIN]):7.1f} {-sum(Qlim[onld]):7.1f}   {-sum(gen[onld, PG]):7.1f} "
            f"{-sum(gen[onld, QG]):7.1f}   {sum(bus[nzld, PD]):7.1f} {sum(bus[nzld, QD]):7.1f}   "
            f"{-sum(gen[onld, PG]) + sum(bus[nzld, PD]):7.1f} {-sum(gen[onld, QG]) + sum(bus[nzld, QD]):7.1f}"
        )
        fd.write("\n")
        fd.write("\nArea      Shunt Inj        Branch      Series Losses      Net Export")
        fd.write("\n Num      MW     MVAr     Charging      MW     MVAr       MW     MVAr")
        fd.write("\n----    ------  ------    --------    ------  ------    ------  ------")
        for i in range(len(s_areas)):
            a = s_areas[i]
            inzsh = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, GS], bus[:, BS]))
            ibrch = find(
                (bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA] == a)
                & (bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA] == a)
                & branch[:, BR_STATUS].astype(bool)
            )
            in_tie = find(
                (bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA] != a)
                & (bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA] == a)
                & branch[:, BR_STATUS].astype(bool)
            )
            out_tie = find(
                (bus[e2i[branch[:, F_BUS].real.astype(int64)], BUS_AREA] == a)
                & (bus[e2i[branch[:, T_BUS].real.astype(int64)], BUS_AREA] != a)
                & branch[:, BR_STATUS].astype(bool)
            )
            fd.write(
                f"\n{a:3d}    {-sum(bus[inzsh, VM] ** 2 * bus[inzsh, GS]):7.1f}"
                f" {sum(bus[inzsh, VM] ** 2 * bus[inzsh, BS]):7.1f}"
                f"    {sum(fchg[ibrch]) + sum(tchg[ibrch]) + sum(fchg[out_tie]) + sum(tchg[in_tie]):7.1f}"
                f"    {sum(real(loss[ibrch])) + sum(real(loss[r_[in_tie, out_tie]])) / 2:7.2f}"
                f" {sum(imag(loss[ibrch])) + sum(imag(loss[r_[in_tie, out_tie]])) / 2:7.2f}   "
                f"{sum(branch[in_tie, PT]) + sum(branch[out_tie, PF]) - sum(real(loss[r_[in_tie, out_tie]])) / 2:7.1f}"
                f" {sum(branch[in_tie, QT]) + sum(branch[out_tie, QF]) - sum(imag(loss[r_[in_tie, out_tie]])) / 2:7.1f}"
            )

        fd.write("\n----    ------  ------    --------    ------  ------    ------  ------")
        fd.write(
            f"\nTot:   {-sum(bus[nzsh, VM] ** 2 * bus[nzsh, GS]):7.1f} {sum(bus[nzsh, VM] ** 2 * bus[nzsh, BS]):7.1f}"
            f"    {sum(fchg) + sum(tchg):7.1f}    {sum(real(loss)):7.2f} {sum(imag(loss)):7.2f}       -       -"
        )
        fd.write("\n")

    ## generator data
    if OUT_GEN:
        if isOPF:
            genlamP = bus[e2i[gen[:, GEN_BUS].astype(int64)], LAM_P]
            genlamQ = bus[e2i[gen[:, GEN_BUS].astype(int64)], LAM_Q]
        
        fd.write(f"\n{'':=^80}")
        fd.write(f"\n|     {'Generator Data':<73}|")
        fd.write(f"\n{'':=^80}")
        fd.write("\n Gen   Bus   Status     Pg        Qg   ")
        if isOPF:
            fd.write("   Lambda ($/MVA-hr)")
        fd.write("\n  #     #              (MW)     (MVAr) ")
        if isOPF:
            fd.write("     P         Q    ")
        fd.write("\n----  -----  ------  --------  --------")
        if isOPF:
            fd.write("  --------  --------")
        for k in range(len(ong)):
            i = ong[k]
            fd.write(f"\n{i:3d} {gen[i, GEN_BUS]:6d}     {gen[i, GEN_STATUS]:2d} ")
            if (gen[i, GEN_STATUS] > 0) & logical_or(gen[i, PG], gen[i, QG]):
                fd.write(f"{gen[i, PG]:10.2f}{gen[i, QG]:10.2f}")
            else:
                fd.write("       -         -  ")
            if isOPF:
                fd.write(f"{genlamP[i]:10.2f}{genlamQ[i]:10.2f}")
        
        fd.write("\n{'--------  --------':>39}")
        fd.write(f"\n            Total: {sum(gen[ong, PG]):9.2f}{sum(gen[ong, QG]):10.2f}")
        fd.write("\n")
        if any(onld + 1):
            fd.write(f"\n{'':=^80}")
            fd.write(f"\n|     {'Dispatchable Load Data':<73}|")
            fd.write(f"\n{'':=^80}")
            fd.write("\n Gen   Bus   Status     Pd        Qd   ")
            if isOPF:
                fd.write("   Lambda ($/MVA-hr)")
            fd.write("\n  #     #              (MW)     (MVAr) ")
            if isOPF:
                fd.write("     P         Q    ")
            fd.write("\n----  -----  ------  --------  --------")
            if isOPF:
                fd.write("  --------  --------")
            for k in range(len(onld)):
                i = onld[k]
                fd.write(f"\n{i:3d} {gen[i, GEN_BUS]:6d}     {gen[i, GEN_STATUS]:2d} ")
                if (gen[i, GEN_STATUS] > 0) & logical_or(gen[i, PG], gen[i, QG]):
                    fd.write(f"{-gen[i, PG]:10.2f}{-gen[i, QG]:10.2f}")
                else:
                    fd.write("       -         -  ")

                if isOPF:
                    fd.write(f"{genlamP[i]:10.2f}{genlamQ[i]:10.2f}")
            fd.write(f"\n{'--------  --------':>39}")
            fd.write(f"\n{'Total:':>18} {-sum(gen[onld, PG]):9.2f}{-sum(gen[onld, QG]):10.2f}")
            fd.write("\n")

    ## bus data
    if OUT_BUS:
        fd.write(f"\n{'':=^80}")
        fd.write(f"\n|     {'Bus Data':<73}|")
        fd.write(f"\n{'':=^80}")
        fd.write("\n Bus      Voltage          Generation             Load        ")
        if isOPF:
            fd.write("  Lambda($/MVA-hr)")
        fd.write("\n  #   Mag(pu) Ang(deg)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)")
        if isOPF:
            fd.write("     P        Q   ")
        fd.write("\n----- ------- --------  --------  --------  --------  --------")
        if isOPF:
            fd.write("  -------  -------")
        for i in range(nb):
            fd.write("\n{:5d}{:7.3f}{:9.3f}".format(*bus[i, [BUS_I, VM, VA]]))
            if bus[i, BUS_TYPE] == REF:
                fd.write("*")
            else:
                fd.write(" ")
            g = find((gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS] == bus[i, BUS_I]) & ~isload(gen))
            ld = find((gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS] == bus[i, BUS_I]) & isload(gen))
            if any(g + 1):
                fd.write(f"{sum(gen[g, PG]):9.2f}{sum(gen[g, QG]):10.2f}")
            else:
                fd.write(f"{'-':>7}{'-':>10}  ")

            if logical_or(bus[i, PD], bus[i, QD]) | any(ld + 1):
                if any(ld + 1):
                    fd.write(f"{bus[i, PD] - sum(gen[ld, PG]):10.2f}*{bus[i, QD] - sum(gen[ld, QG]):9.2f}*")
                else:
                    fd.write("{:10.2f}{:10.2f} ".format(*bus[i, [PD, QD]]))
            else:
                fd.write(f"{'-':>8}{'-':>10}   ")
            if isOPF:
                fd.write(f"{bus[i, LAM_P]:9.3f}")
                if abs(bus[i, LAM_Q]) > ptol:
                    fd.write(f"{bus[i, LAM_Q]:8.3f}")
                else:
                    fd.write("     -")
        fd.write(f"\n{'--------  --------  --------  --------':>62}")
        fd.write(
            f"\n{'Total:':>21} {sum(gen[ong, PG]):9.2f} {sum(gen[ong, QG]):9.2f} "
            f"{sum(bus[nzld, PD]) - sum(gen[onld, PG]):9.2f} {sum(bus[nzld, QD]) - sum(gen[onld, QG]):9.2f}"
        )

    ## branch data
    if OUT_BRANCH:
        fd.write(f"\n{'':=^80}")
        fd.write(f"\n|     {'Branch Data':<73}|")
        fd.write(f"\n{'':=^80}")
        fd.write("\nBrnch   From   To    From Bus Injection   To Bus Injection     Loss (I^2 * Z)  ")
        fd.write("\n  #     Bus    Bus    P (MW)   Q (MVAr)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)")
        fd.write("\n-----  -----  -----  --------  --------  --------  --------  --------  --------")
        for i in range(nl):
            fd.write(
                f"\n{i:4d}{branch[i, F_BUS].real:7d}{branch[i, T_BUS].real:7d}{branch[i, PF].real:10.2f}"
                f"{branch[i, QF].real:10.2f}{branch[i, PT].real:10.2f}{branch[i, QT].real:10.2f}{loss[i].real:10.3f}"
                f"{loss[i].imag:10.2f}"
            )
        fd.write(f"\n{'--------  --------':>79}")
        fd.write(f"\n{'Total:':>58}{sum(real(loss)):10.3f}{sum(imag(loss)):10.2f}")
        fd.write("\n")

    ##-----  constraint data  -----
    if isOPF:
        ctol = ppopt["OPF_VIOLATION"]  ## constraint violation tolerance
        ## voltage constraints
        if (not isDC) & (
            OUT_V_LIM
            == 2
            | (
                OUT_V_LIM
                == 1
                & (
                    any(bus[:, VM] < bus[:, VMIN] + ctol)
                    | any(bus[:, VM] > bus[:, VMAX] - ctol)
                    | any(bus[:, MU_VMIN] > ptol)
                    | any(bus[:, MU_VMAX] > ptol)
                )
            )
        ):
            fd.write(f"\n{'':=^80}")
            fd.write(f"\n|     {'Voltage Constraints':<73}|")
            fd.write(f"\n{'':=^80}")
            fd.write("\nBus #  Vmin mu    Vmin    |V|   Vmax    Vmax mu")
            fd.write("\n-----  --------   -----  -----  -----   --------")
            for i in range(nb):
                if (OUT_V_LIM == 2) | (
                    OUT_V_LIM
                    == 1
                    & (
                        (bus[i, VM] < bus[i, VMIN] + ctol)
                        | (bus[i, VM] > bus[i, VMAX] - ctol)
                        | (bus[i, MU_VMIN] > ptol)
                        | (bus[i, MU_VMAX] > ptol)
                    )
                ):
                    fd.write(f"\n{bus[i, BUS_I]:5d}")
                    if (bus[i, VM] < bus[i, VMIN] + ctol) | (bus[i, MU_VMIN] > ptol):
                        fd.write(f"{bus[i, MU_VMIN]:10.3f}")
                    else:
                        fd.write("      -   ")
                    
                    fd.write("{:8.3f}{:7.3f}{:7.3f}".format(*bus[i, [VMIN, VM, VMAX]]))
                    if (bus[i, VM] > bus[i, VMAX] - ctol) | (bus[i, MU_VMAX] > ptol):
                        fd.write(f"{bus[i, MU_VMAX]:10.3f}")
                    else:
                        fd.write("      -    ")
            fd.write("\n")

        ## generator P constraints
        if (
            (OUT_PG_LIM == 2)
            | (
                (OUT_PG_LIM == 1)
                & (
                    any(gen[ong, PG] < gen[ong, PMIN] + ctol)
                    | any(gen[ong, PG] > gen[ong, PMAX] - ctol)
                    | any(gen[ong, MU_PMIN] > ptol)
                    | any(gen[ong, MU_PMAX] > ptol)
                )
            )
            | (
                (not isDC)
                & (
                    (OUT_QG_LIM == 2)
                    | (
                        (OUT_QG_LIM == 1)
                        & (
                            any(gen[ong, QG] < gen[ong, QMIN] + ctol)
                            | any(gen[ong, QG] > gen[ong, QMAX] - ctol)
                            | any(gen[ong, MU_QMIN] > ptol)
                            | any(gen[ong, MU_QMAX] > ptol)
                        )
                    )
                )
            )
        ):
            fd.write(f"\n{'':=^80}")
            fd.write(f"\n|     {'Generation Constraints':<73}|")
            fd.write(f"\n{'':=^80}")

        if (OUT_PG_LIM == 2) | (
            (OUT_PG_LIM == 1)
            & (
                any(gen[ong, PG] < gen[ong, PMIN] + ctol)
                | any(gen[ong, PG] > gen[ong, PMAX] - ctol)
                | any(gen[ong, MU_PMIN] > ptol)
                | any(gen[ong, MU_PMAX] > ptol)
            )
        ):
            fd.write("\n Gen   Bus                Active Power Limits")
            fd.write("\n  #     #    Pmin mu    Pmin       Pg       Pmax    Pmax mu")
            fd.write("\n----  -----  -------  --------  --------  --------  -------")
            for k in range(len(ong)):
                i = ong[k]
                if (OUT_PG_LIM == 2) | (
                    (OUT_PG_LIM == 1)
                    & (
                        (gen[i, PG] < gen[i, PMIN] + ctol)
                        | (gen[i, PG] > gen[i, PMAX] - ctol)
                        | (gen[i, MU_PMIN] > ptol)
                        | (gen[i, MU_PMAX] > ptol)
                    )
                ):
                    fd.write(f"\n{i:4d}{gen[i, GEN_BUS]:6d} ")
                    if (gen[i, PG] < gen[i, PMIN] + ctol) | (gen[i, MU_PMIN] > ptol):
                        fd.write(f"{gen[i, MU_PMIN]:8.3f}")
                    else:
                        fd.write("     -  ")
                    if gen[i, PG]:
                        fd.write("{:10.2f}{:10.2f}{:10.2f}".format(*gen[i, [PMIN, PG, PMAX]]))
                    else:
                        fd.write("{:10.2f}       -  {:10.2f}".format(*gen[i, [PMIN, PMAX]]))
                    if (gen[i, PG] > gen[i, PMAX] - ctol) | (gen[i, MU_PMAX] > ptol):
                        fd.write(f"{gen[i, MU_PMAX]:9.3f}")
                    else:
                        fd.write("      -  ")
            fd.write("\n")

        ## generator Q constraints
        if (not isDC) & (
            (OUT_QG_LIM == 2)
            | (
                (OUT_QG_LIM == 1)
                & (
                    any(gen[ong, QG] < gen[ong, QMIN] + ctol)
                    | any(gen[ong, QG] > gen[ong, QMAX] - ctol)
                    | any(gen[ong, MU_QMIN] > ptol)
                    | any(gen[ong, MU_QMAX] > ptol)
                )
            )
        ):
            fd.write("\nGen  Bus              Reactive Power Limits")
            fd.write("\n #    #   Qmin mu    Qmin       Qg       Qmax    Qmax mu")
            fd.write("\n---  ---  -------  --------  --------  --------  -------")
            for k in range(len(ong)):
                i = ong[k]
                if (OUT_QG_LIM == 2) | (
                    (OUT_QG_LIM == 1)
                    & (
                        (gen[i, QG] < gen[i, QMIN] + ctol)
                        | (gen[i, QG] > gen[i, QMAX] - ctol)
                        | (gen[i, MU_QMIN] > ptol)
                        | (gen[i, MU_QMAX] > ptol)
                    )
                ):
                    fd.write(f"\n{i:3d}{gen[i, GEN_BUS]:5d}")
                    if (gen[i, QG] < gen[i, QMIN] + ctol) | (gen[i, MU_QMIN] > ptol):
                        fd.write(f"{gen[i, MU_QMIN]:8.3f}")
                    else:
                        fd.write("     -  ")
                    if gen[i, QG]:
                        fd.write("{:10.2f}{:10.2f}{:10.2f}".format(*tuple(gen[i, [QMIN, QG, QMAX]])))
                    else:
                        fd.write("{:10.2f}       -  {:10.2f}".format(*tuple(gen[i, [QMIN, QMAX]])))

                    if (gen[i, QG] > gen[i, QMAX] - ctol) | (gen[i, MU_QMAX] > ptol):
                        fd.write(f"{gen[i, MU_QMAX]:9.3f}")
                    else:
                        fd.write("      -  ")
            fd.write("\n")

        ## dispatchable load P constraints
        if (
            (OUT_PG_LIM == 2)
            | (OUT_QG_LIM == 2)
            | (
                (OUT_PG_LIM == 1)
                & (
                    any(gen[onld, PG] < gen[onld, PMIN] + ctol)
                    | any(gen[onld, PG] > gen[onld, PMAX] - ctol)
                    | any(gen[onld, MU_PMIN] > ptol)
                    | any(gen[onld, MU_PMAX] > ptol)
                )
            )
            | (
                (OUT_QG_LIM == 1)
                & (
                    any(gen[onld, QG] < gen[onld, QMIN] + ctol)
                    | any(gen[onld, QG] > gen[onld, QMAX] - ctol)
                    | any(gen[onld, MU_QMIN] > ptol)
                    | any(gen[onld, MU_QMAX] > ptol)
                )
            )
        ):
            fd.write(f"\n{'':=^80}")
            fd.write(f"\n|     {'Dispatchable Load Constraints':<73}|")
            fd.write(f"\n{'':=^80}")
        if (OUT_PG_LIM == 2) | (
            (OUT_PG_LIM == 1)
            & (
                any(gen[onld, PG] < gen[onld, PMIN] + ctol)
                | any(gen[onld, PG] > gen[onld, PMAX] - ctol)
                | any(gen[onld, MU_PMIN] > ptol)
                | any(gen[onld, MU_PMAX] > ptol)
            )
        ):
            fd.write("\nGen  Bus               Active Power Limits")
            fd.write("\n #    #   Pmin mu    Pmin       Pg       Pmax    Pmax mu")
            fd.write("\n---  ---  -------  --------  --------  --------  -------")
            for k in range(len(onld)):
                i = onld[k]
                if (OUT_PG_LIM == 2) | (
                    (OUT_PG_LIM == 1)
                    & (
                        (gen[i, PG] < gen[i, PMIN] + ctol)
                        | (gen[i, PG] > gen[i, PMAX] - ctol)
                        | (gen[i, MU_PMIN] > ptol)
                        | (gen[i, MU_PMAX] > ptol)
                    )
                ):
                    fd.write(f"\n{i:3d}{gen[i, GEN_BUS]:5d}")
                    if (gen[i, PG] < gen[i, PMIN] + ctol) | (gen[i, MU_PMIN] > ptol):
                        fd.write(f"{gen[i, MU_PMIN]:8.3f}")
                    else:
                        fd.write("     -  ")
                    if gen[i, PG]:
                        fd.write("{:10.2f}{:10.2f}{:10.2f}".format(*gen[i, [PMIN, PG, PMAX]]))
                    else:
                        fd.write("{:10.2f}       -  {:10.2f}".format(*gen[i, [PMIN, PMAX]]))

                    if (gen[i, PG] > gen[i, PMAX] - ctol) | (gen[i, MU_PMAX] > ptol):
                        fd.write(f"{gen[i, MU_PMAX]:9.3f}")
                    else:
                        fd.write("      -  ")
            fd.write("\n")

        ## dispatchable load Q constraints
        if (not isDC) & (
            (OUT_QG_LIM == 2)
            | (
                (OUT_QG_LIM == 1)
                & (
                    any(gen[onld, QG] < gen[onld, QMIN] + ctol)
                    | any(gen[onld, QG] > gen[onld, QMAX] - ctol)
                    | any(gen[onld, MU_QMIN] > ptol)
                    | any(gen[onld, MU_QMAX] > ptol)
                )
            )
        ):
            fd.write("\nGen  Bus              Reactive Power Limits")
            fd.write("\n #    #   Qmin mu    Qmin       Qg       Qmax    Qmax mu")
            fd.write("\n---  ---  -------  --------  --------  --------  -------")
            for k in range(len(onld)):
                i = onld[k]
                if (OUT_QG_LIM == 2) | (
                    (OUT_QG_LIM == 1)
                    & (
                        (gen[i, QG] < gen[i, QMIN] + ctol)
                        | (gen[i, QG] > gen[i, QMAX] - ctol)
                        | (gen[i, MU_QMIN] > ptol)
                        | (gen[i, MU_QMAX] > ptol)
                    )
                ):
                    fd.write(f"\n{i:3d}{gen(i, GEN_BUS):5d}")
                    if (gen[i, QG] < gen[i, QMIN] + ctol) | (gen[i, MU_QMIN] > ptol):
                        fd.write(f"{gen[i, MU_QMIN]:8.3f}")
                    else:
                        fd.write("     -  ")

                    if gen[i, QG]:
                        fd.write("{:10.2f}{:10.2f}{:10.2f}".format(*gen[i, [QMIN, QG, QMAX]]))
                    else:
                        fd.write("{:10.2f}       -  {:10.2f}".format(*gen[i, [QMIN, QMAX]]))

                    if (gen[i, QG] > gen[i, QMAX] - ctol) | (gen[i, MU_QMAX] > ptol):
                        fd.write(f"{gen[i, MU_QMAX]:9.3f}")
                    else:
                        fd.write("      -  ")
            fd.write("\n")

        ## line flow constraints
        if (ppopt["OPF_FLOW_LIM"] == 1) | isDC:  ## P limit
            Ff = branch[:, PF]
            Ft = branch[:, PT]
            strg = "\n  #     Bus    Pf  mu     Pf      |Pmax|      Pt      Pt  mu   Bus"
        elif ppopt["OPF_FLOW_LIM"] == 2:  ## |I| limit
            Ff = abs((branch[:, PF] + 1j * branch[:, QF]) / V[e2i[branch[:, F_BUS].real.astype(int64)]])
            Ft = abs((branch[:, PT] + 1j * branch[:, QT]) / V[e2i[branch[:, T_BUS].real.astype(int64)]])
            strg = "\n  #     Bus   |If| mu    |If|     |Imax|     |It|    |It| mu   Bus"
        else:  ## |S| limit
            Ff = abs(branch[:, PF] + 1j * branch[:, QF])
            Ft = abs(branch[:, PT] + 1j * branch[:, QT])
            strg = "\n  #     Bus   |Sf| mu    |Sf|     |Smax|     |St|    |St| mu   Bus"

        if (OUT_LINE_LIM == 2) | (
            (OUT_LINE_LIM == 1)
            & (
                any((branch[:, RATE_A] != 0) & (abs(Ff) > branch[:, RATE_A] - ctol))
                | any((branch[:, RATE_A] != 0) & (abs(Ft) > branch[:, RATE_A] - ctol))
                | any(branch[:, MU_SF] > ptol)
                | any(branch[:, MU_ST] > ptol)
            )
        ):
            fd.write(f"\n{'':=^80}")
            fd.write(f"\n|     {'Branch Flow Constraints':<73}|")
            fd.write(f"\n{'':=^80}")
            fd.write('\nBranch   From     "From" End        Limit       "To" End        To')
            fd.write(strg)
            fd.write("\n-----  -----  -------  --------  --------  --------  -------  -----")
            for i in range(nl):
                if (OUT_LINE_LIM == 2) | (
                    (OUT_LINE_LIM == 1)
                    & (
                        ((branch[i, RATE_A] != 0) & (abs(Ff[i]) > branch[i, RATE_A] - ctol))
                        | ((branch[i, RATE_A] != 0) & (abs(Ft[i]) > branch[i, RATE_A] - ctol))
                        | (branch[i, MU_SF] > ptol)
                        | (branch[i, MU_ST] > ptol)
                    )
                ):
                    fd.write(f"\n{i:4d}{branch[i, F_BUS].real:7d}")
                    if (Ff[i] > branch[i, RATE_A] - ctol) | (branch[i, MU_SF] > ptol):
                        fd.write(f"{branch[i, MU_SF].real:10.3f}")
                    else:
                        fd.write("      -   ")
                    
                    fd.write(f"{Ff[i]:9.2f}{branch[i, RATE_A].real:10.2f}{Ft[i]:10.2f}")
                    if (Ft[i] > branch[i, RATE_A] - ctol) | (branch[i, MU_ST] > ptol):
                        fd.write(f"{branch[i, MU_ST].real:10.3f}")
                    else:
                        fd.write("      -   ")
                    fd.write(f"{branch[i, T_BUS].real:6d}")
            fd.write("\n")

    ## execute userfcn callbacks for 'printpf' stage
    if have_results_struct and "userfcn" in results:
        if not isOPF:  ## turn off option for all constraints if it isn't an OPF
            ppopt = ppoption(ppopt, "OUT_ALL_LIM", 0)
        run_userfcn(results["userfcn"], "printpf", results, fd, ppopt)

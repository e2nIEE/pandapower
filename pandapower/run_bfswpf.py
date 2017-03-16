import numpy as np
import scipy as sp

from time import time  # alternatively use import timeit.default_timer as time

from scipy.sparse import csr_matrix, csgraph

from pandapower.auxiliary import ppException

from pandapower.pypower_extensions.pfsoln import pfsoln
from pandapower.pypower_extensions.bustypes import bustypes
from pandapower.pypower_extensions.runpf import _import_numba_extensions_if_flag_is_true, _get_pf_variables_from_ppci

from pypower.makeSbus import makeSbus
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VM, VA, REF, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, PF, QF, PT, QT, TAP, BR_STATUS, SHIFT
from pypower.idx_gen import GEN_BUS, PG, QG, PMAX, PMIN, QMAX, QMIN, GEN_STATUS, VG

class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


def _reindex_bus_ppc(ppc, bus_ind_dict):
    """
    reindexing buses according to dictionary old_bus_index -> new_bus_index
    :param ppc: matpower-type power system
    :param bus_ind_dict:  dictionary for bus reindexing
    :return: ppc with bus indices updated
    """
    ppc_bfs = ppc.copy()
    buses = ppc_bfs['bus'].copy()
    branches = ppc_bfs['branch'].copy()
    generators = ppc_bfs['gen'].copy()

    buses[:, BUS_I] = [bus_ind_dict[bus] for bus in buses[:, BUS_I]]

    branches[:, F_BUS] = [bus_ind_dict[bus] for bus in branches[:, F_BUS]]
    branches[:, T_BUS] = [bus_ind_dict[bus] for bus in branches[:, T_BUS]]
    branches[:, F_BUS:T_BUS + 1] = np.sort(branches[:, F_BUS:T_BUS + 1], axis=1)  # sort in order to T_BUS > F_BUS

    generators[:, GEN_BUS] = [bus_ind_dict[bus] for bus in generators[:, GEN_BUS]]

    # sort buses, branches and generators according to new numbering
    ppc_bfs['bus'] = buses[np.argsort(buses[:, BUS_I])]
    ppc_bfs['branch'] = branches[np.lexsort((branches[:, T_BUS], branches[:, F_BUS]))]
    ppc_bfs['gen'] = generators[np.argsort(generators[:, GEN_BUS])]

    return ppc_bfs


def _cut_ppc(ppc, buses):
    ppc_cut = ppc.copy()

    ppc_cut['bus'] = ppc_cut['bus'][buses, :]

    branch_in = (np.in1d(ppc_cut['branch'][:, F_BUS], buses) &
                   np.in1d(ppc_cut['branch'][:, T_BUS], buses) )
    ppc_cut['branch'] = ppc_cut['branch'][branch_in, :]

    gen_in = np.in1d(ppc['gen'][:, GEN_BUS], buses)
    ppc_cut['gen'] = ppc_cut['gen'][gen_in, :]

    return ppc_cut




def _bibc_bcbv(ppc, graph):
    """
    performs depth-first-search bus ordering and creates Direct Load Flow (DLF) matrix
    which establishes direct relation between bus current injections and voltage drops from each bus to the root bus

    :param ppc: matpower-type case data
    :return: DLF matrix DLF = BIBC * BCBV where
                    BIBC - Bus Injection to Branch-Current
                    BCBV - Branch-Current to Bus-Voltage
            ppc with bfs ordering
            original bus names bfs ordered (used to convert voltage array back to normal)
    """

    ppci = ppc
    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    nobus = bus.shape[0]
    nobranch = branch.shape[0]

    # reference bus is assumed as root bus for a radial network
    ref = bus[bus[:,BUS_TYPE]==3, BUS_I]
    root_bus = ref[0]

    G = graph # network graph


    # ordering buses according to breadth-first-search (bfs)
    buses_ordered_bfs, predecs_bfs = csgraph.breadth_first_order(G, ref, directed=False, return_predecessors=True)
    branches_ordered_bfs = zip(predecs_bfs[buses_ordered_bfs[1:]], buses_ordered_bfs[1:])
    # buses_bfs_dict = dict(zip(buses_ordered_bfs, range(0, nobus)))  # old to new bus names dictionary
    G_tree = csgraph.breadth_first_tree(G, ref, directed=False)

    # identify loops if graph is not a tree
    branches_loops = []
    if G_tree.nnz < G.nnz:
        G_nnzs = G.nonzero()
        G_tree_nnzs = G_tree.nonzero()
        branches_loops = set(zip(G_nnzs[0], G_nnzs[1])) - set(zip(G_tree_nnzs[0], G_tree_nnzs[1]))

    # renaming buses in graph and in ppc
    # G = nx.relabel_nodes(G, buses_bfs_dict)
    # root_bus = buses_bfs_dict[root_bus]
    # ppc_bfs = _reindex_bus_ppc(ppci, buses_bfs_dict)
    ppc_bfs = ppci.copy()
    # ordered list of branches


    # searching leaves of the tree
    # succ = nx.bfs_successors(G, root_bus)
    # leaves = set(G.nodes()) - set(succ.keys())

    # dictionary with impedance values keyed by branch tuple (frombus, tobus)
    branches_lst = list(zip(ppc_bfs['branch'][:, F_BUS].real.astype(int), ppc_bfs['branch'][:, T_BUS].real.astype(int)))
    tap = ppc_bfs['branch'][:, TAP]     # * np.exp(1j * np.pi / 180 * branch[:, SHIFT])
    z_ser = (ppc_bfs['branch'][:, BR_R].real + 1j * ppc_bfs['branch'][:, BR_X].real) * tap  # series impedance
    z_brch_dict = dict(zip(branches_lst, z_ser))

    # #------ building BIBC and BCBV martrices ------

    # order branches for BIBC and BCBV matrices and set loop-closing branches to the end
    branches_ind_dict = dict(zip(branches_ordered_bfs, range(0, nobus - 1)))
    branches_ind_dict.update(dict(zip(branches_loops, range(nobus - 1, nobranch))))  # add loop-closing branches

    rowi_BIBC = []
    coli_BIBC = []
    data_BIBC = []
    data_BCBV = []

    brchi = 0
    for brch in branches_ordered_bfs:
        # TODO check if this is faster with connected components instead of bfs
        tree_down, predecs = csgraph.breadth_first_order(G_tree, brch[1], directed=True, return_predecessors=True)
        if len(tree_down) == 1:  # If at leaf
            pass
        if brch in z_brch_dict:
            z_br = z_brch_dict[brch]
        else:
            z_br = z_brch_dict[brch[::-1]]
        rowi_BIBC += [branches_ind_dict[brch]] * len(tree_down)
        coli_BIBC += list(tree_down)
        data_BCBV += [z_br] * len(tree_down)
        data_BIBC += [1] * len(tree_down)



    for loop_i, brch_loop in enumerate(branches_loops):
        path_lens, path_preds = csgraph.shortest_path(G_tree, directed=False,
                                                      indices=brch_loop, return_predecessors=True)
        init, end = brch_loop
        loop = [end]
        while init != end:
            end = path_preds[0, end]
            loop.append(end)

        loop_size = len(loop)
        coli_BIBC += [nobus + loop_i] * loop_size
        for i in range(len(loop)):
            brch = (loop[i-1], loop[i])
            if np.argwhere(buses_ordered_bfs == brch[0]) < np.argwhere(buses_ordered_bfs == brch[1]):
                brch_direct = 1
            else:
                brch_direct = -1
            data_BIBC.append(brch_direct)

            if brch in branches_ind_dict:
                rowi_BIBC.append(branches_ind_dict[brch])
            else:
                rowi_BIBC.append(branches_ind_dict[brch[::-1]])

            if brch in z_brch_dict:
                data_BCBV.append(z_brch_dict[brch] * brch_direct)
            else:
                data_BCBV.append(z_brch_dict[brch[::-1]] * brch_direct)

            brchi += 1

    # construction of the BIBC matrix
    # column indices correspond to buses: assuming root bus is always 0 after ordering indices are subtracted by 1
    BIBC = csr_matrix((data_BIBC, (rowi_BIBC, np.array(coli_BIBC) - 1)),
                  shape=(nobranch, nobranch))
    BCBV = csr_matrix((data_BCBV, (rowi_BIBC, np.array(coli_BIBC) - 1)),
                  shape=(nobranch, nobranch)).transpose()

    if BCBV.shape[0] > nobus - 1:  # if nbrch > nobus - 1 -> network has loops
        DLF_loop = BCBV * BIBC
        # DLF = [A  M.T ]
        #       [M  N   ]
        A = DLF_loop[0:nobus - 1, 0:nobus - 1]
        M = DLF_loop[nobus - 1:, 0:nobus - 1]
        N = DLF_loop[nobus - 1:, nobus - 1:].A
        # considering the fact that number of loops is relatively small, N matrix is expected to be small and dense
        # ...in that case dense version is more efficient, i.e. N is transformed to dense and
        # inverted using sp.linalg.inv(N)
        DLF = A - M.T * csr_matrix(sp.linalg.inv(N)) * M  # Kron's Reduction
    else:  # no loops -> radial network
        DLF = BCBV * BIBC
    return DLF, ppc_bfs, buses_ordered_bfs



def _bfswpf(DLF, bus, gen, branch, baseMVA, Ybus, bus_ord_bfsw, Sbus, V0, ref, pv, pq,
         enforce_q_lims, tolerance_kva, max_iteration, **kwargs):
    """
    distribution power flow solution according to [1]
    :param DLF: direct-Load-Flow matrix which relates bus current injections to voltage drops from the root bus

    :param bus: buses martix
    :param gen: generators matrix
    :param branch: branches matrix
    :param baseMVA:
    :param Ybus: bus admittance matrix
    :param Sbus: vector of power injections
    :param V0: initial voltage state vector
    :param ref: reference bus index
    :param pv: PV buses indices
    :param pq: PQ buses indices

    :return: power flow result

    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions", IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.
    """
    # setting options
    tolerance_mva = tolerance_kva * 1e-3
    max_it = max_iteration  # maximum iterations
    verbose = kwargs["VERBOSE"]     # verbose is set in run._runpppf() #
    # tolerance for the inner loop for PV nodes
    if 'tolerance_kva_pv' in kwargs:
        tol_mva_inner = kwargs['tolerance_kva_pv'] * 1e-3
    else:
        tol_mva_inner = 1.e-2

    if 'max_iter_pv' in kwargs:
        max_iter_pv = kwargs['max_iter_pv']
    else:
        max_iter_pv = 20

    nobus = bus.shape[0]
    ngen = gen.shape[0]

    mask_root = ~ (bus[:, BUS_TYPE] == 3)  # mask for eliminating root bus
    root_bus_i = ref
    Vref = V0[ref]

    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u. and Qsh is the reactive power injected by
    # the shunt at V = 1.0 p.u. then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # vector of shunt admittances
    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    # Line charging susceptance BR_B is also added as shunt admittance:
    # summation of charging susceptances per each bus
    stat = branch[:, BR_STATUS]  ## ones at in-service branches
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])
    ysh = (- branch[:, BR_B].imag + 1j*(branch[:, BR_B].real)) / 2
    tap = branch[:, TAP]    # * np.exp(1j * np.pi / 180 * branch[:, SHIFT])

    ysh_f = Ys * (1-tap)/(tap * np.conj(tap)) + ysh/(tap * np.conj(tap))
    ysh_t = Ys * (tap-1)/tap + ysh

    Gch = (np.bincount(branch[:, F_BUS].real.astype(int), weights=ysh_f.real, minlength=nobus) +
           np.bincount(branch[:, T_BUS].real.astype(int), weights=ysh_t.real, minlength=nobus))
    Bch = (np.bincount(branch[:, F_BUS].real.astype(int), weights=ysh_f.imag, minlength=nobus) +
           np.bincount(branch[:, T_BUS].real.astype(int), weights=ysh_t.imag, minlength=nobus))

    Ysh += Gch + 1j * Bch


    # detect generators on PV buses which have status ON
    gen_pv = np.in1d(gen[:, GEN_BUS], pv) & (gen[:, GEN_STATUS] > 0)
    qg_lim = np.zeros(ngen, dtype=bool)   #initialize generators which violated Q limits

    V_iter = V0[mask_root].copy()  # initial voltage vector without root bus
    V = V0.copy()
    Iinj = np.conj(Sbus / V) - Ysh * V  # Initial current injections

    n_iter = 0
    converged = 0

    if verbose:
        print(' -- AC Power Flow Backward/Forward sweep\n')

    while not converged and n_iter < max_it:
        n_iter_inner = 0
        n_iter += 1

        deltaV = DLF * Iinj[mask_root]
        V_iter = np.ones(nobus - 1) * Vref + deltaV
        # ##
        # inner loop for considering PV buses
        inner_loop_converged = False

        while not inner_loop_converged and len(pv) > 0:

            pvi = pv - 1  # internal PV buses indices, assuming reference node is always 0

            Vmis = (np.abs(gen[gen_pv, VG])) ** 2 - (np.abs(V_iter[pvi])) ** 2
            dQ = (Vmis / (2 * DLF[pvi, pvi].A1.imag)).flatten()

            gen[gen_pv, QG] += dQ

            if enforce_q_lims:  #check Q violation limits
                ## find gens with violated Q constraints
                qg_max_lim = (gen[:, QG] > gen[:, QMAX]) & gen_pv
                qg_min_lim = (gen[:, QG] < gen[:, QMIN]) & gen_pv

                if qg_min_lim.any():
                    gen[qg_min_lim, QG] = gen[qg_min_lim, QMIN]
                    bus[gen[qg_min_lim, GEN_BUS].astype(int), BUS_TYPE] = 1 # convert to PQ bus

                if qg_max_lim.any():
                    gen[qg_max_lim, QG] = gen[qg_max_lim, QMAX]
                    bus[gen[qg_max_lim, GEN_BUS].astype(int), BUS_TYPE] = 1  # convert to PQ bus

                # TODO: correct: once all the PV buses are converted to PQ buses, conversion back to PV is not possible
                qg_lim_new = qg_min_lim | qg_max_lim
                if qg_lim_new.any():
                    pq2pv = (qg_lim != qg_lim_new) &  qg_lim
                    # convert PQ to PV bus
                    if pq2pv.any():
                        bus[gen[qg_max_lim, GEN_BUS].astype(int), BUS_TYPE] = 2  # convert to PV bus

                    qg_lim = qg_lim_new.copy()
                    ref, pv, pq = bustypes(bus, gen)


            Sbus = makeSbus(baseMVA, bus, gen)
            V = np.insert(V_iter, root_bus_i, Vref)
            Iinj = np.conj(Sbus / V) - Ysh * V
            deltaV = DLF * Iinj[mask_root]
            V_iter = np.ones(nobus - 1) * V0[root_bus_i] + deltaV

            if n_iter_inner > max_iter_pv:
                raise LoadflowNotConverged(" FBSW Power Flow did not converge - inner iterations for PV nodes "
                                           "reached maximum value of {0}!".format(max_iter_pv))

            n_iter_inner += 1

            if np.all(np.abs(dQ) < tol_mva_inner):  # inner loop termination criterion
                inner_loop_converged = True

        # testing termination criterion -
        V = np.insert(V_iter, root_bus_i, Vref)
        mis = V * np.conj(Ybus * V) - Sbus
        F = np.r_[mis[pv].real,
                  mis[pq].real,
                  mis[pq].imag]

        # check tolerance
        normF = np.linalg.norm(F, np.Inf)
        if normF < tolerance_mva:
            converged = 1
            if verbose:
                print("\nFwd-back sweep power flow converged in "
                                 "{0} iterations.\n".format(n_iter))
        elif n_iter == max_it:
            raise LoadflowNotConverged(" FBSW Power Flow did not converge - "
                                       "reached maximum iterations = {0}!".format(max_it))

        # updating injected currents
        Iinj = np.conj(Sbus / V) - Ysh * V

    return V, converged



def _get_options(options):
    enforce_q_lims = options['enforce_q_lims']
    tolerance_kva = options['tolerance_kva']
    max_iteration = options['max_iteration']
    calculate_voltage_angles = options['calculate_voltage_angles']
    numba = options["numba"]

    return enforce_q_lims, tolerance_kva, max_iteration, calculate_voltage_angles, numba

def _run_bfswpf(ppc, options, **kwargs):
    """
    SPARSE version of distribution power flow solution according to [1]
    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions", IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.

    :param ppc: matpower-style case data
    :return: results (pypower style), success (flag about PF convergence)
    """
    time_start = time()  # starting pf calculation timing


    tap_shift = ppc['branch'][:, SHIFT].copy().real


    enforce_q_lims, tolerance_kva, max_iteration, calculate_voltage_angles, numba = _get_options(options)

    numba, makeYbus = _import_numba_extensions_if_flag_is_true(numba)

    ppci = ppc

    baseMVA, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    nobus = bus.shape[0]
    nobranch = branch.shape[0]
    # generate results for original bus ordering
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    # get bus index lists of each type of bus
    refs, pv, pq = bustypes(bus, gen)


    # creating networkx graph from list of branches
    bus_from = branch[:, F_BUS].real.astype(int)
    bus_to = branch[:, T_BUS].real.astype(int)
    G = csr_matrix((np.ones(nobranch), (bus_from, bus_to)),
                                      shape=(nobus, nobus))

    V_final = np.zeros(nobus,dtype=complex)
    for subi,ref in enumerate(refs):
        G_tree = csgraph.breadth_first_tree(G, ref, directed=False)
        if len(refs) > 1:
            nographs, subgraph_i = csgraph.connected_components(G, directed=False, return_labels=True)
            nodes = bus[subgraph_i == subi, BUS_I].real.astype(int)
            nobus_sub = len(nodes)
            ppci_sub = _cut_ppc(ppci, nodes)
            branch_sub_mask = np.in1d(bus_from,nodes) & np.in1d(bus_to,nodes)
            nobranch_sub = branch_sub_mask.sum()
            buses_ordered_bfs, predecs_bfs = csgraph.breadth_first_order(G, ref, directed=False,
                                                                         return_predecessors=True)
            buses_bfs_dict = dict(zip(buses_ordered_bfs, range(0, nobus)))  # old to new bus names dictionary
            G_sub = csr_matrix((np.ones(nobranch_sub),
                                (bus_from[branch_sub_mask], bus_to[branch_sub_mask])),
                               shape=(nobus_sub, nobus_sub))

        else:
            ppci_sub = ppci
            G_sub = G

        # depth-first-search bus ordering and generating Direct Load Flow matrix DLF = BCBV * BIBC
        DLF, ppc_bfsw, buses_ordered_bfs = _bibc_bcbv(ppci_sub, G_sub)
        ppc_bfsw['branch'][:, SHIFT] = 0

        baseMVA_bfsw, bus_bfsw, gen_bfsw, branch_bfsw, ref_bfsw, pv_bfsw, pq_bfsw,\
        on, gbus, V0 = _get_pf_variables_from_ppci(ppc_bfsw)


        Sbus_bfsw = makeSbus(baseMVA_bfsw, bus_bfsw, gen_bfsw)

        Ybus_bfsw, Yf_bfsw, Yt_bfsw = makeYbus(baseMVA_bfsw, bus_bfsw, branch_bfsw)

        # #-----  run the power flow  -----
        V_final, success = _bfswpf(DLF, bus_bfsw, gen_bfsw, branch_bfsw, baseMVA, Ybus_bfsw, buses_ordered_bfs,
                                   Sbus_bfsw, V0, ref_bfsw, pv_bfsw, pq_bfsw,
                                   enforce_q_lims, tolerance_kva, max_iteration, **kwargs)

        # V_final = V_res  # return bus voltages in original bus order
        # TODO: find the better way to consider transformer phase shift and remove this workaround
        if calculate_voltage_angles:
            brch_shift_mask = tap_shift!=0
            trafos_shift = dict(list(zip(list(zip(ppc_bfsw['branch'][brch_shift_mask, F_BUS].real.astype(int),
                                         ppc_bfsw['branch'][brch_shift_mask, T_BUS].real.astype(int))),
                                         tap_shift[brch_shift_mask])))
            for trafo_ind, shift_degree in trafos_shift.iteritems():
                if (np.argwhere(buses_ordered_bfs == trafo_ind[0]) <
                        np.argwhere(buses_ordered_bfs == trafo_ind[1])):
                    lv_bus = trafo_ind[1]
                    shift_degree *= -1
                else:
                    lv_bus = trafo_ind[0]

                buses_shifted_from_root = csgraph.breadth_first_order(G_tree, lv_bus, directed=True, return_predecessors=False)
                V_final[buses_shifted_from_root] *= np.exp(1j * np.pi / 180 * shift_degree)

    # #----- output results to ppc ------
    ppci["et"] = time() - time_start    # pf time end



    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V_final, refs, pv, pq)

    ppci["success"] = success

    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch

    return ppci, success



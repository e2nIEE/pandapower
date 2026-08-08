from pandapower.pypower.idx_bus import BUS_I, VM, VA
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS, VG
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.bustypes import bustypes
from numpy import abs as np_abs, flatnonzero as find, pi, exp, int64, hstack, zeros, float64


class ZeroBusVoltageMagnitude(ppException):
    """
    Raised when a bus that carries an in-service generator has an initial voltage
    magnitude of 0. Continuing would silently produce NaN values (via a 0-division)
    instead of a clear, actionable error.
    """
    pass


def _get_pf_variables_from_ppci(ppci, vsc_ref=False):
    ## default arguments
    if ppci is None:
        raise ValueError('ppci is empty')
    # ppopt = ppoption(ppopt)

    # get data for calc
    bus, gen, vsc = ppci["bus"], ppci["gen"], ppci["vsc"]
    branch = ppci["branch"]
    br_shape = branch.shape
    if br_shape[1] < branch_cols:
        branch = hstack([branch, zeros(shape=(br_shape[0], branch_cols - br_shape[1]), dtype=float64)])

    ## get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen, vsc if vsc_ref else None)

    ## generator info
    on = find(gen[:, GEN_STATUS] > 0)  ## which generators are on?
    gbus = gen[on, GEN_BUS].astype(int64)  ## what buses are they at?

    ## initial state
    # V0    = ones(bus.shape[0])            ## flat start
    V0 = bus[:, VM] * exp(1j * pi / 180. * bus[:, VA])
    vm_at_gbus = np_abs(V0[gbus])
    zero_vm = vm_at_gbus == 0
    if zero_vm.any():
        bad_bus_ids = bus[gbus[zero_vm], BUS_I].astype(int64).tolist()
        raise ZeroBusVoltageMagnitude(
            "Cannot initialize power flow: voltage magnitude is 0 at bus(es) "
            f"{bad_bus_ids}, which carry an in-service generator. This usually means "
            "these buses are isolated/disconnected from the rest of the network, or "
            "were not assigned a valid initial voltage. Check net.bus.in_service and "
            "network connectivity for these buses, or try init='flat'."
        )
    V0[gbus] = gen[on, VG] / vm_at_gbus * V0[gbus]

    ref_gens = ppci["internal"]["ref_gens"]
    return ppci["baseMVA"], bus, gen, branch, ppci["svc"], ppci["tcsc"], ppci["ssc"], ppci["vsc"], \
        ref, pv, pq, on, gbus, V0, ref_gens


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, iterations, et):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    ppci["success"] = bool(success)
    ppci["iterations"] = iterations
    ppci["et"] = et
    return ppci

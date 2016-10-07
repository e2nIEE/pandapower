__author__ = "fmeier"
import numpy as np
import pandapower as pp
from pandapower.build_branch import _build_branch_mpc, _switch_branches
from pandapower.build_bus import _build_bus_mpc, _calc_shunts_and_add_on_mpc
from pandapower.build_gen import _build_gen_mpc
from pandapower.run import _set_out_of_service
from pypower.idx_bus import BUS_I, BASE_KV, BUS_TYPE, PD, QD, GS, BS
from pypower.idx_gen import QMIN, QMAX, PMIN, PMAX, GEN_STATUS, GEN_BUS, PG, VG, QG
from pypower.idx_bus import PV, REF, VA, VM, BUS_TYPE, PQ, VMAX, VMIN
from pypower.idx_brch import RATE_A
import pypower.ppoption as ppoption
import numpy.core.numeric as ncn
from pandapower.auxiliary import _sum_by_group
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from scipy import sparse

# TODO: check for unused imports


def _pd2mpc_opf(net, gen_is, eg_is, sg_is):
    """ we need to put the sgens into the gen table instead of the bsu table so we need to change 
    _pd2mpc a little to get the mpc we need for the OPF
    """

    mpc = {"baseMVA": 1.,
           "version": 2,
           "bus": np.array([], dtype=float),
           "branch": np.array([], dtype=np.complex128),
           "gen": np.array([], dtype=float),
           "gencost": np.array([], dtype=float)}

    calculate_voltage_angles = False
    enforce_q_lims = False
    trafo_model = "t"
    bus_lookup = _build_bus_mpc(net, mpc, calculate_voltage_angles, gen_is, eg_is)

    mpc["bus"][:, VMAX] = net["bus"][net.bus.in_service==True]["max_vm_pu"]
    mpc["bus"][:, VMIN] = net["bus"][net.bus.in_service==True]["min_vm_pu"]
    


    _build_gen_opf(net, mpc,  gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is)
    _build_branch_mpc(net, mpc, bus_lookup, calculate_voltage_angles, trafo_model)

    mpc["bus"][mpc["bus"][:, BUS_TYPE] == REF, VMAX] = mpc["bus"][mpc["bus"][:, BUS_TYPE] == REF, VM]
    mpc["bus"][mpc["bus"][:, BUS_TYPE] == REF, VMIN] = mpc["bus"][mpc["bus"][:, BUS_TYPE] == REF, VM]

    if len(net["line"]) > 0:
        mpc["branch"][:len(net["line"]), RATE_A] = net.line.max_loading_percent  / 100 * \
                                  net.line.imax_ka * net.bus.vn_kv[net.line.from_bus].values * np.sqrt(3)
    if len(net["trafo"]) > 0:
        mpc["branch"][len(net["line"]):(len(net["line"]) + len(net["trafo"])), RATE_A] = \
        net.trafo.max_loading_percent  / 100 * net.trafo.sn_kva / 1000
    # at 1.0 p.u. the maximum apparent power [MVA] equals the maximum current [MA]

    _calc_shunts_and_add_on_mpc(net, mpc, bus_lookup)
    _calc_loads_and_add_opf(net, mpc, bus_lookup)
    _switch_branches(net, mpc, bus_lookup)
    _set_out_of_service(net, mpc)

    return mpc, bus_lookup


def _make_objective(mpc, ppopt, objectivetype="maxp"):
    """ Implementaton of diverse objective functions for the OPF of the Form C{N}, C{fparm}, 
        C{H} and C{Cw}

    * mpc . Matpower case of the net
    * objectivetype - string with name of objective function


    ** "maxp" - linear costs of the form  I*p. p represents the active power values of the
                    generators. This then basically is this:
                    max p subject to {vm_min<u<vm_max,min_p_kw<p<pmax,qmin<q<qmax,i<imax
                    
    """
    ng = len(mpc["gen"])  # -
    nref = sum(mpc["bus"][:, BUS_TYPE] == REF)

    if objectivetype == "maxp":

        mpc["gencost"] = np.zeros((ng, 8), dtype=float)
        mpc["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 100, 0])
        mpc["gencost"][nref:ng, :] = np.array([1, 0, 0, 2, 0, 100, 100, 0])

        ppopt = ppoption.ppoption(ppopt,  OPF_FLOW_LIM=2, OPF_VIOLATION = 1e-1, OUT_LIM_LINE = 2, 
                                  PDIPM_GRADTOL = 1e-10, PDIPM_COMPTOL =1e-10, PDIPM_COSTTOL=1e-10)

        
    if objectivetype == "Einspeisemanagement":
        pass
    
    
    
    
    return mpc, ppopt


def _build_gen_opf(net, mpc, gen_is, eg_is, bus_lookup, calculate_voltage_angles, sg_is):
    '''
    Takes the empty mpc network and fills it with the gen values. The gen
    datatype will be float afterwards.

    **INPUT**:
        **net** -The Pandapower format network

        **mpc** - The PYPOWER format network to fill in values
    '''
    eg_end = len(eg_is)
    gen_end = eg_end + len(gen_is)
    sg_end = gen_end + len(sg_is)

    q_lim_default = 1e9  # which is 1000 TW - should be enough for distribution grids.
    p_lim_default = 1e9

    # initialize generator matrix
    mpc["gen"] = np.zeros(shape=(sg_end, 21), dtype=float)
    mpc["gen"][:] = np.array([0, 0, 0, q_lim_default, -q_lim_default, 1., 1., 1, p_lim_default,
                              -p_lim_default, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # add sgens first so pv bus types won't be overwritten
    if sg_end > gen_end:
        mpc["gen"][gen_end:sg_end, GEN_BUS] = pp.get_indices(sg_is["bus"].values, bus_lookup)
        mpc["gen"][gen_end:sg_end, PG] = - sg_is["p_kw"].values * 1e-3 * sg_is["scaling"].values
        mpc["gen"][gen_end:sg_end, QG] = sg_is["q_kvar"].values

        # set bus values for generator busses
        sg_busses = pp.get_indices(sg_is["bus"].values, bus_lookup)
        gen_busses = pp.get_indices(sg_is["bus"].values, bus_lookup)
        mpc["bus"][gen_busses, BUS_TYPE] = PQ

        # set constraints for PV generators
        if "min_q_kvar" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, QMAX] = -sg_is["min_q_kvar"].values * 1e-3
            qmax = mpc["gen"][gen_end:sg_end, [QMIN]]
            ncn.copyto(qmax, -q_lim_default, where=np.isnan(qmax))
            mpc["gen"][gen_end:sg_end, [QMIN]] = qmax

        if "max_q_kvar" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, QMIN] = -sg_is["max_q_kvar"].values * 1e-3
            qmin = mpc["gen"][gen_end:sg_end, [QMAX]]
            ncn.copyto(qmin, q_lim_default, where=np.isnan(qmin))
            mpc["gen"][gen_end:sg_end, [QMAX]] = qmin

        if "min_p_kw" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, PMIN] = -sg_is["min_p_kw"].values * 1e-3
            pmax = mpc["gen"][gen_end:sg_end, [PMIN]]
            ncn.copyto(pmax, -p_lim_default, where=np.isnan(pmax))
            mpc["gen"][gen_end:sg_end, [PMIN]] = pmax
        if "max_p_kw" in sg_is.columns:
            mpc["gen"][gen_end:sg_end, PMAX] = -sg_is["max_p_kw"].values * 1e-3
            min_p_kw = mpc["gen"][gen_end:sg_end, [PMAX]]
            ncn.copyto(min_p_kw, p_lim_default, where=np.isnan(min_p_kw))
            mpc["gen"][gen_end:sg_end, [PMAX]] = min_p_kw

    # add ext grid / slack data
    mpc["gen"][:eg_end, GEN_BUS] = pp.get_indices(eg_is["bus"].values, bus_lookup)
    mpc["gen"][:eg_end, VG] = eg_is["vm_pu"].values
    mpc["gen"][:eg_end, GEN_STATUS] = eg_is["in_service"].values

    # set bus values for external grid busses
    eg_busses = pp.get_indices(eg_is["bus"].values, bus_lookup)
    if calculate_voltage_angles:
        mpc["bus"][eg_busses, VA] = eg_is["va_degree"].values
    mpc["bus"][eg_busses, BUS_TYPE] = REF

    # add generator / pv data
    if gen_end > eg_end:
        mpc["gen"][eg_end:gen_end, GEN_BUS] = pp.get_indices(gen_is["bus"].values, bus_lookup)
        mpc["gen"][eg_end:gen_end, PG] = - gen_is["p_kw"].values * 1e-3 * gen_is["scaling"].values
        mpc["gen"][eg_end:gen_end, VG] = gen_is["vm_pu"].values

        # set bus values for generator busses
        gen_busses = pp.get_indices(gen_is["bus"].values, bus_lookup)
        mpc["bus"][gen_busses, BUS_TYPE] = PV
        mpc["bus"][gen_busses, VM] = gen_is["vm_pu"].values

        # set constraints for PV generators
        mpc["gen"][eg_end:gen_end, QMIN] = -gen_is["max_q_kvar"].values * 1e-3
        mpc["gen"][eg_end:gen_end, QMAX] = -gen_is["min_q_kvar"].values * 1e-3
        mpc["gen"][eg_end:gen_end, PMIN] = -gen_is["min_p_kw"].values * 1e-3
        mpc["gen"][eg_end:gen_end, PMAX] = -gen_is["max_p_kw"].values * 1e-3

        qmin = mpc["gen"][eg_end:gen_end, [QMIN]]
        ncn.copyto(qmin, -q_lim_default, where=np.isnan(qmin))
        mpc["gen"][eg_end:gen_end, [QMIN]] = qmin

        qmax = mpc["gen"][eg_end:gen_end, [QMAX]]
        ncn.copyto(qmax, q_lim_default, where=np.isnan(qmax))
        mpc["gen"][eg_end:gen_end, [QMAX]] = qmax

        min_p_kw = mpc["gen"][eg_end:gen_end, [PMIN]]
        ncn.copyto(min_p_kw, -p_lim_default, where=np.isnan(min_p_kw))
        mpc["gen"][eg_end:gen_end, [PMIN]] = min_p_kw

        pmax = mpc["gen"][eg_end:gen_end, [PMAX]]
        ncn.copyto(pmax, p_lim_default, where=np.isnan(pmax))
        mpc["gen"][eg_end:gen_end, [PMAX]] = pmax


def _calc_loads_and_add_opf(net, mpc, bus_lookup):
    """ we need to exclude controllable sgens from the bus table
    """

    l = net["load"]
    vl = l["in_service"].values * l["scaling"].values.T / np.float64(1000.)
    lp = l["p_kw"].values * vl
    lq = l["q_kvar"].values * vl

    sgen = net["sgen"]
    if not sgen.empty:
        vl = (sgen["in_service"].values & ~sgen["controllable"]) * sgen["scaling"].values.T / \
            np.float64(1000.)
        sp = sgen["p_kw"].values * vl
        sq = sgen["q_kvar"].values * vl
    else:
        sp = []
        sq = []

    b = pp.get_indices(np.hstack([l["bus"].values, sgen["bus"].values]
                                 ), bus_lookup)
    b, vp, vq = _sum_by_group(b, np.hstack([lp, sp]), np.hstack([lq, sq]))

    mpc["bus"][b, PD] = vp
    mpc["bus"][b, QD] = vq

# -*- coding: utf-8 -*-
import warnings
# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd

from pandapower.auxiliary import _sum_by_group, phase_to_sequence, version_check
from pandapower.pypower.idx_bus import BUS_I, BASE_KV, PD, QD, GS, BS, VMAX, VMIN, BUS_TYPE, NONE, \
    VM, VA, CID, CZD, bus_cols, REF, PV
from pandapower.pypower.idx_bus_sc import C_MAX, C_MIN, bus_cols_sc
from .pypower.idx_bus_dc import dc_bus_cols, DC_BUS_TYPE, DC_BUS_AREA, DC_VM, DC_ZONE, DC_VMAX, DC_VMIN, DC_P, DC_BUS_I, \
    DC_BASE_KV, DC_NONE, DC_REF, DC_B2B, DC_PD
from .pypower.idx_ssc import ssc_cols, SSC_BUS, SSC_R, SSC_X, SSC_SET_VM_PU, SSC_X_CONTROL_VA, SSC_X_CONTROL_VM, \
    SSC_STATUS, SSC_CONTROLLABLE, SSC_INTERNAL_BUS
from .pypower.idx_svc import svc_cols, SVC_BUS, SVC_SET_VM_PU, SVC_MIN_FIRING_ANGLE, SVC_MAX_FIRING_ANGLE, SVC_STATUS, \
    SVC_CONTROLLABLE, SVC_X_L, SVC_X_CVAR, SVC_THYRISTOR_FIRING_ANGLE
from .pypower.idx_vsc import vsc_cols, VSC_BUS, VSC_INTERNAL_BUS, VSC_R, VSC_X, VSC_STATUS, VSC_CONTROLLABLE, \
    VSC_BUS_DC, VSC_MODE_AC, VSC_VALUE_AC, VSC_MODE_DC, VSC_VALUE_DC, VSC_MODE_AC_V, VSC_MODE_AC_Q, VSC_MODE_AC_SL, \
    VSC_MODE_DC_V, VSC_MODE_DC_P, VSC_INTERNAL_BUS_DC, VSC_R_DC, VSC_PL_DC

try:
    from numba import jit
    version_check('numba')
except ImportError:
    from .pf.no_numba import jit


@jit(nopython=True, cache=False)
def ds_find(ar, bus):  # pragma: no cover
    while True:
        p = ar[bus]
        if p == bus:
            break
        bus = p
    return p


@jit(nopython=True, cache=False)
def ds_union(ar, bus1, bus2, bus_is_pv, bus_is_active, merged_bus):  # pragma: no cover
    root1 = ds_find(ar, bus1)
    root2 = ds_find(ar, bus2)
    if root1 == root2:
        return
    if bus_is_active[root2] and ~bus_is_pv[root1]:      # if root2 is an active bus (load or gen) and root1 is not a PV bus, it will merge root1 into root2
        ar[root1] = root2
        merged_bus[root1] = True
    else:
        ar[root2] = root1
        merged_bus[root2] = True


@jit(nopython=True, cache=False)
def ds_create(ar, switch_bus, switch_elm, switch_et_bus, switch_closed, switch_z_ohm,
              bus_is_pv, bus_is_active, bus_in_service, merged_bus):  # pragma: no cover
    for i in range(len(switch_bus)):
        if not switch_closed[i] or not switch_et_bus[i] or switch_z_ohm[i] > 0:
            continue
        bus1 = switch_bus[i]
        bus2 = switch_elm[i]
        if bus_in_service[bus1] and bus_in_service[bus2]:
            ds_union(ar, bus1, bus2, bus_is_pv, bus_is_active, merged_bus)


@jit(nopython=True, cache=False)
def fill_bus_lookup(ar, bus_lookup, bus_index):
    for i in range(len(bus_index)):
        bus_lookup[bus_index[i]] = i
    for b in bus_index:
        ds = ds_find(ar, b)
        bus_lookup[b] = bus_lookup[ar[ds]]


def create_bus_lookup_numba(net, bus_index, bus_is_idx):
    max_bus_idx = np.max(bus_index)
    # extract numpy arrays of switch table data
    switch = net["switch"]
    switch_bus = switch["bus"].values
    switch_elm = switch["element"].values
    switch_et_bus = switch["et"].values == "b"
    switch_closed = switch["closed"].values
    switch_z_ohm = switch['z_ohm'].values
    # create array for fast checking if a bus is in_service
    bus_in_service = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_in_service[bus_is_idx] = True
    # create mask arrays to distinguish different grid elements
    _is_elements = net["_is_elements"]
    eg_is_mask = _is_elements['ext_grid']
    gen_is_mask = _is_elements['gen']
    load_is_mask = _is_elements['load']
    motor_is_mask = _is_elements['motor']
    sgen_is_mask = _is_elements['sgen']
    shunt_is_mask = _is_elements['shunt']
    ward_is_mask = _is_elements['ward']
    xward_is_mask = _is_elements['xward']
    # create array for fast checking if a bus is pv bus
    bus_is_pv = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_is_pv[net["ext_grid"]["bus"].values[eg_is_mask]] = True
    bus_is_pv[net["gen"]["bus"].values[gen_is_mask]] = True
    # create array for checking if a bus is active (i.e., it has some element connected to it)
    bus_is_active = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_is_active[net["ext_grid"]["bus"].values[eg_is_mask]] = True
    bus_is_active[net["gen"]["bus"].values[gen_is_mask]] = True
    bus_is_active[net["load"]["bus"].values[load_is_mask]] = True
    bus_is_active[net["motor"]["bus"].values[motor_is_mask]] = True
    bus_is_active[net["sgen"]["bus"].values[sgen_is_mask]] = True
    bus_is_active[net["shunt"]["bus"].values[shunt_is_mask]] = True
    bus_is_active[net["ward"]["bus"].values[ward_is_mask]] = True
    bus_is_active[net["xward"]["bus"].values[xward_is_mask]] = True
    # create array that represents the disjoint set
    ar = np.arange(max_bus_idx + 1)
    merged_bus = np.zeros(len(ar), dtype=bool)
    ds_create(ar, switch_bus, switch_elm, switch_et_bus, switch_closed, switch_z_ohm, bus_is_pv,
              bus_is_active, bus_in_service, merged_bus)
    # finally create and fill bus lookup
    bus_lookup = -np.ones(max_bus_idx + 1, dtype=np.int64)
    fill_bus_lookup(ar, bus_lookup, bus_index)
    return bus_lookup, merged_bus


class DisjointSet(dict):
    def add(self, item):
        self[item] = item

    def find(self, item):
        parent = self[item]
        if self[parent] != parent:
            parent = self.find(parent)
            self[item] = parent
        return parent

    def union(self, item1, item2):
        p1 = self.find(item1)
        p2 = self.find(item2)
        self[p1] = p2


def create_consecutive_bus_lookup(net, bus_index):
    # create a mapping from arbitrary pp-index to a consecutive index starting at zero (ppc-index)
    if len(bus_index) == 0:
        return np.array([], dtype=np.int64)
    consec_buses = np.arange(len(bus_index), dtype=np.int64)
    # bus_lookup as dict:
    # bus_lookup = dict(zip(bus_index, consec_buses))
    # bus lookup as mask from pandapower -> pypower
    bus_lookup = -np.ones(max(bus_index) + 1, dtype=np.int64)
    bus_lookup[bus_index] = consec_buses
    return bus_lookup


def create_bus_lookup_numpy(net, bus_index, closed_bb_switch_mask):
    bus_lookup = create_consecutive_bus_lookup(net, bus_index)
    net._fused_bb_switches = closed_bb_switch_mask & (net["switch"]["z_ohm"].values <= 0)
    merged_bus = np.zeros(len(bus_lookup), dtype=bool)
    if net._fused_bb_switches.any():
        # Note: this might seem a little odd - first constructing a pp to ppc mapping without
        # fused busses and then update the entries. The alternative (to construct the final
        # mapping at once) would require to determine how to fuse busses and which busses
        # are not part of any opened bus-bus switches first. It turns out, that the latter takes
        # quite some time in the average usecase, where #busses >> #bus-bus switches.

        # create mask arrays to distinguish different grid elements
        _is_elements = net["_is_elements"]
        eg_is_mask = _is_elements['ext_grid']
        gen_is_mask = _is_elements['gen']
        load_is_mask = _is_elements['load']
        motor_is_mask = _is_elements['motor']
        sgen_is_mask = _is_elements['sgen']
        shunt_is_mask = _is_elements['shunt']
        ward_is_mask = _is_elements['ward']
        xward_is_mask = _is_elements['xward']
        # Find PV / Slack nodes -> their bus must be kept when fused with a PQ node
        pv_list = [net["ext_grid"]["bus"].values[eg_is_mask], net["gen"]["bus"].values[gen_is_mask]]
        pv_ref = np.unique(np.hstack(pv_list))
        # Find active nodes -> their bus must be possibly kept when fused with a zero injection node
        active_bus = [net["load"]["bus"].values[load_is_mask], net["motor"]["bus"].values[motor_is_mask], \
                    net["sgen"]["bus"].values[sgen_is_mask], net["shunt"]["bus"].values[shunt_is_mask], \
                    net["ward"]["bus"].values[ward_is_mask], net["xward"]["bus"].values[xward_is_mask]]
        active_ref = np.unique(np.hstack(active_bus))
        # get the pp-indices of the buses which are connected to a switch to be fused
        fbus = net["switch"]["bus"].values[net._fused_bb_switches]
        tbus = net["switch"]["element"].values[net._fused_bb_switches]

        # create a mapping to map each bus to itself at frist ...
        ds = DisjointSet({e: e for e in chain(fbus, tbus)})

        # ... to follow each bus along a possible chain of switches to its final bus and update the
        # map
        for f, t in zip(fbus, tbus):
            ds.union(f, t)

        # now we can find out how to fuse each bus by looking up the final bus of the chain they
        # are connected to
        v = defaultdict(set)
        for a in ds:
            v[ds.find(a)].add(a)
        # build sets of buses which will be fused
        disjoint_sets = [e for e in v.values() if len(e) > 1]

        # check if PV buses need to be fused
        # if yes: the sets with PV buses must be found (which is slow)
        # if no: the check can be omitted
        if any(i in fbus or i in tbus for i in pv_ref):
            # for every disjoint set
            for dj in disjoint_sets:
                # check if pv and active buses are in the disjoint set dj
                pv_buses_in_set = set(pv_ref) & dj
                nr_pv_bus = len(pv_buses_in_set)
                active_buses_in_set = set(active_ref) & dj
                nr_active_bus = len(active_buses_in_set)
                if nr_pv_bus > 0:
                    # one pv bus. Get bus from pv_buses_in_set
                    ref_bus = pv_buses_in_set.pop()
                else:
                    if nr_active_bus > 0:
                        # no pv bus but another active bus. Get bus from active_buses_in_set
                        ref_bus = active_buses_in_set.pop()
                    else:
                        # neither pv buses nor active buses. Use any bus in dj
                        ref_bus = dj.pop()
                map_to = bus_lookup[ref_bus]
                for bus in dj:
                    # update lookup
                    bus_lookup[bus] = map_to
                    if bus != ref_bus:
                        merged_bus[bus] = 1
        else:
            # no PV buses in set
            for dj in disjoint_sets:
                active_buses_in_set = set(active_ref) & dj
                nr_active_bus = len(active_buses_in_set)
                if nr_active_bus > 0:
                    # no ov bus but another active bus. Get bus from active_buses_in_set
                    ref_bus = active_buses_in_set.pop()
                else:
                    # neither pv buses nor active busese. Use any bus in dj
                    ref_bus = dj.pop()
                map_to = bus_lookup[ref_bus]
                for bus in dj:
                    # update bus lookup
                    bus_lookup[bus] = map_to
                    if bus != ref_bus:
                        merged_bus[bus] = 1
    return bus_lookup, merged_bus


def create_bus_lookup(net, bus_index, bus_is_idx, numba):
    switches_with_pos_z_ohm = net["switch"]["z_ohm"].values > 0
    if switches_with_pos_z_ohm.any() or not numba:
        # if there are any closed bus-bus switches find them
        closed_bb_switch_mask = (net["switch"]["closed"].values &
                                 (net["switch"]["et"].values == "b") &
                                 np.isin(net["switch"]["bus"].values, bus_is_idx) &
                                 np.isin(net["switch"]["element"].values, bus_is_idx))

    if switches_with_pos_z_ohm.any():
        net._impedance_bb_switches = closed_bb_switch_mask & switches_with_pos_z_ohm
    else:
        net._impedance_bb_switches = np.zeros(switches_with_pos_z_ohm.shape)

    if numba:
        bus_lookup, merged_bus = create_bus_lookup_numba(net, bus_index, bus_is_idx)
    else:
        bus_lookup, merged_bus = create_bus_lookup_numpy(net, bus_index, closed_bb_switch_mask)
    return bus_lookup, merged_bus


def get_voltage_init_vector(net, init_v, mode, sequence=None):
    if isinstance(init_v, str):
        if init_v == "results":
            res_table = "res_bus" if sequence is None else "res_bus_3ph"

            # init voltage possible if bus results are available
            if res_table not in net or not net[res_table].index.equals(net.bus.index):
                # cannot init from results, since sorting of results is different from element table
                # TO BE REVIEWED! Why there was no raise before this commit?
                raise UserWarning("Init from results not possible. Index of %s do not match with "
                                  "bus. You should sort res_bus before calling runpp." % res_table)

            if res_table == "res_bus_3ph":
                vm = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].values.T
                va = net.res_bus_3ph[["va_a_degree", "va_b_degree", "va_c_degree"]].values.T

                voltage_vector = phase_to_sequence(vm * np.exp(1j * np.pi * va / 180.))[sequence, :]

                if mode == "magnitude":
                    return np.abs(voltage_vector)
                elif mode == "angle":
                    return np.angle(voltage_vector) * 180 / np.pi
                else:
                    raise UserWarning(str(mode)+" for initialization not available!")
            else:
                if mode == "magnitude":
                    return net[res_table]["vm_pu"].values.copy()
                elif mode == "angle":
                    return net[res_table]["va_degree"].values.copy()
                else:
                    raise UserWarning(str(mode)+" for initialization not available!")
        if init_v == "flat":
            return None
    elif isinstance(init_v, (float, np.ndarray, list)) and sequence is None or sequence == 1:
        return init_v
    elif isinstance(init_v, pd.Series) and sequence is None or sequence == 1:
        if init_v.index.equals(net.bus.index):
            return init_v.loc[net.bus.index]
        else:
            raise UserWarning("Voltage starting vector indices do not match bus indices")


def _build_bus_ppc(net, ppc, sequence=None):
    """
    Generates the ppc["bus"] array and the lookup pandapower indices -> ppc indices
    """
    init_vm_pu = net["_options"]["init_vm_pu"]
    init_va_degree = net["_options"]["init_va_degree"]
    mode = net["_options"]["mode"]
    numba = net["_options"]["numba"] if "numba" in net["_options"] else False

    # get bus indices
    nr_xward = len(net.xward)
    nr_trafo3w = len(net.trafo3w)
    nr_ssc = len(net.ssc)
    nr_vsc = len(net.vsc)
    aux = dict()
    if nr_xward > 0 or nr_trafo3w > 0 or nr_ssc > 0 or nr_vsc > 0:
        bus_indices = [net["bus"].index.values, np.array([], dtype=np.int64)]
        max_idx = max(net["bus"].index) + 1
        if nr_xward > 0:
            aux_xward = np.arange(max_idx, max_idx + nr_xward, dtype=np.int64)
            aux["xward"] = aux_xward
            bus_indices.append(aux_xward)
        if nr_trafo3w:
            aux_trafo3w = np.arange(max_idx + nr_xward, max_idx + nr_xward + nr_trafo3w)
            aux["trafo3w"] = aux_trafo3w
            bus_indices.append(aux_trafo3w)
        if nr_ssc:
            aux_ssc = np.arange(max_idx + nr_xward + nr_trafo3w, max_idx + nr_xward + nr_trafo3w + nr_ssc)
            aux["ssc"] = aux_ssc
            bus_indices.append(aux_ssc)
        if nr_vsc:
            aux_vsc = np.arange(max_idx + nr_xward + nr_trafo3w + nr_ssc, max_idx + nr_xward + nr_trafo3w + nr_ssc + nr_vsc)
            aux["vsc"] = aux_vsc
            bus_indices.append(aux_vsc)
        bus_index = np.concatenate(bus_indices)
    else:
        bus_index = net["bus"].index.values
    # get in service elements

    if mode == "nx":
        bus_lookup = create_consecutive_bus_lookup(net, bus_index)
    else:
        _is_elements = net["_is_elements"]
        # eg_is_mask = _is_elements['ext_grid']
        # gen_is_mask = _is_elements['gen']
        bus_is_idx = _is_elements['bus_is_idx']
        bus_lookup, merged_bus = create_bus_lookup(net, bus_index, bus_is_idx, numba=numba)

    n_bus_ppc = len(bus_index)
    # init ppc with empty values
    ppc["bus"] = np.zeros(shape=(n_bus_ppc, bus_cols), dtype=np.float64)
    ppc["bus"][:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0., 0.])  # changes of
    # voltage limits (2 and 0) must be considered in check_opf_data

    if sequence is not None and sequence != 1:
        ppc["bus"][:, VM] = 0.

    if mode == "sc":
        bus_sc = np.empty(shape=(n_bus_ppc, bus_cols_sc), dtype=np.float64)
        bus_sc.fill(np.nan)
        ppc["bus"] = np.hstack((ppc["bus"], bus_sc))

    # apply consecutive bus numbers
    ppc["bus"][:, BUS_I] = np.arange(n_bus_ppc)

    n_bus = len(net.bus.index)
    # init voltages from net
    ppc["bus"][:n_bus, BASE_KV] = net["bus"]["vn_kv"].values
    # set buses out of service (BUS_TYPE == 4)
    if nr_xward > 0 or nr_trafo3w > 0 or nr_ssc > 0 or nr_vsc > 0:
        in_service = np.concatenate([net["bus"]["in_service"].values,
                                     net["xward"]["in_service"].values,
                                     net["trafo3w"]["in_service"].values,
                                     net["ssc"]["in_service"].values,
                                     net["vsc"]["in_service"].values])
    else:
        in_service = net["bus"]["in_service"].values
    ppc["bus"][~in_service, BUS_TYPE] = NONE
    if mode != "nx":
        set_reference_buses(net, ppc, bus_lookup, mode)
    vm_pu = get_voltage_init_vector(net, init_vm_pu, "magnitude", sequence=sequence)
    if vm_pu is not None:
        ppc["bus"][:n_bus, VM] = vm_pu

    va_degree = get_voltage_init_vector(net, init_va_degree, "angle", sequence=sequence)
    if va_degree is not None:
        ppc["bus"][:n_bus, VA] = va_degree

    if mode == "sc":
        _add_c_to_ppc(net, ppc)

    if net._options["mode"] == "opf":
        if "max_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMAX] = net["bus"].max_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMAX] = 2.  # changes of VMAX must be considered in check_opf_data
        if "min_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMIN] = net["bus"].min_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMIN] = 0.  # changes of VMIN must be considered in check_opf_data

    if len(net.xward):
        _fill_auxiliary_buses(net, ppc, bus_lookup, "xward", "bus", aux)

    if len(net.trafo3w):
        _fill_auxiliary_buses(net, ppc, bus_lookup, "trafo3w", "hv_bus", aux)

    if len(net.ssc):
        _fill_auxiliary_buses(net, ppc, bus_lookup, "ssc", "bus", aux)

    if len(net.vsc):
        _fill_auxiliary_buses(net, ppc, bus_lookup, "vsc", "bus", aux)

    net["_pd2ppc_lookups"]["bus"] = bus_lookup
    net["_pd2ppc_lookups"]["aux"] = aux
    if mode != "nx":
        net["_pd2ppc_lookups"]["merged_bus"] = merged_bus


def _build_bus_dc_ppc(net, ppc):
    """
    Generates the ppc["bus_dc"] array and the lookup pandapower indices -> ppc indices
    """
    init_vm_pu = net["_options"]["init_vm_pu"]
    mode = net["_options"]["mode"]
    numba = net["_options"]["numba"] if "numba" in net["_options"] else False

    # get bus indices
    bus_index = net["bus_dc"].index.values
    # get in service elements
    aux = dict()
    nr_vsc = len(net.vsc)
    if nr_vsc > 0:
        max_idx = max(net["bus_dc"].index) + 1
        aux_vsc = np.arange(max_idx, max_idx + nr_vsc)
        aux["vsc"] = aux_vsc
        bus_index = np.concatenate([bus_index, aux_vsc])

    bus_lookup = create_consecutive_bus_lookup(net, bus_index)
    # todo if dc switches are added, adjust this part to implement @fuse buses@ for bus_dc:
    #if mode == "nx":
     #   bus_lookup = create_consecutive_bus_lookup(net, bus_index)
    #else:
        # _is_elements = net["_is_elements"]
        #eg_is_mask = _is_elements['ext_grid']
        #gen_is_mask = _is_elements['gen']
        #bus_is_idx = _is_elements['bus_dc_is_idx']
        #bus_lookup = create_bus_lookup(net, bus_index, bus_is_idx,
         #                              gen_is_mask, eg_is_mask, numba=numba)

    n_bus_ppc = len(bus_index)
    # init ppc with empty values
    ppc["bus_dc"] = np.zeros(shape=(n_bus_ppc, dc_bus_cols), dtype=np.float64)
    # changes of voltage limits (2 and 0) must be considered in check_opf_data
    ppc["bus_dc"][:, [DC_BUS_TYPE, DC_BUS_AREA, DC_VM, DC_ZONE, DC_VMAX, DC_VMIN]] = np.array([DC_P, 1, 1, 1, 2, 0], dtype=np.int64)

    # if mode == "sc":
    #     bus_sc = np.empty(shape=(n_bus_ppc, bus_cols_sc), dtype=np.float64)
    #     bus_sc.fill(np.nan)
    #     ppc["bus"] = np.hstack((ppc["bus"], bus_sc))

    # apply consecutive bus numbers
    ppc["bus_dc"][:, DC_BUS_I] = np.arange(n_bus_ppc)

    n_bus_dc = len(net.bus_dc.index)
    # init voltages from net
    ppc["bus_dc"][:n_bus_dc, DC_BASE_KV] = net["bus_dc"]["vn_kv"].values
    # set buses out of service (BUS_TYPE == 4)

    if nr_vsc > 0:
        in_service = np.concatenate([net["bus_dc"]["in_service"].values,
                                 net["vsc"]["in_service"].values])
    else:
        in_service = net["bus_dc"]["in_service"].values

    ppc["bus_dc"][~in_service, DC_BUS_TYPE] = DC_NONE

    # todo: add init vm_pu for DC buses eventually
    #vm_pu = get_voltage_init_vector(net, init_vm_pu, "magnitude", sequence=sequence)
    #if vm_pu is not None:
    #    ppc["bus"][:n_bus_dc, VM] = vm_pu

    if net._options["mode"] == "opf":
        if "max_vm_pu" in net.bus_dc:
            ppc["bus_dc"][:n_bus_dc, DC_VMAX] = net["bus_dc"].max_vm_pu.values
        else:
            ppc["bus_dc"][:n_bus_dc, DC_VMAX] = 2.  # changes of VMAX must be considered in check_opf_data
        if "min_vm_pu" in net.bus_dc:
            ppc["bus_dc"][:n_bus_dc, DC_VMIN] = net["bus_dc"].min_vm_pu.values
        else:
            ppc["bus_dc"][:n_bus_dc, DC_VMIN] = 0.  # changes of VMIN must be considered in check_opf_data

    if nr_vsc > 0:
        _fill_auxiliary_buses(net, ppc, bus_lookup, "vsc", "bus_dc", aux, "bus_dc")

    net["_pd2ppc_lookups"]["bus_dc"] = bus_lookup
    net["_pd2ppc_lookups"]["aux_dc"] = aux

    if mode != "nx":
        set_reference_buses_dc(net, ppc, bus_lookup, mode)


def _fill_auxiliary_buses(net, ppc, bus_lookup, element, bus_column, aux, bus_table="bus"):
    element_bus_idx = bus_lookup[net[element][bus_column].values]
    aux_idx = bus_lookup[aux[element]]
    if bus_table == "bus":
        base_kv, vmin, vmax, vm = BASE_KV, VMIN, VMAX, VM
    elif bus_table == "bus_dc":
        base_kv, vmin, vmax, vm = DC_BASE_KV, DC_VMIN, DC_VMAX, DC_VM
    else:
        raise NotImplementedError(f"bus table {bus_table} not supported")

    ppc[bus_table][aux_idx, base_kv] = ppc[bus_table][element_bus_idx, base_kv]
    if net._options["mode"] == "opf":
        ppc[bus_table][aux_idx, vmin] = ppc[bus_table][element_bus_idx, vmin]
        ppc[bus_table][aux_idx, vmax] = ppc[bus_table][element_bus_idx, vmax]

    if net._options["init_vm_pu"] == "results":
        ppc[bus_table][aux_idx, vm] = net["res_%s" % element][
            "vm_internal_pu" if bus_table == "bus" else "vm_internal_dc_pu"].values
    else:
        ppc[bus_table][aux_idx, vm] = ppc[bus_table][element_bus_idx, vm]
    if net._options["init_va_degree"] == "results" and bus_table == "bus":
        ppc[bus_table][aux_idx, VA] = net["res_%s" % element]["va_internal_degree"].values
    elif bus_table == "bus":
        ppc[bus_table][aux_idx, VA] = ppc[bus_table][element_bus_idx, VA]


def set_reference_buses(net, ppc, bus_lookup, mode):
    if mode == "nx":
        return
    eg_buses = bus_lookup[net.ext_grid.bus.values[net._is_elements["ext_grid"]]]
    ppc["bus"][eg_buses, BUS_TYPE] = REF
    ppc["internal"]["ac_slack_buses"] = set(eg_buses)  # needed later in _select_is_elements_numba
    if mode == "sc":
        gen_slacks = net._is_elements["gen"]  # generators are slacks for short-circuit calculation
    else:
        gen_slacks = net._is_elements["gen"] & net.gen["slack"].values
    if gen_slacks.any():
        slack_buses = net.gen["bus"].values[gen_slacks]
        ppc["bus"][bus_lookup[slack_buses], BUS_TYPE] = REF
        ppc["internal"]["ac_slack_buses"] |= set(bus_lookup[slack_buses])  # needed later in _select_is_elements_numba
    ppc["internal"]["ac_slack_buses"] = list(ppc["internal"]["ac_slack_buses"])

def set_reference_buses_dc(net, ppc, bus_lookup, mode):
    if mode == "nx":
        return
    vsc_dc_slack = net.vsc.control_mode_dc.values == "vm_pu"
    vsc_ac_slack = net.vsc.control_mode_ac.values == "slack"  # VSC that defines AC slack cannot define DC slack
    # todo vsc
    # ref_buses = bus_lookup[net._pd2ppc_lookups["aux_dc"].get("vsc", np.array([], dtype=np.int64))[net._is_elements["vsc"] & vsc_dc_slack & ~vsc_ac_slack]]
    ref_buses = bus_lookup[net.vsc.bus_dc.values[net._is_elements["vsc"] & vsc_dc_slack & ~vsc_ac_slack]]
    ppc["bus_dc"][ref_buses, DC_BUS_TYPE] = DC_REF

    # identify back-to-back converters:
    vsc_dc_p = net.vsc.control_mode_dc.values == "p_mw"
    p_buses = bus_lookup[net._pd2ppc_lookups["aux_dc"].get("vsc", np.array([], dtype=np.int64))[net._is_elements["vsc"] & vsc_dc_p]]
    b2b_buses = np.intersect1d(ref_buses, p_buses)
    ppc["bus_dc"][b2b_buses, DC_BUS_TYPE] = DC_B2B


def _calc_pq_elements_and_add_on_ppc(net, ppc, sequence=None):
    # init values
    b, p, q = np.array([], dtype=np.int64), np.array([]), np.array([])

    _is_elements = net["_is_elements"]
    voltage_depend_loads = net["_options"]["voltage_depend_loads"]
    mode = net["_options"]["mode"]
    pq_elements = ["load", "motor", "sgen", "storage", "ward", "xward"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    for element in pq_elements:
        tab = net[element]
        if len(tab) == 0:
            continue
        active = _is_elements[element].astype(np.float64)
        if element == "load" and voltage_depend_loads:
            if ((tab["const_z_percent"] + tab["const_i_percent"]) > 100).any():
                raise ValueError("const_z_percent + const_i_percent need to "
                                 "be less or equal to 100%!")
            for bus in set(tab["bus"]):
                mask = (tab["bus"] == bus) & active
                no_loads = sum(mask)
                if not no_loads:
                    continue
                ci_sum = sum(tab["const_i_percent"][mask] / 100.)
                ppc["bus"][bus_lookup[bus], CID] = ci_sum / no_loads
                cz_sum = sum(tab["const_z_percent"][mask] / 100.)
                ppc["bus"][bus_lookup[bus], CZD] = cz_sum / no_loads
        sign = -1 if element == "sgen" else 1
        if element == "motor":
            p_mw, q_mvar = _get_motor_pq(net)
            p = np.hstack([p, p_mw])
            q = np.hstack([q, q_mvar])
        elif element.endswith("ward"):
            p = np.hstack([p, tab["ps_mw"].values * active * sign])
            q = np.hstack([q, tab["qs_mvar"].values * active * sign])
        else:
            scaling = tab["scaling"].values
            p = np.hstack([p, tab["p_mw"].values * active * scaling * sign])
            q = np.hstack([q, tab["q_mvar"].values * active * scaling * sign])
        b = np.hstack([b, tab["bus"].values])

    for element in ["asymmetric_load", "asymmetric_sgen"]:
        if len(net[element]) > 0 and mode == "pf":
            p_mw, q_mvar = _get_symmetric_pq_of_unsymetric_element(net, element)
            sign = -1 if element.endswith("sgen") else 1
            p = np.hstack([p, p_mw * sign])
            q = np.hstack([q, q_mvar * sign])
            b = np.hstack([b, net[element]["bus"].values])

    # sum up p & q of bus elements
    if b.size > 0:
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)
        ppc["bus"][b, PD] = vp
        ppc["bus"][b, QD] = vq

    # init values DC
    b_dc, p_dc = np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    p_dc_elements = []  # todo add more DC elements like DC loads, DC sgens
    # p_dc_elements = ["vsc"]  # todo add more DC elements like DC loads, DC sgens
    bus_dc_lookup = net["_pd2ppc_lookups"]["bus_dc"]

    for element in p_dc_elements:
        tab = net[element]
        if len(tab) == 0:
            continue
        active = _is_elements[element].astype(np.float64)
        scaling = tab["scaling"].values if "scaling" in tab.columns else 1.
        sign = -1 if element == "sgen_dc" else 1  # todo add DC sgen etc.
        if element == "vsc":
            # old: we model pl_dc as shsunt resistor instead (here left for reference, can be deleted safely)
            p_dc = np.hstack([p_dc, tab["pl_dc_mw"].values * active * scaling * sign])
            # b_dc = np.hstack([b_dc, tab["bus_dc"].values])
            b_dc = np.hstack([b_dc, net["_pd2ppc_lookups"]["aux_dc"]["vsc"]])
        else:
            p_dc = np.hstack([p_dc, tab["p_dc_mw"].values * active * scaling * sign])
            b_dc = np.hstack([b_dc, tab["bus_dc"].values])

    # sum up p_dc of bus_dc elements
    if b_dc.size > 0:
        b_dc = bus_dc_lookup[b_dc]
        b_dc, vp_dc, _ = _sum_by_group(b_dc, p_dc, p_dc)
        ppc["bus_dc"][b_dc, DC_PD] = vp_dc


def _get_symmetric_pq_of_unsymetric_element(net, element):
    scale = net["_is_elements"][element] * net[element]["scaling"].values.T
    q_mvar = np.sum(net[element][["q_a_mvar", "q_b_mvar", "q_c_mvar"]].values, axis=1)
    p_mw = np.sum(net[element][["p_a_mw", "p_b_mw", "p_c_mw"]].values, axis=1)
    return p_mw*scale, q_mvar*scale


def _get_motor_pq(net):
    tab = net["motor"]
    active = net._is_elements["motor"]
    scale = tab["loading_percent"].values/100 * tab["scaling"].values*active

    efficiency = tab["efficiency_percent"].values
    p_mech = tab["pn_mech_mw"].values
    cos_phi = tab["cos_phi"].values

    p_mw = p_mech / efficiency * 100 * scale
    s_mvar = p_mw / cos_phi
    q_mvar = np.sqrt(s_mvar**2 - p_mw**2)
    return p_mw, q_mvar


def _calc_shunts_and_add_on_ppc(net, ppc):
    # init values
    b, p, q = np.array([], dtype=np.int64), np.array([]), np.array([])
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # Divide base kv by 3 for 3 phase calculation
    mode = net._options["mode"]
    base_multiplier = 1/3 if mode == "pf_3ph" else 1
    # get in service elements
    _is_elements = net["_is_elements"]

    s = net["shunt"]
    if len(s) > 0:
        vl = _is_elements["shunt"]
        # fill missing vn_kv values in shunt table with rated voltage of connected bus
        idx_s_vnkv_isnan = s.loc[np.isnan(s.vn_kv)].index
        s.loc[idx_s_vnkv_isnan, "vn_kv"] = net.bus.loc[s.bus[idx_s_vnkv_isnan].values, "vn_kv"].values
        v_ratio = (ppc["bus"][bus_lookup[s["bus"].values], BASE_KV] / s["vn_kv"].values) ** 2 * base_multiplier

        if "step_dependency_table" in s:
            if np.any(vl & (s.step_dependency_table == True) & (pd.isna(s.id_characteristic_table))):
                raise UserWarning("Shunts with step_dependency_table True and id_characteristic_table NA detected.\n"
                                  "Please set an id_characteristic_table or set step_dependency_table to False.")
            elif np.any(vl & (s.step_dependency_table == False) & (~pd.isna(s.id_characteristic_table))):
                warnings.warn("Shunts with step_dependency_table False but id_characteristic_table detected.",
                              category=UserWarning)
            s.step_dependency_table.fillna(False)
            if s.step_dependency_table.any():
                s_tmp = s.merge(net.shunt_characteristic_table, how="left", left_on="id_characteristic_table",
                                right_on="id_characteristic", suffixes=("", "_table"))
                s = s_tmp.loc[(s_tmp["step"] == s_tmp["step_table"]) | ~s_tmp["step_dependency_table"]]

                q = np.hstack([q, s["q_mvar"].values * s["step"].values * v_ratio * vl * ~s["step_dependency_table"]])
                p = np.hstack([p, s["p_mw"].values * s["step"].values * v_ratio * vl * ~s["step_dependency_table"]])
                q = q + s["q_mvar_table"].fillna(0).to_numpy() * v_ratio * vl
                p = p + s["p_mw_table"].fillna(0).to_numpy() * v_ratio * vl
            else:
                q = np.hstack([q, s["q_mvar"].values * s["step"].values * v_ratio * vl])
                p = np.hstack([p, s["p_mw"].values * s["step"].values * v_ratio * vl])
        else:
            q = np.hstack([q, s["q_mvar"].values * s["step"].values * v_ratio * vl])
            p = np.hstack([p, s["p_mw"].values * s["step"].values * v_ratio * vl])
        b = np.hstack([b, s["bus"].values])

    w = net["ward"]
    if len(w) > 0:
        vl = _is_elements["ward"]
        q = np.hstack([q, w["qz_mvar"].values * base_multiplier * vl])
        p = np.hstack([p, w["pz_mw"].values * base_multiplier * vl])
        b = np.hstack([b, w["bus"].values])

    xw = net["xward"]
    if len(xw) > 0:
        vl = _is_elements["xward"]
        q = np.hstack([q, xw["qz_mvar"].values * base_multiplier * vl])
        p = np.hstack([p, xw["pz_mw"].values * base_multiplier * vl])
        b = np.hstack([b, xw["bus"].values])

    loss_location = net._options["trafo3w_losses"].lower()
    trafo3w = net["trafo3w"]
    if loss_location == "star" and len(trafo3w) > 0:
        pfe_mw = trafo3w["pfe_kw"].values * 1e-3
        i0 = trafo3w["i0_percent"].values
        sn_mva = trafo3w["sn_hv_mva"].values

        q_mvar = (sn_mva * i0 / 100.) ** 2 - pfe_mw ** 2
        q_mvar[q_mvar < 0] = 0
        q_mvar = np.sqrt(q_mvar)

        vn_hv_trafo = trafo3w["vn_hv_kv"].values
        vn_hv_bus = ppc["bus"][bus_lookup[trafo3w.hv_bus.values], BASE_KV]
        v_ratio = (vn_hv_bus / vn_hv_trafo) ** 2 * base_multiplier

        q = np.hstack([q, q_mvar * v_ratio])
        p = np.hstack([p, pfe_mw * v_ratio])
        b = np.hstack([b, net._pd2ppc_lookups["aux"]["trafo3w"]])

    # if array is not empty
    if b.size:
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)

        ppc["bus"][b, GS] = vp
        ppc["bus"][b, BS] = -vq


def _build_svc_ppc(net, ppc, mode):
    length = len(net.svc)
    ppc["svc"] = np.zeros(shape=(length, svc_cols), dtype=np.float64)

    if mode != "pf":
        return

    if length > 0:
        baseMVA = ppc["baseMVA"]
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        f = 0
        t = length

        bus = bus_lookup[net.svc["bus"].values]

        svc = ppc["svc"]
        baseV = ppc["bus"][bus, BASE_KV]
        baseZ = baseV ** 2 / baseMVA

        svc[f:t, SVC_BUS] = bus

        svc[f:t, SVC_X_L] = net["svc"]["x_l_ohm"].values / baseZ
        svc[f:t, SVC_X_CVAR] = net["svc"]["x_cvar_ohm"].values / baseZ
        svc[f:t, SVC_SET_VM_PU] = net["svc"]["set_vm_pu"].values
        svc[f:t, SVC_THYRISTOR_FIRING_ANGLE] = np.deg2rad(net["svc"]["thyristor_firing_angle_degree"].values)
        svc[f:t, SVC_MIN_FIRING_ANGLE] = np.deg2rad(net["svc"]["min_angle_degree"].values)
        svc[f:t, SVC_MAX_FIRING_ANGLE] = np.deg2rad(net["svc"]["max_angle_degree"].values)

        svc[f:t, SVC_STATUS] = net["svc"]["in_service"].values
        svc[f:t, SVC_CONTROLLABLE] = net["svc"]["controllable"].values.astype(bool) & net["svc"][
            "in_service"].values.astype(bool)


def _build_ssc_ppc(net, ppc, mode):
    length = len(net.ssc)
    ppc["ssc"] = np.zeros(shape=(length, ssc_cols), dtype=np.float64)

    if mode != "pf":
        return

    if length > 0:
        baseMVA = ppc["baseMVA"]
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        aux = net["_pd2ppc_lookups"]["aux"]
        f = 0
        t = length

        bus = bus_lookup[net.ssc["bus"].values]
        controllable = net["ssc"]["controllable"].values.astype(bool)

        ssc = ppc["ssc"]
        baseV = ppc["bus"][bus, BASE_KV]
        baseZ = baseV ** 2 / baseMVA

        ssc[f:t, SSC_BUS] = bus
        ssc[f:t, SSC_INTERNAL_BUS] = bus_lookup[aux["ssc"]]

        ssc[f:t, SSC_R] = net["ssc"]["r_ohm"].values / baseZ
        ssc[f:t, SSC_X] = net["ssc"]["x_ohm"].values / baseZ
        ssc[f:t, SSC_SET_VM_PU] = net["ssc"]["set_vm_pu"].values
        ssc[f:t, SSC_X_CONTROL_VA] = np.deg2rad(net["ssc"]["va_internal_degree"].values)
        ssc[f:t, SSC_X_CONTROL_VM] = net["ssc"]["vm_internal_pu"].values

        ssc[f:t, SSC_STATUS] = net._is_elements["ssc"].astype(np.int64)
        ssc[f:t, SSC_CONTROLLABLE] = controllable & net["ssc"]["in_service"].values.astype(bool)
        ppc["bus"][aux["ssc"][~controllable], BUS_TYPE] = PV
        ppc["bus"][aux["ssc"][~controllable], VM] = net["ssc"].loc[~controllable, "vm_internal_pu"].values


def _build_vsc_ppc(net, ppc, mode):
    length = len(net.vsc)
    ppc["vsc"] = np.zeros(shape=(length, vsc_cols), dtype=np.float64)

    if mode != "pf":
        return

    if length == 0:
        return

    baseMVA = ppc["baseMVA"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    bus_lookup_dc = net["_pd2ppc_lookups"]["bus_dc"]
    aux = net["_pd2ppc_lookups"]["aux"]
    aux_dc = net["_pd2ppc_lookups"]["aux_dc"]
    f = 0
    t = length

    bus = bus_lookup[net.vsc["bus"].values]
    bus_dc = bus_lookup_dc[net.vsc["bus_dc"].values]
    controllable = net["vsc"]["controllable"].values.astype(bool)
    mode_ac = net["vsc"]["control_mode_ac"].values
    mode_dc = net["vsc"]["control_mode_dc"].values
    value_ac = net["vsc"]["control_value_ac"].values
    value_dc = net["vsc"]["control_value_dc"].values

    vsc = ppc["vsc"]
    baseV = ppc["bus"][bus, BASE_KV]
    baseZ = baseV ** 2 / baseMVA

    baseVDC = ppc["bus_dc"][bus_dc, DC_BASE_KV]
    baseR = baseVDC ** 2 / baseMVA

    vsc[f:t, VSC_BUS] = bus
    vsc[f:t, VSC_INTERNAL_BUS] = bus_lookup[aux["vsc"]]
    vsc[f:t, VSC_BUS_DC] = bus_dc
    vsc[f:t, VSC_INTERNAL_BUS_DC] = bus_lookup_dc[aux_dc["vsc"]]

    vsc[f:t, VSC_R] = net["vsc"]["r_ohm"].values / baseZ
    vsc[f:t, VSC_X] = net["vsc"]["x_ohm"].values / baseZ
    vsc[f:t, VSC_R_DC] = net["vsc"]["r_dc_ohm"].values / baseR
    vsc[f:t, VSC_PL_DC] = net["vsc"]["pl_dc_mw"].values / baseMVA

    if np.any(~np.isin(mode_ac, ["vm_pu", "q_mvar", "slack"])):
        raise NotImplementedError("VSC element only supports the following control modes for AC side: vm_pu, q_mvar")

    if np.any(~np.isin(mode_dc, ["vm_pu", "p_mw"])):
        raise NotImplementedError("VSC element only supports the following control modes for AC side: vm_pu, q_mvar")

    mode_ac_code = np.where(mode_ac == "vm_pu", VSC_MODE_AC_V,
                            np.where(mode_ac == "q_mvar", VSC_MODE_AC_Q, VSC_MODE_AC_SL))
    vsc[f:t, VSC_MODE_AC] = mode_ac_code
    vsc[f:t, VSC_VALUE_AC] = np.where((mode_ac_code == VSC_MODE_AC_V) | (mode_ac_code == VSC_MODE_AC_SL),
                                      value_ac, value_ac / baseMVA)
    mode_dc_code = np.where(mode_dc == "vm_pu", VSC_MODE_DC_V, VSC_MODE_DC_P)
    vsc[f:t, VSC_MODE_DC] = mode_dc_code
    vsc[f:t, VSC_VALUE_DC] = np.where(mode_dc == "vm_pu", value_dc, value_dc / baseMVA)

    vsc[f:t, VSC_STATUS] = net._is_elements["vsc"].astype(np.int64)
    vsc[f:t, VSC_CONTROLLABLE] = controllable & net["vsc"]["in_service"].values.astype(bool)
    ppc["bus"][aux["vsc"][~controllable], BUS_TYPE] = PV
    # it has a role of REF but internally it is PQ and we set the behavior of REF with the Jacobian and mismatch:
    ppc["bus"][bus[(mode_ac_code == VSC_MODE_AC_SL) & (ppc["bus"][bus, BUS_TYPE] != NONE)], BUS_TYPE] = REF
    # maybe we add this in the future
    # ppc["bus"][aux["vsc"][~controllable], VM] = net["vsc"].loc[~controllable, "vm_internal_pu"].values


# Short circuit relevant routines
def _add_ext_grid_sc_impedance(net, ppc):
    mode = net._options["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if mode == "sc":
        case = net._options["case"]
    else:
        case = "max"
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc = bus_lookup[eg_buses]

    if mode == "sc":
        c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    else:
        c = 1.1
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError(("short circuit apparent power s_sc_%s_mva needs to be specified for "
                          "external grid \n Try: net.ext_grid['s_sc_max_mva'] = 1000") % case)
    s_sc = eg["s_sc_%s_mva" % case].values/ppc['baseMVA']
    if not "rx_%s" % case in eg:
        raise ValueError(("short circuit R/X rate rx_%s needs to be specified for external grid \n"
                          " Try: net.ext_grid['rx_max'] = 0.1") % case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    if mode == 'pf_3ph':
        z_grid = c / (s_sc/3)  # 3 phase power divided to get 1 ph power
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

    y_grid = 1 / (r_grid + x_grid * 1j)
    buses, gs, bs = _sum_by_group(eg_buses_ppc, y_grid.real, y_grid.imag)
    if mode == "sc":
        ppc["bus"][buses, GS] += gs * ppc['baseMVA']
        ppc["bus"][buses, BS] += bs * ppc['baseMVA']
    else:
        ppc["bus"][buses, GS] = gs * ppc['baseMVA']
        ppc["bus"][buses, BS] = bs * ppc['baseMVA']
    return gs * ppc['baseMVA'], bs * ppc['baseMVA']


def _add_motor_impedances_ppc(net, ppc):
    if net._options["case"] == "min":
        return
    motor = net["motor"][net._is_elements["motor"]]
    if motor.empty:
        return
    for par in ["vn_kv", "lrc_pu", "efficiency_n_percent", "cos_phi_n", "rx", "pn_mech_mw"]:
        if any(pd.isnull(motor[par])):
            raise UserWarning("%s needs to be specified for all motors in net.motor.%s" % (
                par, par))
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    motor_buses_ppc = bus_lookup[motor.bus.values]
    vn_net = ppc["bus"][motor_buses_ppc, BASE_KV]

    efficiency = motor.efficiency_n_percent.values
    cos_phi = motor.cos_phi_n.values
    p_mech = motor.pn_mech_mw.values
    vn_kv = motor.vn_kv.values
    lrc = motor.lrc_pu.values
    rx = motor.rx.values

    s_motor = p_mech / (efficiency/100 * cos_phi)
    z_motor_ohm = 1 / lrc * vn_kv**2 / s_motor
    z_motor_pu = z_motor_ohm / vn_net**2

    x_motor_pu = z_motor_pu / np.sqrt(rx ** 2 + 1)
    r_motor_pu = rx * x_motor_pu
    y_motor_pu = 1 / (r_motor_pu + x_motor_pu * 1j)

    buses, gs, bs = _sum_by_group(motor_buses_ppc, y_motor_pu.real, y_motor_pu.imag)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs


def _add_load_sc_impedances_ppc(net, ppc):
    baseMVA = ppc["baseMVA"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    for element_type, sign in (("sgen", -1), ("load", 1)):

        element = net[element_type][net._is_elements[element_type]]

        if element.empty:
            continue

        element_buses_ppc = bus_lookup[element.bus.values]

        vm_pu = ppc["bus"][element_buses_ppc, VM]
        va_degree = ppc["bus"][element_buses_ppc, VA]
        v = vm_pu * np.exp(1j * np.deg2rad(va_degree))  # this is correct!
        # print(np.abs(v), np.angle(v, deg=True))

        s_element_mva = sign * (element.p_mw.values + 1j * element.q_mvar.values) * element.scaling.values
        s_element_pu = s_element_mva / baseMVA
        # S = V * conj(I) -> I = conj(S / V)
        i_element_pu = np.conj(s_element_pu / v)   # this is correct!
        # i_element_ka = -i_element_pu * 100 / (np.sqrt(3) * 110)  # for us to validate

        y_element_pu = i_element_pu / v  # p.u.
        # y_element_s = y_element_pu * baseMVA / v_base**2

        buses, gs, bs = _sum_by_group(element_buses_ppc, y_element_pu.real, y_element_pu.imag)
        # buses, gs, bs = _sum_by_group(element_buses_ppc, s_element_pu.real, s_element_pu.imag)
        ppc["bus"][buses, GS] += gs * baseMVA  # power in p.u. is equal to admittance in p.u.
        ppc["bus"][buses, BS] += bs * baseMVA


def _add_c_to_ppc(net, ppc):
    ppc["bus"][:, C_MAX] = 1.1
    ppc["bus"][:, C_MIN] = 1.
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        lv_tol_percent = net["_options"]["lv_tol_percent"]
        if lv_tol_percent == 10:
            c_ns = 1.1
        elif lv_tol_percent == 6:
            c_ns = 1.05
        else:
            raise ValueError("Voltage tolerance in the low voltage grid has" +
                             " to be either 6% or 10% according to IEC 60909")
        ppc["bus"][lv_buses, C_MAX] = c_ns
        ppc["bus"][lv_buses, C_MIN] = .95

# This function includes various function used for general functionalities such as plotting, grid search

import copy
from typing import List, Tuple

from matplotlib.collections import PatchCollection
from typing_extensions import deprecated

import geojson
import math
from math import isinf
import heapq
import pandas as pd
import numpy as np
import networkx as nx
import logging as log

import pandapower as pp
import pandapower.plotting as plot
from pandapower import pandapowerNet
from pandapower.topology.create_graph import create_nxgraph

import warnings

logger = log.getLogger(__name__)

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

try:
    import mplcursors

    MPLCURSORS_INSTALLED = True
except ImportError:
    MPLCURSORS_INSTALLED = False
    logger.info('could not import mplcursors')

warnings.filterwarnings('ignore')


def _get_coords_from_bus_idx(net: pp.pandapowerNet, bus_idx: pd.Index) -> List[Tuple[float, float]]:
    try:
        bl = net.bus.dropna(subset=["geo"]).loc[bus_idx, 'geo']
        if isinstance(bl, pd.Series):
            return bl.apply(geojson.loads).apply(geojson.utils.coords).apply(next).to_list()
        else:
            return [next(geojson.utils.coords(geojson.loads(bl)))]
    except KeyError:
        logger.error(f"Bus {bus_idx} not found in net.bus.geo")
    return []


def _get_coords_from_line_idx(net: pp.pandapowerNet, line_idx: pd.Index) -> List[Tuple[float, float]]:
    try:
        ll = net.line.dropna(subset=["geo"]).loc[line_idx, 'geo']
        if isinstance(ll, pd.Series):
            return ll.apply(geojson.loads).apply(geojson.utils.coords).apply(next).to_list()
        else:
            return [next(geojson.utils.coords(geojson.loads(ll)))]
    except KeyError:
        logger.error(f"Line {line_idx} not found in net.line.geo")
    return []


def create_sc_bus(net_copy, sc_line_id, sc_fraction):
    # This function creates a short-circuit location (a bus) on a line.
    net = copy.deepcopy(net_copy)
    if sc_fraction < 0 or sc_fraction > 1:
        print("Please select line location that is between 0 and 1")
        return

    max_idx_line = max(net.line.index)
    max_idx_bus = max(net.bus.index)

    aux_line = net.line.loc[sc_line_id]

    # get bus voltage of the short circuit bus
    bus_vn_kv = net.bus.vn_kv.at[aux_line.from_bus]

    # set new virtual short circuit bus with give line and location
    bus_sc = pp.create_bus(net, name="Bus_SC", vn_kv=bus_vn_kv, type="b", index=max_idx_bus + 1)

    # sim bench grids
    if 's_sc_max_mva' not in net.ext_grid:
        print('input s_sc_max_mva or taking 1000')
        net.ext_grid['s_sc_max_mva'] = 1000
    if 'rx_max' not in net.ext_grid:
        print('input rx_max or taking 0.1')
        net.ext_grid['rx_max'] = 0.1
    if 'k' not in net.sgen and len(net.sgen) != 0:
        print('input  Ratio of nominal current to short circuit current- k or  taking k=1')
        net.sgen['k'] = 1

    # set new lines
    sc_line1 = sc_line_id
    net.line.at[sc_line1, 'to_bus'] = bus_sc
    net.line.at[sc_line1, 'length_km'] *= sc_fraction

    sc_line2 = pp.create_line_from_parameters(net, bus_sc, aux_line.to_bus,
                                              length_km=aux_line.length_km * (1 - sc_fraction),
                                              index=max_idx_line + 1, r_ohm_per_km=aux_line.r_ohm_per_km,
                                              x_ohm_per_km=aux_line.x_ohm_per_km, c_nf_per_km=aux_line.c_nf_per_km,
                                              max_i_ka=aux_line.max_i_ka)

    if 'endtemp_degree' in net.line.columns:
        net.line.at[sc_line2, "endtemp_degree"] = net.line.endtemp_degree.at[sc_line1]

    net.line = net.line.sort_index()

    # check if switches are connected to the line and set the switches to new lines
    for switch_id in net.switch.index:
        if (aux_line.from_bus == net.switch.bus[switch_id]) & (net.switch.element[switch_id] == sc_line_id):
            net.switch.loc[switch_id, 'element'] = sc_line1
        elif (aux_line.to_bus == net.switch.bus[switch_id]) & (net.switch.element[switch_id] == sc_line_id):
            net.switch.element[switch_id] = sc_line2

    # set geodata for new bus
    net.bus.loc[max_idx_bus + 1, 'geo'] = None

    x1, y1 = _get_coords_from_bus_idx(net, aux_line.from_bus)[0]
    x2, y2 = _get_coords_from_bus_idx(net, aux_line.to_bus)[0]

    net.bus.geo.at[max_idx_bus + 1] = geojson.dumps(
        geojson.Point((sc_fraction * (x2 - x1) + x1, sc_fraction * (y2 - y1) + y1)), sort_keys=True)
    return net


def calc_faults_at_full_line(net, line, location_step_size=0.01, start_location=0.01, end_location=1, sc_case="min"):
    # functon to create sc at full line
    import pandapower.shortcircuit as sc
    i = 0

    fault_currents = []
    max_bus_idx = max(net.bus.index)
    location = start_location
    while location < end_location:
        net_sc = create_sc_bus(net, line, location)
        sc.calc_sc(net_sc, case=sc_case)

        fault_currents.append(net_sc.res_bus_sc.ikss_ka.at[max_bus_idx + 1])

        i += 1
        location = start_location + i * location_step_size
    return fault_currents


def get_opposite_side_bus_from_switch(net, switch_id):
    # get the frm and to bus of switch
    line_idx = net.switch.element.at[switch_id]
    is_from_bus = get_from_bus_info_switch(net, switch_id)

    if is_from_bus:
        opp_bus_idx = net.line.to_bus.at[line_idx]
    else:
        opp_bus_idx = net.line.from_bus.at[line_idx]

    return opp_bus_idx


def get_opposite_side_bus_from_bus_line(net, bus_idx, line_idx):
    # get the from abd to bus of given line
    is_from_bus = get_from_bus_info_bus_line(net, bus_idx, line_idx)

    if is_from_bus:
        opp_bus_idx = net.line.to_bus.at[line_idx]
    else:
        opp_bus_idx = net.line.from_bus.at[line_idx]

    return opp_bus_idx


def get_from_bus_info_switch(net, switch_id):
    # get the from bus of given switch id
    bus_idx = net.switch.bus.at[switch_id]
    line_idx = net.switch.element.at[switch_id]

    return bus_idx == net.line.from_bus.at[line_idx]


def get_from_bus_info_bus_line(net, bus_idx, line_idx):
    # get bus nfo of given line
    return bus_idx == net.line.from_bus.at[line_idx]


def get_line_impedance(net, line_idx):
    # get line impedence
    line_length = net.line.length_km.at[line_idx]
    line_r_per_km = net.line.r_ohm_per_km.at[line_idx]
    line_x_per_km = net.line.x_ohm_per_km.at[line_idx]
    z_line = complex(line_r_per_km * line_length, line_x_per_km * line_length)  # Z = R + jX
    return z_line


def get_lowest_impedance_line(net: pandapowerNet, lines):
    # get the low impedenceline
    min_imp_line = None
    min_impedance = float('inf')
    for line in lines:
        impedance = abs(get_line_impedance(net, line))
        if impedance < min_impedance:
            min_impedance = impedance
            min_imp_line = line
    return min_imp_line


def check_for_closed_bus_switches(net_copy):
    # closed switches
    net = copy.deepcopy(net_copy)
    closed_bus_switches = net.switch.loc[(net.switch.et == "b") & (net.switch.closed == True)]

    if len(closed_bus_switches) > 0:
        net = fuse_bus_switches(net, closed_bus_switches)

    return net


def fuse_bus_switches(net, bus_switches):
    # get fused switches
    for bus_switch in bus_switches.index:
        bus1 = net.switch.bus.at[bus_switch]
        bus2 = net.switch.element.at[bus_switch]

        pp.fuse_buses(net, bus1, bus2)

    return net


def get_fault_annotation(net: pandapowerNet, fault_current: float = .0, font_size_bus: float = 0.06) -> PatchCollection:
    max_bus_idx = max(net.bus.dropna(subset=['geo']).index)
    fault_text = f'\tI_sc = {fault_current}kA'

    fault_geo_x_y: Tuple[float, float] = next(geojson.utils.coords(geojson.loads(net.bus.geo.at[max_bus_idx])))
    fault_geo_x_y = (fault_geo_x_y[0], fault_geo_x_y[1] - font_size_bus + 0.02)

    # list of new geo data for line (half position of switch)
    fault_annotate: PatchCollection = plot.create_annotation_collection(
        texts=[fault_text],
        coords=[fault_geo_x_y],
        size=font_size_bus,
        prop=None
    )

    return fault_annotate


def get_sc_location_annotation(net: pandapowerNet, sc_location: float, font_size_bus: float = 0.06) -> PatchCollection:
    max_bus_idx = max(net.bus.dropna(subset=['geo']).index)
    sc_text = f'\tsc_location: {sc_location * 100}%'

    # list of new geo data for line (middle of  position of switch)
    sc_geo_x_y = next(geojson.utils.coords(geojson.loads(net.bus.geo.at[max_bus_idx])))
    sc_geo_x_y = (sc_geo_x_y[0], sc_geo_x_y[1] + 0.02)

    sc_annotate: PatchCollection = plot.create_annotation_collection(
        texts=[sc_text],
        coords=[sc_geo_x_y],
        size=font_size_bus,
        prop=None
    )

    return sc_annotate


def plot_tripped_grid(net, trip_decisions, sc_location, bus_size=0.055, plot_annotations=True):
    # plot the tripped grid of net_sc
    if MPLCURSORS_INSTALLED:
        mplcursors.cursor(hover=False)

    # plot grid and color the according switches - instantaneous tripping red, int backup tripping orange and tripping_time_auto backuo-yellow

    ext_grid_busses = net.ext_grid.bus.values
    fault_location = [max(net.bus.index)]

    lc = plot.create_line_collection(net, lines=net.line.index, zorder=0)

    bc_extgrid = plot.create_bus_collection(net, buses=ext_grid_busses, zorder=1, size=bus_size, patch_type="rect")

    bc = plot.create_bus_collection(net, buses=set(net.bus.index) - set(ext_grid_busses) - set(fault_location),
                                    zorder=2, color="black", size=bus_size)

    bc_fault_location = plot.create_bus_collection(net, buses=set(fault_location), zorder=3, color="red", size=bus_size,
                                                   patch_type="circle")

    collection = [lc, bc_extgrid, bc, bc_fault_location]

    tripping_times = []

    for trip_idx in range(len(trip_decisions)):
        trip_time = trip_decisions[trip_idx].get("Trip time [s]")
        tripping_times.append(trip_time)
    tripping_times = [v for v in tripping_times if not isinf(v)]
    backup_tripping_times = copy.deepcopy(tripping_times)
    backup_tripping_times.remove(min(backup_tripping_times)) and backup_tripping_times.remove(
        heapq.nsmallest(2, backup_tripping_times)[-1])

    inst_trip_switches = []
    backup_trip_switches = []
    inst_backup_switches = []

    for trip_idx in range(len(trip_decisions)):
        trip_decision = trip_decisions[trip_idx]
        switch_id = trip_decision.get("Switch ID")
        trip = trip_decision.get("Trip")
        trip_time = trip_decision.get("Trip time [s]")

        if trip_time == heapq.nsmallest(2, tripping_times)[-1]:
            inst_backup_switches.append(switch_id)

        if trip_time == min(tripping_times):
            inst_trip_switches.append(switch_id)

        if trip_time in backup_tripping_times and trip == True:
            backup_trip_switches.append(switch_id)

    dist_to_bus = bus_size * 3.25

    # Inst relay trip, red colour
    if len(inst_trip_switches) > 0:
        sc_inst = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="red",
                                                     switches=inst_trip_switches)
        collection.append(sc_inst)

    # backup relay based on time grade (yellow colour)
    if len(backup_trip_switches) > 0:
        sc_backup = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="yellow",
                                                       switches=backup_trip_switches)

        collection.append(sc_backup)

    # orange colour for inst_backup relay
    if len(inst_backup_switches) > 0:
        instant_sc_backup = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus,
                                                               color="orange", switches=inst_backup_switches)

        collection.append(instant_sc_backup)

    len_sc = len(set(net.switch.index) - set(inst_trip_switches) - set(backup_trip_switches))

    if len_sc != 0:
        # closed switch-black
        sc = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="black",
                                                switches=set(net.switch.index) - set(inst_trip_switches) - set(
                                                    backup_trip_switches))
        collection.append(sc)

    # make annotations optional (if True then annotate else only plot)
    if plot_annotations:
        # line annotations
        line_text = []
        line_geodata = []

        fault_current: float = .0

        # for Switches in trip_decisions:
        for line in net.line.index:

            if line == max(net.line.index):
                break

            # annotate line_id
            text_line = r"  line_" + str(line)  # + ",sw_"+str(Switch_index)

            # get bus_index from the line (from switch)
            get_bus_index = pp.get_connected_buses_at_element(net, element_index=line, element_type='l',
                                                              respect_in_service=False)

            bus_list = list(get_bus_index)
            bus_coords: List[Tuple[float, float]] = [
                geojson.utils.coords(geojson.loads(net.bus.geo.at[bus])) for bus in bus_list
            ]

            # TODO:
            # place annotations on middle of the line
            line_geo_x = (bus_coords[0][0] + bus_coords[1][0]) / 2
            line_geo_y = ((bus_coords[0][1] + bus_coords[1][1]) / 2) + 0.05

            line_geo_x_y = [line_geo_x, line_geo_y]

            # list of new geo data for line (half position of switch)
            line_geodata.append(tuple(line_geo_x_y))

            fault_current = round(net.res_bus_sc['ikss_ka'].at[max(net.bus.index)], 2)
            # round(Switches['Fault Current'],2)

            line_text.append(text_line)

        # line annotations to collections for plotting
        line_annotate = plot.create_annotation_collection(texts=line_text, coords=line_geodata, size=0.06, prop=None)
        collection.append(line_annotate)

        # Bus Annotatations
        bus_text = []
        for i in net.bus.geo.dropna().index:
            bus_texts = 'bus_' + str(i)

            bus_text.append(bus_texts)

        bus_text = bus_text[:-1]

        bus_geodata = net.bus.geo.dropna().apply(geojson.loads).apply(geojson.utils.coords).apply(next).to_list()

        # placing bus
        bus_index = [(x[0] - 0.11, x[1] + 0.095) for x in bus_geodata]

        # TODO:
        bus_annotate = plot.create_annotation_collection(texts=bus_text, coords=bus_index, size=0.06, prop=None)
        collection.append(bus_annotate)

        # Short circuit annotations
        collection.append(get_fault_annotation(net, fault_current))

        # sc_location annotation
        collection.append(get_sc_location_annotation(net, sc_location))

        # switch annotations
        # from pandapower.protection.implemeutility_functions import switch_geodata
        switch_text = []
        for Switches in trip_decisions:
            Switch_index = Switches['Switch ID']

            text_switch = r"sw_" + str(Switch_index)
            switch_text.append(text_switch)

        switch_geodata = switch_geodatas(net, size=bus_size, distance_to_bus=3.25 * bus_size)
        i = 0
        for i in range(len(switch_geodata)):
            switch_geodata[i]['x'] = switch_geodata[i]['x'] - 0.085  # scale the value if annotations overlap
            switch_geodata[i]['y'] = switch_geodata[i]['y'] + 0.055  # scale the value if annotations overlap
            i = i + 1
        switch_annotate = plot.create_annotation_collection(texts=switch_text, coords=switch_geodata, size=0.06,
                                                            prop=None)
        collection.append(switch_annotate)
    plot.draw_collections(collection)


def plot_tripped_grid_protection_device(net, trip_decisions, sc_location, sc_bus, bus_size=0.055,
                                        plot_annotations=True):
    # plot the tripped grid of net_sc with networks using ProtectionDevice class
    if MPLCURSORS_INSTALLED:
        mplcursors.cursor(hover=False)

    # plot grid and color the according switches - instantaneous tripping red, int backup tripping orange and tripping_time_auto backuo-yellow

    ext_grid_busses = net.ext_grid.bus.values
    fault_location = [max(net.bus.index)]

    lc = plot.create_line_collection(net, lines=net.line.index, zorder=0)

    bc_extgrid = plot.create_bus_collection(net, buses=ext_grid_busses, zorder=1, size=bus_size, patch_type="rect")

    bc = plot.create_bus_collection(net, buses=set(net.bus.index) - set(ext_grid_busses) - set(fault_location),
                                    zorder=2, color="black", size=bus_size)

    bc_fault_location = plot.create_bus_collection(net, buses=set(fault_location), zorder=3, color="red", size=bus_size,
                                                   patch_type="circle")

    collection = [lc, bc_extgrid, bc, bc_fault_location]

    tripping_times = []

    for trip_idx in range(len(trip_decisions)):
        trip_time = trip_decisions.trip_melt_time_s.at[trip_idx]
        tripping_times.append(trip_time)
    tripping_times = [v for v in tripping_times if not isinf(v)]
    if len(tripping_times) == 0:
        return
    backup_tripping_times = copy.deepcopy(tripping_times)
    backup_tripping_times.remove(min(backup_tripping_times)) and backup_tripping_times.remove(
        heapq.nsmallest(2, backup_tripping_times)[-1])

    inst_trip_switches = []
    backup_trip_switches = []
    inst_backup_switches = []
    bus_bus_switches = net.switch.index[net.switch.et == "b"]

    # add trafo to collection
    if len(net.trafo) > 0:
        trafo_collection = plot.create_trafo_collection(net, size=2 * bus_size)
        trafo_conn_collection = plot.create_trafo_connection_collection(net)
        collection.append(trafo_collection)
        collection.append(trafo_conn_collection)
    # add load to collection
    if len(net.load) > 0:
        load_collection = plot.create_load_collection(net, size=2 * bus_size)
        collection.append(load_collection)

    for trip_idx in range(len(trip_decisions)):
        trip_decision = trip_decisions.iloc[[trip_idx]]
        switch_id = trip_decision.switch_id.at[trip_idx]
        trip = trip_decision.trip_melt.at[trip_idx]
        trip_time = trip_decision.trip_melt_time_s.at[trip_idx]

        if trip_time == heapq.nsmallest(2, tripping_times)[-1]:
            inst_backup_switches.append(switch_id)

        if trip_time == min(tripping_times):
            inst_trip_switches.append(switch_id)

        if trip_time in backup_tripping_times and trip == True:
            backup_trip_switches.append(switch_id)

    dist_to_bus = bus_size * 3.25

    # Inst relay trip, red colour
    if len(inst_trip_switches) > 0:
        sc_inst = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="red",
                                                     switches=set(inst_trip_switches) - set(bus_bus_switches))
        bb_inst = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="red")
        collection.append(sc_inst)
        collection.append(bb_inst)

    # backup relay based on time grade (yellow colour)
    if len(backup_trip_switches) > 0:
        sc_backup = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="yellow",
                                                       switches=backup_trip_switches)
        bb_backup = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="yellow")
        collection.append(sc_backup)
        collection.append(bb_backup)

    # orange colour for inst_backup relay
    if len(inst_backup_switches) > 0:
        instant_sc_backup = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus,
                                                               color="orange", switches=inst_backup_switches)
        instant_bb_backup = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="orange")
        collection.append(instant_sc_backup)

    len_sc = len(set(net.switch.index) - set(inst_trip_switches) - set(backup_trip_switches))

    if len_sc != 0:
        # closed switch-black
        sc = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="black",
                                                switches=set(net.switch.index) - set(inst_trip_switches) - set(
                                                    backup_trip_switches) - set(bus_bus_switches))
        bb = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="black")

        collection.append(sc)
        collection.append(bb)

    # make annotations optional (if True then annotate else only plot)
    if plot_annotations:
        # line annotations
        line_text = []
        line_geodata = []

        fault_current = None

        # for Switches in trip_decisions:
        for line in net.line.index:

            if line == max(net.line.index):
                break

            # annotate line_id
            text_line = r"  line_" + str(line)  # + ",sw_"+str(Switch_index)

            # get bus_index from the line (from switch)
            get_bus_index = pp.get_connected_buses_at_element(net, element_index=line, element_type='l',
                                                              respect_in_service=False)

            bus_list = list(get_bus_index)

            # place annotations on middle of the line
            bus_coords = list(
                zip(*net.bus.geo.iloc[bus_list[0:2]].apply(geojson.loads).apply(geojson.utils.coords).apply(
                    next).to_list()))
            line_geo_x_y = [sum(x) / 2 for x in bus_coords]
            line_geo_x_y[1] += 0.05

            # list of new geo data for line (half position of switch)
            line_geodata.append(tuple(line_geo_x_y))

            fault_current = round(net.res_bus_sc['ikss_ka'].at[sc_bus], 2)  # round(Switches['Fault Current'],2)

            line_text.append(text_line)

        # line annotations to collections for plotting
        line_annotate = plot.create_annotation_collection(texts=line_text, coords=line_geodata, size=0.06, prop=None)
        collection.append(line_annotate)

        # Bus Annotations
        bus_text = []
        for i in net.bus.index:
            bus_texts = f'bus_{i}'
            bus_text.append(bus_texts)

        bus_text = bus_text[:-1]

        bus_geodata = net.bus.geo.apply(geojson.loads).apply(geojson.utils.coords).apply(next).to_list()

        # placing bus
        bus_geodata = [(x[0] - 0.11, x[1] + 0.095) for x in bus_geodata]

        bus_annotate = plot.create_annotation_collection(texts=bus_text, coords=bus_geodata, size=0.06, prop=None)
        collection.append(bus_annotate)

        max_bus_idx = max(net.bus.dropna(subset=['geo']).index)

        # Short circuit annotations
        collection.append(get_fault_annotation(net, fault_current))

        # sc_location annotation
        collection.append(get_sc_location_annotation(net, sc_location))

        # switch annotations
        # from pandapower.protection.utility_functions import switch_geodata
        switch_text = []
        for switch_id in trip_decisions.switch_id:
            text_switch = r"sw_" + str(switch_id)
            switch_text.append(text_switch)

        switch_geodata = switch_geodatas(net, size=bus_size, distance_to_bus=3.25 * bus_size)
        for i, (x, y) in enumerate(switch_geodata):
            switch_geodata[i] = (x - 0.085, y + 0.055)
        switch_annotate = plot.create_annotation_collection(
            texts=switch_text,
            coords=switch_geodata,
            size=0.06,
            prop=None
        )
        collection.append(switch_annotate)
    plot.draw_collections(collection)


def calc_line_intersection(m1, b1, m2, b2):
    xi = (b1 - b2) / (m2 - m1)
    yi = m1 * xi + b1
    return xi, yi


# get connected lines using bus id
@deprecated("Use pandapower.get_connected_elements(net, 'line', bus_idx) instead!")
def get_connected_lines(net, bus_idx):
    return pp.get_connected_elements(net, "line", bus_idx)


# Returns the index of the second bus an element is connected to, given a
# first one. E.g. the from_bus given the to_bus of a line.
@deprecated("Use pandapower.next_bus(net, bus, element_id instead!")
def next_buses(net, bus, element_id):
    return pp.next_bus(net, bus, element_id)


# get the connected bus listr from start to end bus
def source_to_end_path(net, start_bus, bus_list, bus_order):
    connected_lines = pp.get_connected_elements(net, 'line', start_bus)
    flag = 0
    for line in connected_lines:
        next_connected_bus = pp.next_bus(net, bus=start_bus, element_id=line)
        bus_order_1 = bus_order.copy()
        if next_connected_bus in bus_order:
            continue
        else:
            bus_order.append(next_connected_bus)
            bus_list = source_to_end_path(net, next_connected_bus, bus_list, bus_order)

            bus_order = bus_order_1
            flag = 1
    if flag == 0:
        bus_list.append(bus_order)

    return bus_list


# get connected switches with bus
@deprecated("Use pandapower.get_connected_switches(net, buses, consider='l', status='closed') instead!")
def get_connected_switches(net, buses):
    return pp.get_connected_switches(net, buses, consider='l', status="closed")


# get connected buses with a oven element
@deprecated(
    "Use pandapower.get_connected_buses_at_element(net, element, element_type='l', respect_in_service=False) instead!"
)
def connected_bus_in_line(net, element):
    return pp.get_connected_buses_at_element(net, element, element_type='l', respect_in_service=False)


@deprecated("Use networkx topological search instead! See pandapower docs.")
def get_line_path(net, bus_path, sc_line_id=0):
    """line path from bus path """
    line_path = []
    for i in range(len(bus_path) - 1):
        bus1 = bus_path[i]
        bus2 = bus_path[i + 1]

        # line=net.line [(net.line.from_bus==bus1) & (net.line.to_bus==bus2)].index.item()

        line_path.extend(net.line[((net.line.from_bus == bus1) & (net.line.to_bus == bus2)) | (
                (net.line.from_bus == bus2) & (net.line.to_bus == bus1))].index.to_list())

    return line_path


def switch_geodatas(net, size, distance_to_bus):
    """get the coordinates for switches at middle of the line"""
    switch_geo = []
    switches = []

    if len(switches) == 0:
        lbs_switches = net.switch.index[net.switch.et == "l"]
    else:
        lbs_switches = switches

    for switch in lbs_switches:
        sb = net.switch.bus.loc[switch]
        line = net.line.loc[net.switch.element.loc[switch]]
        fb = line.from_bus
        tb = line.to_bus

        line_buses = {fb, tb}
        target_bus = list(line_buses - {sb})[0]

        pos_sb = _get_coords_from_bus_idx(net, sb)
        if len(pos_sb) > 1:
            ValueError(f'Bus {sb} has multiple geodata entries: {pos_sb}')
        if len(pos_sb) == 0:
            ValueError(f'Bus {sb} has no geodata entry.')
        pos_sb = pos_sb[0]
        pos_tb = np.zeros(2)

        pos_tb = _get_coords_from_bus_idx(net, target_bus)
        if len(pos_sb) > 1:
            ValueError(f'Bus {sb} has multiple geodata entries: {pos_sb}')
        if len(pos_sb) == 0:
            ValueError(f'Bus {sb} has no geodata entry.')
        pos_tb = pos_tb[0]

        # position of switch symbol
        vec = np.array(pos_tb) - np.array(pos_sb)
        mag = np.linalg.norm(vec)

        pos_sw = pos_sb + vec / mag * distance_to_bus
        switch_geo.append(pos_sw)
    return switch_geo


def create_I_t_plot(trip_decisions, switch_id):
    """function create I-T plot using tripping decisions"""
    if not MATPLOTLIB_INSTALLED:
        raise ImportError('matplotlib must be installed to run create_I_t_plot()')

    x = [0, 0, 0, 0]
    y = [0, 0, 0, 0]

    X = []
    for counter, switch_id in enumerate(switch_id):
        lst_I = [trip_decisions[switch_id]['Ig'], trip_decisions[switch_id]['Igg']]
        lst_t = [trip_decisions[switch_id]['tg'], trip_decisions[switch_id]['tgg']]
        fault_current = trip_decisions[switch_id]['Fault Current [kA]']

        label = 'Relay:R' + str(switch_id)
        x[counter] = [lst_I[0], lst_I[0], lst_I[1], lst_I[1], 100]
        y[counter] = [10, lst_t[0], lst_t[0], lst_t[1], lst_t[1]]

        X.append([x[counter], y[counter], label, fault_current])

    plt.figure()

    for count, data in enumerate(X):
        plt.loglog(X[count][0], X[count][1], label=X[count][2])

        plt.axvline(x=X[count][3], ymin=0, ymax=X[0][1][0], color='r')
        plt.text(X[count][3] + 0.01, 0.01, 'Ikss  ' + str(round(fault_current, 1)) + 'kA', rotation=0)

        plt.grid(True, which="both", ls="-")
        plt.title("I-t-plot")
        plt.xlabel("I [kA]")
        plt.ylabel("t [s]")
        plt.xlim(0.1, 11)
        plt.ylim(0.01, 11)

    plt.show()
    plt.legend()

    if MPLCURSORS_INSTALLED:
        # hover the plott
        cursor = mplcursors.cursor(hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            'I:{} kA,t:{} s'.format(round(sel.target[0], 2), round(sel.target[1], 2))))


def power_flow_end_points(net):
    """function calculate end point from meshed grid and the start point from the radial grid to ext grid"""

    pf_net = copy.deepcopy(net)
    pp.runpp(pf_net)
    pf_loop_end_buses = []
    pf_radial_end_buses = []

    for bus in pf_net.bus.index:

        lines = pp.get_connected_elements(pf_net, element_type='l', buses=bus)

        if len(lines) == 1:
            pf_radial_end_buses.append(bus)
        else:
            pf_endpoint = []
            for line in lines:
                if bus == pf_net.line.from_bus.at[line]:
                    if pf_net.res_line.p_from_mw[line] < 0:
                        endpoint = True
                    else:
                        endpoint = False
                else:
                    if pf_net.res_line.p_to_mw[line] < 0:
                        endpoint = True
                    else:
                        endpoint = False
                pf_endpoint.append(endpoint)

            if all(pf_endpoint):
                pf_loop_end_buses.append(bus)

    return pf_loop_end_buses, pf_radial_end_buses


def bus_path_from_to_bus(net, radial_start_bus, loop_start_bus, end_bus):
    """function calculate the bus path from start and end buses"""
    loop_path = []
    radial_path = []

    from pandapower.topology.create_graph import create_nxgraph
    G = create_nxgraph(net)

    for start_buses in radial_start_bus:
        radial = nx.shortest_path(G, source=start_buses, target=end_bus)
        radial_path.append(radial)

    for start_buses in loop_start_bus:
        for path in nx.all_simple_paths(G, source=start_buses, target=end_bus):
            loop_path.append(path)

    bus_path = radial_path + loop_path
    return bus_path


def get_switches_in_path(net, paths):
    """function calculate the switching times from the  bus path"""

    lines_in_path: List[List] = []

    for path in paths:
        lines_at_path: set = set()

        for bus in path:
            lines_at_path.update(pp.get_connected_elements(net, "l", bus))

        lines_at_paths = [
            line for line in lines_at_path
            if net.line.from_bus[line] in path and net.line.to_bus[line] in path
        ]

        lines_in_path.append(lines_at_paths)

    switches_in_path = [
        [net.switch[(net.switch['et'] == 'l') & (net.switch['element'] == line)].index for line in line_path]
        for line_path in lines_in_path
    ]

    return switches_in_path


def get_vi_angle(net: pandapowerNet, switch_id: int, **kwargs) -> float:
    """calculate the angle between voltage and current with reference to voltage"""

    if "powerflow_results" in kwargs:
        logger.warning(
            "The powerflow_results argument is deprecated and will be removed in the future."
        )

    pp.runpp(net)
    line_idx = net.switch.element.at[switch_id]

    if get_from_bus_info_switch(net, switch_id):
        p = net.res_line_sc.p_from_mw.at[line_idx]
        q = net.res_line_sc.q_from_mvar.at[line_idx]
    else:
        p = net.res_line_sc.p_to_mw.at[line_idx]
        q = net.res_line_sc.q_to_mvar.at[line_idx]

    if p > 0 and q > 0:
        vi_angle = math.degrees(math.atan(q / p))
    elif p < 0 <= q:
        vi_angle = math.degrees(math.atan(q / p)) + 180
    elif p < 0 and q < 0:
        vi_angle = math.degrees(math.atan(q / p)) - 180
    elif p == 0 < q:
        vi_angle = 90
    elif p == 0 > q:
        vi_angle = -90
    else:
        vi_angle = math.inf
    return vi_angle


def bus_path_multiple_ext_bus(net):
    G = create_nxgraph(net)
    bus_path = []
    for line_id in net.line.index:

        # line_id=62
        from_bus = net.line.from_bus.at[line_id]
        to_bus = net.line.to_bus.at[line_id]
        max_bus_path = []

        if net.trafo.empty:
            # for ext_bus in net.ext_grid.bus:
            for ext_bus in set(net.ext_grid.bus):
                from_bus_path = nx.shortest_path(G, source=ext_bus, target=from_bus)
                to_bus_path = nx.shortest_path(G, source=ext_bus, target=to_bus)

                if len(from_bus_path) == len(to_bus_path):
                    from_bus_path.append(to_bus_path[-1])
                    max_bus_path.append(from_bus_path)

                elif len(from_bus_path) != len(to_bus_path):
                    if len(from_bus_path) > 1 and len(to_bus_path) > 1:
                        min_len = min(len(from_bus_path), len(to_bus_path))
                        if from_bus_path[min_len - 1] != to_bus_path[min_len - 1]:
                            if len(from_bus_path) < len(to_bus_path):
                                from_bus_path.append(to_bus_path[-1])
                                max_bus_path.append(from_bus_path)
                            else:
                                to_bus_path.append(from_bus_path[-1])
                                max_bus_path.append(to_bus_path)
                        else:
                            max_bus_path.append(max([from_bus_path, to_bus_path]))
                    else:
                        max_bus_path.append(max([from_bus_path, to_bus_path]))

            bus_path.append(sorted(max_bus_path, key=len)[0])

    return bus_path


# get the line path from the given bus path
def get_line_path(net, bus_path):
    """ Function return the list of line path from the given bus path"""
    line_path = []
    for i in range(len(bus_path) - 1):
        bus1 = bus_path[i]
        bus2 = bus_path[i + 1]
        line1 = net.line[(net.line.from_bus == bus1) & (net.line.to_bus == bus2)].index.to_list()
        line2 = net.line[(net.line.from_bus == bus2) & (net.line.to_bus == bus1)].index.to_list()
        if len(line2) == 0:
            line_path.append(line1[0])

        if len(line1) == 0:
            line_path.append(line2[0])
    return line_path


def parallel_lines(net):
    """ Function return the list of parallel lines in the network"""

    parallel = []
    new_parallel = []

    # parallel_lines
    for i in net.line.index:
        for j in net.line.index:
            if i == j:
                continue

            i_from_bus = net.line.loc[i].from_bus
            i_to_bus = net.line.loc[i].to_bus
            j_from_bus = net.line.loc[j].from_bus
            j_to_bus = net.line.loc[j].to_bus

            if (
                    (i_from_bus == j_from_bus and i_to_bus == j_to_bus)
                    or (i_from_bus == j_to_bus and i_to_bus == j_from_bus)
                    or (i_to_bus == j_from_bus and i_from_bus == j_to_bus)
                    or (i_to_bus == j_to_bus and i_from_bus == j_from_bus)
            ):
                parallel.append([i, j])

        parallel_line = [list(i) for i in set(map(tuple, parallel))]

        # remove duplicates
        new_parallel = []
        for line in parallel_line:
            if line not in new_parallel:
                new_parallel.append(line)

    return new_parallel


def read_fuse_from_std_type():
    # write characteristic of fuse from fuse_std_library
    # net,overwrite already existing characteristic
    # raise error if name of fuse is not available in fuse library
    raise NotImplementedError("This function is not implemented yet.")

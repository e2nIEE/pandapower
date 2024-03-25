
#This function includes various function used for general functanalities such as plotting, grid search

import copy
import numpy as np
import math
from math import isinf
import heapq
import networkx as nx
import logging as log

import pandapower as pp
import pandapower.plotting as plot
from pandapower.topology.create_graph import create_nxgraph

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

import warnings
warnings.filterwarnings('ignore')


def create_sc_bus(net_copy, sc_line_id, sc_fraction):
    #    This function creates a short-circuit location (a bus) on a line.
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
    bus_sc = pp.create_bus(net,name = "Bus_SC", vn_kv = bus_vn_kv, type = "b", index = max_idx_bus+1)

    #sim bench grids
    if 's_sc_max_mva' not in net.ext_grid:
        print('input s_sc_max_mva or taking 1000')
        net.ext_grid['s_sc_max_mva'] = 1000
    if 'rx_max' not in net.ext_grid:
        print('input rx_max or taking 0.1')
        net.ext_grid['rx_max'] = 0.1
    if 'k' not in net.sgen and len(net.sgen) !=0:
        print('input  Ratio of nominal current to short circuit current- k or  taking k=1')
        net.sgen['k']=1

    # set new lines
    sc_line1 = sc_line_id
    net.line.at[sc_line1, 'to_bus'] = bus_sc
    net.line.at[sc_line1, 'length_km'] *= sc_fraction

    sc_line2 = pp.create_line_from_parameters(net, bus_sc, aux_line.to_bus, length_km = aux_line.length_km*(1-sc_fraction),
    index = max_idx_line+1, r_ohm_per_km = aux_line.r_ohm_per_km, x_ohm_per_km = aux_line.x_ohm_per_km, c_nf_per_km = aux_line.c_nf_per_km, max_i_ka = aux_line.max_i_ka)

    if 'endtemp_degree' in net.line.columns:
        net.line.at[sc_line2, "endtemp_degree"] = net.line.endtemp_degree.at[sc_line1]

    net.line = net.line.sort_index()

    # check if switches are connected to the line and set the switches to new lines
    for switch_id in net.switch.index:
        if (aux_line.from_bus == net.switch.bus[switch_id]) & (net.switch.element[switch_id] == sc_line_id):
            net.switch.element[switch_id] = sc_line1
        elif (aux_line.to_bus == net.switch.bus[switch_id]) & (net.switch.element[switch_id] == sc_line_id):
            net.switch.element[switch_id] = sc_line2

    # set geodata for new bus
    net.bus_geodata.loc[max_idx_bus+1] = None

    x1 = net.bus_geodata.x.at[aux_line.from_bus] #x-coordinate of from_bus
    x2 = net.bus_geodata.x.at[aux_line.to_bus] #x-coordinate of to_bus
    y1 = net.bus_geodata.y.at[aux_line.from_bus]
    y2 = net.bus_geodata.y.at[aux_line.to_bus]

    net.bus_geodata.at[max_idx_bus+1, "x"] = sc_fraction*(x2-x1) + x1
    net.bus_geodata.at[max_idx_bus+1, "y"] = sc_fraction*(y2-y1) + y1

    return net


def calc_faults_at_full_line(net, line, location_step_size = 0.01, start_location = 0.01, end_location = 1, sc_case = "min"):
    # functon to create sc at full line
    import pandapower.shortcircuit as sc
    i = 0

    fault_currents = []
    max_bus_idx = max(net.bus.index)
    location = start_location
    while location < end_location:
        net_sc = create_sc_bus(net, line, location)
        sc.calc_sc(net_sc, case = sc_case)

        fault_currents.append(net_sc.res_bus_sc.ikss_ka.at[max_bus_idx + 1])

        i+=1
        location = start_location + i*location_step_size
    return fault_currents


def get_line_idx(net, switch_id):
    # get the line id from swithc id
    line_idx = net.switch.element.at[switch_id]
    return line_idx


def get_bus_idx(net, switch_id):
    # get the bus id using switch if
    bus_idx = net.switch.bus.at[switch_id]
    return bus_idx


def get_opposite_side_bus_from_switch(net, switch_id):
    # get the frm and to bus of switch
    line_idx = get_line_idx(net, switch_id)
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
    bus_idx = get_bus_idx(net,switch_id)
    line_idx = get_line_idx(net, switch_id)

    for line in net.line.index: # can be written better
        if line == line_idx:
            if bus_idx == net.line.from_bus.at[line_idx]: # asks if switch is at from_bus
                is_from_bus = True
                # sc_fraction = 0.95
            else:                                             # else it is at to_bus
                is_from_bus = False
                # sc_fraction = 0.05

    return is_from_bus


def get_from_bus_info_bus_line(net, bus_idx, line_idx):
    # get bus nfo of given line
    for line in net.line.index: # can be written better
        if line == line_idx:
            if bus_idx == net.line.from_bus.at[line_idx]: # asks if switch is at from_bus
                is_from_bus = True
                # sc_fraction = 0.95
            else:                                             # else it is at to_bus
                is_from_bus = False
                # sc_fraction = 0.05

    return is_from_bus


def get_line_impedance(net, line_idx):
    # get line impedence
    line_length = net.line.length_km.at[line_idx]
    line_r_per_km = net.line.r_ohm_per_km.at[line_idx]
    line_x_per_km = net.line.x_ohm_per_km.at[line_idx]
    Z_line = complex(line_r_per_km*line_length, line_x_per_km*line_length) # Z = R + jX
    return Z_line


def get_lowest_impedance_line(net, lines):
    # get the low impedenceline
    i=0
    for line in lines:
        impedance = abs(get_line_impedance(net, line))
        if i==0:
            min_imp_line = line
            min_impedance = impedance
        else:
            if impedance < min_impedance:
                min_impedance = impedance
                min_imp_line = line
        i+=1
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

        pp.fuse_buses(net,bus1,bus2)

    return net


def plot_tripped_grid(net, trip_decisions, sc_location, bus_size = 0.055, plot_annotations=True):
    # plot the tripped grid of net_sc
    if MPLCURSORS_INSTALLED:
        mplcursors.cursor(hover=False)

    # plot grid and color the according switches - instantaneous tripping red, int backup tripping orange and tripping_time_auto backuo-yellow

    ext_grid_busses = net.ext_grid.bus.values
    fault_location = [max(net.bus.index)]

    lc = plot.create_line_collection(net, lines = net.line.index, zorder=0)

    bc_extgrid = plot.create_bus_collection(net, buses = ext_grid_busses, zorder=1, size = bus_size, patch_type = "rect")

    bc = plot.create_bus_collection(net, buses = set(net.bus.index) - set(ext_grid_busses) - set(fault_location), zorder=2, color = "black", size = bus_size)

    bc_fault_location = plot.create_bus_collection(net, buses = set(fault_location), zorder=3, color = "red", size = bus_size, patch_type = "circle")

    collection = [lc, bc_extgrid, bc, bc_fault_location ]

    tripping_times = []

    for trip_idx in range(len(trip_decisions)):
       trip_time= trip_decisions[trip_idx].get("Trip time [s]")
       tripping_times.append(trip_time)
    tripping_times = [v for v in tripping_times if not isinf(v)]
    backup_tripping_times=copy.deepcopy(tripping_times)
    backup_tripping_times.remove(min(backup_tripping_times)) and  backup_tripping_times.remove(heapq.nsmallest(2,backup_tripping_times)[-1])

    inst_trip_switches = []
    backup_trip_switches = []
    inst_backup_switches=[]

    for trip_idx in range(len(trip_decisions)):
       trip_decision = trip_decisions[trip_idx]
       switch_id = trip_decision.get("Switch ID")
       trip = trip_decision.get("Trip")
       trip_time=trip_decision.get("Trip time [s]")

       if trip_time== heapq.nsmallest(2,tripping_times)[-1]:
               inst_backup_switches.append(switch_id)

       if trip_time==min(tripping_times):
               inst_trip_switches.append(switch_id)

       if trip_time in backup_tripping_times and trip==True:

           backup_trip_switches.append(switch_id)

    dist_to_bus = bus_size * 3.25

    #Inst relay trip, red colour
    if  len(inst_trip_switches)>0:

        sc_inst = plot.create_line_switch_collection(net, size = bus_size, distance_to_bus = dist_to_bus, color = "red", switches = inst_trip_switches)
        collection.append(sc_inst)

    #backup relay based on time grade (yellow colour)
    if  len(backup_trip_switches)>0:

        sc_backup = plot.create_line_switch_collection(net, size = bus_size, distance_to_bus = dist_to_bus, color = "yellow", switches = backup_trip_switches)

        collection.append(sc_backup)

    # orange colour for inst_backup relay
    if  len(inst_backup_switches)>0:

        instant_sc_backup = plot.create_line_switch_collection(net, size = bus_size, distance_to_bus = dist_to_bus, color = "orange", switches = inst_backup_switches)

        collection.append(instant_sc_backup)

    len_sc=len( set(net.switch.index) - set(inst_trip_switches)- set(backup_trip_switches))

    if len_sc!=0:

        #closed switch-black
        sc = plot.create_line_switch_collection(net, size = bus_size, distance_to_bus = dist_to_bus, color = "black",
                                            switches = set(net.switch.index) - set(inst_trip_switches)- set(backup_trip_switches))
        collection.append(sc)

    #make annotations optional (if True then annotate else only plot)
    if plot_annotations:
        # line annotations
        line_text=[]
        line_geodata=[]

        #for Switches in trip_decisions:
        for line in net.line.index:

            if line== max(net.line.index):
                break

            #annotate line_id
            text_line = r"  line_"+str(line)#+ ",sw_"+str(Switch_index)

            #get bus_index from the line (from switch)
            get_bus_index= pp.get_connected_buses_at_element(net, element_index=line, element_type='l', respect_in_service=False)

            bus_list=list(get_bus_index)

            #place annotations on middle of the line
            line_geo_x=(net.bus_geodata.iloc[bus_list[0]].x+ net.bus_geodata.iloc[bus_list[1]].x)/2

            line_geo_y=((net.bus_geodata.iloc[bus_list[0]].y+ net.bus_geodata.iloc[bus_list[1]].y)/2)+0.05

            line_geo_x_y=[line_geo_x,line_geo_y]

            # list of new geo data for line (half position of switch)
            line_geodata.append(tuple(line_geo_x_y))


            fault_current= round(net.res_bus_sc['ikss_ka'].at[max(net.bus.index)],2)  #round(Switches['Fault Current'],2)

            line_text.append(text_line)

        #line annotations to collections for plotting
        line_annotate=plot.create_annotation_collection( texts=line_text, coords=line_geodata, size=0.06, prop=None)
        collection.append(line_annotate)

        #Bus Annotatations
        bus_text=[]
        for i in net.bus_geodata.index:
            bus_texts='bus_'+str(i)

            bus_text.append(bus_texts)

        bus_text=bus_text[:-1]

        bus_geodata = net.bus_geodata[['x', 'y']]

        #placing bus
        bus_geodata['x'] = bus_geodata['x'] - 0.11
        bus_geodata['y'] = bus_geodata['y'] + 0.095

        bus_index= [tuple(x) for x in bus_geodata.to_numpy()]
        bus_annotate=plot.create_annotation_collection( texts=bus_text, coords=bus_index, size=0.06, prop=None)
        collection.append(bus_annotate)

        # Short circuit annotations
        fault_geodata=[]

        fault_text=[]

        fault_texts='    I_sc = '+str(fault_current)+'kA'

        font_size_bus=0.06  # font size of fault location  text

        fault_geo_x=net.bus_geodata.iloc[max(net.bus_geodata.index)][0]
        fault_geo_y=net.bus_geodata.iloc[max(net.bus_geodata.index)][1]-font_size_bus+0.02

        fault_geo_x_y=[fault_geo_x,fault_geo_y]

        # list of new geo data for line (half position of switch)
        fault_geodata.append(tuple(fault_geo_x_y))

        fault_text.append(fault_texts)
        fault_annotate=plot.create_annotation_collection( texts=fault_text, coords=fault_geodata, size=0.06, prop=None)

        collection.append(fault_annotate)

        # sc_location annotation
        sc_text=[]
        sc_geodata=[]

        sc_texts='   sc_location: '+str(sc_location*100)+'%'

        #font_size_bus=0.06  # font size of sc location

        sc_geo_x=net.bus_geodata.iloc[max(net.bus_geodata.index)][0]

        sc_geo_y=net.bus_geodata.iloc[max(net.bus_geodata.index)][1]+0.02

        sc_geo_x_y=[sc_geo_x,sc_geo_y]

        # list of new geo data for line (middle of  position of switch)
        sc_geodata.append(tuple(sc_geo_x_y))

        sc_text.append(sc_texts)
        sc_annotate=plot.create_annotation_collection( texts=sc_text, coords=sc_geodata, size=0.06, prop=None)

        collection.append(sc_annotate)

        # switch annotations
        #from pandapower.protection.implemeutility_functions import switch_geodata
        switch_text=[]
        for Switches in trip_decisions:

            Switch_index=Switches['Switch ID']

            text_switch= r"sw_"+str(Switch_index)
            switch_text.append(text_switch)

        switch_geodata= switch_geodatas(net, size= bus_size, distance_to_bus=3.25* bus_size)
        i=0
        for i in range(len(switch_geodata)):

            switch_geodata[i]['x'] = switch_geodata[i]['x'] - 0.085  #scale the value if annotations overlap
            switch_geodata[i]['y'] = switch_geodata[i]['y'] + 0.055  #scale the value if annotations overlap
            i=i+1
        switch_annotate=plot.create_annotation_collection( texts=switch_text, coords=switch_geodata, size=0.06, prop=None)
        collection.append(switch_annotate)
    plot.draw_collections(collection)


def plot_tripped_grid_protection_device(net, trip_decisions, sc_location, sc_bus, bus_size=0.055, plot_annotations=True):
    # plot the tripped grid of net_sc with networks using ProtectionDevice class
    if MPLCURSORS_INSTALLED:
        mplcursors.cursor(hover=False)

    # plot grid and color the according switches - instantaneous tripping red, int backup tripping orange and tripping_time_auto backuo-yellow

    ext_grid_busses = net.ext_grid.bus.values
    fault_location = [max(net.bus.index)]

    lc = plot.create_line_collection(net, lines = net.line.index, zorder=0)

    bc_extgrid = plot.create_bus_collection(net, buses = ext_grid_busses, zorder=1, size = bus_size, patch_type = "rect")

    bc = plot.create_bus_collection(net, buses = set(net.bus.index) - set(ext_grid_busses) - set(fault_location), zorder=2, color = "black", size = bus_size)

    bc_fault_location = plot.create_bus_collection(net, buses = set(fault_location), zorder=3, color = "red", size = bus_size, patch_type = "circle")

    collection = [lc, bc_extgrid, bc, bc_fault_location ]

    tripping_times = []

    for trip_idx in range(len(trip_decisions)):
       trip_time= trip_decisions.trip_melt_time_s.at[trip_idx]
       tripping_times.append(trip_time)
    tripping_times = [v for v in tripping_times if not isinf(v)]
    if len(tripping_times) == 0:
        return
    backup_tripping_times=copy.deepcopy(tripping_times)
    backup_tripping_times.remove(min(backup_tripping_times)) and backup_tripping_times.remove(heapq.nsmallest(2,backup_tripping_times)[-1])

    inst_trip_switches = []
    backup_trip_switches = []
    inst_backup_switches=[]
    bus_bus_switches = net.switch.index[net.switch.et == "b"]

    # add trafo to collection
    if len(net.trafo) > 0:
        trafo_collection = plot.create_trafo_collection(net, size=2*bus_size)
        trafo_conn_collection = plot.create_trafo_connection_collection(net)
        collection.append(trafo_collection)
        collection.append(trafo_conn_collection)
    # add load to collection
    if len(net.load) > 0:
        load_collection = plot.create_load_collection(net, size=2*bus_size)
        collection.append(load_collection)

    for trip_idx in range(len(trip_decisions)):
       trip_decision = trip_decisions.iloc[[trip_idx]]
       switch_id = trip_decision.switch_id.at[trip_idx]
       trip = trip_decision.trip_melt.at[trip_idx]
       trip_time=trip_decision.trip_melt_time_s.at[trip_idx]

       if trip_time== heapq.nsmallest(2,tripping_times)[-1]:
               inst_backup_switches.append(switch_id)

       if trip_time==min(tripping_times):
               inst_trip_switches.append(switch_id)

       if trip_time in backup_tripping_times and trip==True:

           backup_trip_switches.append(switch_id)

    dist_to_bus = bus_size * 3.25

    #Inst relay trip, red colour
    if  len(inst_trip_switches)>0:

        sc_inst = plot.create_line_switch_collection(net, size = bus_size, distance_to_bus = dist_to_bus, color = "red", switches = set(inst_trip_switches) - set(bus_bus_switches))
        bb_inst = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="red")
        collection.append(sc_inst)
        collection.append(bb_inst)

    #backup relay based on time grade (yellow colour)
    if len(backup_trip_switches) > 0:

        sc_backup = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="yellow", switches = backup_trip_switches)
        bb_backup = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="yellow")
        collection.append(sc_backup)
        collection.append(bb_backup)

    # orange colour for inst_backup relay
    if len(inst_backup_switches)>0:

        instant_sc_backup = plot.create_line_switch_collection(net, size=bus_size, distance_to_bus=dist_to_bus, color="orange", switches = inst_backup_switches)
        instant_bb_backup = plot.create_bus_bus_switch_collection(net, size=bus_size, helper_line_color="orange")
        collection.append(instant_sc_backup)

    len_sc=len( set(net.switch.index) - set(inst_trip_switches)- set(backup_trip_switches))

    if len_sc != 0:

        #closed switch-black
        sc = plot.create_line_switch_collection(net, size = bus_size, distance_to_bus = dist_to_bus, color = "black",
                                            switches = set(net.switch.index) - set(inst_trip_switches)- set(backup_trip_switches) - set(bus_bus_switches))
        bb = plot.create_bus_bus_switch_collection(net, size = bus_size, helper_line_color="black")

        collection.append(sc)
        collection.append(bb)

    #make annotations optional (if True then annotate else only plot)
    if plot_annotations:
        # line annotations
        line_text=[]
        line_geodata=[]

        #for Switches in trip_decisions:
        for line in net.line.index:

            if line== max(net.line.index):
                break

            #annotate line_id
            text_line = r"  line_"+str(line)#+ ",sw_"+str(Switch_index)

            #get bus_index from the line (from switch)
            get_bus_index= pp.get_connected_buses_at_element(net, element_index=line, element_type='l', respect_in_service=False)

            bus_list=list(get_bus_index)

            #place annotations on middle of the line
            line_geo_x=(net.bus_geodata.iloc[bus_list[0]].x+ net.bus_geodata.iloc[bus_list[1]].x)/2

            line_geo_y=((net.bus_geodata.iloc[bus_list[0]].y+ net.bus_geodata.iloc[bus_list[1]].y)/2)+0.05

            line_geo_x_y=[line_geo_x,line_geo_y]

            # list of new geo data for line (half position of switch)
            line_geodata.append(tuple(line_geo_x_y))


            fault_current= round(net.res_bus_sc['ikss_ka'].at[sc_bus],2)  #round(Switches['Fault Current'],2)

            line_text.append(text_line)

        #line annotations to collections for plotting
        line_annotate=plot.create_annotation_collection( texts=line_text, coords=line_geodata, size=0.06, prop=None)
        collection.append(line_annotate)

        #Bus Annotatations
        bus_text=[]
        for i in net.bus_geodata.index:
            bus_texts='bus_'+str(i)

            bus_text.append(bus_texts)

        bus_text=bus_text[:-1]

        bus_geodata = net.bus_geodata[['x', 'y']]

        #placing bus
        bus_geodata['x'] = bus_geodata['x'] - 0.11
        bus_geodata['y'] = bus_geodata['y'] + 0.095

        bus_index= [tuple(x) for x in bus_geodata.to_numpy()]
        bus_annotate=plot.create_annotation_collection( texts=bus_text, coords=bus_index, size=0.06, prop=None)
        collection.append(bus_annotate)

        # Short circuit annotations
        fault_geodata=[]

        fault_text=[]

        fault_texts='    I_sc = '+str(fault_current)+'kA'

        font_size_bus=0.06  # font size of fault location  text

        fault_geo_x=net.bus_geodata.iloc[max(net.bus_geodata.index)][0]
        fault_geo_y=net.bus_geodata.iloc[max(net.bus_geodata.index)][1]-font_size_bus+0.02

        fault_geo_x_y=[fault_geo_x,fault_geo_y]

        # list of new geo data for line (half position of switch)
        fault_geodata.append(tuple(fault_geo_x_y))

        fault_text.append(fault_texts)
        fault_annotate=plot.create_annotation_collection( texts=fault_text, coords=fault_geodata, size=0.06, prop=None)

        collection.append(fault_annotate)

        # sc_location annotation
        sc_text=[]
        sc_geodata=[]

        sc_texts='   sc_location: '+str(sc_location*100)+'%'

        #font_size_bus=0.06  # font size of sc location

        sc_geo_x=net.bus_geodata.iloc[max(net.bus_geodata.index)][0]

        sc_geo_y=net.bus_geodata.iloc[max(net.bus_geodata.index)][1]+0.02

        sc_geo_x_y=[sc_geo_x,sc_geo_y]

        # list of new geo data for line (middle of  position of switch)
        sc_geodata.append(tuple(sc_geo_x_y))

        sc_text.append(sc_texts)
        sc_annotate=plot.create_annotation_collection( texts=sc_text, coords=sc_geodata, size=0.06, prop=None)

        collection.append(sc_annotate)

        # switch annotations
        #from pandapower.protection.implemeutility_functions import switch_geodata
        switch_text=[]
        for switch_id in trip_decisions.switch_id:

            text_switch= r"sw_"+str(switch_id)
            switch_text.append(text_switch)

        switch_geodata= switch_geodatas(net, size= bus_size, distance_to_bus=3.25* bus_size)
        i=0
        for i in range(len(switch_geodata)):

            switch_geodata[i]['x'] = switch_geodata[i]['x'] - 0.085  #scale the value if annotations overlap
            switch_geodata[i]['y'] = switch_geodata[i]['y'] + 0.055  #scale the value if annotations overlap
            i=i+1
        switch_annotate=plot.create_annotation_collection( texts=switch_text, coords=switch_geodata, size=0.06, prop=None)
        collection.append(switch_annotate)
    plot.draw_collections(collection)

def calc_line_intersection(m1, b1, m2, b2):
    xi = (b1-b2) / (m2-m1)
    yi = m1 * xi + b1

    return (xi, yi)


# get connected lines using bus id
def get_connected_lines(net, bus_idx):
    connected_lines = pp.get_connected_elements(net, "line", bus_idx)
    #connected_lines.remove(line_idx)
    return connected_lines

    #Returns the index of the second bus an element is connected to, given a
    #first one. E.g. the from_bus given the to_bus of a line.


def next_buses(net, bus, element_id):
    next_connected_bus=pp.next_bus(net,bus,element_id)
    return next_connected_bus


# get the connected bus listr from start to end bus
def source_to_end_path(net,start_bus,bus_list,bus_order):

    connected_lines=get_connected_lines(net,start_bus)
    flag=0
    for line in connected_lines:
        next_connected_bus=next_buses(net, bus=start_bus,element_id=line)
        bus_order_1=bus_order.copy()
        if next_connected_bus in bus_order:
            continue
        else:
            bus_order.append(next_connected_bus)
            bus_list= source_to_end_path(net, next_connected_bus, bus_list, bus_order)

            bus_order=bus_order_1
            flag=1
    if flag==0:
        bus_list.append(bus_order)

    return bus_list

#get connected switches with bus
def get_connected_switches(net, buses):

    connected_switches=pp.get_connected_switches(net, buses, consider=('l'), status="closed")
    return connected_switches

# get connected buses with a oven element
def connected_bus_in_line(net, element):
     get_bus_line=pp.get_connected_buses_at_element(net, element, et='l', respect_in_service=False)

     return get_bus_line


def get_line_path(net, bus_path,sc_line_id=0):
    """line path from bus path """
    line_path=[]

    for i in range(len(bus_path)-1):
        bus1=bus_path[i]
        bus2=bus_path[i+1]

        #line=net.line [(net.line.from_bus==bus1) & (net.line.to_bus==bus2)].index.item()

        line=net.line [((net.line.from_bus==bus1) & (net.line.to_bus==bus2)) | (
                        (net.line.from_bus==bus2) & (net.line.to_bus==bus1))].index.item()

        line_path.append(line)

    return line_path


def switch_geodatas(net, size, distance_to_bus):
    """get the coordinates for switches at moddile of the line"""

    switch_geo=[]

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

        pos_sb = net.bus_geodata.loc[sb, ["x", "y"]].values
        pos_tb = np.zeros(2)

        pos_tb = net.bus_geodata.loc[target_bus, ["x", "y"]]

        # position of switch symbol
        vec = pos_tb - pos_sb
        mag = np.linalg.norm(vec)

        pos_sw = pos_sb + vec / mag * distance_to_bus
        switch_geo.append(pos_sw)
    return switch_geo


def create_I_t_plot(trip_decisions,switch_id):
    """function create I-T plot using tripping decsions"""
    if not MATPLOTLIB_INSTALLED:
        raise ImportError('matplotlib must be installed to run create_I_t_plot()')

    x = [0, 0, 0, 0]
    y = [0, 0, 0, 0]

    X = []
    for counter, switch_id in enumerate(switch_id):

        lst_I=[trip_decisions[switch_id]['Ig'],trip_decisions[switch_id]['Igg']]
        lst_t=[trip_decisions[switch_id]['tg'],trip_decisions[switch_id]['tgg']]
        fault_current=trip_decisions[switch_id]['Fault Current [kA]']


        label='Relay:R'+str(switch_id)
        x[counter] = [lst_I[0], lst_I[0] , lst_I[1], lst_I[1],100]
        y[counter] = [10, lst_t[0], lst_t[0], lst_t[1], lst_t[1]]

        X.append([x[counter],y[counter],label,fault_current])

    plt.figure()

    for count, data in enumerate(X):

        plt.loglog(X[count][0], X[count][1],label=X[count][2])

        plt.axvline(x =X[count][3],ymin=0, ymax=X[0][1][0],  color = 'r',)
        plt.text(X[count][3]+0.01,0.01,'Ikss  '+str(round(fault_current,1))+'kA',rotation=0)

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
        'I:{} kA,t:{} s'.format(round(sel.target[0],2), round(sel.target[1],2))))


def power_flow_end_points(net):
    """function calculate end point from meshed grid and the start point from the radial grid to ext grid"""

    pf_net = copy.deepcopy(net)
    pp.runpp(pf_net)
    pf_loop_end_buses = []
    pf_radial_end_buses = []

    for bus in pf_net.bus.index:

        lines = pp.get_connected_elements(pf_net, element = "l", buses = bus)

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


def bus_path_from_to_bus(net,radial_start_bus, loop_start_bus, end_bus):
    """#function calcuulate the bus path from start and end buses"""
    loop_path=[]
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


def get_switches_in_path(net, pathes):
    """# function calculate the switching times from the  bus path"""

    Lines_in_path = []

    for path in pathes:
        Lines_at_path = []

        for bus in path:
            Lines_at_paths = []
            lines_at_bus = pp.get_connected_elements(net, "l", bus)

            for line in lines_at_bus:
                Lines_at_path.append(line)

            for Line1 in Lines_at_path:
                if net.line.from_bus[Line1] in path:
                    if net.line.to_bus[Line1] in path:
                        if Line1 not in Lines_at_paths:
                            Lines_at_paths.append(Line1)

        Lines_in_path.append(Lines_at_paths)

    switches_in_net = net.switch.index
    switches_in_path = []

    for Linepath in Lines_in_path:
        switches_at_path = []

        for Line in Linepath:

            for switch in switches_in_net:
                if net.switch.et[switch] == "l":
                    if net.switch.element[switch] == Line:
                        switches_at_path.append(switch)
        switches_in_path.append(switches_at_path)

    return switches_in_path


def get_vi_angle(net,switch_id,powerflow_results=False):
    """calculate the angle betwen voltage and current with reference to voltage"""

    pp.runpp(net)
    line_idx =get_line_idx(net, switch_id)
    bus_idx = get_bus_idx(net,switch_id)

    if powerflow_results:

        if  get_from_bus_info_switch(net, switch_id):

            P = net.res_line.p_from_mw.at[line_idx]
            Q = net.res_line.q_from_mvar.at[line_idx]

            vm = net.bus.vn_kv.at[bus_idx] * net.res_line.vm_from_pu.at[line_idx]
        else:
            P = net.res_line.p_to_mw.at[line_idx]
            Q = net.res_line.q_to_mvar.at[line_idx]

            vm = net.bus.vn_kv.at[bus_idx] * net.res_line.vm_to_pu.at[line_idx]
    else:

        if  get_from_bus_info_switch(net, switch_id):

                P = net.res_line_sc.p_from_mw.at[line_idx]
                Q = net.res_line_sc.q_from_mvar.at[line_idx]

                vm = net.bus.vn_kv.at[bus_idx] * net.res_line_sc.vm_from_pu.at[line_idx]

        else:
                P = net.res_line_sc.p_to_mw.at[line_idx]
                Q = net.res_line_sc.q_to_mvar.at[line_idx]
                vm = net.bus.vn_kv.at[bus_idx] * net.res_line_sc.vm_to_pu.at[line_idx]

    if P>0 and Q>0:
        vi_angle = math.degrees(math.atan(Q/P))
    elif P<0 and Q>=0:
        vi_angle = math.degrees(math.atan(Q/P))+180
    elif P<0 and Q<0:
        vi_angle = math.degrees(math.atan(Q/P))-180
    elif P==0 and Q>0:
          vi_angle =90
    elif P==0 and Q<0:
          vi_angle =-90
    else:
        vi_angle =math.inf
    return vi_angle


def bus_path_multiple_ext_bus(net):
    G = create_nxgraph(net)
    bus_path = []
    for line_id in net.line.index:

        #line_id=62
        from_bus =net.line.from_bus.at[line_id]
        to_bus =net.line.to_bus.at[line_id]
        max_bus_path=[]

        if net.trafo.empty:
            #for ext_bus in net.ext_grid.bus:
            for ext_bus in set(net.ext_grid.bus):
                from_bus_path = nx.shortest_path(G, source=ext_bus, target=from_bus)
                to_bus_path=nx.shortest_path(G, source=ext_bus, target=to_bus)

                if len(from_bus_path) == len(to_bus_path):
                    from_bus_path.append(to_bus_path[-1])
                    max_bus_path.append(from_bus_path)

                elif len(from_bus_path) != len(to_bus_path):
                    if len(from_bus_path) > 1 and len(to_bus_path) > 1:
                        minlen = min(len(from_bus_path), len(to_bus_path))
                        if from_bus_path[minlen-1] != to_bus_path[minlen-1]:
                            if len(from_bus_path) < len(to_bus_path):
                                from_bus_path.append(to_bus_path[-1])
                                max_bus_path.append(from_bus_path)
                            else:
                                to_bus_path.append(from_bus_path[-1])
                                max_bus_path.append(to_bus_path)
                        else:
                            max_bus_path.append(max([from_bus_path , to_bus_path]))
                    else:
                        max_bus_path.append(max([from_bus_path , to_bus_path]))

            bus_path.append(sorted(max_bus_path, key=len)[0])

    return bus_path


  # get the line path from the given bus path
def get_line_path(net, bus_path):
    """ Function return the list of line path from the given bus path"""
    line_path=[]
    for i in range(len(bus_path)-1):
        bus1=bus_path[i]
        bus2=bus_path[i+1]
        line1=net.line[(net.line.from_bus==bus1) & (net.line.to_bus==bus2)].index.to_list()
        line2=net.line[(net.line.from_bus==bus2) & (net.line.to_bus==bus1)].index.to_list()
        if len(line2)==0:
            line_path.append(line1[0])

        if len(line1)==0:
            line_path.append(line2[0])
    return line_path


def parallel_lines(net):
    """ Function return the list of parallel lines in the network"""

    parallel=[]

    #parallel_lines
    for i in net.line.index:
        for j in net.line.index:

            if i!=j:

                if net.line.loc[i].from_bus==net.line.loc[j].from_bus :

                    if net.line.loc[i].to_bus==net.line.loc[j].to_bus:

                        parallel.append([i,j])

                if net.line.loc[i].from_bus==net.line.loc[j].to_bus :

                    if  net.line.loc[i].to_bus==net.line.loc[j].from_bus :

                        parallel.append([i,j])

                if net.line.loc[i].to_bus==net.line.loc[j].from_bus :

                    if  net.line.loc[i].from_bus==net.line.loc[j].to_bus :

                        parallel.append([i,j])

                if net.line.loc[i].to_bus==net.line.loc[j].to_bus :

                    if  net.line.loc[i].from_bus==net.line.loc[j].from_bus :

                        parallel.append([i,j])

        parallel_line=[list(i) for i in set(map(tuple, parallel))]

        #remove duplicates
        new_parallel = []
        for l in parallel_line:
            if l not in new_parallel:
                new_parallel.append(l)

    return new_parallel

def read_fuse_from_std_type():
    # write characteristic of fuse from fuse_std_library
    # net,overwrite already existing characteristic
    # raise error if name of fuse is not available in fuse library


    pass
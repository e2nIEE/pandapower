# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import networkx as nx
import pandas as pd
from pandapower.topology.create_graph import create_nxgraph

def connected_component(mg, bus, notravbuses=[]):

    """
    Finds all buses in a NetworkX graph that are connected to a certain bus. 
     
    INPUT:
     
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.
        
        **bus** (integer) - Index of the bus at which the search for connected components originates
         
         
    OPTIONAL:
     
     **notravbuses** (list/set) - Indeces of notravbuses: lines connected to these buses are
                                     not being considered in the graph  
         
    RETURN:
     
        **cc** (generator) - Returns a generator that yields all buses connected to the input bus
        
    EXAMPLE:
        
         import pandapower.topology as top
         
         mg = top.create_nx_graph(net)
         
         cc = top.connected_component(mg, 5)
         
    """    
    
    yield bus
    visited = {bus}
    stack = [(bus, iter(mg[bus]))]
    while stack:
        parent, children = stack[-1]
        try:
            child = next(children)
            if child not in visited:
                yield child
                visited.add(child)
                if not child in notravbuses:
                    stack.append((child, iter(mg[child])))
        except StopIteration:
            stack.pop()


def connected_components(mg, notravbuses=set()):
    
    """
     Clusters all buses in a NetworkX graph that are connected to each other.  
     
     INPUT:
     
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.
         
         
     OPTIONAL:
     
     **notravbuses** (set) - Indeces of notravbuses: lines connected to these buses are
                                       not being considered in the graph  
         
     RETURN:
     
        **cc** (generator) - Returns a generator that yields all clusters of buses connected 
                             to each other.
        
     EXAMPLE:
        
         import pandapower.topology as top
         
         mg = top.create_nx_graph(net)         
         
         cc = top.connected_components(net, 5)
         
    """ 

    nodes = set(mg.nodes()) - notravbuses
    while nodes:
        cc = set(connected_component(mg, nodes.pop(), notravbuses=notravbuses))
        yield cc
        nodes -= cc
    # the above does not work if two notravbuses are directly connected
    for f, t in mg.edges():
        if f in notravbuses and t in notravbuses:
            yield set([f, t])
  
    
def calc_distance_to_bus(net, bus, respect_switches=True, nogobuses=None,
                         notravbuses=None):      
    """
        Calculates the shortest distance between a source bus and all buses connected to it.  
     
     INPUT:
     
        **net** (PandapowerNet) - Variable that contains a pandapower network.
        
        **bus** (integer) - Index of the source bus.
         
         
     OPTIONAL:
     
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored
                                                        
        **nogobuses** (integer/list, None) - nogobuses are not being considered
        
        **notravbuses** (integer/list, None) - lines connected to these buses are not being
                                              considered
                                                     
     RETURN:
     
        **dist** - Returns a pandas series with containing all distances to the source bus 
                   in km.
         
     EXAMPLE:
        
         import pandapower.topology as top
         
         dist = top.calc_distance_to_bus(net, 5)
         
    """
    g = create_nxgraph(net, respect_switches=respect_switches,
                       nogobuses=nogobuses, notravbuses=None)
    return pd.Series(nx.single_source_dijkstra_path_length(g, bus))
    

def unsupplied_buses(net, mg=None, in_service_only=False, slacks=None):
    
    """
     Finds buses, that are not connected to an external grid.  
     
     INPUT:
     
        **net** (PandapowerNet) - variable that contains a pandapower network
         
     OPTIONAL:
     
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.
         
     RETURN:
     
        **ub** (set) - unsupplied buses
        
     EXAMPLE:
        
         import pandapower.topology as top
         
         top.unsupplied_buses(net)
    """
     
    mg = mg or create_nxgraph(net)
    slacks = slacks or set(net.ext_grid[net.ext_grid.in_service==True].bus.values)
    not_supplied = set()
    for cc in nx.connected_components(mg):
        if not set(cc) & slacks:
            not_supplied.update(set(cc))
    return not_supplied


def find_bridges(g, roots):
    discovery = {root:0 for root in roots} # "time" of first discovery of node during search
    low = {root:0 for root in roots}
    visited = set(roots)
    stack = [(root, root, iter(g[root])) for root in roots]
    bridges = set()
    while stack:
        grandparent, parent, children = stack[-1]
        try:
            child = next(children)
            if grandparent == child:
                continue
            if child in visited:
                if discovery[child] <= discovery[parent]: # back edge
                    low[parent] = min(low[parent], discovery[child])
            else:
                low[child] = discovery[child] = len(discovery)
                visited.add(child)
                stack.append((parent, child, iter(g[child])))
        except StopIteration:
            stack.pop()
            if low[parent] >= discovery[grandparent]:
                bridges.add((grandparent, parent))
            low[grandparent] = min(low[parent], low[grandparent])
    return bridges

    
def get_2connected_buses(g, roots):
    bridges = find_bridges(g, roots)
    if not bridges:
        return set(g.nodes())
    visited = set(roots)
    stack = [(root, root, iter(g[root])) for root in roots]
    while stack:
        grandparent, parent, children = stack[-1]
        try:
            child = next(children)
            if child == grandparent or (parent, child) in bridges or (child, parent) in bridges:
                continue
            if child not in visited:
                visited.add(child)
                stack.append((parent, child, iter(g[child])))
        except StopIteration:
            stack.pop()
    return visited


def determine_stubs(net, roots=None, mg=None):
    
    """
     Finds stubs in a network. Open switches are being ignored. Results are being written in a new 
     column in the bus table ("on_stub") and line table ("is_stub") as True/False value.
     
     
     INPUT:
     
        **net** (PandapowerNet) - Variable that contains a pandapower network.
         
     OPTIONAL:
     
        **roots** (integer/list, None) - Indeces of buses that should be excluded (by default, the
                                         ext_grid buses will be set as roots)
         
     RETURN:
     
        None
        
     EXAMPLE:
        
         import pandapower.topology as top
         
         top.determine_stubs(net, roots = [0, 1])

         
    """
    if mg is None: 
        mg = create_nxgraph(net, respect_switches=False)
    # remove buses with degree lower 2 until none left
    roots = roots or set(net.ext_grid.bus)
#    mg.add_edges_from((a, b) for a, b in zip(list(roots)[:-1], list(roots)[1:]))
#    while True:
#        dgo = {g for g, d in list(mg.degree().items()) if d < 2} #- roots
#        if not dgo:
#            break
#        mg.remove_nodes_from(dgo)
#    n1_buses = mg.nodes()
    n1_buses = get_2connected_buses(mg, roots)
    net.bus["on_stub"] = True
    net.bus.loc[n1_buses, "on_stub"] = False
    net.line["is_stub"] = ~((net.line.from_bus.isin(n1_buses)) & (net.line.to_bus.isin(n1_buses)))
    stubs = set(net.bus.index) - set(n1_buses)
    return stubs
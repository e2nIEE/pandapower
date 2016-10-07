from builtins import next
# -*- coding: utf-8 -*-
__author__ = 'ascheidler, lthurner'

import networkx as nx
import pandas as pd
from pandapower.topology.create_graph import create_nxgraph

def connected_component(mg, bus, notravbusses=[]):

    """
     Finds all busses in a NetworkX graph that are connected to a certain bus. 
     
     INPUT:
     
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.
        
        **bus** (integer) - Index of the bus at which the search for connected components originates
         
         
     OPTIONAL:
     
     **notravbusses** (list/set) - Indeces of notravbusses: lines connected to these busses are
                                     not being considered in the graph  
         
     RETURN:
     
        **cc** (generator) - Returns a generator that yields all busses connected to the input bus
        
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
                if not child in notravbusses:
                    stack.append((child, iter(mg[child])))
        except StopIteration:
            stack.pop()


def connected_components(mg, notravbusses=set()):
    
    """
     Clusters all busses in a NetworkX graph that are connected to each other.  
     
     INPUT:
     
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.
         
         
     OPTIONAL:
     
     **notravbusses** (set) - Indeces of notravbusses: lines connected to these busses are
                                       not being considered in the graph  
         
     RETURN:
     
        **cc** (generator) - Returns a generator that yields all clusters of busses connected 
                             to each other.
        
     EXAMPLE:
        
         import pandapower.topology as top
         
         mg = top.create_nx_graph(net)         
         
         cc = top.connected_components(net, 5)
         
    """ 

    nodes = set(mg.nodes()) - notravbusses
    while nodes:
        cc = set(connected_component(mg, nodes.pop(), notravbusses=notravbusses))
        yield cc
        nodes -= cc
    # the above does not work if two notravbusses are directly connected
    for f, t in mg.edges():
        if f in notravbusses and t in notravbusses:
            yield set([f, t])
  
    
def calc_distance_to_bus(net, bus, respect_switches=True, nogobusses=None,
                         notravbusses=None):      
    """
        Calculates the shortest distance between a source bus and all busses connected to it.  
     
     INPUT:
     
        **net** (PandapowerNet) - Variable that contains a pandapower network.
        
        **bus** (integer) - Index of the source bus.
         
         
     OPTIONAL:
     
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored
                                                        
        **nogobusses** (integer/list, None) - nogobusses are not being considered
        
        **notravbusses** (integer/list, None) - lines connected to these busses are not being
                                              considered
                                                     
     RETURN:
     
        **dist** - Returns a pandas series with containing all distances to the source bus 
                   in km.
         
     EXAMPLE:
        
         import pandapower.topology as top
         
         dist = top.calc_distance_to_bus(net, 5)
         
    """
    g = create_nxgraph(net, respect_switches=respect_switches,
                       nogobusses=nogobusses, notravbusses=None)
    return pd.Series(nx.single_source_dijkstra_path_length(g, bus))
    

def unsupplied_busses(net, mg=None, in_service_only=False, slacks=None):
    
    """
     Finds busses, that are not connected to an external grid.  
     
     INPUT:
     
        **net** (PandapowerNet) - variable that contains a pandapower network
         
     OPTIONAL:
     
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.
         
     RETURN:
     
        **ub** (set) - unsupplied busses
        
     EXAMPLE:
        
         import pandapower.topology as top
         
         top.unsupplied_busses(net)
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

    
def get_2connected_busses(g, roots):
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
     
        **roots** (integer/list, None) - Indeces of busses that should be excluded (by default, the
                                         ext_grid busses will be set as roots)
         
     RETURN:
     
        None
        
     EXAMPLE:
        
         import pandapower.topology as top
         
         top.determine_stubs(net, roots = [0, 1])

         
    """
    if mg is None: 
        mg = create_nxgraph(net, respect_switches=False)
    # remove busses with degree lower 2 until none left
    roots = roots or set(net.ext_grid.bus)
#    mg.add_edges_from((a, b) for a, b in zip(list(roots)[:-1], list(roots)[1:]))
#    while True:
#        dgo = {g for g, d in list(mg.degree().items()) if d < 2} #- roots
#        if not dgo:
#            break
#        mg.remove_nodes_from(dgo)
#    n1_busses = mg.nodes()
    n1_busses = get_2connected_busses(mg, roots)
    net.bus["on_stub"] = True
    net.bus.loc[n1_busses, "on_stub"] = False
    net.line["is_stub"] = ~((net.line.from_bus.isin(n1_busses)) & (net.line.to_bus.isin(n1_busses)))
    stubs = set(net.bus.index) - set(n1_busses)
    return stubs
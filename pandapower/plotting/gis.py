# -*- coding: utf-8 -*-
from __future__ import division
__author__ = "Alexander Scheidler"

import pandapower.topology as top
import networkx as nx

def find_gis_components(net):
    g = top.create_nxgraph(net, respect_switches=False)
    sbci = set(net.bus_geodata.index)
    for cc in top.connected_components(g, notravbusses=sbci):
        sg = g.subgraph(cc)
        apsp = nx.all_pairs_dijkstra_path_length(sg)
        max_dist_in_sg = max(vv for v in apsp.values() for vv in v.values())
        nodes_with_gis = cc & sbci
        inner_nodes = cc - sbci
        lines = [sg[a][b][0]["key"] for a, b in sg.edges() if sg[a][b][0]["type"]=="l"]
        if max_dist_in_sg > 0:
            yield (nodes_with_gis, inner_nodes, lines, max_dist_in_sg)
   
def perpendicular_line(coords):
    from shapely.geometry import LineString
    from shapely import affinity
    line = LineString(coords)
    rotated = affinity.rotate(line, 90, origin='centroid')
    sf = 100. / rotated.length
    scaled = affinity.scale(rotated, xfact=sf, yfact=sf, zfact=sf, origin="centroid")
    return scaled.coords[:]

def direct_connections(net, gis_components, trennstellen="no"): # no, yes, only
    for onodes, _, lines, _ in gis_components:
        coords = net.bus_geodata
        if len(onodes) == 2:
            a = onodes.pop()
            b = onodes.pop()
            res = [(coords.x.at[a], coords.y.at[a]), (coords.x.at[b], coords.y.at[b])]
            if res[0] == res[1]:
                continue
            if not trennstellen == "only":
                yield res
            if not trennstellen == "no":
                if len(net.switch[(net.switch.element.isin(lines))
                                & (net.switch.element_type=="l")
                                & (net.switch.closed==0)]):
                    yield perpendicular_line(res)
            continue
        elif len(onodes) > 0:
            x = net.bus_geodata.x.loc[onodes].sum() / len(onodes)
            y = net.bus_geodata.y.loc[onodes].sum() / len(onodes)
#        x = sum(p.x for p in j) / len(j)
#        y = sum(p.y for p in j) / len(j)
            for p in onodes:
                res = [(coords.x.at[p], coords.y.at[p]), (x, y)]
                if not trennstellen == "only":
                    yield res
                if trennstellen == "no":
                    continue
                for line in lines:
                    if p in net.line.loc[line, ["from_bus", "to_bus"]]:
                        if len(net.switch[(net.switch.element == line)
                                        & (net.switch.element_type=="l")
                                        & (net.switch.closed==0)]):
                            yield perpendicular_line(res)
                            
                            

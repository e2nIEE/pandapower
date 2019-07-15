# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:40:07 2016

@author: thurner
"""
import pandapower as pp

net = pp.create_empty_network()

linetypes = pp.available_std_types(net, "line")
columns = [c for c in net.line.columns if c in linetypes.columns] + ["q_mm2", "alpha"]
linetypes = linetypes.reindex(columns, axis=1)
linetypes.to_csv("linetypes.csv", sep=";")

trafotypes = pp.available_std_types(net, "trafo")
trafotypes = trafotypes.reindex([c for c in net.trafo.columns if c in trafotypes.columns], axis=1)
trafotypes.to_csv("trafotypes.csv", sep=";")

trafo3wtypes = pp.available_std_types(net, "trafo3w")
trafo3wtypes = trafo3wtypes.reindex([c for c in net.trafo3w.columns if c in trafo3wtypes.columns], axis=1)
trafo3wtypes.to_csv("trafo3wtypes.csv", sep=";")
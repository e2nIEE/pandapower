# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 13:40:07 2016

@author: thurner
"""
import pandapower as pp

net = pp.create_empty_network()

linetypes = pp.available_std_types(net, "line")
linetypes.to_csv("linetypes.csv", sep=";")

trafotypes = pp.available_std_types(net, "trafo")
trafotypes.to_csv("trafotypes.csv", sep=";")

trafo3wtypes = pp.available_std_types(net, "trafo3w")
trafo3wtypes.to_csv("trafo3wtypes.csv", sep=";")

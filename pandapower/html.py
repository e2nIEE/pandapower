# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.
# File created my Massimo Di Pierro

from itertools import combinations
from cgi import escape
import json

class Raw(object):
    def __init__(self, html):
        self.html = html

class Tag(object):
    def __init__(self, name):
        self.name = name
    def __call__(self, *args, **kwargs):
        attr = ' '+' '.join('%s="%s"' % (k,escape(v)) for k,v in kwargs.items())
        contents = ''.join(a.html if isinstance(a, Raw) else escape(str(a)) for a in args)
        return Raw('<%s%s>%s</%s>' % (self.name, attr.rstrip(), contents, self.name))

def to_html(net, respect_switches=True, include_lines=True, include_trafos=True, show_tables=True):

    """
     Converts a pandapower network into an html page which contains a simplified representation
     of a network's topology, reduced to nodes and edges. Busses are being represented by nodes
     (Note: only buses with in_service = 1 appear in the graph), edges represent physical
     connections between buses (typically lines or trafos).

     INPUT:
        **net** (pandapowerNet) - variable that contains a pandapower network


     OPTIONAL:
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored

        **include_lines** (boolean, True) - determines, whether lines get converted to edges

        **include_trafos** (boolean, True) - determines, whether trafos get converted to edges

        **show_tables** (boolean, True) - shows pandapower element tables

     EXAMPLE:

         from pandapower.html import to_html
         html = to_html(net, respect_switches = False)
         open('C:\\index.html', "w").write(html)

    """

    nodes = [{'id':int(id), 'label':str(id)} for id in net.bus[net.bus.in_service==1].index]
    edges = []

    if include_lines:
        # lines with open switches can be excluded
        nogolines = set(net.switch.element[(net.switch.et == "l") &  (net.switch.closed == 0)]) \
                    if respect_switches else set()
        edges += [{'from':int(fb),
                       'to':int(tb),
                       'label':'weight %f, key: %i, type %s, capacity: %f, path: %i' %
                       (l, idx, 'l', imax, 1)}
                      for fb, tb, l, idx, inservice, imax in
                      list(zip(net.line.from_bus, net.line.to_bus, net.line.length_km,
                               net.line.index, net.line.in_service, net.line.max_i_ka))
                      if inservice == 1 and not idx in nogolines]
        edges += [{'from':int(fb),
                   'to':int(tb),
                   'label': 'key: %i, type %s, path: %i' %(idx, 'i', 1)}
                  for fb, tb, idx, inservice in
                  list(zip(net.impedance.from_bus, net.impedance.to_bus,
                           net.impedance.index, net.impedance.in_service))
                  if inservice == 1]

    if include_trafos:
        nogotrafos = set(net.switch.element[(net.switch.et == "t") & (net.switch.closed == 0)])
        edges += [{'from':int(hvb),
                   'to':int(lvb),
                   'label':'weight %f, key: %i, type %s' % (0, idx, 't')}
                  for hvb, lvb, idx, inservice in
                  list(zip(net.trafo.hv_bus, net.trafo.lv_bus,
                           net.trafo.index, net.trafo.in_service))
                  if inservice == 1 and not idx in nogotrafos]
        for trafo3, t3tab in net.trafo3w.iterrows():
            edges += [{'from':int(bus1),
                       'to':int(bus2),
                       'label':'weight %f, key: %i, type %s' % (0, trafo3, 't3')}
                      for bus1, bus2 in combinations([t3tab.hv_bus,t3tab.mv_bus, t3tab.lv_bus], 2)
                      if t3tab.in_service]

    # add bus-bus switches
    bs = net.switch[(net.switch.et == "b") &
                    ((net.switch.closed == 1) | (not respect_switches))]
    edges += [{'from':int(b),
               'to':int(e),
               'label':'weight %f, key: %i, type %s' % (0, i, 's')}
              for b, e, i in list(zip(bs.bus, bs.element, bs.index))]

    HTML, HEAD, STYLE, BODY, DIV = Tag('html'), Tag('head'), Tag('style'), Tag('body'), Tag('div')
    TABLE, TR, TH, TD, SCRIPT = Tag('table'), Tag('tr'), Tag('th'), Tag('td'), Tag('script')
    H2 = Tag('h2')

    style = 'tr:first {background:#e1e1e1;} th,td {text-align:center; border:1px solid #e1e1e1;}'

    script = "var data = {nodes: new vis.DataSet(%s), edges: new vis.DataSet(%s)};" % (
        json.dumps(nodes), json.dumps(edges))
    script += "var container = document.getElementById('net');"
    script += "var network = new vis.Network(container, data, {zoomable: false});"

    tables = []
    if show_tables:
        for name in ['bus', 'trafo', 'line', 'load', 'ext_grid',
                     'res_bus', 'res_trafo', 'res_line', 'res_load', 'res_ext_grid']:
            item = getattr(net, name)
            table = TABLE(TR(*map(TH, item.columns)),
                          *[TR(*map(TD, row)) for row in item.values])
            tables.append(DIV(H2(name), table))

    page = HTML(
        HEAD(STYLE(style)),
        BODY(DIV(*tables),DIV(id='net',style="border:1px solid #f1f1f1;max-width:90%")),
        SCRIPT(src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.18.1/vis.min.js"),
        SCRIPT(Raw(script))
        )

    return page.html
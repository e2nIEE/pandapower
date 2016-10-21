# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.
import pandapower as pp
import random


def random_empty_grid(num_buses, p):
    """
    Creates a random line network with one line to every bus (tree structure). The probability whether
    the next created bus is connected to the last created bus (1-p) or randomly to any of the already
    created buses (p) is given by p.

    INPUT:
        **num_buses** (int) - number of buses

        **p** (float) - probability that the currently created bus is connected to any of the already created \
             buses and not to the last created bus

    RETURN:

         **net** - Returns the random empyt grid

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.networksrandom_empty_grid(5, 0)
    """
    linetypes = ["NAYY 4x120 SE", "NAYY 4x150 SE", "NA2XS2Y 1x95 RM/25 12/20 kV"]
    net = pp.create_empty_network()
    cb = pp.create_bus(net, name="slack", vn_kv=20.)
    pp.create_ext_grid(net, cb)
    for i in list(range(num_buses)):
        bidx = pp.create_bus(net, name="bus %s" % (i+1), vn_kv=20)
        pp.create_line(net, cb, bidx, random.uniform(0.1, 2.),
                       name="line %s" % i, std_type=random.choice(linetypes))
        cb = bidx if random.random() > p or i == 0 else random.randint(0, i)
    return net


def setup_grid(num_buses, seed=None, p=0, num_loads=None, deviation=None, **kwargs):
    """
    Creates a random network by using function random_empty_grid and fills it with loads. The loads
    are randomly assigned to the nodes and always have an active power of 200 kW. The number of
    loads can be given directly or by a voltage value which must be reached by the load dependent
    voltage drop.

    INPUT:
        **num_buses** (int) - number of buses

    Optional:
        **seed** (int) - initialization of the random

        **p** (float) - probability that the currently created bus is connected to any of the \
            already created buses and not to the last created bus (consider random_empty_grid)

        **num_loads** - number of loads to fill the random network

        **deviation** - alternative way to determine the number of loads

    RETURN:
        **net** - Return the random grid with loads

    EXAMPLE:

        import pandapower.networks as pn
        net = pn.networks.setup_grid(5, p=0.3, num_loads=7)
    """
    if not num_loads and not deviation:
        raise UserWarning("specify either num_loads or deviation!")
    random.seed(seed)
    net = random_empty_grid(num_buses, p)
    if num_loads:
        for _ in range(num_loads):
            pp.create_load(net, random.randrange(1, num_buses), p_kw=200)
        return net
    else:
        pp.runpp(net)
        while net.res_bus.vm_pu.min() > 0.985 - deviation:
            pp.create_load(net, random.randrange(1, num_buses), p_kw=200)
            pp.runpp(net)
        return net


def _chose_from_range(minmax_list):
    return minmax_list[0] + (minmax_list[1] - minmax_list[0])*random.random()


def random_line_network(voltage_level=20., nr_buses_main=5, p_pv=0.5, p_wp=0.5,
                        p_pv_range=[0, 0], q_pv_range=[0, 0], p_wp_range=[0, 0],
                        q_wp_range=[0, 0], p_load_range=[200, 1600], q_load_range=[0, 0],
                        line_length_range=[0.1, 1.4], linetypes=[], branches=[]):
    """
    Creates a random line network (without transformer) with one load at every bus. Dependent on the\
    propabilities for pv plant and wp plant each bus has a pv and/or wp plant.
    The network can be either a simple one line network or a branched network.

    INPUT:
        **voltage_level** (float, default 20.0) - network voltage level

        **nr_buses_main** (int, default 5) - number of main buses to create

        **p_pv** (float, default 0.5) - probability of PV generator creation

        **p_wp** (float, default 0.5) - probability of WP generator creation

        **line_length_range** (list, default [0.1, 1.4])  - range for line lengths: [l_min_km, l_max_km]

        **p_pv_range** (list, default [0, 0]) - range for the active power of each pv plant: [min_p_kw, max_p_kw]

        **q_pv_range** (list, default [0, 0]) - range for the reactive power of each pv plant: [min_q_kvar, max_q_kvar]

        **p_wp_range** (list, default [0, 0]) - range for the active power of each wp plant: [min_p_kw, max_p_kw]

        **q_wp_range** (list, default [0, 0]) - range for the reactive power of each wp plant: [q_min_kw, q_max_kw]

        **p_load_range** (list, default [200, 1600]) - range for the active power of each load: [min_p_kw, max_p_kw]

        **q_load_range** (list, default [0, 0]) - range for the reactive power of each load: [min_q_kvar, max_q_kvar]

        **linetypes** (list, default []) - list of all possible linetypes used in the grid. For each line one of these types is \
                    randomly chosen. If empty, one of the overall available types is chosen.

        **branches** (list, default []) - list of tuples to add additional branches to the network. Each tuple has to be \
                    in the form of (start_bus, nr_buses), so branches=[(2,5), (4,3)] adds one \
                    branch that is 5 lines long and starts at bus 2 and one branch that is 3 lines \
                    long and starts at bus 4. If branches is an empty list, the network is a simple \
                    one line network.

    RETURN:

        **net** - randomly created pandapower network

    EXAMPLE:

         import pandapower.networks as pn

         net = pn.random_line_network()


         **With defined standard types and probabilities:**

         net = pn.random_line_networ(p_pv=0.7, p_wp=0.4, linetypes=["NAYY 4x120 SE", \
             "NAYY 4x150 SE", "NA2XS2Y 1x95 RM/25 12/20 kV"])
    """
    net = pp.create_empty_network()

    avail_std_types = None
    if not linetypes:
        avail_std_types = pp.available_std_types(net, element='line').index
    else:
        avail_std_types = linetypes

    cb = pp.create_bus(net, name="slack", vn_kv=voltage_level)
    pp.create_ext_grid(net, cb)
    for cb, nr_buses in [(0, nr_buses_main)] + branches:
        start_bus = cb
        for i in range(nr_buses):
            ind_str = "%s" % (i + 1) if start_bus == 0 else "%s.%s" % (start_bus, i + 1)
            bidx = pp.create_bus(net, name="Bus " + ind_str, vn_kv=voltage_level)
            pp.create_load(net, bidx, p_kw=_chose_from_range(p_load_range),
                           q_kvar=_chose_from_range(q_load_range), name="Load " + ind_str)
            if p_pv > random.random():
                pp.create_sgen(net, bidx, p_kw=-_chose_from_range(p_pv_range),
                               q_kvar=_chose_from_range(q_pv_range), name="PV " + ind_str, type='PV')
            if p_wp > random.random():
                pp.create_sgen(net, bidx, p_kw=-_chose_from_range(p_wp_range),
                               q_kvar=_chose_from_range(q_wp_range), name="WP " + ind_str, type='WP')
            pp.create_line(net, cb, bidx, _chose_from_range(line_length_range),
                           name="Line " + ind_str, std_type=random.choice(avail_std_types))
            cb = bidx
    return net

if __name__ == '__main__':
    net = random_line_network(branches=[(2, 5), (4, 3)])

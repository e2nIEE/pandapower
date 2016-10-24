.. _realistic_example:

=====================================
Multi-Voltage Level Example Network
=====================================

The following example contains all elements that are supported by the pandapower format. It is a more 
realistic network than the simple example and of course more complex. Using typically voltage levels 
for european distribution networks (high, medium and low voltage) the example relates characteristic 
topologies, utility types, line lengths and generator type distribution to the various voltage levels. 
To set network size limits the quantity of nodes in every voltage level is restricted and one medium 
voltage open ring and only two low voltage feeder are considered. Other feeders are represented by 
equivalent loads. As an example one double busbar and one single busbar are considered.

This example is also available as an interactive jupyter notebook.

The finally generated network of this example may be generated and illustrated as below.
 
.. image:: /pandapower/pics/example_network.png
	:width: 42em
	:alt: alternate Text
	:align: center
	
|

In this example a new network will be created. It starts by typing:


.. code:: python

 import pandapower as pp
 import pandas as pd

 net = pp.create_empty_network()
 
|

Further on each voltage level will be shown and described individually. 
 
---------------------------------

High voltage level
==================

Create buses
-------------
.. note:: 
 The double busbars and the single busbar need a more detailed bus implementation than the main picture shows.

.. code:: python

    # Double busbar
    pp.create_bus(net, name='Double Busbar 1', vn_kv=380, type='b')
    pp.create_bus(net, name='Double Busbar 2', vn_kv=380, type='b')
    for i in range(10):
        pp.create_bus(net, name='Bus DB T%s' % i, vn_kv=380, type='n')
    for i in range(1, 5):
        pp.create_bus(net, name='Bus DB %s' % i, vn_kv=380, type='n')

    # Single busbar
    pp.create_bus(net, name='Single Busbar', vn_kv=110, type='b')
    for i in range(1, 6):
        pp.create_bus(net, name='Bus SB %s' % i, vn_kv=110, type='n')
    for i in range(1, 6):
        for j in [1, 2]:
            pp.create_bus(net, name='Bus SB T%s.%s' % (i, j), vn_kv=110, type='n')

    # Remaining buses
    for i in range(1, 5):
        pp.create_bus(net, name='Bus HV%s' % i, vn_kv=110, type='n')

.. image:: /pandapower/pics/example_network_buses_hv_detail.png
	:width: 35em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create lines
------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.12\linewidth}|p{0.10\linewidth}|p{0.12\linewidth}|p{0.25\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/hv_lines.csv
   :delim: ;
   :widths: 7, 15, 10, 15, 35, 7, 7
   
   

.. note:: 
    The table above shows the information which is stored in the pandas dataframes to create the needed elements. 
    You can either load this information from a '.csv'-table or initialize your own dataframe.
   
.. code:: python

    hv_lines = pd.read_csv('hv_lines.csv', sep=';', header=0, decimal=',')
    for _, hv_line in hv_lines.iterrows():
        from_bus = pp.get_bus_by_name(net, hv_line.from_bus)
        to_bus = pp.get_bus_by_name(net, hv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=hv_line.length, std_type=hv_line.std_type, name=hv_line.line_name, parallel=hv_line.parallel)
					   
.. image:: /pandapower/pics/example_network_lines_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center

---------------------------------

Create transformer
------------------

.. code:: python
	
    hv_bus = pp.get_bus_by_name(net, "Bus DB 2")
    lv_bus = pp.get_bus_by_name(net, "Bus SB 1")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus, sn_kva=300000, vn_hv_kv=380, vn_lv_kv=110, vscr_percent=0.06, vsc_percent=8, pfe_kw=0, i0_percent=0, tp_pos=0, shift_degree=0, name='EHV-HV-Trafo')
										  
.. image:: /pandapower/pics/example_network_trafos_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create switches
---------------

.. note:: 

 - Between each bus and line or transformer you can create a switch. These switches are only shown as examples but are implemented in this grid.
 - The double busbars and the single busbar need a more detailed switch implementation than the main picture shows.
 
  See example picture!

.. tabularcolumns:: |p{0.07\linewidth}|p{0.10\linewidth}|p{0.20\linewidth}|p{0.15\linewidth}|p{0.10\linewidth}|p{0.05\linewidth}|p{0.10\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/hv_bus_sw.csv
   :delim: ;
   :widths: 7, 10, 20, 15, 10, 10, 10
   
   
.. code:: python

    # Bus-bus switches
    hv_bus_sw = pd.read_csv('hv_bus_sw.csv', sep=';', header=0, decimal=',')
    for _, switch in hv_bus_sw.iterrows():
        from_bus = pp.get_bus_by_name(net, switch.from_bus)
        to_bus = pp.get_bus_by_name(net, switch.to_bus)
        pp.create_switch(net, from_bus, to_bus, et=switch.et, closed=switch.closed, type=switch.type, name=switch.bus_name)

    # Bus-line switches
    hv_buses = net.bus[(net.bus.vn_kv == 380) | (net.bus.vn_kv == 110)].index
    hv_ls = net.line[(net.line.from_bus.isin(hv_buses)) & (net.line.to_bus.isin(hv_buses))]
    for _, line in hv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    # Trafo-line switches
    pp.create_switch(net, pp.get_bus_by_name(net, 'Bus DB 2'), pp.get_trafo_by_name(net, 'EHV-HV-Trafo'), et='t', closed=True, type='LBS', name='Switch DB2 - EHV-HV-Trafo')
    pp.create_switch(net, pp.get_bus_by_name(net, 'Bus SB 1'), pp.get_trafo_by_name(net, 'EHV-HV-Trafo'), et='t', closed=True, type='LBS', name='Switch SB1 - EHV-HV-Trafo')

.. image:: /pandapower/pics/example_network_switches_hv.png
	:alt: alternate Text
	:align: center
	
---------------------------------

Create external grids
---------------------

.. code:: python

    pp.create_ext_grid(net, pp.get_bus_by_name(net, 'Double Busbar 1'), vm_pu=1.03, va_degree=0, name='External grid', sk_max_mva=10000, rx_max=0.1, rx_min=0.1)
	
.. image:: /pandapower/pics/example_network_ext_grids_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	

---------------------------------
	
Create loads
------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.12\linewidth}|p{0.10\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/hv_loads.csv
   :delim: ;
   :widths: 7, 12, 10, 7, 7
   
   
.. code:: python

    hv_loads = pd.read_csv('hv_loads.csv', sep=';', header=0, decimal=',')
    for _, load in hv_loads.iterrows():
        bus_idx = pp.get_bus_by_name(net, load.bus)
        pp.create_load(net, bus_idx, p_kw=load.p, q_kvar=load.q, name=load.load_name)

	
.. image:: /pandapower/pics/example_network_loads_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create generators
-----------------

.. code:: python

	pp.create_gen(net, pp.get_bus_by_name(net, 'Bus HV4'), vm_pu=1.03, p_kw=-1e5, name='Gas turbine')
				   
.. image:: /pandapower/pics/example_network_gens_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create static generators
------------------------

.. code:: python

	pp.create_sgen(net, pp.get_bus_by_name(net, 'Bus SB 5'), p_kw=-20000, q_kvar=-4000, sn_kva=45000, type='WP', name='Wind Park')
				   
.. image:: /pandapower/pics/example_network_sgens_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create shunt
------------

.. code:: python

    pp.create_shunt(net, pp.get_bus_by_name(net, 'Bus HV1'), p_kw=0, q_kvar=-960, name='Shunt')
	
.. image:: /pandapower/pics/example_network_shunts_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	
	
Create external network equivalents
-----------------------------------

.. note:: Ward and Extended Ward equivalents can be created the same way. They are a practical representation 
    of adjacent external networks. If there are more than one border bus impedances between all border buses 
    represent the correlating load flows.

.. code:: python

    # Impedance
    pp.create_impedance(net, pp.get_bus_by_name(net, 'Bus HV3'), pp.get_bus_by_name(net, 'Bus HV1'), r_pu=0.074873, x_pu=0.198872, sn_kva=100000, name='Impedance')
    
    # xwards
    pp.create_xward(net, pp.get_bus_by_name(net, 'Bus HV3'), ps_kw=23942, qs_kvar=-12241.87, pz_kw=2814.571, qz_kvar=0, r_ohm=0, x_ohm=12.18951, vm_pu=1.02616, name='XWard 1')
    pp.create_xward(net, pp.get_bus_by_name(net, 'Bus HV1'), ps_kw=3776, qs_kvar=-7769.979, pz_kw=9174.917, qz_kvar=0, r_ohm=0, x_ohm=50.56217, vm_pu=1.024001, name='XWard 2')
	
.. image:: /pandapower/pics/example_network_ext_equi_hv.png
	:width: 25em
	:alt: alternate Text
	:align: center
	
---------------------------------


Medium Voltage Level
====================

Create buses
-------------

.. code:: python

    pp.create_bus(net, name='Bus MV0 20kV', vn_kv=20, type='n')
    for i in range(8):
        pp.create_bus(net, name='Bus MV%s' % i, vn_kv=10, type='n')

.. image:: /pandapower/pics/example_network_buses_mv.png
	:width: 30em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create lines
------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.12\linewidth}|p{0.10\linewidth}|p{0.10\linewidth}|p{0.07\linewidth}|p{0.32\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/mv_lines.csv
   :delim: ;
   :widths: 7, 10, 10, 11, 7, 32
   
   
.. code:: python
	
    mv_lines = pd.read_csv('mv_lines.csv', sep=';', header=0, decimal=',')
    for _, mv_line in mv_lines.iterrows():
        from_bus = pp.get_bus_by_name(net, mv_line.from_bus)
        to_bus = pp.get_bus_by_name(net, mv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=mv_line.length, std_type=mv_line.std_type, name=mv_line.line_name)

.. image:: /pandapower/pics/example_network_lines_mv.png
	:width: 30em
	:alt: alternate Text
	:align: center

---------------------------------

Create 3 windings transformer
-----------------------------

.. code:: python
	
    hv_bus = pp.get_bus_by_name(net, "Bus HV2")
    mv_bus = pp.get_bus_by_name(net, "Bus MV0 20kV")
    lv_bus = pp.get_bus_by_name(net, "Bus MV0")
    pp.create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus, vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10, sn_hv_kva=40000, sn_mv_kva=15000, sn_lv_kva=25000, vsc_hv_percent=10.1, vsc_mv_percent=10.1, vsc_lv_percent=10.1, vscr_hv_percent=0.266667, vscr_mv_percent=0.033333, vscr_lv_percent=0.04, pfe_kw=0, i0_percent=0, shift_mv_degree=30, shift_lv_degree=30, tp_side=0, tp_mid=0, tp_min=-8, tp_max=8, tp_st_percent=1.25, tp_pos=0, name='HV-MV-MV-Trafo')
	
.. image:: /pandapower/pics/example_network_trafos_mv.png
	:width: 30em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create switches
---------------

.. code:: python
	
    # Bus-line switches
    mv_buses = net.bus[(net.bus.vn_kv == 10) | (net.bus.vn_kv == 20)].index
    mv_ls = net.line[(net.line.from_bus.isin(mv_buses)) & (net.line.to_bus.isin(mv_buses))]
    for _, line in mv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

	# open switch
    open_switch_id = net.switch[(net.switch.name == 'Switch Bus MV5 - MV Line5')].index
    net.switch.closed.loc[open_switch_id] = False
	
	

---------------------------------

Create loads
------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.18\linewidth}|p{0.15\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/mv_loads.csv
   :delim: ;
   :widths: 7, 18, 15, 7, 7
   
   
.. code:: python
   
    mv_loads = pd.from_csv('mv_loads')
    for _, load in mv_loads.iterrows():
        bus_idx = pp.get_bus_by_name(net, load.bus)
        pp.create_load(net, bus_idx, p_kw=load.p, q_kvar=load.q, name=load.load_name)

	
.. image:: /pandapower/pics/example_network_loads_mv.png
	:width: 30em
	:alt: alternate Text
	:align: center
	
---------------------------------


Create static generators
------------------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.20\linewidth}|p{0.15\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/mv_sgens.csv
   :delim: ;
   :widths: 7, 20, 15, 7, 7, 7, 7
   

.. code:: python

    mv_sgens = pd.from_csv('mv_sgens.csv', sep=';', header=0, decimal=',')
    for _, sgen in mv_sgens.iterrows():
        bus_idx = pp.get_bus_by_name(net, sgen.bus)
        pp.create_sgen(net, bus_idx, p_kw=sgen.p, q_kvar=sgen.q, sn_kva=sgen.sn,
                       type=sgen.type, name=sgen.sgen_name)

.. image:: /pandapower/pics/example_network_sgens_mv.png
	:width: 30em
	:alt: alternate Text
	:align: center	
	
---------------------------------

Low voltage level
=================

Create buses
-------------
	
.. code:: python

    pp.create_bus(net, name='Bus LV0', vn_kv=0.4, type='n')
    for i in range(1, 6):
        pp.create_bus(net, name='Bus LV1.%s' % i, vn_kv=0.4, type='m')
    for i in range(1, 5):
        pp.create_bus(net, name='Bus LV2.%s' % i, vn_kv=0.4, type='m')
    pp.create_bus(net, name='Bus LV2.2.1', vn_kv=0.4, type='m')
    pp.create_bus(net, name='Bus LV2.2.2', vn_kv=0.4, type='m')
	
.. image:: /pandapower/pics/example_network_buses_lv.png
	:width: 30em
	:alt: alternate Text
	:align: center
	
---------------------------------
	
Create lines
------------
	
.. tabularcolumns:: |p{0.07\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.07\linewidth}|p{0.20\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/lv_lines.csv
   :delim: ;
   :widths: 7, 12, 12, 12, 7, 20
   
	
.. code:: python

    lv_lines = pd.read_csv('lv_lines.csv', sep=';', header=0, decimal=',')
    for _, lv_line in lv_lines.iterrows():
        from_bus = pp.get_bus_by_name(net, lv_line.from_bus)
        to_bus = pp.get_bus_by_name(net, lv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=lv_line.length, std_type=lv_line.std_type, name=lv_line.line_name)
					   
.. image:: /pandapower/pics/example_network_lines_lv.png
	:width: 30em
	:alt: alternate Text
	:align: center

---------------------------------


Create transformer
------------------

.. code:: python

    hv_bus = pp.get_bus_by_name(net, "Bus MV4")
    lv_bus = pp.get_bus_by_name(net, "Bus LV0")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus, sn_kva=400, vn_hv_kv=10, vn_lv_kv=0.4, vscr_percent=1.325, vsc_percent=4, pfe_kw=0.95, i0_percent=0.2375, tp_side=1, tp_mid=0, tp_min=-2, tp_max=2, tp_st_percent=2.5, tp_pos=0, shift_degree=150, name='MV-LV-Trafo')
										  
.. image:: /pandapower/pics/example_network_trafos_lv.png
	:width: 30em
	:alt: alternate Text
	:align: center	
	
---------------------------------


Create switches
---------------

.. code:: python

    # Bus-line switches
    lv_buses = net.bus[net.bus.vn_kv == 0.4].index
    lv_ls = net.line[(net.line.from_bus.isin(lv_buses)) & (net.line.to_bus.isin(lv_buses))]
    for _, line in lv_ls.iterrows():
            pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
            pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))

    # Trafo-line switches
    pp.create_switch(net, pp.get_bus_by_name(net, 'Bus MV4'), pp.get_trafo_by_name(net, 'MV-LV-Trafo'), et='t', closed=True, type='LBS', name='Switch MV4 - MV-LV-Trafo')
    pp.create_switch(net, pp.get_bus_by_name(net, 'Bus LV0'), pp.get_trafo_by_name(net, 'MV-LV-Trafo'), et='t', closed=True, type='LBS', name='Switch LV0 - MV-LV-Trafo')
---------------------------------

Create loads
------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.25\linewidth}|p{0.12\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/lv_loads.csv
   :delim: ;
   :widths: 7, 25, 12, 7, 7
   
   
.. code:: python
	
    lv_loads = pd.from_csv('lv_loads.csv', sep=';', header=0, decimal=',')
    for _, load in lv_loads.iterrows():
        bus_idx = pp.get_bus_by_name(net, load.bus)
        pp.create_load(net, bus_idx, p_kw=load.p, q_kvar=load.q, name=load.load_name)

	
.. image:: /pandapower/pics/example_network_loads_lv.png
	:width: 30em
	:alt: alternate Text
	:align: center
	
---------------------------------

Create static generators
------------------------

.. tabularcolumns:: |p{0.07\linewidth}|p{0.12\linewidth}|p{0.12\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|p{0.07\linewidth}|
.. csv-table:: 
   :file: example_realistic_tables/lv_sgens.csv
   :delim: ;
   :widths: 7, 12, 12, 7, 7, 7, 7
   
   
.. code:: python

    lv_sgens = pd.from_csv('lv_sgens.csv', sep=';', header=0, decimal=',')
    for _, sgen in lv_sgens.iterrows():
        bus_idx = pp.get_bus_by_name(net, sgen.bus)
        pp.create_sgen(net, bus_idx, p_kw=sgen.p, q_kvar=sgen.q, sn_kva=sgen.sn, type=sgen.type, name=sgen.sgen_name)
					   
.. image:: /pandapower/pics/example_network_sgens_lv.png
	:width: 30em
	:alt: alternate Text
	:align: center
	
---------------------------------

Element Tables
============================

**Bus table**

.. code:: python

	net.bus

    # Output
    index  name             vn_kv  type  zone  in_service
	
    0      Double Busbar 1  380    b     None  1
    1      Double Busbar 2  380    b     None  1
    2      Bus DB T0        380    n     None  1
    3      Bus DB T1        380    n     None  1
    4      Bus DB T2        380    n     None  1
    5      Bus DB T3        380    n     None  1
    6      Bus DB T4        380    n     None  1
    7      Bus DB T5        380    n     None  1
    8      Bus DB T6        380    n     None  1
    9      Bus DB T7        380    n     None  1
    10     Bus DB T8        380    n     None  1
    11     Bus DB T9        380    n     None  1
    12     Bus DB 1         380    n     None  1
    13     Bus DB 2         380    n     None  1
    14     Bus DB 3         380    n     None  1
    15     Bus DB 4         380    n     None  1
    16     Single Busbar    110    b     None  1
    17     Bus SB 1         110    n     None  1
    18     Bus SB 2         110    n     None  1
    19     Bus SB 3         110    n     None  1
    20     Bus SB 4         110    n     None  1
    21     Bus SB 5         110    n     None  1
    22     Bus SB T1.1      110    n     None  1
    23     Bus SB T1.2      110    n     None  1
    24     Bus SB T2.1      110    n     None  1
    25     Bus SB T2.2      110    n     None  1
    26     Bus SB T3.1      110    n     None  1
    27     Bus SB T3.2      110    n     None  1
    28     Bus SB T4.1      110    n     None  1
    29     Bus SB T4.2      110    n     None  1
    30     Bus SB T5.1      110    n     None  1
    31     Bus SB T5.2      110    n     None  1
    32     Bus HV1          110    n     None  1
    33     Bus HV2          110    n     None  1
    34     Bus HV3          110    n     None  1
    35     Bus HV4          110    n     None  1
    36     Bus MV0 20kV     20     n     None  1
    37     Bus MV0          10     n     None  1
    38     Bus MV1          10     n     None  1
    39     Bus MV2          10     n     None  1
    40     Bus MV3          10     n     None  1
    41     Bus MV4          10     n     None  1
    42     Bus MV5          10     n     None  1
    43     Bus MV6          10     n     None  1
    44     Bus MV7          10     n     None  1
    45     Bus LV0          0.4    n     None  1
    46     Bus LV1.1        0.4    m     None  1
    47     Bus LV1.2        0.4    m     None  1
    48     Bus LV1.3        0.4    m     None  1
    49     Bus LV1.4        0.4    m     None  1
    50     Bus LV1.5        0.4    m     None  1
    51     Bus LV2.1        0.4    m     None  1
    52     Bus LV2.2        0.4    m     None  1
    53     Bus LV2.3        0.4    m     None  1
    54     Bus LV2.4        0.4    m     None  1
    55     Bus LV2.2.1      0.4    m     None  1
    56     Bus LV2.2.2      0.4    m     None  1

	
---------------------------------


**Line table**

.. code:: python

	net.line
	
	# Output
	index  name          std_type                      from_bus  to_bus   length_km 
	
	0      HV Line1      184-AL1/30-ST1A 110.0         18        32       30        
	1      HV Line2      184-AL1/30-ST1A 110.0         32        33       20        
	2      HV Line3      184-AL1/30-ST1A 110.0         33        35       30        
	3      HV Line4      184-AL1/30-ST1A 110.0         32        35       15        
	4      HV Line5      184-AL1/30-ST1A 110.0         34        35       25        
	5      HV Line6      184-AL1/30-ST1A 110.0         19        34       30        
	6      MV Line1      NA2XS2Y 1x185 RM/25 12/20 kV  37        38       1.5       
	7      MV Line2      NA2XS2Y 1x185 RM/25 12/20 kV  38        39       1.5       
	8      MV Line3      NA2XS2Y 1x185 RM/25 12/20 kV  39        40       1.5       
	9      MV Line4      NA2XS2Y 1x185 RM/25 12/20 kV  40        41       1.5       
	10     MV Line5      NA2XS2Y 1x185 RM/25 12/20 kV  41        42       1.5       
	11     MV Line6      NA2XS2Y 1x185 RM/25 12/20 kV  42        43       1.5       
	12     MV Line7      NA2XS2Y 1x185 RM/25 12/20 kV  43        44       1.5       
	13     MV Line8      NA2XS2Y 1x185 RM/25 12/20 kV  37        44       1.5       
	14     LV Line1.1    NAYY 4x120 SE                 45        46       0.08      
	15     LV Line1.2    NAYY 4x120 SE                 46        47       0.08      
	16     LV Line1.3    NAYY 4x120 SE                 47        48       0.08      
	17     LV Line1.4    NAYY 4x120 SE                 48        49       0.08      
	18     LV Line1.6    NAYY 4x120 SE                 49        50       0.08      
	19     LV Line2.1    NAYY 4x120 SE                 45        51       0.12      
	20     LV Line2.2    NAYY 4x120 SE                 51        52       0.12      
	21     LV Line2.3    15-AL1/3-ST1A 0.4             52        53       0.12      
	22     LV Line2.4    15-AL1/3-ST1A 0.4             53        54       0.12      
	23     LV Line2.2.1  15-AL1/3-ST1A 0.4             52        55       0.12      
	24     LV Line2.2.2  15-AL1/3-ST1A 0.4             55        56       0.12      


	
	index  ...  r_ohm_per_km  x_ohm_per_km  c_nf_per_km  imax_ka  df  parallel  type  
                
	0      ...  0.1571        0.4           8.8          0.535    1   1         ol    
	1      ...  0.1571        0.4           8.8          0.535    1   1         ol    
	2      ...  0.1571        0.4           8.8          0.535    1   1         ol    
	3      ...  0.1571        0.4           8.8          0.535    1   1         ol    
	4      ...  0.1571        0.4           8.8          0.535    1   1         ol    
	5      ...  0.1571        0.4           8.8          0.535    1   1         ol    
	6      ...  0.161         0.117         273          0.362    1   1         cs    
	7      ...  0.161         0.117         273          0.362    1   1         cs    
	8      ...  0.161         0.117         273          0.362    1   1         cs    
	9      ...  0.161         0.117         273          0.362    1   1         cs    
	10     ...  0.161         0.117         273          0.362    1   1         cs    
	11     ...  0.161         0.117         273          0.362    1   1         cs    
	12     ...  0.161         0.117         273          0.362    1   1         cs    
	13     ...  0.161         0.117         273          0.362    1   1         cs    
	14     ...  0.225         0.08          264          0.242    1   1         cs    
	15     ...  0.225         0.08          264          0.242    1   1         cs    
	16     ...  0.225         0.08          264          0.242    1   1         cs    
	17     ...  0.225         0.08          264          0.242    1   1         cs    
	18     ...  0.225         0.08          264          0.242    1   1         cs    
	19     ...  0.225         0.08          264          0.242    1   1         cs    
	20     ...  0.225         0.08          264          0.242    1   1         cs    
	21     ...  1.8769        0.35          11           0.105    1   1         ol    
	22     ...  1.8769        0.35          11           0.105    1   1         ol    
	23     ...  1.8769        0.35          11           0.105    1   1         ol    
	24     ...  1.8769        0.35          11           0.105    1   1         ol    
	
	
	index  ...  in_service
	
	0      ...  True
	1      ...  True
	2      ...  True
	3      ...  True
	4      ...  True
	5      ...  True
	6      ...  True
	7      ...  True
	8      ...  True
	9      ...  True
	10     ...  True
	11     ...  True
	12     ...  True
	13     ...  True
	14     ...  True
	15     ...  True
	16     ...  True
	17     ...  True
	18     ...  True
	19     ...  True
	20     ...  True
	21     ...  True
	22     ...  True
	23     ...  True
	24     ...  True
	
---------------------------------

**Trafo table**

.. code:: python

	net.trafo

	# Output
	index  name          std_type  hv_bus  lv_bus  sn_kva  vn_hv_kv  vn_lv_kv 
	
	0      EHV-HV-Trafo  None      13      17      300000  380     110    
	1      MV-LV-Trafo   None      41      45      400     10      0.4    

	
	index   ...  vsc_percent  vscr_percent  pfe_kw  i0_percent  tp_side  
			     
	0       ...  8            0.06          0       0                    
	1       ...  4            1.325         0.95    0.2375      1       

	
	index   ...  tp_mid   tp_min  tp_max  tp_st_percent  tp_pos  in_service  shift_degree
    		     
	0       ...           NaN     NaN     NaN            NaN     True        0
	1       ...  0       -2       2       2.5            0       True        150
	
	
	net.trafo3w
	
	# Output
	index  name          std_type  hv_bus  mv_bus  lv_bus  vn_hv_kv  vn_mv_kv 
	 
	0      HV-MV-MV-Trafo          33      36      37      110     20     

	
	index  ...  vn_lv_kv  sn_hv_kva  sn_mv_kva  sn_lv_kva  vsc_hv_percent  vsc_mv_percent  
                                                                   
	0      ...  10      40000    15000    25000    10.1            10.1            
	
	
	index  ...  vsc_lv_percent  vscr_hv_percent  vscr_mv_percent  vscr_lv_percent  
	                                                              
	0      ...  10.1            0.266667         0.033333         0.04             
	
	
	index  ...  pfe_kw  i0_percent shift_mv_degree  shift_lv_degree  tp_side  
	               
	0      ...  0.0     0.0        30.0             30.0             0        
	
	
	index  ...  tp_mid   tp_min  tp_max  tp_st_percent  tp_pos  in_service
	                                                            
	0      ...  0       -8       8       1.25           0       True
	
---------------------------------

**Switch table**

.. code:: python

    net.switch
    
    # Output
    index  bus  element  et  type  closed  name
	
    0      1     2       b   DS    True    DB DS0                            
    1      0     3       b   DS    True    DB DS1                            
    2      1     5       b   DS    True    DB DS2                            
    3      0     5       b   DS    False   DB DS3                            
    4      1     7       b   DS    True    DB DS4                            
    5      0     7       b   DS    False   DB DS5                            
    6      1     9       b   DS    True    DB DS6                            
    7      0     9       b   DS    False   DB DS7                            
    8      1     11      b   DS    True    DB DS8                            
    9      0     11      b   DS    False   DB DS9                            
    10     4     12      b   DS    True    DB DS10                           
    11     6     13      b   DS    True    DB DS11                           
    12     8     14      b   DS    True    DB DS12                           
    13     10    15      b   DS    True    DB DS13                           
    14     2     3       b   CB    True    DB CB0                            
    15     5     4       b   CB    True    DB CB1                            
    16     7     6       b   CB    True    DB CB2                            
    17     9     8       b   CB    True    DB CB3                            
    18     11    10      b   CB    True    DB CB4                            
    19     22    17      b   DS    True    SB DS1.1                          
    20     16    23      b   DS    True    SB DS1.2                          
    21     24    18      b   DS    True    SB DS2.1                          
    22     16    25      b   DS    True    SB DS2.2                          
    23     26    19      b   DS    True    SB DS3.1                          
    24     16    27      b   DS    True    SB DS3.2                          
    25     28    20      b   DS    True    SB DS4.1                          
    26     16    29      b   DS    True    SB DS4.2                          
    27     30    21      b   DS    True    SB DS5.1                          
    28     16    31      b   DS    True    SB DS5.2                          
    29     23    22      b   CB    True    SB CB1                            
    30     25    24      b   CB    True    SB CB2                            
    31     27    26      b   CB    True    SB CB3                            
    32     29    28      b   CB    True    SB CB4                            
    33     31    30      b   CB    True    SB CB5                            
    34     18    0       l   LBS   True    Switch Bus SB 2 - HV Line1        
    35     32    0       l   LBS   True    Switch Bus HV1 - HV Line1         
    36     32    1       l   LBS   True    Switch Bus HV1 - HV Line2         
    37     33    1       l   LBS   True    Switch Bus HV2 - HV Line2         
    38     33    2       l   LBS   True    Switch Bus HV2 - HV Line3         
    39     35    2       l   LBS   True    Switch Bus HV4 - HV Line3         
    40     32    3       l   LBS   True    Switch Bus HV1 - HV Line4         
    41     35    3       l   LBS   True    Switch Bus HV4 - HV Line4         
    42     34    4       l   LBS   True    Switch Bus HV3 - HV Line5         
    43     35    4       l   LBS   True    Switch Bus HV4 - HV Line5         
    44     19    5       l   LBS   True    Switch Bus SB 3 - HV Line6        
    45     34    5       l   LBS   True    Switch Bus HV3 - HV Line6         
    46     37    6       l   LBS   True    Switch Bus MV0 - MV Line1         
    47     38    6       l   LBS   True    Switch Bus MV1 - MV Line1         
    48     38    7       l   LBS   True    Switch Bus MV1 - MV Line2         
    49     39    7       l   LBS   True    Switch Bus MV2 - MV Line2         
    50     39    8       l   LBS   True    Switch Bus MV2 - MV Line3         
    51     40    8       l   LBS   True    Switch Bus MV3 - MV Line3         
    52     40    9       l   LBS   True    Switch Bus MV3 - MV Line4         
    53     41    9       l   LBS   True    Switch Bus MV4 - MV Line4         
    54     41    10      l   LBS   True    Switch Bus MV4 - MV Line5         
    55     42    10      l   LBS   False   Switch Bus MV5 - MV Line5         
    56     42    11      l   LBS   True    Switch Bus MV5 - MV Line6         
    57     43    11      l   LBS   True    Switch Bus MV6 - MV Line6         
    58     43    12      l   LBS   True    Switch Bus MV6 - MV Line7         
    59     44    12      l   LBS   True    Switch Bus MV7 - MV Line7         
    60     37    13      l   LBS   True    Switch Bus MV0 - MV Line8         
    61     44    13      l   LBS   True    Switch Bus MV7 - MV Line8         
    62     45    14      l   LBS   True    Switch Bus LV0 - LV Line1.1       
    63     46    14      l   LBS   True    Switch Bus LV1.1 - LV Line1.1     
    64     46    15      l   LBS   True    Switch Bus LV1.1 - LV Line1.2     
    65     47    15      l   LBS   True    Switch Bus LV1.2 - LV Line1.2     
    66     47    16      l   LBS   True    Switch Bus LV1.2 - LV Line1.3     
    67     48    16      l   LBS   True    Switch Bus LV1.3 - LV Line1.3     
    68     48    17      l   LBS   True    Switch Bus LV1.3 - LV Line1.4     
    69     49    17      l   LBS   True    Switch Bus LV1.4 - LV Line1.4     
    70     49    18      l   LBS   True    Switch Bus LV1.4 - LV Line1.6     
    71     50    18      l   LBS   True    Switch Bus LV1.5 - LV Line1.6     
    72     45    19      l   LBS   True    Switch Bus LV0 - LV Line2.1       
    73     51    19      l   LBS   True    Switch Bus LV2.1 - LV Line2.1     
    74     51    20      l   LBS   True    Switch Bus LV2.1 - LV Line2.2     
    75     52    20      l   LBS   True    Switch Bus LV2.2 - LV Line2.2     
    76     52    21      l   LBS   True    Switch Bus LV2.2 - LV Line2.3     
    77     53    21      l   LBS   True    Switch Bus LV2.3 - LV Line2.3     
    78     53    22      l   LBS   True    Switch Bus LV2.3 - LV Line2.4     
    79     54    22      l   LBS   True    Switch Bus LV2.4 - LV Line2.4     
    80     52    23      l   LBS   True    Switch Bus LV2.2 - LV Line2.2.1   
    81     55    23      l   LBS   True    Switch Bus LV2.2.1 - LV Line2.2.1 
    82     55    24      l   LBS   True    Switch Bus LV2.2.1 - LV Line2.2.2 
    83     56    24      l   LBS   True    Switch Bus LV2.2.2 - LV Line2.2.2 
    84     13    0       t   LBS   True    Switch DB2 - EHV-HV-Trafo         
    85     17    0       t   LBS   True    Switch SB1 - EHV-HV-Trafo         
    86     41    1       t   LBS   True    Switch MV4 - MV-LV-Trafo          
    87     45    1       t   LBS   True    Switch LV0 - MV-LV-Trafo          
	
---------------------------------

**External grid table**

.. code:: python

	net.ext_grid
	
	# Output
	index  bus  vm_pu  va_degree  name           sk_max_mva  sk_min_mva  rx_max
	
	0      14   1.03   0.0        External grid  10000.0     NaN         0.1
	
	index  ...  rx_min  in_service
 
	0      ...  0.1     True

	
---------------------------------

**Load table**

.. code:: python

    net.load
	
    # Output
    index  name                     bus  p_kw   q_kvar    sn_kva  scaling  in_service
	
    0      MV Net 0                 20   38000  6000      NaN     1        True
    1      MV Net 1                 32   38000  6000      NaN     1        True
    2      MV Net 2                 33   38000  6000      NaN     1        True
    3      MV Net 3                 34   38000  6000      NaN     1        True
    4      MV Net 4                 35   38000  6000      NaN     1        True
    5      Further MV-Rings         37   6000   2000      NaN     1        True
    6      Industry Load            36   18000  4000      NaN     1        True
    7      LV Net 1                 38   400    100       NaN     1        True
    8      LV Net 2                 39   400    60        NaN     1        True
    9      LV Net 3                 40   400    60        NaN     1        True
    10     LV Net 5                 42   400    60        NaN     1        True
    11     LV Net 6                 43   400    60        NaN     1        True
    12     LV Net 7                 44   400    60        NaN     1        True
    13     Further LV-Feeders Load  45   100    10        NaN     1        True
    14     Residential Load         46   10     3         NaN     1        True
    15     Residential Load(1)      47   10     3         NaN     1        True
    16     Residential Load(2)      48   10     3         NaN     1        True
    17     Residential Load(3)      49   10     3         NaN     1        True
    18     Residential Load(4)      50   10     3         NaN     1        True
    19     Rural Load               51   10     3         NaN     1        True
    20     Rural Load(1)            52   10     3         NaN     1        True
    21     Rural Load(2)            53   10     3         NaN     1        True
    22     Rural Load(3)            54   10     3         NaN     1        True
    23     Rural Load(4)            55   10     3         NaN     1        True
    24     Rural Load(5)            56   10     3         NaN     1        True
	
---------------------------------

**Generator table**

.. code:: python

    net.gen

    #Output
    index  name         bus   p_kw      vm_pu  sn_kva  min_q_kvar  max_q_kvar  scaling 
    
    0      Gas turbine  35   -100000.0  1.03   NaN     NaN        NaN        1.0      
    
    
    index  ...  in_service  type
    
    0      ...  True        sync

---------------------------------
	
**Static generator table**

.. code:: python

    net.sgen

    # Output
    index  name                  bus   p_kw     q_kvar  sn_kva  scaling 
    
    0      Wind Park             21   -20000   -4000    45000   1       
    1      Biogas plant          43   -500      0       750     1       
    2      Further MV Generator  37   -500     -50      1000    1       
    3      Industry Generator    36   -15000   -3000    20000   1       
    4      PV Park               42   -2000    -100     5000    1       
    5      PV                    46   -6        0       12      1       
    6      PV(1)                 48   -5        0       10      1       
    7      PV(2)                 53   -5        0       10      1       
    8      PV(3)                 54   -5        0       10      1       
    9      PV(4)                 55   -5        0       10      1       
    10     PV(5)                 56   -5        0       10      1      
    
    
    index  ...  in_service  type
                
    0      ...  True        WP
    1      ...  True        SGEN
    2      ...  True        SGEN
    3      ...  True        SGEN
    4      ...  True        PV
    5      ...  True        PV
    6      ...  True        PV
    7      ...  True        PV
    8      ...  True        PV
    9      ...  True        PV
    10     ...  True        PV





	
	
---------------------------------
	
**Shunt table**

.. code:: python

    net.shunt
    
    # Output
    index  bus  name   p_kw   q_kvar  in_service
    
    0      32   Shunt  0.0   -960.0   1
    
---------------------------------
	
**Network eqivalent table**
	
.. code:: python
    
    net.impedance
    
    # Output
    index  name       from_bus  to_bus  r_pu      x_pu      sn_kva    in_service
    
    0      Impedance  34        32      0.074873  0.198872  100000.0  True
    
    
    net.xward
    
    # Output
    index  name     bus  ps_kw     qs_kvar    pz_kw     qz_kvar  r_ohm  x_ohm
    
    0      XWard 1  34   23942.0  -12241.870  2814.571  0.0      0.0    12.18951
    1      XWard 2  32   3776.0   -7769.979   9174.917  0.0      0.0    50.56217
	
	
    index  ...  vm_pu     in_service  
    0      ...  1.026160  1  
    1      ...  1.024001  1 
    
---------------------------------
	
Powerflow and Result Tables
==============================

To calculate the power flow run:

.. code:: python

	pp.runpp(net, init='dc', calculate_voltage_angles=True)
	
.. note:: Because of the non radial topology of the high voltage network the loadflow calculation should consider the right voltage angles. To ensure loadflow convergence an initialization by a DC loadflow is recommended.
	
---------------------------------

**Bus result table**

.. code:: python

	net.res_bus

	# Output
	index  vm_pu      va_degree   p_kw            q_kvar
	
	0      1.030000   0.000000    0.000000        0.000000
	1      1.030000   0.000000    0.000000        0.000000
	2      1.030000   0.000000    0.000000        0.000000
	3      1.030000   0.000000    0.000000        0.000000
	4      1.030000   0.000000    0.000000        0.000000
	5      1.030000   0.000000    0.000000        0.000000
	6      1.030000   0.000000    0.000000        0.000000
	7      1.030000   0.000000    0.000000        0.000000
	8      1.030000   0.000000    0.000000        0.000000
	9      1.030000   0.000000    0.000000        0.000000
	10     1.030000   0.000000    0.000000        0.000000
	11     1.030000   0.000000    0.000000        0.000000
	12     1.030000   0.000000    0.000000        0.000000
	13     1.030000   0.000000    0.000000        0.000000
	14     1.030000   0.000000    0.000000        0.000000
	15     1.030000   0.000000   -120792.011150   9018.761039
	16     1.032575  -1.736474    0.000000        0.000000
	17     1.032575  -1.736474    0.000000        0.000000
	18     1.032575  -1.736474    0.000000        0.000000
	19     1.032575  -1.736474    0.000000        0.000000
	20     1.032575  -1.736474    38000.000000    6000.000000
	21     1.032575  -1.736474   -20000.000000   -4000.000000
	22     1.032575  -1.736474    0.000000        0.000000
	23     1.032575  -1.736474    0.000000        0.000000
	24     1.032575  -1.736474    0.000000        0.000000
	25     1.032575  -1.736474    0.000000        0.000000
	26     1.032575  -1.736474    0.000000        0.000000
	27     1.032575  -1.736474    0.000000        0.000000
	28     1.032575  -1.736474    0.000000        0.000000
	29     1.032575  -1.736474    0.000000        0.000000
	30     1.032575  -1.736474    0.000000        0.000000
	31     1.032575  -1.736474    0.000000        0.000000
	32     1.022189  -3.963880    51362.605035   -3216.223733
	33     1.014620  -4.628114    38000.000000    6000.000000
	34     1.025561  -3.522127    64902.295399   -6851.853557
	35     1.030000  -3.266803   -62000.000000   -12604.41768
	36     1.001993  -36.383570   3000.000000     1000.000000
	37     1.002942  -36.236992   5500.000000     1950.000000
	38     0.999218  -36.346819   400.000000      100.000000
	39     0.996617  -36.429063   400.000000      60.000000
	40     0.995070  -36.477681   400.000000      60.000000
	41     0.994577  -36.492365   0.000000        0.000000
	42     1.013857  -35.781827  -1600.000000    -40.000000
	43     1.009969  -35.932650  -100.000000      60.000000
	44     1.005931  -36.100943   400.000000      60.000000
	45     0.984188  172.591812   100.000000      10.000000
	46     0.979053  172.599363   4.000000        3.000000
	47     0.974501  172.596337   10.000000       3.000000
	48     0.971226  172.597077   5.000000        3.000000
	49     0.968654  172.589454   10.000000       3.000000
	50     0.967367  172.585629   10.000000       3.000000
	51     0.976027  172.628905   10.000000       3.000000
	52     0.969779  172.672240   10.000000       3.000000
	53     0.953259  173.033013   5.000000        3.000000
	54     0.944973  173.218135   5.000000        3.000000
	55     0.953259  173.033013   5.000000        3.000000
	56     0.944973  173.218135   5.000000        3.000000
	
---------------------------------

**Line result table**

.. code:: python

	net.res_line

	# Output
	index   p_from_kw      q_from_kvar   p_to_kw       q_to_kvar   pl_kw  
	                                                    
	0       39789.14439   -4545.20740   -39204.91000   4973.46900  584.23521
	1       19772.73911    3692.45340   -19671.52000  -4128.6260   101.22115
	2      -26941.65698   -5372.47387    27225.21000   5045.56000  283.55529
	3      -27804.55829   -5285.78949    27953.36000   5136.34400  148.79971
	4      -6804.67979    -3263.27249    6821.43000    2422.51400  16.74994
	5       62975.20714   -10161.39992  -62235.70000   9918.76600  739.49742
	6       1390.80090     208.79883    -1386.04600   -218.23580   4.75529
	7       986.04561      118.23579    -983.65630    -129.31080   2.38933
	8       583.65628      69.31076     -582.81410    -81.45682    0.84221
	9       182.81407      21.45682     -182.73070    -34.12819    0.08340
	10      0.00010       -12.72574      0.00000       0.00000     0.00010
	11      1600.00000     40.00000     -1593.98000   -48.79868    6.01966
	12      1693.98034    -11.20132     -1687.18600    3.06830     6.79393
	13     -1283.22453     52.96818      1287.18600   -63.06830    3.96188
	14      39.52183       15.18051     -39.31365     -15.10751    0.20818
	15      35.31365       12.10751     -35.15008     -12.05036    0.16357
	16      25.15008       9.05037      -25.06545     -9.02128     0.08464
	17      20.06545       6.02128      -20.01310     -6.00366     0.05234
	18      10.01310       3.00366      -10.00000     -3.00000     0.01310
	19      41.09293       18.29533     -40.74042     -18.17152    0.35250
	20      30.74042       15.17152     -30.53225     -15.09901    0.20817
	21      10.26612       6.04951      -10.05360     -6.00994     0.21253
	22      5.05360        3.00994      -5.00000      -3.00000     0.05360
	23      10.26612       6.04951      -10.05360     -6.00994     0.21253
	24      5.05360        3.00994      -5.00000      -3.00000     0.05360


	index  ...   ql_kvar      i_ka      loading_percent
	            
	0      ...   428.261734  0.203566  38.049708
	1      ...  -436.172485  0.103978  19.435141
	2      ...  -326.913843  0.142113  26.563254
	3      ...  -149.445594  0.145325  27.163565
	4      ...  -840.758711  0.038623  7.219187
	5      ...  -242.634198  0.324248  30.303467
	6      ...  -9.436960    0.081073  22.395755
	7      ...  -11.074972   0.057474  15.876927
	8      ...  -12.146064   0.034144  9.432095
	9      ...  -12.671367   0.010791  2.980913
	10     ...  -12.725742   0.000739  0.204068
	11     ...  -8.798678    0.091163  25.183102
	12     ...  -8.133026    0.096839  26.751042
	13     ...  -10.100115   0.073966  20.432659
	14     ...   0.072997    0.062091  25.657248
	15     ...   0.057144    0.055037  22.742477
	16     ...   0.029088    0.039590  16.359462
	17     ...   0.017612    0.031134  12.865351
	18     ...   0.003664    0.015578  6.437040
	19     ...   0.123805    0.065969  27.260092
	20     ...   0.072509    0.050696  20.948704
	21     ...   0.039570    0.017735  16.890669
	22     ...   0.009935    0.008906  8.482235
	23     ...   0.039570    0.017735  16.890669
	24     ...   0.009935    0.008906  8.482235
	
	
	
	
	
	
	
	
---------------------------------

**Trafo result table**

.. code:: python

	net.res_trafo

	# Output
	index  p_hv_kw         q_hv_kvar     p_lv_kw         q_lv_kvar     pl_kw        
	                                                                              
	0      120792.011232  -9018.761088  -120764.351608   12706.607349  27.659625    
	1      182.730641      46.853859    -180.614757     -43.475831     2.115810     

	
	index  ...  ql_kvar      i_hv_ka   i_lv_ka   loading_percent
                                       
	0      ...  3687.846257  0.178675  0.617241  39.200074
	1      ...  3.378100     0.010951  0.272449  47.417621
	
	
	net.res_trafo3w

	# Output
	index  p_hv_kw       q_hv_kvar     p_mv_kw       q_mv_kvar     p_lv_kw      
	
	0      8613.175019   3501.099685  -3000.000000  -1000.000000  -5607.576453  

	
	index  ...   q_lv_kvar    pl_kw     i_hv_ka   i_mv_ka   i_lv_ka   loading_percent
	                           
	0      ...  -2211.766936  5.598566  0.048096  0.091106  0.347006  24.041287
	
---------------------------------

**External grid result table**

.. code:: python

	net.res_ext_grid

	# Output
	index   p_kw           q_kvar
	
	0      -120792.011232  9018.761039 
	
---------------------------------

**Load result table**

.. code:: python

	net.res_load

	# Output
	index  p_kw     q_kvar
	
	0      38000.0  6000.0
	1      38000.0  6000.0
	2      38000.0  6000.0
	3      38000.0  6000.0
	4      38000.0  6000.0
	5      6000.0   2000.0
	6      18000.0  4000.0
	7      400.0    100.0
	8      400.0    60.0
	9      400.0    60.0
	10     400.0    60.0
	11     400.0    60.0
	12     400.0    60.0
	13     100.0    10.0
	14     10.0     3.0
	15     10.0     3.0
	16     10.0     3.0
	17     10.0     3.0
	18     10.0     3.0
	19     10.0     3.0
	20     10.0     3.0
	21     10.0     3.0
	22     10.0     3.0
	23     10.0     3.0
	24     10.0     3.0

---------------------------------

**Static generator result table**

.. code:: python

	net.res_sgen

	# Output
	index   p_kw      q_kvar
	
	0      -20000.0  -4000.0
	1      -500.0     0.0
	2      -500.0    -50.0
	3      -15000.0  -3000.0
	4      -2000.0   -100.0
	5      -6.0       0.0
	6      -5.0       0.0
	7      -5.0       0.0
	8      -5.0       0.0
	9      -5.0       0.0
	10     -5.0       0.0

---------------------------------

**Shunt result table**

.. code:: python

	net.res_shunt
	
	# Output
	index  p_kw       q_kvar       vm_pu     
	                               
	0      0.000000  -1003.076195  1.022189  
	
---------------------------------
	
**Network equivalents result table**
	
.. code:: python
	
	net.res_impedance
	
	# Output
	index  p_from_kw    q_from_kvar   p_to_kw       q_to_kvar   pl_kw      ql_kvar  
	
	0      4138.094119  196.360323   -4125.876687  -163.909306  12.217432  32.451018
	
	
	index  ...  i_from_ka  i_to_ka 
	
	0      ...  0.021202   0.021202
	
	
	net.res_xward
	
	# Output
	index  p_kw           q_kvar        vm_pu
	
	0      26902.295399  -12851.853556  1.025561
	1      13362.605035  -8213.147536   1.022189
	

.. _simple_example:

=====================================
Simple Example Network
=====================================

The following example contains all basic elements that are supported by the pandapower format. It is a simple example to show the 
basic principles of creating a pandapower network.

This example is also available as an interactive jupyter notebook. 

The finally generated network of this example may be generated and illustrated as below.

.. code:: python

 import pandapower as pp

 net_complete = pp.create_example_simple()
 
|

.. image:: /pics/example_network_simple.png
	:alt: alternate Text
	:align: center  
	
|


Create Network
================

 In this example a new network will be created. It starts by typing:
 
|
 
.. code:: python 
  
 import pandapower as pp
 
 net = pp.create_empty_network()	

-----------



**Create buses**

.. In the first step we create all the buses within the network. For all the buses at the 110kV high-voltage level (bus1, bus2, bus3) we select vn_kv = 110. 
   The remaining buses are related to the 20kV medium-voltage level. So we choose vn_kv = 20. Bus 3 and bus 4 are nodes. The type parameter is set to type = "n". All other buses are
   busbars (type = "b").




.. code:: python 

 bus1 = pp.create_bus(net, name = "HV Busbar", vn_kv = 110, type = "b")
 bus2 = pp.create_bus(net, name = "HV Busbar 2", vn_kv = 110, type = "b")
 bus3 = pp.create_bus(net, name = "HV Transformer Bus", vn_kv = 110, type = "n")
 bus4 = pp.create_bus(net, name = "MV Transformer Bus", vn_kv = 20, type = "n")
 bus5 = pp.create_bus(net, name = "MV Station 1", vn_kv = 20, type = "b")
 bus6 = pp.create_bus(net, name = "MV Station 2", vn_kv = 20, type = "b")
 bus7 = pp.create_bus(net, name = "MV Station 3", vn_kv = 20, type = "b")
 bus8 = pp.create_bus(net, name = "MV Station 4", vn_kv = 20, type = "b")

.. image:: /pics/example_network_simple_buses.png
	:alt: alternate Text
	:align: center   

*bus table:*	
	
.. code:: python 

    index  name                vn_kv  type  zone  in_service	
	
    0      HV Busbar           110    b     None  True
    1      HV Busbar 2         110    b     None  True
    2      HV Transformer Bus  110    n     None  True
    3      MV Transformer Bus  20     n     None  True
    4      MV Station 1        20     b     None  True
    5      MV Station 2        20     b     None  True
    6      MV Station 3        20     b     None  True
    7      MV Station 4        20     b     None  True
	
-------------	
	
**Create external grid**

.. The network is connected to a superordinate 110kV External Grid (point of connection = bus1). The external networks voltage value is 1.02 per unit (vm_pu) 
   and the volage angle is 20 degree (va_degree).



.. code:: python 

 pp.create_ext_grid(net, bus1, va_degree = 20)

.. image:: /pics/example_network_simple_ext_grids.png
	:alt: alternate Text
	:align: center   

*external grid table:*

.. code:: python 

 index  bus  vm_pu  va_degree  name  sk_max  sk_min  rx_max  rx_min  in_service
 
 0      0    1      20         NaN   NaN     NaN     NaN     NaN     True

 
----
 
**Create transformer**

.. The transformer connects the medium-voltage with the high-voltage side of the grid. The high voltage bus of the transformer is connected to Bus 3 and on the low voltage side the transformer is linked
   to Bus 4. In this case we elect a standard type transformer from the pandapower standard type library (std_type = XXXXXXXX). 
   The standard type includes all transformer parameters which can be find here: :ref:`transformer_table`.




.. code:: python 

 pp.create_transformer(net, bus3, bus4, name = "110kV/20kV transformer", std_type = "HV_MV_Feeder1")

.. image:: /pics/example_network_simple_trafos.png
	:alt: alternate Text
	:align: center   
	
*transformer table:*

.. code:: python 

    index  name                    std_type      hv_bus  lv_bus  sn_kva   
	                                                                   
    0      110kV/20kV transformer  HV_MV_Feeder1 2       3       25000    
	
	
    index  ...  vn_hv_kv  vn_lv_kv  vsc_percent  vscr_percent  pfe_kw  i0_percent    
                                     
    0      ...  110.0   20      12.0         0.16          0.0     0.0           
	
	
    index  ...  tp_side tp_mid  tp_min tp_max  tp_st_percent  tp_pos  in_service	
                
    0      ...  1       0       -2     2       2.5            0.0     True     
	
----	
	
**Create lines**


.. The network contains four lines with several lengths (length_km) and different standard types (std_type). The standard type includes all line parameters which can be find here: 
  :ref:`line_table`. The overhead line "AL 50" for intance contains the following parameters:{"r_ohm_per_km": 0.571, "conductor": "Al", "isolation": "PVC", "imax_ka": 0.225, "endtmp_deg": 200.0, "x_ohm_per_km": 0.392, "ices": 0.04461946, "q_mm2": 50.0  
   You also need to specify to which buses the line is connected to. (line1 -> link between bus1 and bus2) 


.. code:: python 

  line1 = pp.create_line(net, bus1, bus2, 0.225, std_type = "N2XS(FL)2Y 1x300RM/25 64/110kV it", name = "Line 1")
  line2 = pp.create_line(net, bus5, bus6, 0.075, std_type = "NA2YSY 1x300rm 12/20kV it", name = "Line 2")
  line3 = pp.create_line(net, bus5, bus7, 0.125, std_type = "NA2YSY 1x300rm 12/20kV it", name = "Line 3")
  line4 = pp.create_line(net, bus5, bus8, 0.175, std_type = "NA2YSY 1x300rm 12/20kV it", name = "Line 4")

.. image:: /pics/example_network_simple_lines.png
	:alt: alternate Text
	:align: center  
	
*line table:*	
	
.. code:: python 	
	
 index  name    std_type                           from bus  to bus  length_km    	
 
 0      Line 1  N2XS(FL)2Y 1x300RM/25 64/110kV it  0         1       0.225        
 1      Line 2  NA2YSY 1x300rm 12/20kV it          4         5       0.075        
 2      Line 3  NA2YSY 1x300rm 12/20kV it          4         6       0.125        
 3      Line 4  NA2YSY 1x300rm 12/20kV it          4         7       0.175        
 
 
 index  ...  r_ohm_per_km  x_ohm_per_km  c_nf_per_km  imax_ka  df  type  in_service
 
 0      ...  0.0613        0.144513      150.0000     0.594    1   cs    True  
 1      ...  0.1042        0.106814      340.0000     0.440    1   cs    True  
 2      ...  0.1042        0.106814      340.0000     0.440    1   cs    True  
 3      ...  0.1042        0.106814      340.0000     0.440    1   cs    True

 
---- 
	
**Create switches**

.. The switches within the grid can be assigned to two different groups of element types. The circuit breakers on the high and low voltage side of the transformer are located between
   two buses (bus2/bus3 and bus4/bus5). These switches represent bus-bus switches (et = "b"). The remaining load break switches are assigned as line-bus switches (et = "l") Switches with this
   element type connect one line with one bus.


.. code:: python 

 # (Circuit breaker)

 pp.create_switch(net, bus2, bus3, "b", type = "CB")
 pp.create_switch(net, bus4, bus5, "b", type = "CB")

 
.. code:: python 

 # (Load break switches)
 
 pp.create_switch(net, bus5, line2, "l", type = "LBS")
 pp.create_switch(net, bus6, line2, "l", type = "LBS")
 pp.create_switch(net, bus5, line3, "l", type = "LBS")
 pp.create_switch(net, bus7, line3, "l", type = "LBS")
 pp.create_switch(net, bus5, line4, "l", type = "LBS")
 pp.create_switch(net, bus8, line4, "l", type = "LBS")

.. image:: /pics/example_network_simple_switches.png
	:alt: alternate Text
	:align: center   

	
	
	
*switch table:*	
	
.. code:: python 

     index  bus  element  et  type  closed  	
	 
     0      1    2        b   CB    1
     1      3    4        b   CB    1
     2      4    1        l   LBS   1
     3      5    1        l   LBS   1
     4      4    2        l   LBS   1
     5      6    2        l   LBS   1
     6      4    3        l   LBS   1
     7      7    3        l   LBS   1

----	 
	
**Create generator / static generator / load**


.. In the last step we create three components to the different busbars (bus6, bus7, bus8)

.. code:: python 

 pp.create_gen(net, bus6, p_kw = -6000, vm_pu = 1.05)

.. code:: python 

 pp.create_sgen(net, bus7, p_kw = -2000)


.. code:: python 

 pp.create_load(net, bus8, p_kw = 20000, q_kvar = 4000, scaling = 0.6)

.. image:: /pics/example_network_simple_gens_sgens_loads.png
	:alt: alternate Text
	:align: center   
 
 
*gen table:*	

.. code:: python 
 
 index  name  bus   p_kw  vm_pu  sn_kva  scaling  in_service  type 
 
 0      None  5    -6000  1.05   None    1        True        sync
 
*sgen table:*	

.. code:: python 
 
 index  name  bus   p_kw  q_kvar  sn_kva  scaling  in_service  type 
 
 0      None  6    -2000  0       None    1        True        PV
 
 
*load table:*	

.. code:: python 
 
 index  name  bus  p_kw   q_kvar  sn_kva  scaling  in_service
                   
 0      NaN   7    20000  4000    NaN     0.6      True
 
---- 
 
**Create shunt**

.. code:: python

 pp.create_shunt(net, bus3, p_kw=0, q_kvar=-960, name='Shunt')

.. image:: /pics/example_network_simple_shunts.png
	:alt: alternate Text
	:align: center 



*shunt table:*

.. code:: python 
 
 index  bus  name   p_kw   q_kvar  in_service

 0      2    Shunt  0.0   -960.0   1

---- 
 
Powerflow and Result Tables
========================================

By executing a loadflow, the following pandapower tables contain all the results of the different components (bus, line, trafo, 
ext_grid, gen, sgen, load):


.. code:: python  
 
 pp.runpp(net)

.. code:: python 


 res_bus:

 index  vm_pu     va_degree   p_kw           q_kvar
                                            
 0      1.000000  20.000000  -4019.069403   -124.661744
 1      0.999995  19.999394   0.000000       0.000000
 2      0.999995  19.999394   0.000000       0.000000
 3      1.049846  18.878819   0.000000       0.000000
 4      1.049846  18.878819   0.000000       0.000000
 5      1.050000  18.882804  -6000.000000   -2224.389105
 6      1.049908  18.882284  -2000.000000    0.000000
 7      1.049218  18.855331   12000.000000   2400.000000

 
 
 res_line:
 
 index   p_from_kw      q_from_kvar   p_to_kw       q_to_kvar      i_ka      

 0       4019.069403    124.661744   -4019.05095   -2.529124e+02   0.021136  
 1      -5999.274219   -2227.177476   6000.00000    2.224389e+03   0.175962  
 2      -1999.881838   -5.765624      2000.00000   -5.017320e-09   0.054991  
 3       12006.200877   2398.120414  -12000.00000  -2.400000e+03   0.336699  
 
 
 index  ...  loading_percent
 
 0      ...  3.558313
 1      ...  39.991403
 2      ...  12.497907
 3      ...  76.522474
 

 
 res_load:
 
 index  p_kw   q_kvar
 
 0      12000  2400

 
 res_sgen:
 
 index   p_kw  q_kvar
 
 0      -2000  0
 
 
 res_gen:
 
 index   p_kw   q_kvar       va_degree
                             
 0      -6000  -2224.389105  18.882804

 
 res_shunt:
 
 index  vm_pu      p_kw   q_kvar
        
 0      1.000028  0.0    -960.053767
 
 
 res_trafo:
 
 index  p_hv_kw     q_hv_kvar    p_lv_kw      q_lv_kvar   i_ka      loading_percent
 
 0      4019.05095  252.912379  -4007.04482  -165.177314  0.110275  10.067553

 
 
 res_ext_grid:
 
 index   p_kw          q_kvar
 
 0      -4019.069403  -124.661744
 
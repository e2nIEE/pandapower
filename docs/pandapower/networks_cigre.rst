==============
CIGRE Networks
==============

CIGRE-Networks were developed by the CIGRE Task Force C6.04.02 to "facilitate the analysis 
and validation of new methods and techniques" that aim to "enable the economic, robust and 
environmentally responsible integration of DER" (Distributed Energy Resources).
CIGRE-Networks are a set of comprehensive reference systems to allow the "analysis of DER 
integration at high voltage, medium voltage and low voltage and at the desired degree of detail".

[Source: Task Force C6.04.02, *Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources*]

High voltage transmission network
---------------------------------

.. code:: python

 import pandapower.networks as pn
 
 # You have to specify a length for the connection line between busses 6a and 6b
 net = pn.create_cigre_network_hv(length_km_6a_6b)
 
 '''
 This pandapower network includes the following parameter tables:
   - shunt (3 elements)
   - trafo (6 elements)
   - bus (13 elements)
   - line (9 elements)
   - load (5 elements)
   - ext_grid (1 elements)
   - gen (3 elements)
 '''
 
 
.. image:: /pandapower/pics/cigre_network_hv.png
	:width: 42em
	:alt: alternate Text
	:align: center
    

    

Medium voltage distribution network
-----------------------------------

.. code:: python

 import pandapower.networks as pn
 
 net = pn.create_cigre_network_mv(with_der=False)
 
 '''
 This pandapower network includes the following parameter tables:
   - switch (8 elements)
   - load (18 elements)
   - ext_grid (1 elements)
   - line (15 elements)
   - trafo (2 elements)
   - bus (15 elements)
 '''
 
 
.. image:: /pandapower/pics/cigre_network_mv.png
	:width: 42em
	:alt: alternate Text
	:align: center


---------------------------


Medium voltage distribution network with DER
--------------------------------------------

.. note:: This network contains additional 9 distributed energy resources compared to medium voltage distribution network:

			- 8 photovoltaic generators 
			- 1 wind turbine

Compared to the CIGRE Task Force C6.04.02 paper 2 Batteries, 2 residential fuel cells, 1 CHP diesel and 1 CHP fuel cell are neglected.

.. code:: python

    import pandapower.networks as pn
    
    net = pn.create_cigre_network_mv(with_der=True)
    
    '''
    This pandapower network includes the following parameter tables:
      - switch (8 elements)
      - load (18 elements)
      - ext_grid (1 elements)
      - sgen (9 elements)
      - line (15 elements)
      - trafo (2 elements)
      - bus (15 elements)
    '''
 
.. image:: /pandapower/pics/cigre_network_mv_der.png
	:width: 42em
	:alt: alternate Text
	:align: center


---------------------------




Low voltage distribution network
---------------------------------

.. code:: python

 import pandapower.networks as pn
 
 net = pn.create_cigre_network_lv()
 
 '''
 This pandapower network includes the following parameter tables:
   - switch (3 elements)
   - load (15 elements)
   - ext_grid (1 elements)
   - line (37 elements)
   - trafo (3 elements)
   - bus (44 elements)
 '''
 
 
.. image:: /pandapower/pics/cigre_network_lv.png
	:width: 42em
	:alt: alternate Text
	:align: center
	
	
---------------------------

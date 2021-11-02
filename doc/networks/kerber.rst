===============
Kerber Networks
===============


The kerber networks are based on the grids used in the dissertation "Aufnahmefähigkeit von Niederspannungsverteilnetzen
für die Einspeisung aus Photvoltaikanlagen" (Capacity of low voltage distribution networks
with increased feed-in of photovoltaic power) by Georg Kerber.
The following introduction shows the basic idea behind his network concepts and demonstrate how you can use them in pandapower.


*"The increasing amount of new distributed power plants demands a reconsideration of conventional planning
strategies in all classes and voltage levels of the electrical power networks. To get reliable results on
loadability of low voltage networks statistically firm network models are required. A strategy for the classification
of low voltage networks, exemplary results and a method for the generation of reference networks are shown."*
(source: https:/mediatum.ub.tum.de/doc/681082/681082.pdf)


.. warning::

    The representative grids for sub-urban areas (Vorstadt) were deduced as open-ring grids from meshed grids.
    They are therefore only valid under the assumption of homogeneous load and generation profiles, and not for
    inhomogeneous operation or even short-circuit situations.

.. seealso::

	- Georg Kerber, `Aufnahmefähigkeit von Niederspannungsverteilnetzen für die Einspeisung aus Photovoltaikkleinanlagen <https://mediatum.ub.tum.de/doc/998003/998003.pdf>`__, Dissertation
	- Georg Kerber, `Statistische Analyse von NS-Verteilungsnetzen und Modellierung von Referenznetzen <https://mediatum.ub.tum.de/doc/681082/681082.pdf>`__



Average Kerber networks
========================


**Kerber Landnetze:**

 - Low number of loads per transformer station
 - High proportion of agriculture and industry
 - Typical network topologies: line

**Kerber Dorfnetz:**

 - Higher number of loads per transformer station (compared to Kerber Landnetze)
 - Lower proportion of agriculture and industry
 - Typical network topologies: line, open ring

**Kerber Vorstadtnetze:**

 - Highest number of loads per transformer station (compared to Kerber Landnetze/Dorfnetz)
 - no agriculture and industry
 - high building density
 - Typical network topologies: open ring, meshed networks



.. tabularcolumns:: |l|l|l|l|l|
.. csv-table::
   :file: kerber.csv
   :delim: ;

You can include the kerber networks by simply using:



.. code:: python

 import pandapower.networks as pn

 net1 = pn.create_kerber_net()






Kerber Landnetze
----------------



.. code:: python

 import pandapower.networks as pn

 net1 = pn.create_kerber_landnetz_freileitung_1()

 '''
 This pandapower network includes the following parameter tables:
   - load (13 elements) p_load_in_mw=8,  q_load_in_mw=0
   - bus (15 elements)
   - line (13 elements) std_type="Al 120", l_lines_in_km=0.021
   - trafo (1 elements)  std_type="0.125 MVA 10/0.4 kV Dyn5 ASEA"
   - ext_grid (1 elements)
 '''

 net2 = pn.create_kerber_landnetz_freileitung_2()

 '''
 This pandapower network includes the following parameter tables:
   - load (8 elements) p_load_in_mw=8,  q_load_in_mw=0
   - bus (10 elements)
   - line (8 elements)  std_type="AL 50", l_lines_1_in_km=0.038, l_lines_2_in_km=0.081
   - trafo (1 elements)  std_type="0.125 MVA 10/0.4 kV Dyn5 ASEA"
   - ext_grid (1 elements)
 '''





.. image:: /pics/networks/kerber//kerber_landnetz_freileitung.png
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

--------------------


.. code:: python

 import pandapower.networks as pn

 net1 = pn.create_kerber_landnetz_kabel_1()

 '''
 This pandapower network includes the following parameter tables:
   - load (8 elements)  p_load_in_mw=8,  q_load_in_mw=0
   - bus (18 elements)
   - line (16 elements)  std_type="NAYY 150", std_type_branchout_line="NAYY 50"
   - trafo (1 elements)  std_type = "0.125 MVA 10/0.4 kV Dyn5 ASEA"
   - ext_grid (1 elements)
 '''

 net2 = pn.create_kerber_landnetz_kabel_2()

 '''
 This pandapower network includes the following parameter tables:
  - load (14 elements)  p_load_in_mw=8,  q_load_in_mw=0
  - bus (30 elements)
  - line (28 elements)  std_type="NAYY 150", std_type_branchout_line="NAYY 50"
  - trafo (1 elements)  std_type="0.125 MVA 10/0.4 kV Dyn5 ASEA"
  - ext_grid (1 elements)
 '''


.. image:: /pics/networks/kerber//kerber_landnetz_kabel.png
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

---------------------------

Kerber Dorfnetz
---------------


.. code:: python

 import pandapower.networks as pn

 net = pn.create_kerber_dorfnetz()

 '''
 This pandapower network includes the following parameter tables:
   - load (57 elements) p_load_in_mw=6,  q_load_in_mw=0
   - bus (116 elements)
   - line (114 elements) std_type="NAYY 150"; std_type_branchout_line="NAYY 50"
   - trafo (1 elements) std_type="0.4 MVA 10/0.4 kV Yyn6 4 ASEA"
   - ext_grid (1 elements)
 '''



.. image:: /pics/networks/kerber//kerber_dorfnetz_1.PNG
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center


---------------------------

Kerber Vorstadtnetze
--------------------


.. code:: python

 import pandapower.networks as pn

 net1 = pn.create_kerber_vorstadtnetz_kabel_1()

 '''
 This pandapower network includes the following parameter tables:
   - load (146 elements) p_load_in_mw=2,  q_load_in_mw=0
   - bus (294 elements)
   - line (292 elements) std_type="NAYY 150", std_type_branchout_line_1="NAYY 50", std_type_branchout_line_2="NYY 35"
   - trafo (1 elements) std_type="0.63 MVA 20/0.4 kV Yyn6 wnr ASEA"
   - ext_grid (1 elements)
 '''




.. image:: /pics/networks/kerber//kerber_vorstadtnetz_a.PNG
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

--------------------


.. code:: python

 import pandapower.networks as pn

 net2 = pn.create_kerber_vorstadtnetz_kabel_2()

 '''
 This pandapower network includes the following parameter tables:
   - load (144 elements) p_load_in_mw=2,  q_load_in_mw=0
   - bus (290 elements)
   - line (288 elements) std_type="NAYY 150", std_type_branchout_line_1="NAYY 50", std_type_branchout_line_2="NYY 35"
   - trafo (1 elements) "std_type=0.63 MVA 20/0.4 kV Yyn6 wnr ASEA"
   - ext_grid (1 elements)
 '''




.. image:: /pics/networks/kerber//kerber_vorstadtnetz_b.PNG
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center



---------------------------

Extreme Kerber networks
=======================


The typical kerber networks represent the most common low-voltage distribution grids.
To produce statements of universal validity or check limit value, a significant part of all existing grids have to be involved.
The following grids obtain special builds of parameters (very high line length, great number of branches or
high loaded transformers). These parameters results in high loaded lines and low voltage magnitudes within the
extreme network. By including the extreme networks, kerber reached the 95% confidence interval.

Therefore 95% of all parameter results in an considered distribution grid are equal or better compared to the outcomes from kerber extreme networks.
Besides testing for extreme parameters you are able to check for functional capability of reactive power control.
Since more rare network combination exist, the total number of extreme grids is higher than the amount of typical kerber networks.

.. seealso::

	- Georg Kerber, `Aufnahmefähigkeit von Niederspannungsverteilnetzen für die Einspeisung aus Photovoltaikkleinanlagen <http:/mediatum.ub.tum.de/doc/998003/998003.pdf>`_, Dissertation
	- Georg Kerber, `Statistische Analyse von NS-Verteilungsnetzen und Modellierung von Referenznetzen <http:/mediatum.ub.tum.de/doc/681082/681082.pdf>`_

.. tabularcolumns:: |l|l|l|l|l|
.. csv-table::
   :file: kerber_extreme.csv
   :delim: ;

--------------

The Kerber extreme networks are categorized into two groups:

 **Type I:** Kerber networks with extreme lines

 **Type II:** Kerber networks with extreme lines and high loaded transformer




.. note:: Note that all Kerber exteme networks (no matter what type / territory) consist of various branches, linetypes or line length.



Extreme Kerber Landnetze
------------------------



.. code:: python

 import pandapower.networks as pn

 '''Extrem Landnetz Freileitung Typ I'''
 net = pn.kb_extrem_landnetz_freileitung()


 '''Extrem Landnetz Kabel Typ I'''
 net = pn.kb_extrem_landnetz_kabel()





.. image:: /pics/networks/kerber//kerber_extrem_landnetz_typ_1.png
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

---------------------------


.. code:: python

 import pandapower.networks as pn

 '''Extrem Landnetz Freileitung Typ II'''
 net = pn.kb_extrem_landnetz_freileitung_trafo()


 '''Extrem Landnetz Kabel Typ II'''
 net = pn.kb_extrem_landnetz_kabel_trafo()




.. image:: /pics/networks/kerber//kerber_extrem_landnetz_typ_2.png
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

Extreme Kerber Dorfnetze
------------------------



.. code:: python

 import pandapower.networks as pn

 '''Extrem Dorfnetz Kabel Typ I'''
 net = pn.kb_extrem_dorfnetz()





.. image:: /pics/networks/kerber//kerber_extrem_dorfnetz_typ_1.png
	:height: 918.0px
	:width: 1282.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

---------------------------

.. code:: python

 import pandapower.networks as pn

 '''Extrem Dorfnetz Kabel Typ II'''
 net = pn.kb_extrem_dorfnetz_trafo()



.. image:: /pics/networks/kerber//kerber_extrem_dorfnetz_typ_2.png
	:height: 918.0px
	:width: 1582.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

Extreme Kerber Vorstadtnetze
----------------------------

.. code:: python

 import pandapower.networks as pn

 '''Extrem Vorstadtnetz Kabel_a Typ I'''
 net = pn.kb_extrem_vorstadtnetz_1()


.. image:: /pics/networks/kerber//kerber_extrem_vorstadt_a_typ_1.png
	:height: 718.0px
	:width: 1402.0px
	:scale: 52%
	:alt: alternate Text
	:align: center

---------------------------


.. code:: python

 import pandapower.networks as pn

 '''Extrem Vorstadtnetz Kabel_b Typ I'''
 net = pn.kb_extrem_vorstadtnetz_2()


.. image:: /pics/networks/kerber//kerber_extrem_vorstadt_b_typ_1.png
	:height: 818.0px
	:width: 1452.0px
	:scale: 52%
	:alt: alternate Text
	:align: center


---------------------------

.. code:: python

 import pandapower.networks as pn

 '''Extrem Vorstadtnetz Kabel_c Typ II'''
 net = pn.kb_extrem_vorstadtnetz_trafo_1()


.. image:: /pics/networks/kerber//kerber_extrem_vorstadt_c_typ_2.png
	:height: 918.0px
	:width: 1482.0px
	:scale: 52%
	:alt: alternate Text
	:align: center


---------------------------

.. code:: python

 import pandapower.networks as pn

 '''Extrem Vorstadtnetz Kabel_d Typ II'''
 net = pn.kb_extrem_vorstadtnetz_trafo_2()


.. image:: /pics/networks/kerber//kerber_extrem_vorstadt_d_typ_2.png
	:height: 918.0px
	:width: 1482.0px
	:scale: 52%
	:alt: alternate Text
	:align: center
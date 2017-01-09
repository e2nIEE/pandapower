==========
IEEE cases
==========

.. note::

    All IEEE case files were converted from PYPOWER


Case 4gs
------------

.. autofunction:: pandapower.networks.case4gs()

---------------------------

Case 6ww
--------------

.. autofunction:: pandapower.networks.case6ww()
---------------------------

Case 9
--------

.. autofunction:: pandapower.networks.case9()

---------------------------

Case 9Q
---------

.. autofunction:: pandapower.networks.case9Q()
---------------------------

Case 14
---------

.. autofunction:: pandapower.networks.case14()
---------------------------

Case 24_ieee_rts
---------

.. autofunction:: pandapower.networks.case24_ieee_rts()
---------------------------

Case 30
--------

.. autofunction:: pandapower.networks.case30()
---------------------------

Case 30pwl
-----------

.. autofunction:: pandapower.networks.case30pwl()

---------------------------

Case 30Q
---------

.. autofunction:: pandapower.networks.case30Q()

---------------------------

Case 39
---------

.. autofunction:: pandapower.networks.case39()

---------------------------

Case 57
---------

.. autofunction:: pandapower.networks.case57()

---------------------------

Case 118
---------

Case118 is not provided in pandapower because two transformer branches would act capacitively.
Possibly the data are this way to model the whole substation with reactive power compensation devices.
However if you want to receive case118 you may use the following code to get a tiny changed data set
with b_shunt=0 for transformer branches from node 68 to 116 and from 86 to 87.

.. code:: python

 import pandapower.test as pt
 import pandapower.converter as pc
 ppc_case118_adapt = pt.case118()
 pp_case118_adapt = pc.ppc2pp(case118)

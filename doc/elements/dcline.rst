.. _gen:

=============
DC Line
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_dcline

Input Parameters
=====================

*net.dcline*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: dcline_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

.. |br| raw:: html

   <br />
   
\*necessary for executing a power flow calculation |br| \*\*optimal power flow parameter 

.. note::
    DC line is only able to model one-directional loadflow for now, which is why p_kw / max_p_kw have to be > 0.
   
Electric Model
=================

Generators are modelled as PV-nodes in the power flow:

.. image:: gen.png
	:width: 12em
	:alt: alternate Text
	:align: center

Voltage magnitude and active power are defined by the input parameters in the generator table:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{gen} &= p\_kw * scaling \\
    v_{bus} &= vm\_pu
   \end{align*}
    
Result Parameters
==========================
*net.res_dcline*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: dcline_res.csv
   :delim: ;
   :widths: 10, 10, 50
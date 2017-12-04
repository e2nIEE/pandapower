.. _dcline:

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

A DC line is modelled as two generators in the loadflow:

.. image:: dcline1.png
	:width: 20em
	:alt: alternate Text
	:align: center

.. image:: dcline2.png
	:width: 20em
	:alt: alternate Text
	:align: center
    
The active power at the from side is defined by the parameters in the dcline table. The active power at the to side is equal to the active power on the from side minus the losses of the DC line.

.. math::
   :nowrap:
   
   \begin{align*}
    P_{from} &= p\_kw \\
    P_{to} &= - (p\_kw - loss\_kw) \cdot (1 - \frac{loss\_percent}{100})
    
   \end{align*}

The voltage control with reactive power works just as described for the generator model. Maximum and Minimum reactive power limits are considered in the OPF, and in the PF if it is run with enforce_q_lims=True.
   
Result Parameters
==========================
*net.res_dcline*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: dcline_res.csv
   :delim: ;
   :widths: 10, 10, 50
   
.. math::
   :nowrap:
   
   \begin{align*}
    p\_from\_kw &= P_{from} \\
    p\_to\_kw &= P_{to} \\
    pl\_kw &= p\_from\_kw + p\_to\_kw \\
    q\_from\_kvar &= Q_{from} \\
    q\_to\_kvar &= Q_{to} \\
    va\_from\_degree &= \angle \underline{v}_{from} \\
    va\_to\_degree &= \angle \underline{v}_{to} \\
    vm\_from\_degree &= |\underline{v}_{from}| \\    
    vm\_to\_degree &= |\underline{v}_{to}| \\
   \end{align*}

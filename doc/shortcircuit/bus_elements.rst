Bus Elements
================

External Grid
-----------------

When calculating maximum short-circuit currents, the impedance of an external grid connection is given as:

.. math::

  z_{eg} =& \frac{c_{max}}{s\_sc\_max\_mva} \\[1em]
  x_{sg} =& \frac{z_{sg}}{\sqrt{1 - (rx\_max)^2}} \\[1em]
  r_{sg} =& rx\_max \cdot x_{sg} 
  
where :math:`rx\_max` and :math:`s\_sc\_max\_mva` are parameters in the ext_grid table and :math:`c_{max}` is the :ref:`voltage correction factor <c>` of the
external grid bus.

In case of minimal short-circuit currents, the impedance is calculated accordingly:

.. math::

  z_{eg} =& \frac{c_{min}}{s\_sc\_min\_mva} \\[1em]
  x_{sg} =& \frac{z_{sg}}{\sqrt{1 - (rx\_min)^2}} \\[1em]
  r_{sg} =& rx\_min \cdot x_{sg} 

Static Generator
-----------------------
Not all inverter based elements contribute to the short-circuit current. That is why it can be chosen in the runsc function if static generators
are to be considered or not. If they are considered, the short-circuit impedance is defined according to the standard as:

.. math::

    Z_{sg} = \frac{1}{3} \cdot \frac{vn\_kv^2 \cdot 1000}{sn\_kva} \\
    X_{sg} = \frac{Z_{sg}}{\sqrt{1 - 0.1^2}} \\
    R_{sg} = 0.1 \cdot X_{sg}

Synchronous Generator
-----------------------

Loads and Shunts
-----------------
The contribution of loads and shunts are negligible according to the standard and therefore neglected in the short-circuit calculation.
=======================
Short-Circuit Currents
=======================

The short-circuit currents are calculated with the equivalent voltage source at the fault location.  
For an explanation of the theory behind short-circuit calculations according to IEC 60909
please refer to the norm or secondary literature:

.. seealso::

    `IEC 60909-0:2016 <https://webstore.iec.ch/publication/24100>`_ Short-circuit currents in three-phase a.c. systems
    
    `According to the IEC 60909 <http://www.openelectrical.org/wiki/index.php?title=According_to_the_IEC_60909>`_ on openelectrical


Initial Short-Circuit Current
==========================================
The initial short-circuit current is calculated as:

.. math::
   
   I''_{k} = \frac{c \cdot V_N}{\sqrt{3} \cdot Z_k}

where c is the :ref:`voltage correction factor <c>` and :math:`Z_k` is the short-circuit impedance at the fault bus. 
The short-circuit impedance at the fault location is defined by the impedances of the :ref:` <network elements>` plus the defined fault impedance.
:math:`Z_k` is calculated from the impedance matrix, which is the inverse of the nodal point admittance matrix. 


Peak Short-Circuit Current
==========================================

The peak short-circuit current is calculated as:

.. math::

    i_p = \kappa \cdot \sqrt{2} \cdot I''_k

where the factor :math:`\kappa` depends on the short-circuit impedance as defined :ref:`here <kappa>`.
    
Thermal Short-Circuit Current
==========================================

The equivalent 

.. math::

    I_{th} = I''_k \cdot \sqrt{m + n}

where the factors m and n are defined :ref:`here <mn>`.

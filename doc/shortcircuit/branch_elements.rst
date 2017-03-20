Branch Elements
=================


Line
-----------------------

.. math::
   :nowrap:

   \begin{align*}
    \underline{R} &= r\_ohm\_per\_km \cdot \frac{length\_km}{parallel} \cdot c_{R, line}\\
    \underline{X} &= x\_ohm\_per\_km \cdot \frac{length\_km}{parallel} 
   \end{align*}
   
where the correction factor  :math:`c_{R, line}` is defined as:

.. math::

  c_{R, line}=\left\{
  \begin{array}{@{}ll@{}}
            1 & \text{ for maximum short-circuit calculations} \\
            1 + 0.04 K^{-1} (endtemp\_degree - 20°C) & \text{ for minimum short-circuit calculations} 
  \end{array}\right.

The end temperature in degree after a fault has to be defined with the parameter endtemp\_degre in the line table.

The shunt capacitance is neglected for short-circuit calculations, so :math:`c_nf_per_km` does not have any influence on the short-circuit current.

Two-Winding Transformer
-------------------------

The short-circuit impedance is calculated as:

.. math::
   :nowrap:

   \begin{align*}
   z_k &= \frac{vsc\_percent}{100} \cdot \frac{1000}{sn\_kva} \cdot c_{trafo} \\
   r_k &= \frac{vscr\_percent}{100} \cdot \frac{1000}{sn\_kva} \cdot c_{trafo} \\
   x_k &= \sqrt{z^2 - r^2} \\
   \end{align*}    

where the correction factor :math:`c_{trafo}` is defined in the standard as:

.. math::

    c_{trafo} = 0.95 \frac{c_{max}}{1 + 0.6 x_T}

where :math:`c_{max}` is the :ref:`voltage correction factor <c>` on the low voltage side of the transformer and :math:`x_T` is the transformer impedance relative to the
rated values of the transformer.

The ratio of the transformer is considered to be the nominal ratio, the tap changer positions are not considered according to the standard. The shunt 
admittance is neglected for short-circuit calculations, so :math:`pfe_kw` and :math:`i0_percent`doe not have any influence on the short-circuit current.


Three-Winding Transformer
--------------------------


Impedance
--------------------------
The impedance element is a generic element that can be used 

.. |variable| replace:: 1

|variable|




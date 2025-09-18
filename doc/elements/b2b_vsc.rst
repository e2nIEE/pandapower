.. _b2b_vsc:

==============================================
Back2Back Voltage Source Converter (B2B VSC)
==============================================

The B2B VSC is implemented as the name suggests by using two VSC, which are connected on the AC side.

.. seealso::
	:ref:`Voltage Source Converter (VSC) <vsc>`

The Back2Back Voltage Source Converter (VSC), is a power electronic device used to convert alternating current (AC) to direct
current (DC) and vice versa. It connects an AC system to a dual DC system. Normally it is used to create multi terminal HVDC systems.
For example for modelling wind park interconnects with a metallic return line. Currently this construct is not possible,
due to a limitation in the underlying pypower modelling approach. Otherwise this model employs two VSC which are connected
to a single AC bus.
In pandapower the VSC model include a coupling transformer, therefore the input parameter are split to both VSC, so each have:
r\_ohm/2, x\_ohm/2 and r_dc_ohm/2.


.. seealso::
	:ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create.create_b2b_vsc

Input Parameters
=====================

*net.b2b_vsc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: b2b_vsc_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

   
Electric Model
=================


.. imagesvg:: b2b_vsc.svg
	:alt: B2B VSC electric model
	:align: center
	:tagtype: object

Image was created with https://www.circuit2tikz.tf.fau.de/designer/.
Back2Back Voltage Source Converters are self-commutated converters to connect HVAC and HVDC systems using devices suitable
for high power electronic applications, such as IGBTs.


Limitations
=================

Since the powerflow equations are modelled that every component is connected to the same ground, topologies employing a lifted
or virtual ground are not currently supported. For example, one could attach two B2B VSC on the minus and plus side,
and therefore "lift" one of the VSC to create a virtual ground for a metallic return line scenario. But in this case,
the first lower VSC will be shorted by second upper VSC. Currently a workaround is employed by creating the topology,
but setting the correspoinding metallic return line out of service. Then a specialized controller needs to be employed,
which calculates the resulting currents and updates the out-of-service line.
See test_facts_b2b_vsc.py: test_hvdc_interconnect_with_dmr() for an example.


Result Parameters
==========================
*net.res_b2b_vsc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: b2b_vsc_res.csv
   :delim: ;
   :widths: 10, 10, 40

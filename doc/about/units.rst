.. _conventions:

============================    
Unit System and Conventions
============================

**Naming Conventions**

Parameters are always named in the form of <parameter>_<unit>, such as:

.. tabularcolumns:: |l|l|
.. csv-table:: 
   :file: nameing1.csv
   :delim: ;
   :widths: 30, 30

Constraint parameters are always named with max or min as the prefix to the variable which is constrained, for example:

.. tabularcolumns:: |l|l|
.. csv-table:: 
   :file: nameing2.csv
   :delim: ;
   :widths: 30, 30

It is advised to keep consistent with these naming conventions when extending the framework and introducing new parameters.
   
**Three Phase System**

For the three phase system, the following conventions apply:

    - voltage values are given as phase-to-phase voltages
    - current values are given as phase currents
    - power values are given as three-phase power flows

The power equation in the three phase system is therefore given as :math:`S = \sqrt3 \cdot V \cdot I`.

Since pandapower was developed for distribution systems, all power values are given in kW or kVar.

**Per Unit System**

Bus voltages are given in the per unit system. The per unit values are relative to the phase-to-phase voltages defined in net.bus.vn_kv for each bus. 

Internally, pandapower calculates with a nominal apparent power of :math:`S_{N} = 1 MVA` for the per unit system, which however should not be relevant for the user since all power values are given in physical units.

**Signing System**

For all bus-based power values, the signing is based on the consumer viewpoint:

    - positive active power is power consumption, negative active power is power generation
    - positive reactive power is inductive consumption, negative reactive power is capacitive consumption

The power flow values for branch elements (lines & transformer) are always defined as the power flow into the branch element.

**Frequency**

The frequency can be defined when creating an empty network. The frequency is only used to calculate the shunt admittance of lines, since the line reactance is given directly in ohm per kilometer.

The standard frequency in pandapower is 50 Hz, and the pandapower standard types are also chosen for 50 Hz systems. If you use a different frequency, please be aware that the line reactance values might not be realistic.
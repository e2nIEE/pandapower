=======================
Short-Circuit Currents
=======================

The short-circuit currents are calculated with the equivalent voltage source at the fault location.
For an explanation of the theory behind short-circuit calculations according to IEC 60909
please refer to the norm or secondary literature:

.. seealso::

    `IEC 60909-0:2016 <https://webstore.iec.ch/publication/24100>`_ Short-circuit currents in three-phase a.c. systems


pandapower currently implements symmetrical and two-phase faults. One phase faults and two-phase faults with earthing are not yet available.

Note that the currents are calculated as complex domain, which allows obtaining the branch-related values for current magnitude and current angle.

We implemented the superposition method that considers pre-fault voltage vector. This method is activated by passing the parameter "use_pre_fault_voltage" as True.
In this case, the values of shunt impedance of inverter-based generators and loads are considered, which are calculated based on their pre-fault current values.


.. toctree::
    :maxdepth: 1

    ikss
    ip
    ith
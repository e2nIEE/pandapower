=======================
Short-Circuit Currents
=======================

The short-circuit currents are calculated according to the **IEC 60909** international standard. pandapower supports two different calculation methods, in accordance with the IEC 60909 standard: 

1. equivalent voltage source method (i.e. by introducing the Thevenin's voltage source) at the fault location and 
2. superposition method that considers the pre-fault voltage vector.

For a detailed explanation of the theory behind short-circuit calculations according to IEC 60909
please refer to the mentioned normative document or secondary literature:

.. seealso::

    `IEC 60909-0:2016 <https://webstore.iec.ch/publication/24100>`_ Short-circuit currents in three-phase a.c. systems


**Ad 1)** As far as the equivalent voltage source method is concerned, pandapower currently implements only three-phase (symmetrical) and two-phase (asymmetrical) faults. Single-phase fault and two-phase fault with earthing are not yet available. The pandapower short-circuit calculation supports following elements:

* sgen (as motor, or as full converter generator, or as asynchronous machine, or as doubly-fed asynchronous machine),
* gen (as synchronous generator),
* ext_grid (external grid),
* line (transmission line or cable),
* trafo (two-winding transformer),
* trafo3w (three-winding transformer),
* impedance,

with the associated correction factors as defined in IEC 60909. Loads and shunts are neglected as per standard. The pandapower switch model is fully integrated into the short-circuit calculation. The calculation, furthermore, enables computing maximum (case="max") and minimum (case="min") short-circuit currents by furnishing the appropriate value for the "case" argument in the "calc_sc" function call.

The following short-circuit currents can be calculated:

* ikss (i.e. Initial symmetrical short-circuit current),
* ip (i.e. Peak short-circuit current),
* ith (i.e. Equivalent thermal short-circuit current),

either as:

* symmetrical three-phase or
* asymmetrical two-phase

short-circuit current. Currents "ip" and "ith" are only implemented for short-circuits far from the synchronous generators.

Calculations are available for the meshed as well as for the radial networks. pandapower includes a meshing detection that automatically detects the meshing (topology="auto") for each short-circuit location during the "calc_sc" function call. Alternatively, the topology can be set to "radial" or "meshed" in order to circumvent the check and shorten the computation time (for large networks). This is achieved by appropriately setting the "topology" argument in the "calc_sc" function call. 

It is also possible to specify a fault impedance in the short-circuit calculation, by providing the values for parameters "r_fault_ohm" and/or "x_fault_ohm" in the "calc_sc" function call, which, respectively, define resistance and reactance at the point of the short-circuit. For the phase-to-phase (i.e. three-phase and two-phase) short-circuits this will be the arc resistance.

Note that the short-circuit currents are calculated in the complex domain, which allows obtaining the branch-related values for current magnitude and phase angle.

**Ad 2)** The superposition method (per IEC 60909) considers the pre-fault voltage vector. This method is activated by passing the argument "use_pre_fault_voltage" as True to the "calc_sc" function call.

Note that the user needs to explicitly carry out the power flow calculation (i.e. by invoking the "runpp" function call) on the network, before proceeding to the short-circuit calculation with the superposition method. We rely on the user to execute this step explicitly so that the user is fully aware that a power flow calculation is executed, and also has control over all the relevant options for the power flow calculation. Results provide branch currents magnitude and angle, along with active and reactive power flows, and bus voltages magnitude and angle values.

In the case of the superposition method, the values of shunt impedance of inverter-based generators and loads are considered, which are calculated based on their pre-fault current values. The following differences to the worst-case scenario calculation should apply:

- transformer correction factor is not applied,
- load, sgen are additionally modelled as shunt admittances (calculated based on their pre-fault currents),
- the grid must contain the results of a successful power flow calculation.

The results for all elements and different short-circuit currents have been tested against commercial software to ensure that correction factors are correctly applied.


.. toctree::
    :maxdepth: 1

    ikss
    ip
    ith

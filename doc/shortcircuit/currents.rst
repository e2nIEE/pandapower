=======================
Short-Circuit Currents
=======================

The short-circuit currents are calculated according to the **IEC 60909** international standard. `pandapower` supports two different calculation methods, in accordance with the IEC 60909 standard: 

1. equivalent voltage source method (i.e. by introducing the Thevenin's voltage source) at the fault location and 
2. superposition method that considers the pre-fault voltage vector.

For both methods, the possiboility to use LU factorization is implemented (option inverse_y=False). This improves the performance of the short-circuit calculation and reduces the required RAM, especially for large grid models, if the number of fault buses is low.

For a detailed explanation of the theory behind short-circuit calculations according to IEC 60909
please refer to the mentioned normative document or secondary literature:

.. seealso::

    `IEC 60909-0:2016 <https://webstore.iec.ch/publication/24100>`_ Short-circuit currents in three-phase a.c. systems

Note: `pandapower` currently implements three-phase (symmetrical), two-phase (asymmetrical), and single-phase (Line-to-Ground) faults. Two-phase fault with earthing is not yet available.

**Ad 1)** The pandapower short-circuit calculation supports following elements:

* `sgen` (as motor, or as full converter generator, or as asynchronous machine, or as doubly-fed asynchronous machine),
* `gen` (as synchronous generator),
* `ext_grid` (external network equivalent),
* `line` (transmission line or cable),
* `trafo` (two-winding transformer),
* `trafo3w` (three-winding transformer),
* `impedance` (arbitrary impedance),

with the associated correction factors as defined in IEC 60909. Loads and shunts are neglected as per standard. The pandapower switch model is fully integrated into the short-circuit calculation. The calculation, furthermore, enables computing maximum (case="max") and minimum (case="min") short-circuit currents by furnishing the appropriate value for the "case" argument in the "calc_sc" function call.

Calculations are available for the meshed as well as for the radial networks. pandapower includes a meshing detection that automatically detects the meshing (topology="auto") for each short-circuit location during the "calc_sc" function call. Alternatively, the topology can be set to "radial" or "meshed" in order to circumvent the check and shorten the computation time (for large networks). This is achieved by appropriately setting the "topology" argument in the "calc_sc" function call. 

It is also possible to specify a fault impedance in the short-circuit calculation, by providing the values for parameters "r_fault_ohm" and/or "x_fault_ohm" in the "calc_sc" function call, which, respectively, define resistance and reactance parts of the impedance at the point of the short-circuit. For the phase-to-phase (i.e. three-phase and two-phase) short-circuits this will be the arc resistance.

The power system units can be considered by setting the parameters "power_station_unit" and "oltc" of the transformer (:code:`net.trafo`) and "power_station_trafo" of generator (:code:`net.gen`).

Note that the short-circuit currents are calculated in the complex domain, which allows obtaining the branch-related values for current magnitude and phase angle. In the case of transformers with rated voltage values unequal to the bus rated voltage values, only the current results are available because the voltage results are not valid in this case. For such configurations, only the superposition method can be used to obtain voltage, active and reactive power results (see below).

**Ad 2)** The superposition method (per IEC 60909) considers the pre-fault voltage vector. This method is activated by passing the argument "use_pre_fault_voltage" as True to the "calc_sc" function call. The superposition method is only inplemented for three-phase symmetric fault (fault="3ph").

Note that the user needs to explicitly carry out the power flow calculation (i.e. by invoking the "runpp" function call) on the network, before proceeding to the short-circuit calculation with the superposition method. We rely on the user to execute this step explicitly so that the user is fully aware that a power flow calculation is executed, and also has control over all the relevant options for the power flow calculation. Results provide branch currents magnitude and angle, along with active and reactive power flows, and bus voltages magnitude and angle values.

In the case of the superposition method, the values of shunt impedance of inverter-based generators and loads are considered, which are calculated based on their pre-fault current values. The following differences to the worst-case scenario calculation apply:

- transformer correction factor is not applied,
- load, sgen are additionally modelled as shunt admittances (calculated based on their pre-fault currents),
- the grid must contain the results of a successful power flow calculation.

The results for all elements and different short-circuit currents have been tested against commercial software to ensure that correction factors are correctly applied.

Both methods allow following short-circuit currents to be calculated:

* ikss (i.e. Initial symmetrical short-circuit current),
* ip (i.e. Peak short-circuit current),
* ith (i.e. Equivalent thermal short-circuit current),

either as:

- symmetrical three-phase or
- asymmetrical two-phase

short-circuit currents. Currents "ip" and "ith" are only implemented for short-circuits far from the synchronous generators.

.. toctree::
    :maxdepth: 1

    ikss
    ip
    ith

Kindly follow the tutorial on the basics of the Short-Circuit Analysis for more details:
https://github.com/e2nIEE/pandapower/blob/develop/tutorials/shortcircuit/shortcircuit.ipynb


**Results**

The results of the short-circuit calculations are stored in the dedicated results tables:

:code:`net.res_bus_sc` for the results for fault buses, :code:`net.res_line_sc` for line results, :code:`net.res_trafo_sc` and :code:`net.res_trafo3w_sc` for transformer results.
The branch results include, in addition to the short-circuit current, the voltage magnitude and angle at the connected buses, and the active and reactive power flowing in and out of the branch.
Only the currents are shown for three-winding transformers at the moment.

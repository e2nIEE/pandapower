Balanced AC Power Flow
=======================

pandapower uses PYPOWER to solve the power flow problem:

.. image:: /pics/flowcharts/pandapower_power flow.png
        :width: 40em
        :alt: alternate Text
        :align: center

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in
    :code:`net["_ppc"]`. However all necessary information is written into the pandapower format net, so the pandapower
    user should not usually have to deal with pypower.

Possible Solver Algorithm
-----------------------------

Pandapower brings a variety of powerflow algorithm, some through the pypower core, other via external libraries:

- "nr" Newton-Raphson (pypower implementation with numba accelerations)
- "iwamoto_nr" Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust)
- "bfsw" backward/forward sweep (specially suited for radial and weakly-meshed networks)
- "gs" gauss-seidel (pypower implementation)
- "fdbx" fast-decoupled (pypower implementation)
- "fdxb" fast-decoupled (pypower implementation)
- "helm" holomorphic embedded loadflow method (Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com)

An algorithm might be better suited for a peculiar use case. The user can select an algorithm via :code:`runpp(algorithm="nr", ...)`
If the algorithm option is not given, "nr" is selected as a default. Please note, not all features / power system elements
are equally supported in all solvers. The most feature complete at the moment is "nr".

Accelerating Packages
-------------------------

Two external packages are available which let accelerate pandapower's power flow command :code:`runppp`:

1. numba
2. lightsim2grid

If available, i.e. installed on the operating computer, the code will check by default all prerequisites to use the
external packages. numba is a python JIT compiler, cf. `link <https://pypi.org/project/numba/>`__. In contrast, the
library lightsim2grid is used as a backend for power flow simulation instead of the implementation in pandapower. It
leads to a boost in performance. The library lightsim2grid is implemented in C++ and can either be installed with
:code:`pip install lightsim2grid`, or built from source. More about the library and the installation guide can be found
in the `documentation <https://lightsim2grid.readthedocs.io/en/latest/>`__ or its GitHub
`repository <https://github.com/BDonnot/lightsim2grid>`__.

lightsim2grid Compatibility
```````````````````````````````

lightsim2grid is supported if all the following conditions are met:

1. The lightsim2grid library is installed and available.
2. The selected power flow algorithm is Newton-Raphson (algorithm='nr').
3. Voltage-dependent loads are not enabled (voltage_depend_loads=False).

    .. attention::
        | Voltage-dependent loads have been overhauled and are now used more frequently without explicitly setting the
          required columns. Due to this change setting :code:`voltage_depend_loads=False` manually is required more
          frequently to use lightsim2grid.
        | With :code:`lightsim2grid=True` you will get an error if voltage-dependent loads have not been disabled.
        | With :code:`lightsim2grid="auto"` you will not be warned when it is disabled due to voltage-dependent loads.

4. Either:

    * There is only one slack bus in the network, or
    * Distributed slack is enabled (distributed_slack=True).

5. None of the following elements are present in the grid model:

    * Controllable shunts, including SVC, SSC, or VSC elements.
    * Controllable impedances, such as TCSC elements.
    * DC elements, including DC buses (bus_dc) or DC lines (line_dc).

6. Temperature-Dependent Power Flow is not requested (tdpf=False).

When lightsim2grid is Not Supported
```````````````````````````````````````

If any of the above conditions are not met, lightsim2grid cannot be used. In such cases:

* If lightsim2grid='auto' (default), the fallback to the standard pandapower implementation occurs without a detailed message.
* If lightsim2grid=True is explicitly set, an appropriate error or warning is raised or logged, depending on the condition.

Common Limitations of lightsim2grid
````````````````````````````````````````

lightsim2grid does not currently support:

* Algorithms other than Newton-Raphson
* Voltage-dependent loads
* Multiple slack buses without distributed slack
* Grids containing any of the following advanced elements:

    * Controllable shunts (SVC, SSC, VSC)
    * Controllable impedances (TCSC)
    * DC buses or DC lines

* Temperature-Dependent Power Flow (tdpf=True)

Temperature-Dependent Power Flow (TDPF)
---------------------------------------

pandapower supports Temperature Dependent Power Flow (TDPF) with consideration of thermal inertia.
TDPF is implemented based on the following publications:

* S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-dependent power flow", IEEE Transactions on Power Systems, vol.
  28, no. 4, pp. 4007-4018, Nov 2013.
* B. Ngoko, H. Sugihara and T. Funaki, "A Temperature Dependent Power Flow Model Considering Overhead Transmission Line
  Conductor Thermal Inertia Characteristics," 2019 IEEE International Conference on Environment and Electrical
  Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC / I&CPS Europe), 2019, pp. 1-6, doi:
  10.1109/EEEIC.2019.8783234.

Additional parameters in net.line are required. If missing, common assumptions are used to add the parameters to
:code:`net.line`.
The column "tdpf" (bool) must be provided in :code:`net.line` to designate which lines are relevant for TDPF.
The parameter "outer_diameter_m" (float) must be provided if the weather model is used (runpp parameter
tdpf_update_r_theta=True).
Otherwise, the parameter "r_theta_kelvin_per_mw" (float) must be specified. It can be calculated using the function
"pandapower.pf.create_jacobian_tdpf.calc_r_theta_from_t_rise" For consideration of thermal inertia, pp.runpp parameter
"tdpf_delay_s" specifies the time after a step change of current. The parameter "mc_joule_per_m_k" describes the mass *
thermal capacity of the conductor per unit length and it must be provided in :code:`net.line`.

More detailed information about TDPF can be found in the tutorials:

- `TDPF introduction <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/temperature_dependent_power_flow.ipynb>`__
- `Sensitivity of TDPF parameters <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/temperature_dependent_power_flow_parameters.ipynb>`__
- `Transformer temperatures <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/temperature_dependent_power_flow_trafo.ipynb>`__
- `Switch temperatures <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/temperature_dependent_power_flow_switchgear.ipynb>`__

.. _tdpf_transformer_temperatures:

Transformer temperatures
````````````````````````

In addition to the conductor temperature of the lines, TDPF reports the top-oil and hot-spot temperature
of every in-service transformer (:code:`net.trafo` and :code:`net.trafo3w`). The calculation is based on
the IEC 60076-7 loading guide and utilizes the python package
`transformer-thermal-model <https://github.com/alliander-opensource/transformer-thermal-model>`__ developed by Alliander.

This is an optional feature and requires an optional dependency. Install it with:

.. code-block:: bash

    pip install pandapower[tdpf]

If the package is not installed, TDPF logs a warning and skips transformer temperature calculations.
Line TDPF continues to work as before. Transformer temperature result columns are not created,
so line-only TDPF users do not see columns filled with NaN values.

Two additional input columns are read from :code:`net.trafo` and :code:`net.trafo3w`:

.. tabularcolumns:: |p{0.30\linewidth}|p{0.15\linewidth}|p{0.45\linewidth}|
.. csv-table::
    :header: "Column", "Type", "Description"
    :widths: 30, 15, 45

    ``cooling_type``,string,"cooling type of the transformer, either ONAN or ONAF (default: ONAN)"
    ``air_temperature_degree_celsius``,float,"ambient air temperature in °C (default: 35)"

The temperature is *reported*, not fed back into the power flow. In contrast to the line TDPF, the
transformer resistance :code:`r_sc` is never modified, because the thermal model runs as post-processing
on the solved transformer loading. The results are therefore not iterated with the power flow and do not
change voltages, flows or losses.

The results are written to additional columns of :code:`net.res_trafo` and :code:`net.res_trafo3w`:

.. tabularcolumns:: |p{0.35\linewidth}|p{0.30\linewidth}|p{0.35\linewidth}|
.. csv-table::
    :header: "Column", "Table", "Description"
    :widths: 35, 30, 35

    ``top_oil_temperature_degree_celsius``,"res_trafo, res_trafo3w","top-oil temperature in °C"
    ``hot_spot_temperature_degree_celsius``,"res_trafo","hot-spot temperature in °C"
    ``hot_spot_temperature_degree_celsius_hv``,"res_trafo3w","hot-spot temperature of the HV winding in °C"
    ``hot_spot_temperature_degree_celsius_mv``,"res_trafo3w","hot-spot temperature of the MV winding in °C"
    ``hot_spot_temperature_degree_celsius_lv``,"res_trafo3w","hot-spot temperature of the LV winding in °C"
    ``temperature_degree_celsius``,"res_trafo, res_trafo3w","hottest hot-spot temperature in °C, used by the contingency limit checks"

For a three winding transformer, :code:`temperature_degree_celsius` is the maximum of the three winding
hot-spot temperatures, mirroring how :code:`loading_percent` aggregates the windings.

Whether the model is solved for the steady state or for a transient depends on the :code:`tdpf_delay_s`
power flow parameter, the same parameter that controls the thermal inertia of the lines:

* :code:`tdpf_delay_s=None` (the default) gives the steady state temperature for the current loading.
* A finite :code:`tdpf_delay_s` gives the temperature after this time step, starting from a cold
  transformer. In a time series run (:code:`pandapower.timeseries.run_timeseries`), the top-oil temperature
  of the previous time step is used as the initial condition, so the transformer warms up and cools down
  continuously over the simulation.

.. note::

    The ONAN and ONAF cooling types share the same steady state temperature. They only differ in the
    transient behaviour: an ONAF transformer has shorter oil and winding time constants and therefore
    reaches its final temperature faster, not at a lower temperature.

.. note::

    Only the nameplate data that pandapower already stores is passed to the thermal model: the no-load
    loss, the load loss, the rated apparent power and the rated voltages of each winding, which yield
    the nominal currents. The optional specifications of the loading guide -- above all
    :code:`winding_oil_gradient`, :code:`hot_spot_fac` and the winding and oil time constants -- are
    left at the IEC 60076-7 defaults of the library (17 K, 1.3 and the values for the selected cooling
    type).

    The top-oil temperature depends only on the nameplate data and therefore matches a calculation with
    modified optional specifications exactly. The hot-spot temperature does depend on them, so it will
    differ from a calculation that uses a transformer-specific winding-oil gradient or hot-spot factor.
    A wider set of input columns is not part of this implementation.

.. _tdpf_switchgear_temperatures:

Switch temperatures
```````````````````

TDPF also reports the temperature of every current carrying switch (:code:`net.switch`). The
calculation uses the first order thermal model of IEC 62271-306 and the python package
`switchgear-thermal-model <https://github.com/alliander-opensource/switchgear-thermal-model>`__
developed by Alliander, the same organisation that maintains the transformer thermal model.

This is an optional feature and requires an optional dependency. Install it with:

.. code-block:: bash

    pip install pandapower[tdpf]

The switch thermal model currently requires Python 3.12 or newer. If the package is not installed,
TDPF logs a warning and skips switch temperature calculations. Line and transformer TDPF continue to
work as before, and the switch temperature result column is not created, so users who do not model
switches do not see a column filled with NaN values.

Unlike a transformer, a switch has no nameplate data from which the thermal model could be derived.
Four additional input columns are therefore read from :code:`net.switch`:

.. tabularcolumns:: |p{0.38\linewidth}|p{0.12\linewidth}|p{0.40\linewidth}|
.. csv-table::
    :header: "Column", "Type", "Description"
    :widths: 38, 12, 40

    ``measured_temperature_rise_kelvin``,float,"temperature rise above ambient at rated current (default: 75)"
    ``temperature_rise_exponent``,float,"exponent that scales the temperature rise with the current (default: 1.8)"
    ``thermal_time_constant_minutes``,float,"thermal time constant of the switch in minutes (default: 90)"
    ``air_temperature_degree_celsius``,float,"ambient temperature in °C (default: 35)"

The rated current :code:`rated_current` of the model is the existing :code:`in_ka` column, converted to
ampere. These four values are assumptions when they are missing, so pandapower logs which defaults it
filled in.

The temperature is *reported*, not fed back into the power flow. The switch thermal model runs as
post-processing on the solved switch current, so it does not change voltages, flows or losses.

The results are written to an additional column of :code:`net.res_switch`:

.. tabularcolumns:: |p{0.35\linewidth}|p{0.30\linewidth}|p{0.35\linewidth}|
.. csv-table::
    :header: "Column", "Table", "Description"
    :widths: 35, 30, 35

    ``temperature_degree_celsius``,"res_switch","switch temperature in °C"

Only switches with a closed state, a finite :code:`in_ka` greater than zero and a finite solved current
in :code:`res_switch.i_ka` are calculated. All other switches stay :code:`NaN`. In particular, a fused
bus-bus switch (:code:`z_ohm=0`) has no branch and therefore no current, and a bus-trafo3w switch
(:code:`et="t3"`) is not supported. They are skipped with a single aggregated log message.
:code:`compute_switch_flows` is not called implicitly. If the current of a fused switch is needed, call
:code:`pandapower.toolbox.compute_switch_flows(net)` after the load flow and run the thermal
calculation again.

Whether the model is solved for the steady state or for a transient depends on the
:code:`tdpf_delay_s` power flow parameter, as for the lines and transformers:

* :code:`tdpf_delay_s=None` (the default) gives the steady state temperature for the present current.
* A finite :code:`tdpf_delay_s` gives the temperature after this time step, starting from ambient. In a
  time series run, the switch temperature of the previous time step is used as the initial condition.

.. autofunction:: pandapower.run.runpp

.. note::

    `enforce_q_lims` optional parameter now considers the reactive power limits of sgen elements too!

.. _pgmpowerflow:

Balanced AC Power Flow using power-grid-model
---------------------------------------------

AC power flow can also be computed using the power-grid-model library.
Power-grid-model has a C++ core which leads to a higher performance (
`pgm documentation <https://power-grid-model.readthedocs.io/en/stable/>`__)
The power-grid-model and conversion library is required for running this function.
They can be installed using `pip install power-grid-model-io`.
Currently the conversion does not support the elements of Generator, DC lines, SVC, TCSC, extended ward and impedance
elements in the net. Check
`pgm conversion documentation <https://power-grid-model-io.readthedocs.io/en/stable/converters/pandapower_converter.html>`__
for details on the conversion and attribute support.

.. autofunction:: pandapower.run.runpp_pgm

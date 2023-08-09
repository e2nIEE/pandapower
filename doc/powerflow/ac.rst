Balanced AC Power Flow
=======================

pandapower uses PYPOWER to solve the power flow problem:

.. image:: /pics/flowcharts/pandapower_power flow.png
		:width: 40em
		:alt: alternate Text
		:align: center

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.

If available, the librabry lightsim2grid is used as a backend for power flow simulation instead of the
implementation in pandapower, leading to a boost in performance. The library lightsim2grid is implemented in C++ and
can either be installed with pip install lightsim2grid, or built from source. More about the library and the
installation guide can be found in the `documentation <https://lightsim2grid.readthedocs.io/en/latest/>`_ or
its GitHub `repository <https://github.com/BDonnot/lightsim2grid>`_.

Temperature-Dependent Power Flow (TDPF)
---------------------------------------

pandapower supports Temperature Dependent Power Flow (TDPF) with consideration of thermal inertia.
TDPF is implemented based on the following publications:

* S. Frank, J. Sexauer and S. Mohagheghi, "Temperature-dependent power flow", IEEE Transactions on Power Systems, vol. 28, no. 4, pp. 4007-4018, Nov 2013.
* B. Ngoko, H. Sugihara and T. Funaki, "A Temperature Dependent Power Flow Model Considering Overhead Transmission Line Conductor Thermal Inertia Characteristics," 2019 IEEE International Conference on Environment and Electrical Engineering and 2019 IEEE Industrial and Commercial Power Systems Europe (EEEIC / I&CPS Europe), 2019, pp. 1-6, doi: 10.1109/EEEIC.2019.8783234.

Additional parameters in net.line are required. If missing, common assumptions are used to add the parameters to net.line.
The column "tdpf" (bool) must be provided in net.line to designate which lines are relevant for TDPF.
The parameter "outer_diameter_m" (float) must be provided if the weather model is used (pp.runpp parameter tdpf_update_r_theta=True).
Otherwise, the parameter "r_theta_kelvin_per_mw" (float) must be specified. It can be calculated using the function "pandapower.pf.create_jacobian_tdpf.calc_r_theta_from_t_rise"
For consideration of thermal inertia, pp.runpp parameter "tdpf_delay_s" specifies the time after a step change of current.
The parameter "mc_joule_per_m_k" describes the mass * thermal capacity of the conductor per unit length and it must be provided in net.line.

More detailed information about TDPF can be found in the tutorials:

- `TDPF introduction <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/temperature_dependent_power_flow.ipynb>`_
- `Sensitivity of TDPF parameters <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/temperature_dependent_power_flow_parameters.ipynb>`_

.. autofunction:: pandapower.run.runpp


.. _pgmpowerflow:

Balanced AC Power Flow using power-grid-model
---------------------------------------------

AC power flow can also be computed using the power-grid-model library.
Power-grid-model has a C++ core which leads to a higher performance (`pgm documentation <https://power-grid-model.readthedocs.io/en/stable/>`_)
The power-grid-model and conversion library is required for running this function.
They can be installed using `pip install power-grid-model-io`.
Currently the conversion does not support the elements of Generator, DC lines, SVC, TCSC, extended ward and impedance elements in the net.
Check `pgm conversion documentation <https://power-grid-model-io.readthedocs.io/en/stable/converters/pandapower_converter.html>`_ for details on the conversion and attribute support.

.. autofunction:: pandapower.run.runpp_pgm
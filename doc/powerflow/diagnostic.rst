Diagnostic Function
====================

A power flow calculation on a pandapower network can fail to converge for a vast variety of reasons, which often makes debugging difficult, annoying and time consuming.
To help with that, the diagnostic function automatically checks pandapower networks for the most common issues leading to errors. It provides logging output and diagnoses with a controllable
level of detail.

.. autofunction:: pandapower.diagnostic

Usage ist very simple: Just call the function and pass the net you want to diagnose as an argument. Optionally you can specify if you want detailed logging output or summaries only and if the diagnostic should
log all checks performed vs. errors only.

Check functions
----------------

The diagnostic function includes the following checks:

- invalid values (e.g. negative element indices)
- check, if at least one external grid exists
- check, if there are buses with more than one gen and/or ext_grid
- overload: tries to run a power flow calculation with loads scaled down to 0.1 %
- switch_configuration: tries to run a power flow calculation with all switches closed
- inconsistent voltages: checks, if there are lines or switches that connect different voltage levels
- lines with impedance zero
- closed switches between in_service and out_of_service buses
- components whose nominal voltages differ from the nominal voltages of the buses they're connected to
- elements, that are disconnected from the network
- usage of wrong reference system for power values of loads and gens 

Logging Output
----------------

Here are a few examples of what logging output looks like:

**detailed_report = True/False**

Both reports show the same result, but on the left hand picture with detailed information, on the right hand picture summary only.

.. image:: /pics/diagnostic/diag_detailed_report.png
	:width: 42em
	:alt: alternate Text
	:align: center


*warnings_only = True/False*

.. image:: /pics/diagnostic/diag_warnings_only.png
	:width: 42em
	:alt: alternate Text
	:align: center

Result Dictionary
------------------

Additionally all check results are returned in a dict to allow simple access to the indeces of all element where
errors were found.

.. image:: /pics/diagnostic/diag_results_dict.png
	:width: 42em
	:alt: alternate Text
	:align: center

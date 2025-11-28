Diagnostic Function
====================

A power flow calculation on a pandapower network can fail to converge for a vast variety of reasons, which often makes
debugging difficult, annoying and time consuming.
To help with that, a range of diagnostic functions can automatically check pandapower networks for the most common
issues leading to errors.

For convienience a function is provided that creates the Diagnostic object and runs the diagnostic.

.. autofunction:: pandapower.diagnostic.diagnostic_helpers.diagnostic

Diagnostic Class
----------------

A class provides automated execution and a reporting functionality for ease of use.

.. autoclass:: pandapower.diagnostic.Diagnostic

.. autofunction:: pandapower.diagnostic.Diagnostic.register_function

.. autofunction:: pandapower.diagnostic.Diagnostic.diagnose_network

.. autofunction:: pandapower.diagnostic.Diagnostic.compact_report

.. autofunction:: pandapower.diagnostic.Diagnostic.detailed_report

.. autofunction:: pandapower.diagnostic.diagnostic.Diagnostic.report

Usage ist very simple: Just call the function and pass the net you want to diagnose as an argument.
Optionally you can specify if you want detailed logging output or summaries only and if the diagnostic should
log all checks performed vs. errors only.

Default Diagnostic functions
----------------------------

The DiagnosticFunction implementing classes included with pandapower.
These will be added to Diagnostic instances by default.

.. autoclass:: pandapower.diagnostic.diagnostic_functions.DeviationFromStdType
.. autoclass:: pandapower.diagnostic.diagnostic_functions.DifferentVoltageLevelsConnected
.. autoclass:: pandapower.diagnostic.diagnostic_functions.DisconnectedElements
.. autoclass:: pandapower.diagnostic.diagnostic_functions.ImplausibleImpedanceValues
.. autoclass:: pandapower.diagnostic.diagnostic_functions.InvalidValues
.. autoclass:: pandapower.diagnostic.diagnostic_functions.MissingBusIndices
.. autoclass:: pandapower.diagnostic.diagnostic_functions.MultipleVoltageControllingElementsPerBus
.. autoclass:: pandapower.diagnostic.diagnostic_functions.NoExtGrid
.. autoclass:: pandapower.diagnostic.diagnostic_functions.NominalVoltagesMismatch
.. autoclass:: pandapower.diagnostic.diagnostic_functions.NumbaComparison
.. autoclass:: pandapower.diagnostic.diagnostic_functions.OptimisticPowerflow
.. autoclass:: pandapower.diagnostic.diagnostic_functions.Overload
.. autoclass:: pandapower.diagnostic.diagnostic_functions.ParallelSwitches
.. autoclass:: pandapower.diagnostic.diagnostic_functions.SlackGenPlacement
.. autoclass:: pandapower.diagnostic.diagnostic_functions.SubNetProblemTest
.. autoclass:: pandapower.diagnostic.diagnostic_functions.TestContinuousBusIndices
.. autoclass:: pandapower.diagnostic.diagnostic_functions.WrongLineCapacitance
.. autoclass:: pandapower.diagnostic.diagnostic_functions.WrongReferenceSystem
.. autoclass:: pandapower.diagnostic.diagnostic_functions.WrongSwitchConfiguration

Logging Output
----------------

Here are a few examples of what logging output looks like:

**detailed_report = True/False**

Both reports show the same result, but on the left hand picture with detailed information, on the right hand picture
summary only.

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

Additionally all check results are returned in a dict to allow simple access to the indices of all element where
errors were found.

.. image:: /pics/diagnostic/diag_results_dict.png
    :width: 42em
    :alt: alternate Text
    :align: center

Custom Diagnostic Functions
---------------------------

Create a class that implements the DiagnosticFunction mete class.

.. autoclass:: pandapower.diagnostic.DiagnosticFunction

.. autofunction:: pandapower.diagnostic.DiagnosticFunction.diagnostic

.. autofunction:: pandapower.diagnostic.DiagnosticFunction.report

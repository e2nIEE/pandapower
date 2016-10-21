.. _std_types:

================================================
Standard Type Libraries
================================================

Lines and transformer have two different categories of parameters: parameter that depend on the specific element (like the length of a line or the bus to which a transformer is connected to etc.) and parameter that only 
depend on the type of line or transformer which is used (like the rated power of a transformer or the resistance per kilometer line).

The standard type library provides a database of different types for transformer and lines, so that you only have to chose a certain type and not 
define all parameters individually for each line or transformer. The standard types are saved in the network as a dictionary in the form of: ::

    net.std_types = {"line": {"standard_type": {"parameter": value, ...},..}, "trafo": {"standard_type": {"parameter": value, ...},..}, "trafo3w": {"standard_type": {"parameter": value, ...},..}}

The create_line and create_transformer functions use this database when you create a line or transformer with a certain standard type.
You can also use the standard type functions directly to create new types in the database, directly load type data, change types or check if a certain type exists.
You can also add additional type parameters which are not added to the pandas table by default (e.g. diameter of the conductor).

For a introduction on how to use the standard type library, see the interactive tutorial on standard types.

The pandapower standard types that every pandapower network is initialized as:



.. note ::
    The pandapower standard types are compatible with 50 Hz systems, please be aware that the standard type values might not be realistic for 60 Hz (or other) power systems.
    
Show all Available Standard Types
------------------

.. autofunction:: pandapower.available_std_types


Create Standard Type
------------------

.. autofunction:: pandapower.create_std_type

Copy Standard Types
------------------

.. autofunction:: pandapower.copy_std_types


Load Standard Types
------------------

.. autofunction:: pandapower.load_std_type


Check if Standard Type Exists
------------------

.. autofunction:: pandapower.std_type_exists

Change Standard Type
------------------

.. autofunction:: pandapower.change_std_type


Load Additional Parameter from Library
------------------

.. autofunction:: pandapower.parameter_from_std_type


Find Standard Type
-------------------

.. autofunction:: pandapower.find_std_type_by_parameter

Delete Standard Type
------------------

.. autofunction:: pandapower.delete_std_type



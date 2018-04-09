.. _std_types:

============================
Standard Type Libraries
============================

Lines and transformers have two different categories of parameters: parameter that depend on the specific element (like the length of a line or the bus to which a transformer is connected to etc.) and parameter that only 
depend on the type of line or transformer which is used (like the rated power of a transformer or the resistance per kilometer line).

The standard type library provides a database of different types for transformer and lines, so that you only have to chose a certain type and not 
define all parameters individually for each line or transformer. The standard types are saved in the network as a dictionary in the form of: ::

    net.std_types = {"line": {"standard_type": {"parameter": value, ...},..}, 
                    "trafo": {"standard_type": {"parameter": value, ...},..}, 
                    "trafo3w": {"standard_type": {"parameter": value, ...},..}}

The create_line and create_transformer functions use this database when you create a line or transformer with a certain standard type.
You can also use the standard type functions directly to create new types in the database, directly load type data, change types or check if a certain type exists.
You can also add additional type parameters which are not added to the pandas table by default (e.g. diameter of the conductor).

For a introduction on how to use the standard type library, see the interactive tutorial on standard types.

.. toctree:: 
    :maxdepth: 2
    
    std_types/basic
    std_types/manage

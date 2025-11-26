.. _gridequivalentmodule:

#############################
Grid Equivalent
#############################
The module is used to reduce an area of the observed pandapower grid. 

Due to limitations of computational resources, data confidentiality, and security-relevant reasons, studies and simulations based on fully detailed grid models are, in practice, not easily achieved. The common solution is using equivalent grids. As the following picture shows, a  power system can be divided into the **internal subsystem** and the **external subsystem**. The former, in which engineers are interested, remains unmodified, and a simple model represents the latter using grid equivalent methods. This module allows for the reduction of the user-defined **external subsystem** by (x)Ward- and REI-equivalent. 

.. image:: /pics/gridequivalent/schema_geq.png
	:width: 42em
	:alt: alternate Text
	:align: center

|

.. toctree::
    :maxdepth: 2

    gridequivalent/gridequivalent_overview
    gridequivalent/gridequivalent_example
    gridequivalent/run_function
    gridequivalent/tutorials
    



.. _controller:

###################################################
Predefined Controllers
###################################################


Basic Controller
==================

The basic controller is the base controller class that should be subclassed when implementing custom controllers.

.. autoclass:: pandapower.control.basic_controller.Controller
    :members:
	
ConstControl
==============
.. _ConstControl:
This controller is made for the use with the time series module to read data from a DataSource and write it to the net.
The controller can write the values either to a column of an element table (e.g. net.load.p_mw) or an attribute of another object that is
stored in an element table (e.g. another controller, net.controller.object). To change a controller attribute, the variable must be defined
in the format "object.attribute" (e.g. "object.set_vm_pu").

.. autoclass:: pandapower.control.controller.const_control.ConstControl
    :members:


TrafoController
===============

The following controllers to control tap changers are predefined within the pandapower control module.

**********************
Continuous Tap Control
**********************
.. autoclass:: pandapower.control.controller.trafo.ContinuousTapControl.ContinuousTapControl
    :members:
    
-------------

**********************
Discrete Tap Control
**********************

.. autoclass:: pandapower.control.controller.trafo.DiscreteTapControl.DiscreteTapControl
    :members:
    

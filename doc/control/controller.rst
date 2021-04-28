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
    
CharacteristicControl
===============

The following controllers that use characteristics are predefined within the pandapower control module.

**********************
CharacteristicControl
**********************
.. autoclass:: pandapower.control.controller.characteristic_control.CharacteristicControl
    :members:

-------------

**********************
USetTapControl
**********************

.. autoclass:: pandapower.control.controller.trafo.USetTapControl.USetTapControl
    :members:
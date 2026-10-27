.. _controller:

###################################################
Predefined Controllers
###################################################


Basic Controller
==================

The basic controller is the base controller class that should be subclassed when implementing custom controllers.

.. autoclass:: pandapower.control.basic_controller.Controller
    :members:
    :inherited-members:
    :show-inheritance:

ConstControl
==============
.. _ConstControl:

This controller is made for the use with the time series module to read data from a DataSource and write it to the net.
The controller can write the values either to a column of an element table (e.g. net.load.p_mw) or an attribute of another object that is
stored in an element table (e.g. another controller, net.controller.object). To change a controller attribute, the variable must be defined
in the format "object.attribute" (e.g. "object.set_vm_pu").
Note that ConstControl writes values to net in time_step, in order to set the values of the time step before the initial power flow.
If ConstControl is used without a data_source, it will reset the controlled values to the initial values, preserving the initial net state.

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


DER Controller
===============
.. autoclass:: pandapower.control.controller.DERController.der_control.DERController
    :members:


CharacteristicControl
=====================

The following controllers that use characteristics are predefined within the pandapower control module.

**********************
CharacteristicControl
**********************
.. autoclass:: pandapower.control.controller.characteristic_control.CharacteristicControl
    :members:

-------------

**********************
VmSetTapControl
**********************

.. autoclass:: pandapower.control.controller.trafo.VmSetTapControl.VmSetTapControl
    :members:

-------------

**********************
TapDependentImpedance
**********************

.. autoclass:: pandapower.control.controller.trafo.TapDependentImpedance.TapDependentImpedance
    :members:


Characteristic
==============

The following classes enable the definition of characteristics for the controllers.

***************
Characteristic
***************

.. autoclass:: pandapower.control.util.characteristic.Characteristic
    :members:

-------------

********************
SplineCharacteristic
********************

.. autoclass:: pandapower.control.util.characteristic.SplineCharacteristic
    :members:


Station Controller
==================
The following controllers are used for the representation of station controllers as used in PowerFactory. The Vdroop is
a new controller class used for the local droop voltage control.

**********************
Binary Search Control
**********************

.. autoclass:: pandapower.control.controller.station_control.BinarySearchControl
    :members:

***************
Droop Control
***************

.. autoclass:: pandapower.control.controller.station_control.DroopControl
    :members:

**********************
Local VDroop Control
**********************

.. autoclass:: pandapower.control.controller.station_control.VDroopControl_local
    :members:

Shunt Controller
==================

**********************
Discrete Shunt Control
**********************

.. autoclass:: pandapower.control.controller.shunt_control.DiscreteShuntController
    :members:

Storage Controllers
===================

``StorageController`` models a battery: it clips the requested grid-side power to the power limits and the remaining
energy and tracks the state of charge. ``SelfConsumptionController`` is a rule-based battery that charges from the
surplus and discharges to cover the deficit of a site of loads and static generators on the storage bus. Its rule is
instantaneous; it uses no forecasts and is not an optimal dispatch.

Both controllers work in time series simulation. The state of charge is committed once per time step, after the
output writer and only if the power flow converged, so the OutputWriter logs the state of charge at the start of each
step. Each run starts from the initial state of charge, and ``runpp`` or ``run_control`` outside a time series does not
change it. Power the battery cannot realise is reported as ``energy_not_charged_mwh`` and
``energy_not_discharged_mwh`` (grid-side MWh per step, with run totals ``*_total_mwh``); for a
``SelfConsumptionController`` this energy is exchanged with the grid.

.. autoclass:: pandapower.control.controller.storage_control.StorageController
    :members:

.. autoclass:: pandapower.control.controller.self_consumption_control.SelfConsumptionController
    :members:

Usage example, with the built-in Kerber LV network. The profile values are illustrative:

.. code-block:: python

    import pandas as pd

    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.control import ConstControl, SelfConsumptionController
    from pandapower.timeseries import DFData, OutputWriter, run_timeseries

    net = pn.create_kerber_vorstadtnetz_kabel_1()
    load = 0
    bus = net.load.at[load, "bus"]
    pv = pp.create_sgen(net, bus, p_mw=0.0)
    battery = pp.create_storage(net, bus, p_mw=0.0, max_e_mwh=0.01, soc_percent=50.0, min_e_mwh=0.001)

    # illustrative hourly values in MW: household load and PV feed-in, 24 steps of one hour
    profiles = pd.DataFrame(
        {
            "load": [0.0015] * 6
            + [0.0020, 0.0030, 0.0025, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020]
            + [0.0025, 0.0030, 0.0035, 0.0035, 0.0030, 0.0025, 0.0020, 0.0015, 0.0015],
            "pv": [0.0] * 7
            + [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
            + [0.0] * 5,
        },
        index=range(24),
    )
    data_source = DFData(profiles)
    ConstControl(net, "load", "p_mw", element_index=load, data_source=data_source, profile_name="load")
    ConstControl(net, "sgen", "p_mw", element_index=pv, data_source=data_source, profile_name="pv")

    # the battery charges from the PV surplus and covers the load deficit
    ctrl = SelfConsumptionController(
        net,
        battery,
        max_charge_mw=0.003,
        max_discharge_mw=0.003,
        eta_charge=0.95,
        eta_discharge=0.9,
        delta_t_h=1.0,
        load_indices=[load],
        sgen_indices=[pv],
    )

    time_steps = range(24)
    ow = OutputWriter(net, time_steps, output_path=None, output_file_type=".json")
    ow.log_variable("storage", "p_mw")  # power per step in MW, positive = charging
    ow.log_variable("storage", "soc_percent")  # state of charge at the start of each step in %
    run_timeseries(net, time_steps, verbose=False)

    print("power [MW]:", ow.output["storage.p_mw"][0].values.round(4))
    print("soc at start of each step [%]:", ow.output["storage.soc_percent"][0].values.round(1))
    print("final soc [%]:", round(float(net.storage.at[battery, "soc_percent"]), 1))

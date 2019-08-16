#################
The Control loop
#################

Elements of a network which contain a control loop can be modelled as so called Controller.
They implement a controlling mechanism as well as a convergence check and are being registered at
the used network ``net.controller``. When a simulation is being invoked, the ``run_control()`` method iteratively
calls the controlling-method ``control_step()`` on each initialized controller until all of them are converged. The possibility to operate
different controllers in a specific sequence is given by the variables ``net.controller.order`` and ``net.controller.level``, which 
we will go into later. This ``control_step()`` can be calculated for a single point in time as well as a simulation of discrete consecutive points in time using the **timeseries** module 
method ``run_timeseries()``. Following picture describes the dependence of both methods.

.. image:: /pics/control/run_timeseries_reglerstrecke.svg
    :width: 400 px
    :align: center


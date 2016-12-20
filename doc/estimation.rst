############################
State Estimation
############################

The module provides a state estimation for pandapower networks.

Theoretical Background
===========================
State Estimation is a process to estimate the electrical state of a network by eliminating inaccuracies and errors from measurement data. Various measurements are placed around the network and transferred to the operational control center via SCADA. Unfortunately measurements are not perfect: There are tolerances for each measurement device, which lead to an inherent inaccuracy in the measurement value. Analog transmission of data can change the measurement values through noise. Faulty devices can return completely wrong measurement values. To account for the measurement errors, the state estimation processes all available measurements and uses a regression method to identify the likely real state of the electrical network. 
The **output** of the state estimator is therefore a **set of voltage absolutes and voltage angles** for all buses in the grid. The **input** is the **network** in pandapower format and a number of **measurements**.

Amount of Measurements
-----------------------

There is a minimum amount of required measurements necessary for the regression to be mathematically possible.
Assuming the network contains :math:`n` buses, the network is then described by :math:`2n` variables, namely :math:`n` voltage absolute values and :math:`n` voltage angles.
A slack bus serves as the reference, its voltage angle is set to zero or the value provided in the corresponding *net.ext_grid.va_degree* entry (see *init* parameter) and is not altered in the estimation process.
The voltage angles of the other network buses are relative to the voltage angles of the connected slack bus. The state estimation therefore has to find :math:`2n-k` variables, where :math:`k` is the number of defined slack buses.
The minimum amount of measurements :math:`m_{min}` needed for the method to work is therefore:

:math:`m_{min} = 2n-k`

To perform well however, the number of redundant measurements should be higher. A value of :math:`m \approx 4n` is often considered reasonable for practical purposes.

Standard Deviation
---------------------
Since each measurement device may have a different tolerance and a different path length it has to travel to the control center, the accuracy of each measurement can be different.
Therefore each measurement is assigned an accuracy value in the form of a standard deviation. Typical measurement errors are 1 % for voltage measurements and 1-3 % for power measurements.

For a more in-depth explanation of the internals of the state estimation method, please see the following sources:  

.. seealso::
	- *Power System State Estimation: Theory and Implementation* by Ali Abur, Antonio Gómez Expósito, CRC Press, 2004.   
	- *State Estimation in Electric Power Systems - A Generalized Approach* by A. Monticelli, Springer, 1999.  


Defining Measurements
===========================

Measurements are defined via the pandapower *"create_measurement"* function. There are different physical properties, which can be measured at different elements. The following lists and table clarify the possible combinations. Bus power injection measurements are given in the producer system. Generated power is positive, consumed power is negative.   

**Types of Measurements**

 - *"v"* for voltage measurements (in per-unit)
 - *"p"* for active power measurements (in kW)
 - *"q"* for reactive power measurements (in kVar)
 - *"i"* for electrical current measurements at a line (in A)
 

**Element Types**

 - *"bus"* for bus measurements
 - *"line"* for line measurements
 - *"transformer"* for transformer measurements 

   
**Available Measurements per Element**

+---------------+------------------------------+
| Element Type  |  Available Measurement Types |
+===============+==============================+
| bus           | v, p, q                      |
+---------------+------------------------------+
| line          | i, p, q                      |
+---------------+------------------------------+
| transformer   | i, p, q                      |
+---------------+------------------------------+

The *"create_measurement"* function is defined as follows:

.. autofunction:: pandapower.create.create_measurement

Running the State Estimation
=============================

The state estimation can be used with the wrapper function *"estimate"*, which prevents the need to deal with the state_estimation class object and functions. It can be imported from *"estimation.state_estimation"*.

.. autofunction:: pandapower.estimation.estimate
 
Example
=============================

As an example, we will define measurements for a simple pandapower network *net* with 4 buses. Bus 4 is out-of-service. The external grid is connected at bus 1.

There are multiple measurements available, which have to be defined for the state estimator. There are two voltage measurements at buses 1 and 2. There are two power measurements (active and reactive power) at bus 2. There are also line power measurements at bus 1. The measurements are both for active and reactive power and are located on the line from bus 1 to bus 2 and from bus 1 to bus 3. This yields the following code: 

:: 

	pp.create_measurement(net, "v", "bus", 1.006, .004, bus1)      # V at bus 1
	pp.create_measurement(net, "v", "bus", 0.968, .004, bus2)      # V at bus 2

	pp.create_measurement(net, "p", "bus", -501, 10, bus2)         # P at bus 2
	pp.create_measurement(net, "q", "bus", -286, 10, bus2)         # Q at bus 2

	pp.create_measurement(net, "p", "line", 888, 8, bus=bus1, element=line1)    # Pline (bus 1 -> bus 2) at bus 1
	pp.create_measurement(net, "p", "line", 1173, 8, bus=bus1, element=line2)   # Pline (bus 1 -> bus 3) at bus 1
	pp.create_measurement(net, "q", "line", 568, 8, bus=bus1, element=line1)    # Qline (bus 1 -> bus 2) at bus 1
	pp.create_measurement(net, "q", "line", 663, 8, bus=bus1, element=line2)    # Qline (bus 1 -> bus 3) at bus 1

Now that the data is ready, the state_estimation can be initialized and run. We want to use the flat start condition, in which all voltages are set to 1.0 p.u..

:: 

	success = estimate(net, init="flat")
	V, delta = net.res_bus_est.vm_pu, net.res_bus_est.va_degree 


The resulting variables now contain the voltage absolute values in *V*, the voltage angles in *delta*, an indication of success in *success*.
The bus power injections can be accessed similarly with *net.res_bus_est.p_kw* and *net.res_bus_est.q_kvar*. Line data is also available in the same format as defined in *res_line*.

.. Class state_estimation
   ======================

.. commentout automodule:: pandapower.estimation.state_estimation
    :members:

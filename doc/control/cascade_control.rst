################
Cascade control
################
Each Controller has the variables **order** and **level** with the default value **0**. Both variables
enable the possibility to operate different controllers in a specific sequence. 
This operation is called a *cascade control*. Following flow chart explains the general usage.

.. image:: /pics/control/cascade_control.svg
        :align: left

+------------------+---------------+--------------+
| Controller       | Order         | Level        |
+==================+===============+==============+
| Controller A     | 1             | 1,2          |
+------------------+---------------+--------------+
| Controller B     | 2             | 2            |
+------------------+---------------+--------------+



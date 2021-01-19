==============
MV Oberrhein
==============

.. note::

    The MV Oberrhein network is a generic network assembled from openly available data supplemented with parameters based on experience.

.. autofunction:: pandapower.networks.mv_oberrhein()

The geographical representation of the network looks like this:

.. image:: /pics/plotting/plotting_tutorial1.png
	:width: 30em
	:align: center

The different colors of the MV/LV stations indicate the feeders which are galvanically seperated by open switches.
If you are interested in how to make plots such as these, check out the pandapower tutorial on plotting.

The power flow results of the network in the different worst case scenarios look like this:

.. image:: /pics/networks/oberrhein_loadcases.png
	:width: 40em
	:align: center
    
As you can see, the network is designed to comply with a voltage band of 0.975 < u < 1.03 and line loading of <60 % in the high
load case (for n-1 security) and <100% in the low load case.
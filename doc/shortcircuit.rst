############################
Short-Circuit *beta!*
############################

The shortcircuit module is used to calculate short-circuits according to DIN/IEC EN 60909.

.. warning:: The pandapower short circuit module is in beta stadium! It is neither fully functional nor fully tested, so please proceed with caution. If you have any question about the shortcircuit module or want to contribute to its development please contact leon.thurner@uni-kassel.de


The shortcircuit calculation currently considers the following elements with the correction factors as described within the norm:

  - line
  - trafo
  - gen
  - ext_grid
  - switch
  
It has not been tested with any other pandapower elements, such as trafo3w, xward, shunt etc. Block transformers are also not yet implemented, so that generators and transformers are always considered seperately.

The following currents can be calculated:

   - Ikss
   - Ith (does not yet work for generator)
   - ip (does not yet work for generators)


Additional to the standard load flow parameters, the following parameters have to be defined for the different elements:

   - short circuit end temperature for lines in net.line.endtemp_degree (only if minimal short-circuit current is calculated)
   - 

.. autofunction:: pandapower.shortcircuit.runsc



.. code:: python

	import shortcircuit as sc
	import networks as nw

	net = nw.mv_network("ring")
	sc.runsc(net, case="min", ip=True)
	print(net.res_bus.loc[1,5,10])

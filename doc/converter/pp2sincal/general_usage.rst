######################################
Usage of the pp2sincal converter
######################################
    
This section explains, which steps are needed to convert a pandapowerNet to a Sincal network.


========================
General Usage
========================
To convert a pandapowerNet three important steps are mainly needed:

1. Initialization
    In this step the function ``create_simulation_environment`` and ``initialize_net`` from the file `pp2sincal/initialization.py`
    are beeing executed. The function ``initialize`` represents this step.

.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.initialize

	
2. Conversion
    After the initialization, the actual conversion can be conducted.  The function ``convert_pandapower_net`` from `pp2sincal.py`
    converts each element from the pandapowerNet to the Sincal network model.

.. autofunction::pandapower.converter.sincal.pp2sincal.util.main.convert_pandapower_net


3. Finalization
    The last step ends the conversion and closes Database Objects, reloads the open Sincal Document and closes
    the Application Object. The function ``write_to_net`` and ``close_net`` from `pp2sincal/pp2sincal.py`
    contain these processes.
 
.. autofunction:: pandapower.converter.sincal.pp2sincal.util.main.finalize

========================
Minimal Example
========================
In this section we convert a simple pandapowerNet to Sincal using the three main steps.
The source code, with some extensions, is also available in an interactive *jupyter notebook* (*sincal/tutorials/minimalExample.ipynb*) file or in the *sincal/minimal_example.py*. You can execute the
code in the jupyter notebook or just open the example in your python interpreter. 
Launch jupyter notebook by navigating with an open anaconda prompt shell in the *sincal/tutorials* folder and execute "jupyter notebook". Select the tutorial in your browser and explore the code.


.. code-block:: python

  import os
  
  from pandapower.test import test_path
  from pandapower.converter.sincal.pp2sincal.util.main import initialize, convert_pandapower_net, finalize
  from pandapower.converter.sincal.pp2sincal.pp2sincal import pp2sincal
  from pandapower.plotting import simple_plot
  
  import pandapower as pp
  
  # create empty net
  net_pp = pp.create_empty_network()
  
  # create buses
  bus1 = pp.create_bus(net_pp, vn_kv=20., name="Bus 1")
  bus2 = pp.create_bus(net_pp, vn_kv=0.4, name="Bus 2")
  bus3 = pp.create_bus(net_pp, vn_kv=0.4, name="Bus 3")
  
  
  # create bus elements
  pp.create_ext_grid(net_pp, bus=bus1, vm_pu=1.02, name="Grid Connection")
  pp.create_load(net_pp, bus=bus3, p_mw=0.100, q_mvar=0.05, name="Load")
  
  # create branch elements
  trafo = pp.create_transformer(net_pp, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV",
                                name="Trafo")
  line = pp.create_line(net_pp, from_bus=bus2, to_bus=bus3, length_km=0.1, std_type="NAYY 4x50 SE",
                        name="Line")
  
  # Set Output path
  output_folder = os.path.join(test_path, 'results', 'minimal_example')
  file_name = 'minimal_example_wo_main' + '.sin'
  use_active_net = False
  use_ui = False
  sincal_interaction = False
  plotting = True
  delete_files = True
  
  
  # 1. Initialization
  net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui,
                                  sincal_interaction, delete_files)
  
  # 2. Conversion
  convert_pandapower_net(net, net_pp, doc, plotting)
  
  # 3. Finalization
  finalize(net, output_folder, file_name, app, sim, doc, sincal_interaction)
  
  file_name = 'minimal_example_with_main' + '.sin'
  
  # Plotting network| generate generic coordinates
  simple_plot(net_pp)
  
  # Drop generic coordinates from first Conversion
  net_pp.bus_geodata.drop(index=net_pp.bus.index, inplace=True)
  
  # pp2sincal method combines the three steps initialize, convert and finalize
  pp2sincal(net_pp, output_folder, file_name, use_active_net, plotting, sincal_interaction)

  
The converted network from this minimal example will be saved as a .sin file in the test_path `pandapower/test/converter/sincal/pp2sincal/results/minimal_example/`.
It should look like this:

.. image:: /pics/converter/pp_to_sincal/minimal_example.png
		:width: 750em
		:align: left 

.. |br| raw:: html

    <br />

|br|    
|br|    
|br|
  
  
  
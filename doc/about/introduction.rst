=====================
A Short Introduction
=====================

pandapower combines the data analysis library `pandas <http://pandas.pydata.org/>`_ and the power flow solver `PYPOWER <https://pypi.python.org/pypi/PYPOWER>`_ to create an easy to use network calculation tool 
aimed at automation of analysis and optimization in power systems.

**Datastructure**

A network in pandapower is represented in a pandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a pandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

We consider the following simple 3-bus example network as a minimal example:

.. image:: /pics/3bus-system.png
		:width: 40em
		:alt: alternate Text
		:align: center 

To create this network in pandapower, we first create an empty network with three buses: ::
    
    import pandapower as pp
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

We then create the bus elements, namely a grid connection at Bus 1 and an load at Bus 3: ::

    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b3, p_kw=100, q_kvar=50, name="Load")
  
We now create the branch elements. First, we create the transformer from the type data as it is given in the network description: ::

    tid = pp.create_transformer_from_parameters(net, sn_kva=400.,
                                                hv_bus=b1, lv_bus=b2,  
                                                vn_hv_kv=20., vn_lv_kv=0.4,
                                                vsc_percent=6., vscr_percent=1.425,
                                                i0_percent=0.3375, pfe_kw=1.35,
                                                name="Trafo")

Note that you do not have to calculate any impedances or tap ratio for the equivalent circuit, this is handled internally by pandapower according to the pandapower `transformer model <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/elements/trafo.html#electric-model>`_.

The `standard type library <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/std_types.html>`_ allows even easier creation of the transformer. 
The parameters given above are the parameters of the transformer "0.4 MVA 20/0.4 kV" from the pandapower `basic standard types <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/std_types/basic.html>`_. The transformer can be created from the standard type library like this: ::

    tid = pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV",
                                name="Trafo")

The same applies to the line, which can either be created by parameters: ::

    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, 
                                   r_ohm_per_km=0.642, x_ohm_per_km=0.083,
                                   c_nf_per_km=210, imax_ka=0.142, name="Line")

or from the standard type library: ::    

    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=0.1, name="Line",
                   std_type="NAYY 4x50 SE")     

the pandapower representation now looks like this:

.. image:: /pics/pandapower_datastructure.png
		:width: 40em
		:alt: alternate Text
		:align: center

This is the version where transformer and line have been created through the standard type libraries, which is why the line has a specified type (cs for cable system) and the transformer has a tap changer, both of
which are defined in the `type data <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/std_types/basic.html>`_.

**Running a Power Flow**  

A powerflow can be carried out with the `runpp function <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/powerflow/ac.html>`_: ::
     
    pp.runpp(net)
    
When a power flow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the power flow.
The results are then processed and written back into pandapower:
        
.. image:: /pics/pandapower_power flow.png
		:width: 40em
		:alt: alternate Text
		:align: center

For the 3-bus example network, the result tables look like this:

.. image:: /pics/pandapower_results.png
		:width: 40em
		:alt: alternate Text
		:align: center
        
You can download the python script that creates this 3-bus system :download:`here  <pandapower_3bus_system.py>`.

For a more in depth introduction into pandapower modeling and analysis functionality, see the `pandapower tutorials <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/getting_started/tutorials.html>`_
about network creation, standard type libraries, power flow, topological searches, plotting and more.
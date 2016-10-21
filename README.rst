What is pandapower?
=====================

pandapower combines the data analysis library `pandas <http://pandas.pydata.org/>`_ and the power flow solver `PYPOWER <https://pypi.python.org/pypi/PYPOWER>`_ to create an easy to use network calculation program.
pandapower is aimed at automation of power system analysis and optimization in distribution and sub-transmission networks.

pandapower is based on electric elements rather than on generic loadflow attributes. For example, in PYPOWER buses have a power demand and shunt admittance, even though these are in reality the attributes of electric
elements (such as loads, pv generators or capacitor banks) which are connected to the buses. In pandapower, we model each electric bus element instead of considering summed values for each bus.
The same goes for branches: in reality, buses in a network are connected by electric elements like lines and transformers that can be defined by a length and cable type (lines) or short circuit 
voltages and rated power (transformers). Since the electric models for lines and transformers are implemented in pandapower, it is possible to model the electric elements with these common nameplate
attributes. All parameters which are necessary for the loadflow (like branch per unit impedances, shunt impedances, bus power, bus loadflow type etc.) are then calculated and handled internally by pandapower.

A network in pandapower is represented in a PandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a PandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

For the following simple 2-bus example network:

.. image:: /docs/pandapower/pics/2bus-system.png
		:width: 20em
		:alt: alternate Text
		:align: center 

the pandapower representation looks like this:

.. image:: /docs/pandapower/pics/pandapower_datastructure.png
		:width: 40em
		:alt: alternate Text
		:align: center

The network can be created with the pandapower create functions, but it also possible to directly manipulate data in the pandapower dataframes.

When a loadflow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the loadflow. The results are then processed and written back into pandapower:
        
.. image:: /docs/pandapower/pics/pandapower_loadflow.png
		:width: 40em
		:alt: alternate Text
		:align: center

For the 2-bus example network, the result tables look like this:

.. image:: /docs/pandapower/pics/pandapower_results.png
		:width: 40em
		:alt: alternate Text
		:align: center

       
Why pandapower?
=====================

There are various reasons why using pandapower is more comfortable than using pypower directly:
   
1. Electric Models
    - pandapower comes with static equivalent circuit models for lines, 2-Winding transformers, 3-Winding transformers, ward-equivalents etc.
    - Input parameters are intuitive and commonly used model plate parameters (such as line length and resistance per kilometer) instead of parameters like total branch resistance in per unit
    - the pandapower switch model allows modelling of ideal bus-bus switches as well as bus-line / bus-trafo switches
    - the loadflow results are processed to include not only the classic loadflow results (such as bus voltages and apparent power branch flows), but also line loading or transformer losses

2. pandapower API
    - the pandapower API provides create functions for each element to allow automized step-by-step construction of networks
    - the standard type library allows simplified creation of lines, 2-Winding transformers and 3-Winding transformers
    - networks can be saved and loaded to the hard drive with the pickle library

3. pandapower Datastructure
    - since variables of any datatype can be stored in the pandas dataframes, electric parameters (integer / float) can be stored together with names (strings), status variables (boolean) etc.
    - variables can be accessed by name instead of by column number of a matrix
    - since all information is stored in pandas tables, all inherent pandas methods can be used to
    
        - `access <http://pandas.pydata.org/pandas-docs/stable/indexing.html>`_,
        - `query <http://pandas.pydata.org/pandas-docs/stable/indexing.html#boolean-indexing>`_,
        - `statistically evaluate <http://pandas.pydata.org/pandas-docs/version/0.17.1/api.html#api-dataframe-stats>`_,
        - `iterate over <http://pandas.pydata.org/pandas-docs/stable/basics.html#iteration>`_,
        - `visualize <http://pandas.pydata.org/pandas-docs/stable/visualization.html>`_,
        -  etc.
        
      any information that is stored in the pandapower dataframes - be it element parameters, loadflow results or a combination of both.

4. Topological Searches
    - pandapower networks can be translated into `networkx <https://networkx.github.io/>` multigraphs for fast topological searches
    - all native `networkx algorithms <https://networkx.readthedocs.io/en/stable/reference/algorithms.html>`can be used to perform graph searches on pandapower networks
    - pandapower provides some search algorithms specialiced on electric power networks

5. Plotting and geographical data
    - geographical data for buses and lines can be stored in the pandapower datastructure
    - networks with geographic information can be plotted using matplotlib
    - if no geographical information is available for the buses, artificial coordinates can be created through a `python-igraph <http://igraph.org/python/>` interface
      
License
=========

.. highlight:: none

pandapower is licensed under the following 3-clause BSD-License: ::
    
    Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for
    Wind Energy and Power Systems Technology (IWES) Kassel and individual
    contributors (see AUTHORS file for details).
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
.. highlight:: python

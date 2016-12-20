================================
Advantages and Contributions
================================
 
1. Electric Models
    - pandapower comes with static equivalent circuit models for lines, 2-Winding transformers, 3-Winding transformers, ward-equivalents etc. (see :ref:`element documentation <elements>` for a complete list).
    - Input parameters are intuitive and commonly used model plate parameters (such as line length and resistance per kilometer) instead of parameters like total branch resistance in per unit
    - the pandapower :ref:`switch model <switch_model>` allows modelling of ideal bus-bus switches as well as bus-line / bus-trafo switches
    - the power flow results are processed to include not only the classic power flow results (such as bus voltages and apparent power branch flows), but also line loading or transformer losses

2. pandapower API
    - the pandapower API provides create functions for each element to allow automized step-by-step construction of networks
    - the :ref:`standard type library <std_types>` allows simplified creation of lines, 2-Winding transformers and 3-Winding transformers
    - networks can be saved and loaded to the hard drive with the pickle library

3. pandapower Datastructure
    - since variables of any datatype can be stored in the pandas dataframes, electric parameters (integer / float) can be stored together with names (strings), status variables (boolean) etc.
    - variables can be accessed by name instead of by column number of a matrix
    - since all information is stored in pandas tables, all inherent pandas methods can be used to
    
        - `access <http:/pandas.pydata.org/pandas-docs/stable/indexing.html>`_,
        - `query <http:/pandas.pydata.org/pandas-docs/stable/indexing.html#boolean-indexing>`_,
        - `statistically evaluate <http:/pandas.pydata.org/pandas-docs/version/0.17.1/api.html#api-dataframe-stats>`_,
        - `iterate over <http:/pandas.pydata.org/pandas-docs/stable/basics.html#iteration>`_,
        - `visualize <http:/pandas.pydata.org/pandas-docs/stable/visualization.html>`_,
        -  etc.
        
      any information that is stored in the pandapower dataframes - be it element parameters, power flow results or a combination of both.

4. Topological Searches
    - pandapower networks can be translated into `networkx <https:/networkx.github.io/>`_ multigraphs for fast topological searches
    - all native `networkx algorithms <https:/networkx.readthedocs.io/en/stable/reference/algorithms.html>`_ can be used to perform graph searches on pandapower networks
    - pandapower provides some search algorithms specialiced on electric power networks

5. Plotting and Geographical Data
    - geographical data for buses and lines can be stored in the pandapower datastructure
    - networks with geographic information can be plotted using matplotlib
    - if no geographical information is available for the buses, generic coordinates can be created through a `python-igraph <http:/igraph.org/python/>`_ interface

6. State Estimation
    - data structure to manage measurements for real-time simulations
    - WLS state estimation generates an exact grid state out of unexact measurements
    - WLS as the industry standard is a good reference for evaluating new state estimation developments
    - bad data detection and filtering methods improve performance of the state estimator (upcoming)

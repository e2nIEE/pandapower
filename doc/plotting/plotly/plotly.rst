#############################
Plotly Network Plots
#############################

pandapower provides interactive network plots using `Plotly <https://plot.ly/python/>`_.
These plots are built with arguments and functionalities to be as much as possible analogous with pandapower's
matlpotlib plotting library.
There is a functionality to translate pandapower network elements into plotly collections (traces).
The different collections for lines, buses or transformers can then be drawn.

In order to get idea about interactive plot features and possibilities see the `tutorial <http://nbviewer.jupyter.org/github/e2nIEE/pandapower/blob/develop/tutorials/plotly_built-in.ipynb>`_.

If a network has geocoordinates, there is a possibility to represent interactive plots on `MapLibre <https://maplibre.org/>`_ maps.

.. note::

    Previously Plotly library used `Mapbox <https://www.mapbox.com/>`_ services which required
    a Mapbox account and a `Mapbox Access Token <https://www.mapbox.com/studio>`_.
    Setting and getting the token is deprecated and will be removed in a future
    pandapower releases.

.. toctree::
    :maxdepth: 1

    built-in_plots
    create_traces
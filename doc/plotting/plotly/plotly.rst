#############################
Plotly Network Plots
#############################

pandapower provides interactive network plots using `Plotly <https://plotly.com/python/>`__. These plots are built with
arguments and functionalities to be as much as possible analogous with pandapower's matlpotlib plotting library.
There is a functionality to translate pandapower network elements into plotly collections (traces). The different
collections for lines, buses or transformers can then be drawn.

In order to get idea about interactive plot features and possibilities see the
`tutorial <https://nbviewer.org/github/e2nIEE/pandapower/blob/develop/tutorials/plotly_built-in.ipynb>`__.

If a network has geocoordinates, there is a possibility to represent interactive plots on
`MapLibre <https://maplibre.org/>`__ maps.

.. toctree::
    :maxdepth: 1

    built-in_plots
    create_traces

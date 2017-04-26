#############################
Plotly Network Plots
#############################

pandapower provides interactive network plots using `Plotly <https://plot.ly/python/>`_.
These plots are built with arguments and functionalities to be as much as possible analogous with pandapower's
matlpotlib plotting library.
There is a functionality to translate pandapower network elements into plotly collections (traces).
The different collections for lines, buses or transformers can than be drawn.

If a network has geocoordinates, there is a possibility to represent interactive plots on `Mapbox <https://www.mapbox.com/>`_ maps.

.. note::

    Plots on Mapbox maps are available only considering you have a Mapbox account and a `Mapbox Access Token <https://www.mapbox.com/studio>`_ which you can add to your pandapower.plotting settings.



.. toctree::
    :maxdepth: 1

    built-in_plots
    create_collections
    geo_data_to_latlong
.. _file_io:

========================
Save and Load Networks
========================

pandapower networks can be saved and loaded using the pickle library or with an excel file.

pickle painlessly stores all datatypes, which is why the network will be exactly the same after saving/loading a network with the pickle library.

Excel has the upside that it provides a human readable format.
However since Excel only accepts table-type inputs, some data mangling is necessary to save  and load pandapower network through excel.
Even though the relevant information is conserved, the process is not as robust as saving networks with pickle.

.. important::
    Always use the pickle format unless you need a human readable file as output!


pickle
-----------

.. autofunction:: pandapower.to_pickle

.. autofunction:: pandapower.from_pickle


Excel
-----------

.. autofunction:: pandapower.to_excel

.. autofunction:: pandapower.from_excel


Json
-----------

.. autofunction:: pandapower.to_json

.. autofunction:: pandapower.from_json
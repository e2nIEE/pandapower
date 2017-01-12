=================
Installation
=================

**Python**

pandapower is tested with Python 2.7, 3.4 and 3.5, but should probably (!) work with all python versions > 2.5.

We recommend the `Anaconda Distribution <https://www.continuum.io/downloads>`_, which already contains a lot of modules for scientific computing that are needed for working with pandapower.

In order to install Anaconda and open a python shell do the following:

1. Go to the `Anaconda Website <https://www.continuum.io/downloads>`_
2. Download Anaconda for your OS and install it
3. Open the command prompt by clicking **Start** and typing **Anaconda Prompt** on Windows systems

**pandapower**

pandapower itself can be installed with the following steps:

1. Download and unzip the `pandapower source distribution <http://www.uni-kassel.de/eecs/fachgebiete/e2n/software/pandapower.html>`_ to your local hard drive.

2. Open a command prompt (e.g. Windows cmd or Anaconda Command Prompt) and navigate to the folder that contains the setup.py file with the command cd <folder> ::

    cd %path_to_pandapower%\pandapower-x.x.x\

3. Install pandapower by running ::

    python setup.py install

.. image:: /pics/install.png
		:width: 40em
		:alt: alternate Text
		:align: center 

    This will install the following dependencies:
        - pypower>=5.0.1
        - pandas
        - networkx
        
4.  To use all of pandapowers functionalites, you will need the following additional packages:
 (for topological )
        - numba>=0.25.0 (for accelerated loadflow calculation)
        - matplotlib (for plotting)
        - python-igraph (for plotting networks without geographical information)
        - xlrd (for loading/saving files from/to excel)
        - openpyxl (for loading/saving files from/to excel)

.. note::
    All of these packages except pypower and python-igraph are already included in the anacaonda distribution. 
    
If you are having problems with the dependencies, you can also install these packages manually with pip install or any 
other way you like.

5. open a python shell or an editor (e.g. Spyder) and test if all pandapower submodules import without error: ::

        import pandapower
        import pandapower.networks
        import pandapower.topology
        import pandapower.plotting
        import pandapower.converter
        import pandapower.estimation

    Some submodules only import correctly with all optional dependencies named above installed.
    If you want to be really sure that everything works fine, you can run the pandapower test suite (pytest module is needed): ::
    
        import pandapower.test
        pandapower.test.run_all_tests()
    
  If everything is installed correctly, all tests should pass.    

=================
Installation
=================

**Python**

pandapower is tested with Python 2.7, 3.4 and 3.5, but should probably (!) work with all python versions > 2.5.

We recommend the anaconda distribution, which already contains a lot of modules for scientific computing that are needed for working with pandapower.

In order to install Anaconda and open a python shell do the following:

1. Go to the `Anaconda Website <https://www.continuum.io/downloads>`_
2. Download Anaconda for your OS and install it
3. Open the command prompt by clicking **Start** and typing **anaconda prompt** on Windows systems

**pandapower**

pandapower itself can be installed with the following steps:

1. Download and unzip the pandapower source distribution to your local hard drive.

2. Open a python shell (e.g. Anaconda prompt, see above) and navigate to the folder that contains the setup.py file with the command cd <folder>

3. Install pandapower by running ::

    python setup.py install
    
4. Install the dependencies needed for pandapower by running (in the same folder): ::

    pip install -r requirements.txt

   This will install the following packages (if not already installed):

        - pypower >= 5.0.1
        - pandas
        - numpy
        - scipy
        - networkx
        - numba >= 0.25.0
        - matplotlib
        - openpyxl
        - xlrd

.. note::
    All of these packages except pypower are included in the anacaonda distribution. 
    
If you are having problems with the dependencies, you can also install these packages manually with pip install or any 
other way you like.

5. open a python shell or a script and test if all pandapower submodules import without error: ::

        import pandapower
        import pandapower.networks
        import pandapower.topology
        import pandapower.plotting

  If you want to be really sure that everything works fine, you can run the pandapower test suite (pytest module is needed): ::
    
        import pandapower.test
        pandapower.test.run_all_tests()
    
  If everything is installed correctly, all tests should pass.    

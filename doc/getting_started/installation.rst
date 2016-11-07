=================
Installation
=================

**Python**

pandapower is tested with Python 2.7, 3.4 and 3.5, but should probably (!) work with all python versions > 2.5.

We recommend the anaconda distribution, which already contains a lot of modules for scientific computing that are needed for working with pandapower.

**pandapower**

pandapower itself can be installed with the following steps:

1. Download and unzip the pandapower source distribution to your local hard drive.

2. Open a python shell (e.g. Anaconda prompt) and navigate to the folder that contains the setup.py file

3. Install pandapower by running ::

    python setup.py install
    
4. Install the dependencies needed for pandapower by running (in the same folder): ::

    pip install -r requirements.txt

   This will install the following packages (if not already installed):

        - pypower >= 5.0.0
        - attrdict
        - numpy*
        - scipy*
        - networkx*
        - numba >=0.2.8*
        - matplotlib*
           
    If you are having problems with the dependencies, you can also install these packages manually with pip install or any 
    other way you like.

*included in anaconda

        

5. open a python shell or a script and test if all pandapower submodules import without error: ::

        import pandapower
        import pandapower.networks
        import pandapower.topology
        import pandapower.plotting

  If you want to be really sure that everything works fine, you can run the pandapower test suite (pytest module is needed): ::
    
        import pandapower.test
        pandapower.test.run_all_tests()
    
  If everything is installed correctly, all tests should pass.    

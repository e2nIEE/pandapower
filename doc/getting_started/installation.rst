=================
Installation
=================

**Python**

pandapower is tested with Python 2.7, 3.4, 3.5 and 3.6.  We recommend the `Anaconda Distribution <https://www.continuum.io/downloads>`_, which already contains a lot of modules for scientific computing 
that are needed for working with pandapower.

Here are the installation instructions depending on what your system looks like or which version of pandapower you want to install:

.. toctree:: 
    :maxdepth: 1
    
    installation_scratch
    installation_anaconda
    installation_python
    installation_without_pip
    installation_develop
    
The easiest way to test your installation is to import all pandapower submodules to see if all dependencies are available:

    import pandapower
    import pandapower.networks
    import pandapower.topology
    import pandapower.plotting
    import pandapower.converter
    import pandapower.estimation

If you want to be really sure that everything works fine, you can run the pandapower test suite (pytest module is needed): ::
    
    import pandapower.test
    pandapower.test.run_all_tests()
    
If everything is installed correctly, all tests should pass.    

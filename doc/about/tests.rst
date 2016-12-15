================================
pandapower Test Suite
================================

pandapower is tested with pytest. There are currently over 100 unit tests testing all kinds of pandapower functionality.

The complete test suit can be run with: ::

        import pandapower.test
        pandapower.test.run_all_tests()
    
If all packages are installed correctly, all tests should pass.

Loadflow Result Tests
=========================
To ensure that pandapower loadflow results are correct, all pandapower element behaviour is tested against DIgSILENT PowerFactory. 

TODO
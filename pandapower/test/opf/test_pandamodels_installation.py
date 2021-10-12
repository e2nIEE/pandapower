# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

@pytest.mark.slow
def test_julia_installation():
    
    try:
        from julia.core import UnsupportedPythonError
    except ImportError:
        UnsupportedPythonError = Exception
    
    try:
        from julia import Main   
        status = True        
    except (ImportError, RuntimeError, UnsupportedPythonError) as e:
        status = False
        print(e)

    assert status 
    
    return status
 
    
 
julia_installed = test_julia_installation()
       

@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_julia_connection():
       
    try:
        import julia
    except:
        assert False
        
    try:
        julia.Julia()
    except:
        assert False

       
@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.dependency(depends=['test_julia_connection'])
def test_pandamodels_installation():
    
    from julia import Main
    from julia import Pkg
    from julia import Base

    if Base.find_package("PandaModels"):  
        # remove PandaModels to reinstall it
        Pkg.rm("PandaModels")
        Pkg.resolve()

    else:
        print("PandaModels is not installed yet!")  
      
    Pkg.Registry.update()
    Pkg.add("PandaModels")  
    Pkg.build()
    Pkg.resolve()
    print("PandaModels is added to julia packages")

    try:
        Main.using("PandaModels")
        print("using PandaModels in its base mode!")
    except:
        assert False
        
        
@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.dependency(depends=['test_pandamodels_installation'])
def test_pandamodels_dev_mode(): 
    
    from julia import Main
    from julia import Pkg
    from julia import Base

    if Base.find_package("PandaModels"):  
        # remove PandaModels to reinstall it
        Pkg.rm("PandaModels")
        Pkg.resolve()
        
    Pkg.Registry.update()
    Pkg.add("PandaModels")  
    print("installing dev mode is a slow process!")  
    Pkg.resolve()
    Pkg.develop("PandaModels")
    # add pandamodels dependencies: slow process
    Pkg.instantiate()            
    Pkg.build()
    Pkg.resolve()
    print("dev mode of PandaModels is added to julia packages")

    try:
        Pkg.activate("PandaModels")
        Main.using("PandaModels")
        print("using PandaModels in its dev mode!")
    except:
        assert False
      
    # activate julia base mode
    Pkg.activate()
    # remove dev mod
    Pkg.rm("PandaModels")
    Pkg.resolve()
    # reinstall base mode
    Pkg.Registry.update()
    Pkg.add("PandaModels")  
    Pkg.build()
    Pkg.resolve()
    print("PandaModels is added to julia packages")
    
if __name__ == '__main__':
    
    pytest.main([__file__])
    # test_julia_installation()
    # test_julia_connection()
    # test_pandamodels_installation()
    # test_pandamodels_dev_mode()
    
    
# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest

try:
    from juliacall import JuliaError
    UnsupportedPythonError = JuliaError
except ImportError:
    UnsupportedPythonError = Exception

try:
    from juliacall import Main
    from juliacall import Base
    from juliacall import Pkg
    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False

import logging

logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
def test_julia_connection():
    try:
        import juliacall
    except:
        raise ImportError("install juliacall properlly to run PandaModels.jl")


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
# @pytest.mark.dependency(depends=['test_julia_connection'])
def test_pandamodels_installation():
    if Base.find_package("PandaModels"):
        # remove PandaModels to reinstall it
        Pkg.rm("PandaModels")
        Pkg.resolve()

    else:
        logger.info("PandaModels is not installed yet!")

    Pkg.Registry.update()
    Pkg.add("PandaModels")
    Pkg.build()
    Pkg.resolve()
    logger.info("PandaModels is added to julia packages")

    try:
        # Main.using("PandaModels")
        Main.seval("using PandaModels")
        logger.info("using PandaModels in its base mode!")
    except ImportError:
        raise ImportError("cannot use PandaModels in its base mode")


@pytest.mark.slow
@pytest.mark.skipif(not julia_installed, reason="requires julia installation")
# @pytest.mark.dependency(depends=['test_julia_connection'])
def test_pandamodels_dev_mode():
    if Base.find_package("PandaModels"):
        # remove PandaModels to reinstall it
        Pkg.rm("PandaModels")
        Pkg.resolve()

    Pkg.Registry.update()
    Pkg.add("PandaModels")
    logger.info("installing dev mode is a slow process!")
    Pkg.resolve()
    Pkg.develop("PandaModels")
    # add pandamodels dependencies: slow process
    Pkg.instantiate()
    Pkg.build()
    Pkg.resolve()
    logger.info("dev mode of PandaModels is added to julia packages")

    try:
        Pkg.activate("PandaModels")
        # Main.using("PandaModels")
        Main.seval("using PandaModels")
        logger.info("using PandaModels in its dev mode!")
    except ImportError:
        # assert False
        raise ImportError("cannot use PandaModels in its dev mode")

    # activate julia base mode
    Pkg.activate()
    # Pkg.free("PandaModels")
    Pkg.resolve()


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

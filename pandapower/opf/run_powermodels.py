import os
from pandapower.converter.powermodels.to_pm import convert_to_pm_structure, dump_pm_json
from pandapower.converter.powermodels.from_pm import read_pm_results_to_net

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _runpm(net, delete_buffer_file=True, pm_file_path=None, pdm_dev_mode=False): 
    """
    Converts the pandapower net to a pm json file, saves it to disk, runs a PandaModels.jl, and reads
    the results back to the pandapower net:
    INPUT
    ----------
    **net** - pandapower net
    OPTIONAL
    ----------
    **delete_buffer_file** (bool, True) - deletes the pm buffer json file if True.
    **pm_file_path** -path to save the converted net json file.
    **pdm_dev_mode** (bool, False) - If True, the develope mode of PdM is called.
    """
    # logger = logging.getLogger("run_pm")
    # logger.setLevel(logging.WARNING)
    # convert pandapower to power models file -> this is done in python
    net, pm, ppc, ppci = convert_to_pm_structure(net)
    # call optinal callback function
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    # writes pm json to disk, which is loaded afterwards in julia
    buffer_file = dump_pm_json(pm, pm_file_path)
    print("the json file for convertet net is stored in: ", buffer_file)
    # run power models optimization in julia
    result_pm = _call_pandamodels(buffer_file, net._options["julia_file"], pdm_dev_mode)
    # read results and write back to net
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    if pm_file_path is None and delete_buffer_file:
        # delete buffer file after calculation
        os.remove(buffer_file)
        print("the json file for convertet net is deleted from ", buffer_file)


def _call_pandamodels(buffer_file, julia_file, dev_mode):  # pragma: no cover

    try:
        import julia
        from julia import Main
        from julia import Pkg
        from julia import Base
    except ImportError:
        raise ImportError(
            "Please install pyjulia properlly to run pandapower with PandaModels.jl.")
        
    try:
        julia.Julia()
    except:
        raise UserWarning(
            "Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured")
              
    if not Base.find_package("PandaModels"):
        print("PandaModels.jl is not installed in julia. It is added now!")
        Pkg.Registry.update()
        Pkg.add("PandaModels")  
        
        if dev_mode:
            print("installing dev mode is a slow process!")
            Pkg.resolve()
            Pkg.develop("PandaModels")
            Pkg.instantiate()
            
        Pkg.build()
        Pkg.resolve()
        print("Successfully added PandaModels")

    # if dev_mode or "julia\dev\PandaModels" in Base.find_package("PandaModels").split(".jl")[0]:
    #     Pkg.activate("PandaModels")
    if dev_mode:
        if ".julia\dev\PandaModels" or ".julia/dev/PandaModels" in Base.find_package("PandaModels"):
            Pkg.activate("PandaModels")
        else:
            Pkg.develop("PandaModels")
            # add pandamodels dependencies: slow process
            Pkg.build()
            Pkg.resolve()
            Pkg.activate("PandaModels")
    try:
        Main.using("PandaModels")
    except ImportError:
        raise ImportError("cannot use PandaModels")

    Main.buffer_file = buffer_file
    result_pm = Main.eval(julia_file + "(buffer_file)")
    return result_pm



import os

# from pandapower import pp_dir
from pandapower.converter.powermodels.to_pm import convert_to_pm_structure, dump_pm_json
from pandapower.converter.powermodels.from_pm import read_pm_results_to_net

try:
    import pplog as logging
except ImportError:
    import logging


def _runpm(net, delete_buffer_file=True, pm_file_path=None):  # pragma: no cover
    """
    Converts the pandapower net to a pm json file, saves it to disk, runs a PowerModels.jl julia function and reads
    the results back to the pandapower net
    INPUT
    ----------
    **net** - pandapower net
    OPTIONAL
    ----------
    **delete_buffer_file** (bool, True) - deletes the pm buffer json file if True.
    """
    logger = logging.getLogger("run_pm")
    logger.setLevel(logging.WARNING)
    # convert pandapower to power models file -> this is done in python
    net, pm, ppc, ppci = convert_to_pm_structure(net)
    # call optinal callback function
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    # writes pm json to disk, which is loaded afterwards in julia
    buffer_file = dump_pm_json(pm, pm_file_path)
    # run power models optimization in julia
    result_pm = _call_pandamodels(buffer_file, net._options["julia_file"])
    # read results and write back to net
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    if pm_file_path is None and delete_buffer_file:
        # delete buffer file after calculation
        os.remove(buffer_file)

# def _add_julia_package():
    #     Pkg.add("PandaModels")
    #     Pkg.build("PandaModels")
    #     Pkg.update()
    #     Pkg.resolve()

# def _activate_dev_mode()
    # if str(type(Base.find_package("PandaModels"))) == "<class 'NoneType'>":
    #     print("PandaModels is not exist")
    #     Pkg.Registry.update()
    #     Pkg.add(url = "https://github.com/e2nIEE/PandaModels.jl") 
    #     # Pkg.build()
    #     Pkg.resolve()
    #     Pkg.develop("PandaModels")
    #     Pkg.build()
    #     Pkg.resolve()
    #     print("add PandaModels")
    # Pkg.activate("PandaModels")
    # Pkg.instantiate()
    # Pkg.resolve()
    # print("activate PandaModels")    


def _call_powermodels(buffer_file, julia_file):  # pragma: no cover
# def _call_powermodels(buffer_file, julia_file, pkg = False, dev = False, instantiate = False):  # pragma: no cover
    try:
        import julia
        from julia import Main
        from julia import Pkg
        from julia import Base
    except ImportError:
        raise ImportError(
            "Please install pyjulia properlly to run pandapower with PowerModels.jl. \nMore info on https://pandapower.readthedocs.io/en/v2.6.0/opf/powermodels.html")

    try:
        j = julia.Julia()
    except:
        raise UserWarning(
            "Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured")
       
    # if Pkg:
    # _add_julia_package()

    inst, dev_mode = None, False
    if not Base.find_package("PandaModels"):
        inst = input("PandaModels is not installed in julia. Add now? (Y/n)")
        if inst in ["Y", "y"]:
            
            Pkg.Registry.update()
            Pkg.add(url="https://github.com/e2nIEE/PandaModels.jl")
            # Pkg.build()
            Pkg.resolve()
            deve = input("Add PandaModels in develop mode? (Y/n)")
            if deve in ["Y", "y"]:
                dev_mode = True
                Pkg.develop("PandaModels")
            Pkg.build()
            print("Successfully added PandaModels")
        else:
            raise ImportError("PandaModels not found")

    # print(Base.find_package("PandaModels"))
    Pkg_path = Base.find_package("PandaModels").split(".jl")[0]
    dev_mode = True if "julia/dev/PandaModels" in Pkg_path else False

    if dev_mode:
        Pkg.activate("PandaModels")
        Pkg.instantiate()
        Pkg.resolve()
        # print("activate PandaModels")

    try:
        Main.using("PandaModels")
        # print("using PandaModels")
    except ImportError:
        raise ImportError("cannot use PandaModels")
        # if not os.path.isfile(julia_file):
    #     raise UserWarning("File %s could not be imported" % julia_file)
    Main.buffer_file = buffer_file
    result_pm = Main.eval(julia_file + "(buffer_file)")
    return result_pm

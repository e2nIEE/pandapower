from pandapower import pp_dir
from pandapower.converter.powermodels.from_pm import (
    read_pm_results_to_net,
)
from pandapower.converter.powermodels.to_pm import (
    convert_to_pm_structure,
    dump_pm_json,
)
from pathlib import Path

try:
    import pplog as logging
except ImportError:
    import logging


def _runpm(
    net, sysimage_file, delete_buffer_file=True,
):  # pragma: no cover
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
    # convert pandapower to power models file -> this is done in python
    (net, pm, ppc, ppci,) = convert_to_pm_structure(net)
    # call optinal callback function
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](
            net, ppci, pm,
        )
    # writes pm json to disk, which is loaded afterwards in julia
    buffer_file = dump_pm_json(pm)
    # run power models optimization in julia
    result_pm = _call_powermodels(
        buffer_file, net._options["julia_file"], sysimage_file,
    )
    # read results and write back to net
    read_pm_results_to_net(
        net, ppc, ppci, result_pm,
    )
    if delete_buffer_file:
        # delete buffer file after calculation
        Path.unlink(buffer_file)


def _call_powermodels(
    buffer_file, julia_file, sysimage_file=None,
):  # pragma: no cover
    # checks if julia works, otherwise raises an error
    try:
        import julia
        from julia.api import Julia
    except ImportError:
        raise ImportError(
            "Please install pyjulia to run pandapower with PowerModels.jl"
        )
    try:
        if sysimage_file and Path.isfile(sysimage_file):
            Julia(sysimage=sysimage_file)
        else:
            julia.Julia()
    except:
        raise UserWarning(
            "Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured. Use a sysimage file if you are inside a conda environment"
        )

    from julia import Main

    # import two julia scripts and runs powermodels julia_file
    Main.include(Path(pp_dir).resolve() / "opf" / "pp_2_pm.jl")
    try:
        run_powermodels = Main.include(julia_file)
    except ImportError:
        raise UserWarning(
            "File %s could not be imported" % julia_file
        )
    result_pm = run_powermodels(buffer_file)
    return result_pm

import json
import os
import warnings

from pandapower.converter.pandamodels.to_pm import convert_to_pm_structure, dump_pm_json
from pandapower.converter.pandamodels.from_pm import read_pm_results_to_net
from pandapower.optimal_powerflow import OPFNotConverged
from pandapower.auxiliary import pandapowerNet

import logging
logger = logging.getLogger(__name__)

# PandaModels.jl is declared in pandapower/juliapkg.json and installed by juliapkg on juliacall import.
# Data is sent in MATPOWER units, which PandaModels.jl converts since this version.
MIN_PANDAMODELS_VERSION = "0.10.0"

_pandamodels = None


def _runpm(
    net: pandapowerNet,
    delete_buffer_file: bool = True,
    pm_file_path: str | None = None,
    pdm_dev_mode: bool = False,
    **kwargs
):
    """
    Converts the pandapower net to a pm json file, saves it to disk, runs a PandaModels.jl, and reads
    the results back to the pandapower net

    Parameters:
        net: the pandapower net
        delete_buffer_file: deletes the pm buffer json file if True.
        pm_file_path: path to save the converted net json file.
        pdm_dev_mode: deprecated, without effect. A local PandaModels.jl checkout is configured with
            juliapkg, see :func:`_load_pandamodels`.

    Keyword Arguments:
        passed to :func:`convert_to_pm_structure`
    """
    if pdm_dev_mode:
        warnings.warn("pdm_dev_mode is deprecated and has no effect. To use a local PandaModels.jl "
                      "checkout run juliapkg.add('PandaModels', '2dbab86a-7cbf-476f-9afe-75ffd3079e7c', "
                      "path=<checkout>, dev=True) once.", DeprecationWarning, stacklevel=3)
    # convert pandapower to power models file -> this is done in python
    net, pm, ppc, ppci = convert_to_pm_structure(net, **kwargs)
    # call optional callback function
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    # writes pm json to disk, which is loaded afterwards in julia
    buffer_file = dump_pm_json(pm, pm_file_path)
    logger.debug("the json file for converted net is stored in: %s" % buffer_file)
    # run power models optimization in julia
    result_pm = _call_pandamodels(buffer_file, net._options["julia_file"])

    logger.info("Optimization ('"+net._options["julia_file"]+"') " +
                "is finished in %s seconds:" % round(result_pm["solve_time"], 2))
    # read results and write back to net
    try:
        read_pm_results_to_net(net, ppc, ppci, result_pm)
    except OPFNotConverged:
        _delete_buffer_file(buffer_file, pm_file_path, delete_buffer_file)
        raise
    _delete_buffer_file(buffer_file, pm_file_path, delete_buffer_file)


def _delete_buffer_file(buffer_file, pm_file_path, delete_buffer_file):
    if pm_file_path is None and delete_buffer_file:
        os.remove(buffer_file)
        logger.debug("the json file for converted net is deleted from %s" % buffer_file)


def _load_pandamodels():  # pragma: no cover
    """
    Returns the PandaModels.jl module, loaded once per process.

    PandaModels.jl is installed by juliapkg from pandapower/juliapkg.json. To work with a local
    checkout of PandaModels.jl instead, register it once in the juliapkg project:

        import juliapkg
        juliapkg.add("PandaModels", "2dbab86a-7cbf-476f-9afe-75ffd3079e7c", path="<checkout>", dev=True)
    """
    global _pandamodels
    if _pandamodels is not None:
        return _pandamodels
    try:
        from juliacall import Main  # type: ignore
    except ImportError:
        raise ImportError(
            "Please install juliacall properly to run pandapower with PandaModels.jl.")
    try:
        Main.seval("using PandaModels")
    except Exception as e:
        raise ImportError("PandaModels.jl could not be loaded. It is installed by juliapkg from "
                          "pandapower/juliapkg.json, run juliapkg.resolve(force=True) to "
                          "reinstall it.") from e
    if not Main.seval(f'pkgversion(PandaModels) >= v"{MIN_PANDAMODELS_VERSION}"'):
        raise ImportError("PandaModels.jl %s is installed, but pandapower requires >= %s. Run "
                          "juliapkg.resolve(force=True) to update it."
                          % (Main.seval("string(pkgversion(PandaModels))"), MIN_PANDAMODELS_VERSION))
    _pandamodels = Main.PandaModels
    return _pandamodels


def _call_pandamodels(buffer_file, julia_file):  # pragma: no cover
    pandamodels = _load_pandamodels()
    try:
        run_function = getattr(pandamodels, julia_file)
    except AttributeError:
        raise ValueError("PandaModels.jl has no function %s" % julia_file) from None
    # one JSON transfer instead of element-wise access to the Julia Dict (NaN arrives as None)
    return json.loads(pandamodels.json_result(run_function(buffer_file)))

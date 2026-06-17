import logging
import time

from pandapower.auxiliary import pandapowerNet
from pandapower.converter.pypowsybl.pypowsybl_converter import PyPowSyBlConverter

logger = logging.getLogger("pypowsyl.from_pypowsybl")

def from_pypowsybl(
        pypowsybl_file: str,
        debug: bool = False,
        debug_run: bool = False,
        default_shift_degree: float = 0.0,
        default_length_km: float = 1.0
) ->pandapowerNet:
    """
    Converts net data stored as a powsybl XIIDM file to a pandapower net.
    
    :param str pypowsybl_file: path to the powsybl .xiidm file which includes the grid data
    :param bool debug: decides whether static transfer comparison tables are printed
    :param bool debug_run: decides whether powsybl and pandapower AC load-flow results are compared
    :param float default_shift_degree: fallback phase-shiftangle in degrees
    :param float default_length_km: fallback line length in kilometres
    
    :return: A pandapower net
    :rtype: pandapowerNet
    
    :example:
        >>> from pandapower.converter.pypowsybl.from_pypowsybl import from_pypowsybl
        >>>
        >>> net = from_pypowsybl("network.xiidm")
    """
    time_start_converting = time.time()

    pypowsybl_converter = PyPowSyBlConverter()

    pp_net, _, _ = pypowsybl_converter._powsybl_to_pandapower(
        filename=pypowsybl_file,
        debug=debug,
        debug_run=debug_run,
        default_shift_degree=default_shift_degree,
        default_length_km=default_length_km
    )

    time_end_converting = time.time()

    logger.info("Needed time for converting from pypowsybl: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time (from_pypowsybl()): %s" % (time_end_converting - time_start_converting))

    return pp_net
from pandapower.pypower_extensions.runpf import _runpf
from pandapower.run_newton_raphson import _run_newton_raphson
from pandapower.run_bfswpf import _run_bfswpf
from pandapower.run_dc_pf import _run_dc_pf
from pandapower.auxiliary import ppException


class AlgorithmUnknown(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def _run_pf_algorithm(ppci, options, **kwargs):
    algorithm = options["algorithm"]
    ac = options["ac"]

    if ac:
        # ----- run the powerflow -----
        if algorithm == 'bfsw':  # forward/backward sweep power flow algorithm
            result = _run_bfswpf(ppci, options, **kwargs)[0]
        elif algorithm == 'nr':
            result = _run_newton_raphson(ppci, options)
        elif algorithm in ['fdBX', 'fdXB', 'gs']:  # algorithms existing within pypower
            result = _runpf(ppci, options, **kwargs)[0]
        else:
            raise AlgorithmUnknown("Algorithm {0} is unknown!".format(algorithm))
    else:
        result = _run_dc_pf(ppci)

    return result

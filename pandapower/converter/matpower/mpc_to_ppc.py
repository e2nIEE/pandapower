import scipy.io
import numpy as np
import pplog

logger = pplog.getLogger(__name__)


def mpc2ppc(mpc_file):
    # load mpc from file
    mpc = scipy.io.loadmat(mpc_file, squeeze_me=True, struct_as_record=False)

    # init empty ppc
    ppc = dict()

    _copy_data_from_mpc_to_ppc(ppc, mpc)
    _adjust_ppc_indices(ppc)
    _change_ppc_TAP_value(ppc)

    return ppc


def _adjust_ppc_indices(ppc):
    # adjust indices of ppc, since ppc must start at 0 rather than 1 (matlab)
    ppc["bus"][:, 0] -= 1
    ppc["branch"][:, 0] -= 1
    ppc["branch"][:, 1] -= 1
    ppc["gen"][:, 0] -= 1


def _copy_data_from_mpc_to_ppc(ppc, mpc):
    if 'mpc' in mpc:
        # if struct contains a field named mpc
        ppc['version'] = mpc['mpc'].version
        ppc["baseMVA"] = mpc['mpc'].baseMVA
        ppc["bus"] = mpc['mpc'].bus
        ppc["gen"] = mpc['mpc'].gen
        ppc["branch"] = mpc['mpc'].branch

        try:
            ppc['gencost'] = mpc['mpc'].gencost
        except:
            logger.info('gencost is not in mpc')

    elif 'bus' in mpc \
            and 'branch' in mpc \
            and 'gen' in mpc \
            and 'baseMVA' in mpc \
            and 'version' in mpc:

        # if struct contains bus, branch, gen, etc. directly
        ppc['version'] = mpc['version']
        ppc["baseMVA"] = mpc['baseMVA']
        ppc["bus"] = mpc['bus']
        ppc["gen"] = mpc['gen']
        ppc["branch"] = mpc['branch']

        if 'gencost' in mpc:
            ppc['gencost'] = mpc['gencost']
        else:
            logger.info('gencost is not in mpc')

    else:
        logger.error('Matfile does not contain a valid mpc structure')


def _change_ppc_TAP_value(ppc):
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    ppc["branch"][np.where(ppc["branch"][:, 8] == 0), 8] = 1

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
import numpy as np
import pandas as pd
import scipy.io

from pandapower.converter.pypower import from_ppc

try:
    from matpowercaseframes import CaseFrames
    matpowercaseframes_imported = True
except ImportError:
    matpowercaseframes_imported = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def from_mpc(mpc_file, f_hz=50, casename_mpc_file='mpc', validate_conversion=False, **kwargs):
    """
    This function converts a matpower case file version 2 to a pandapower net.

    Note: If 'mpc_file' ends with '.m' the python package 'matpowercaseframes' is used. If
    'mpc_file' ends with '.mat' 'scipy.io.loadmat' is used. Other file endings are not supported.
    In that other cases, please, rename the file ending or use the internal subfunctions.

    Note: python is 0-based while Matlab is 1-based.

    INPUT:

        **mpc_file** - path to a matpower case file (.mat format not .m script).

    OPTIONAL:

        **f_hz** (int, 50) - The frequency of the network.

        **casename_mpc_file** (str, 'mpc') - The name of the variable in .mat file which contain
        the matpower case structure, i.e. the arrays "gen", "branch" and "bus".

        ****kwargs** - key word arguments for from_ppc()

    OUTPUT:

        **net** - The pandapower network

    EXAMPLE:

        import pandapower.converter as pc

        pp_net1 = cv.from_mpc('case9.mat', f_hz=60)
        pp_net2 = cv.from_mpc('case9.m', f_hz=60)

    """
    ending = os.path.splitext(os.path.basename(mpc_file))[1]
    if ending == ".mat":
        ppc = _mat2ppc(mpc_file, casename_mpc_file)
    elif ending == ".m":
        ppc = _m2ppc(mpc_file, casename_mpc_file)
    net = from_ppc(ppc, f_hz=f_hz, validate_conversion=validate_conversion, **kwargs)
    if "mpc_additional_data" in ppc:
        if "_options" not in net:
            net["_options"] = dict()
        net._options.update(ppc["mpc_additional_data"])
        logger.info('added fields %s in net._options' % list(ppc["mpc_additional_data"].keys()))

    return net


def _mpc2ppc(mpc_file, casename_mpc_file):
    raise DeprecationWarning("_mpc2ppc() has been renamed by _mat2ppc().")


def _mat2ppc(mpc_file, casename_mpc_file):
    # load mpc from file
    mpc = scipy.io.loadmat(mpc_file, squeeze_me=True, struct_as_record=False)

    # init empty ppc
    ppc = dict()

    _copy_data_from_mpc_to_ppc(ppc, mpc, casename_mpc_file)
    _adjust_ppc_indices(ppc)
    _change_ppc_TAP_value(ppc)

    return ppc


def _m2ppc(mpc_file, casename_mpc_file):
    if not matpowercaseframes_imported:
        raise NotImplementedError(
            "matpowercaseframes is used to convert .m file. Please install that python "
            "package, e.g. via 'pip install matpowercaseframes'.")
    mpc_frames = CaseFrames(mpc_file)
    ppc = {key: mpc_frames.__getattribute__(key) if not isinstance(
        mpc_frames.__getattribute__(key), pd.DataFrame) else mpc_frames.__getattribute__(
        key).values for key in mpc_frames._attributes}
    _adjust_ppc_indices(ppc)
    return ppc


def _adjust_ppc_indices(ppc):
    # adjust indices of ppc, since ppc must start at 0 rather than 1 (matlab)
    ppc["bus"][:, 0] -= 1
    ppc["branch"][:, 0] -= 1
    ppc["branch"][:, 1] -= 1
    # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
    if len(ppc["gen"].shape) == 1:
        ppc["gen"] = np.array(ppc["gen"], ndmin=2)
    ppc["gen"][:, 0] -= 1


def _copy_data_from_mpc_to_ppc(ppc, mpc, casename_mpc_file):
    if casename_mpc_file in mpc:
        # if struct contains a field named mpc
        ppc['version'] = mpc[casename_mpc_file].version
        ppc["baseMVA"] = mpc[casename_mpc_file].baseMVA
        ppc["bus"] = mpc[casename_mpc_file].bus
        ppc["gen"] = mpc[casename_mpc_file].gen
        ppc["branch"] = mpc[casename_mpc_file].branch

        try:
            ppc['gencost'] = mpc[casename_mpc_file].gencost
        except:
            logger.info('gencost is not in mpc')

        for k in mpc[casename_mpc_file]._fieldnames:
           if k not in ppc:
               ppc.setdefault("mpc_additional_data", dict())[k] = getattr(mpc[casename_mpc_file], k)

    else:
        logger.error('Matfile does not contain a valid mpc structure.')


def _change_ppc_TAP_value(ppc):
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    ppc["branch"][np.where(ppc["branch"][:, 8] == 0), 8] = 1


if "__main__" == __name__:
    pass

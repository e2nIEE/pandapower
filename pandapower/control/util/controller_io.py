# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import logging as pplog
logger=pplog.getLogger(__name__)

try:
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy
except ImportError:
    logger.debug('could not import jsonpickle')
import pandapower.file_io as fio
import os
import copy as cp


def save_snapshot(ch, path, save_res=False):
    """
    Saves a snapshot of the control-handler to a specified folder.
    The handler will be pickled in JSON the net will be saved in HDF5.
    """
    jsonpickle_numpy.register_handlers()

    if not os.path.exists(path):
        os.makedirs(path)

    fio.to_hdf5(ch.net, path+"\\net.h5", save_res=save_res)

    # remove net reference
    ch_cpy = cp.deepcopy(ch)
    ch_cpy.net = None

    with open(path+"\\ch.json", "w") as output:
        output.write(jsonpickle.encode(ch_cpy))

    jsonpickle_numpy.unregister_handlers()


def load_snapshot(path):
    """
    Loads and returns a previously saved snapshot of a control-handler.
    Expects a ch.json and a net.h5 file to be present at the specified location.
    """
    jsonpickle_numpy.register_handlers()

    net = fio.from_hdf5(path+"\\net.h5")

    with open(path+"\\ch.json", "r") as input:
        json = input.read()

    ch = jsonpickle.decode(json)

    # add net references
    ch.net = net
    for c in ch.controller:
        c.net = net

    jsonpickle_numpy.unregister_handlers()

    return ch


def controller_to_json(controller, path=None):
    """
    Encodes the internal state of a controller
    in JSON.

    :param controller: The controller or list of controllers to encode.
    :param path: Optionally saves the JSON-String in a specified
    file.
    :return: The encoded JSON-String
    """
    logger.warning('the function controller_to_json is deprecated')
    jsonpickle_numpy.register_handlers()

    json = jsonpickle.encode(controller)

    if path is not None:
        with open(path, "w") as output:
            output.write(json)

    jsonpickle_numpy.unregister_handlers()

    return json


def json_to_controller(net, path=None, json=None):
    """
    Decodes the internal state of a controller
    from JSON. Optionally loads the JSON-String in a specified
    file.

    :param path: A path to a file containing JSON-Strings for decoding
    :param json: A JSON-String for decoding
    :return:
    """
    jsonpickle_numpy.register_handlers()

    if path is not None:
        with open(path, "r") as input:
            json = input.read()
    elif json is None:
        raise UserWarning("Either a JSON-String or a file path must be specified!")

    controller = jsonpickle.decode(json)

    for c in controller:
        c.net = net

    jsonpickle_numpy.unregister_handlers()

    return controller


def dump_controller(net, time_step, dump_controller=False):
    # ToDo: @Rieke -> Implement dumping function
    if dump_controller:
        # save information about unconverged controller_order
        unconverged = [ctrl for ctrl in net.controller.controller if not ctrl.is_converged()]
        unconverged.append((time_step, unconverged))

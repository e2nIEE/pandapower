# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import csv
import random
from functools import reduce
from pandapower.auxiliary import ADict

import numpy as np
import pandas

import pandas as pd
from pandas import Int64Index

# import control as ct
try:
    import pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)

"""
This module contains a few functions for building the Controller-Objects
for a simulation. You can explore these functions and use them as an
example for your own setup.

If you wrote a helper function that you find very useful, you might want to add
it here for other people. If you do so, please make sure that it is well documented
as well as leaving a note, who the author is, so people know, who to turn to if they
have questions.

For creating Controllers there is a basic schematic workflow:
    1. Read the parameters for the *Controller* from the data-structure
    2. If you want profiles you need to:
        - Create a *DataSource* from a CSV-file or a profile-generator for
          the actual profiles
        - Read a profile-table relating the profiles to the Controllers
    3. Create Controllers with the retrieved parameters and add them to
       your *ControlHandler*
"""


def asarray(val, dtype=np.float64):
    """
    make sure the value is a numpy array of a certain dtype
    in contrast to np.asarray, it also converts a scalar value to an array,
    e.g. asarray(0) = array([0])
    :param val: scalar or list-like or pandas index input
    :param dtype: np.dtype for the output array
    :return: numpy array of the dtype set
    """
    # val is a list, tuple etc
    if not np.isscalar(val) and np.ndim(val) > 0:
        np_val = np.asarray(val, dtype=dtype)
    else:
        # val is a scalar number
        np_val = np.asarray([val], dtype=dtype)

    return np_val


def check_controller_frame(net):
    """
    check if net has controller DataFrame and if not, add it.
    Purpose: to define at 1 place
    """
    if 'controller' not in net.keys():
        net['controller'] = pd.DataFrame(
           columns=['controller', 'in_service', 'order', 'level', 'recycle'])
        net.controller.in_service.astype(bool)
        net.controller.recycle.astype(bool)

def mute_control(net):
    """
    Use this function to set all controllers in net out of service, e. g. when you want to use the
    net with new controllers
    :param net: pandapowerNet
    :return:
    """
    check_controller_frame(net)
    net.controller['in_service'] = False


def get_controller_by_element(net, element, element_index):
    """
    Returns the indices of controllers controlling the given elements.
    """
    # TODO

    # till now the element_index in the controller are names differently (tid, gid, ...)
    # after consistent renaming this function can be coded
    idx = None

    return idx


def get_controller_index_by_type(net, ctrl_type, idx=[]):
    """
    Returns controller indices of a given type as list.
    """
    check_controller_frame(net)
    idx = idx if len(idx) else net.controller.index
    return [i for i in idx if isinstance(net.controller.controller.loc[i], ctrl_type)]


def get_controller_index_by_typename(net, typename, idx=[], case_sensitive=False):
    """
    Returns controller indices of a given name of type as list.
    """
    check_controller_frame(net)
    idx = idx if len(idx) else net.controller.index
    if case_sensitive:
        return [i for i in idx if str(net.controller.controller.loc[i]).split(" ")[0] == typename]
    else:
        return [i for i in idx if str(net.controller.controller.loc[i]).split(" ")[0].lower() ==
                typename.lower()]


def _controller_attributes_query(controller, parameters):
    """
    Returns a boolean if the controller attributes matches given parameter dict data
    """
    match = True
    for key in parameters.keys():
        if key not in controller.__dict__:
            logger.debug(str(key) + " is no attribute of controller object " + str(controller))
            return False
        try:
            match &= bool(controller.__getattribute__(key) == parameters[key])
        except ValueError:
            try:
                match &= all(controller.__getattribute__(key) == parameters[key])
            except ValueError:
                match &= bool(len(set(controller.__getattribute__(key)) & set(parameters[key])))
    return match


def get_controller_index(net, ctrl_type=None, parameters=None, idx=[]):
    """ Returns indices of searched controllers. Parameters can specify the search query.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTINAL:
        **controller_type** (controller object or string name of controller object)

        **parameters** (None, dict) - Dict of parameter names, which are in the controller object or
            net.controller DataFrame

        **idx** ([], list) - list of indices in net.controller to be searched for. If list is empty
            all indices are considered.

    OUTPUT:
        **idx** (list) - index / indices of controllers in net.controller which are in idx and
            matches given ctrl_type or parameters
    """
    #    logger.debug(ctrl_type, parameters, idx)
    check_controller_frame(net)
    idx = idx if len(idx) else net.controller.index
    if ctrl_type is not None:
        if isinstance(ctrl_type, str):
            idx = get_controller_index_by_typename(net, ctrl_type, idx)
        else:
            idx = get_controller_index_by_type(net, ctrl_type, idx)
    if isinstance(parameters, dict):
        df_keys = [k for k in parameters.keys() if k in net.controller.columns]
        attributes_keys = list(set(parameters.keys()) - set(df_keys))
        attributes_dict = {k: parameters[k] for k in attributes_keys}
        # query of parameters in net.controller dataframe
        idx = Int64Index(idx)
        for df_key in df_keys:
            idx &= net.controller.index[net.controller[df_key] == parameters[df_key]]
        # query of parameters in controller object attributes
        idx = [i for i in idx if _controller_attributes_query(
            net.controller.controller.loc[i], attributes_dict)]
    return idx


def log_same_type_existing_controllers(net, this_ctrl_type, index=None, matching_params=None,
                                       **kwargs):
    """
    Logs same type controllers, if a controller is created.
    INPUT:
        **net** - pandapower net

        **this_ctrl_type** (controller object or string name of controller object)

    OPTIONAL:
        **index** (int) - index in net.controller of the controller to be created

        **matching_params** (dict) - parameters, which must be equal if same type controller should
            be logged.

        ****kwargs** - unused arguments, given to avoid unexpected input arguments
    """
    index = str(index)
    if isinstance(matching_params, dict):
        same_type_existing_ctrl = get_controller_index(net, ctrl_type=this_ctrl_type,
                                                       parameters=matching_params)
        if len(same_type_existing_ctrl):
            logger.info("Controller " + index + " has same type and matching parameters like " +
                        "controllers " + str(['%i' % idx for idx in same_type_existing_ctrl]))
    else:
        logger.info("Creating controller " + index + " of type %s " % this_ctrl_type)
        logger.debug("no matching parameters are given to check whether problematic, " +
                    "same type controllers already exist.")


def drop_same_type_existing_controllers(net, this_ctrl_type, index=None, matching_params=None,
                                        **kwargs):
    """
    Drops same type controllers to create a new controller of this type.
    INPUT:
        **net** - pandapower net

        **this_ctrl_type** (controller object or string name of controller object)

    OPTIONAL:
        **index** (int) - index in net.controller of the controller to be created

        **matching_params** (dict) - parameters, which must be equal if same type controller should
            be logged.

        ****kwargs** - unused arguments, given to avoid unexpected input arguments
    """
    index = str(index)
    if isinstance(matching_params, dict):
        same_type_existing_ctrl = get_controller_index(net, ctrl_type=this_ctrl_type,
                                                       parameters=matching_params)
        if len(same_type_existing_ctrl):
            net.controller.drop(same_type_existing_ctrl, inplace=True)
            logger.debug("Controllers " + str(['%i' % idx for idx in same_type_existing_ctrl]) +
                         "got removed because of same type and matching parameters as new " +
                         "controller " + index + ".")
    else:
        logger.info("Creating controller " + index + " of type %s, " % this_ctrl_type +
                    "no matching parameters are given to check which " +
                    "same type controllers should be dropped.")


def get_slack_trafo_id(net):
    try:
        return net.trafo.loc[net.trafo.hv_bus == net.ext_grid.bus[0]].index.values[0]
    except:
        raise UserWarning("Could not locate slack trafo!")


def cmp(x, y):
    return ((x > y) - (x < y))


def convert_csv_to_hdf5(csv_path, hdf5_path=None):
    """
    Converter for csv to hdf5 files.

    |   **csv_path** - The csv path
    |   **hdf5_path** - The hdf5 path (default will be the same as csv)
    """
    # use the same path, just replace ends with h5
    if not hdf5_path:
        hdf5_path = csv_path[:csv_path.rfind(".")] + ".h5"

    df = pd.DataFrame(pd.read_csv(csv_path, index_col=None), dtype=float)

    df.to_hdf(hdf5_path, "df", complevel=1, complib="zlib")


def convert_csv_to_pickle(csv_path, pickle_path=None):
    """
    Converter for csv to hdf5 files.

    |   **csv_path** - The csv path
    |   **hdf5_path** - The hdf5 path (default will be the same as csv)
    """
    # use the same path, just replace ends with h5
    if not pickle_path:
        pickle_path = csv_path[:csv_path.rfind(".")] + ".pkl"

    df = pd.DataFrame(pd.read_csv(csv_path, index_col=None), dtype=float)

    df.to_pickle(pickle_path)


def build_ctrl_list(ch, strat, p_ac, max_error):
    """
    This function will read from the loadinfo of a net and creating
    controllers accordingly. The strategy and simultaneity factor
    for all PV-controllers will be set by the passed parameters.

    Note: it will create a PV-controller for every entry in loadinfo!
    You might want to read at net["loadinfo"][load_idx, 1] to identify
    if the loadtype is actually a PV-generator.

    Hint: loadinfo will be abandoned in the indefinite future, so you might
    want to keep access to these information at a minimum and at one place to
    make future changes easy.

    |   **ch** - The ControlHandler for which we want to create controllers
    |   **strat** - A string for chosing different strategies
    |   **p_ac** - The simultaneity factor

    author: Julian Dollichon
    """

    # for each one of them
    for idx in ch.net["sgen"].index:
        # read their data from the loadinfo
        bus = ch.net.sgen.at[idx, "bus"]
        p = ch.net.sgen.at[idx, "p_kw"]
        q = ch.net.sgen.at[idx, "q_kvar"]
        rated_power = ch.net.sgen.at[idx, "sn_kva"]

        # and create a Controller according to the strategy set,
        # as well as adding it to the ControlHandler
        if strat == "cosphi":
            ch.add_controller(ct.CosphiPv(bus, p, q, rated_power, max_error=max_error, id=idx,
                                          p_ac=p_ac))
        elif strat == "QPofU":
            ch.add_controller(ct.PqofuPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                         p_ac=p_ac))
        elif strat == "Q(U)_P_70%_control":
            ch.add_controller(ct.QofuLimPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                           p_ac=p_ac))
        elif strat == "Q(U)_control":
            ch.add_controller(ct.QofuPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                        p_ac=p_ac))
        elif strat == "Q(U)_control":
            ch.add_controller(ct.QofuPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                        p_ac=p_ac))


def build_ctrl_with_profile(ch, max_error, p_ac=1., data_source=None, profile_table_path=None):
    """
    This function will read from the loadinfo of a net and creating
    controllers accordingly. The strategy and simultaneity factor
    for all PV-controllers will be set by the passed parameters.

    Additionally it will read from the given profile table to relate
    PV-controllers to their designated profiles.

    Hint: loadinfo will be abandoned in the indefinite future, so you might
    want to keep access to these information at a minimum and at one place to
    make future changes easy.

    |   **ch** - The ControlHandler for which we want to create controllers
    |   **p_ac** - The simultaneity factor
    |   **data_sources** - The DataSource we expect to get profile data from
    |   **profile_table_path** - The path to a CSV-File relating a Controller to
    its designated profile

    author: Julian Dollichon
    """
    # this will be our ConstLoad later ;)
    const = None

    # for each one of them
    for idx in ch.net["sgen"].index:
        # read their data from the loadinfo
        bus = ch.net.sgen.at[idx, "bus"]
        p = ch.net.sgen.at[idx, "p_kw"]
        q = ch.net.sgen.at[idx, "q_kvar"]
        rated_power = ch.net.sgen.at[idx, "sn_kva"]
        load_name = ch.net.sgen.at[idx, "name"]

        # read and create profile_table from file
        profile_table = pd.DataFrame(pd.read_csv(profile_table_path, header=0, sep=';',
                                                 index_col=0,
                                                 decimal='.',
                                                 thousands=None))

        # read profile name, strategy etc from profile.csv
        # Note: you can add columns to the CSV-file and read them in the same manner
        # if you would like to specify further options
        profile_name = profile_table.loc[load_name]["profile"]
        strat = profile_table.loc[load_name]["strategy"]
        cos_phi = profile_table.loc[load_name]["cosphi"]
        scale = profile_table.loc[load_name]["scale"]

        # decide which Controller should be added and which parameters should be used
        # data_sources = the data source we created from our CSV profiles
        # profile_name = the name of the column of the profile we want to use
        # profile_scale = use to scale profiles (default = 1.)
        if strat == "cosphi":
            ch.add_controller(ct.CosphiPv(bus, p, q, rated_power, max_error=max_error, id=idx,
                                          p_ac=p_ac, data_source=data_source,
                                          profile_name=profile_name, profile_scale=scale))
        elif strat == "QPofU":
            ch.add_controller(ct.PqofuPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                         p_ac=p_ac, data_source=data_source,
                                         profile_name=profile_name, profile_scale=scale))
        elif strat == "QofU70":
            ch.add_controller(ct.QofuLimPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                           p_ac=p_ac, data_source=data_source,
                                           profile_name=profile_name, profile_scale=scale))
        elif strat == "QofU":
            ch.add_controller(ct.QofuPv(bus, p, q, rated_power, max_error=max_error, gid=idx,
                                        p_ac=p_ac, data_source=data_source,
                                        profile_name=profile_name, profile_scale=scale))
        elif strat == "pvnoctrl":
            ch.add_controller(ct.PvNoControl(bus, p, q, rated_power, max_error=max_error, id=idx,
                                             p_ac=p_ac, data_source=data_source,
                                             profile_name=profile_name, profile_scale=scale))
        elif strat == "constload":
            # TODO: rework for pandapower i.e. index
            if const is None:
                const = ct.ConstLoadMulti(data_source=data_source, cos_phi=cos_phi)
                ch.add_controller(const)
            const.add_load(load_name, bus, p, q, profile_name=profile_name, scaling=scale)


def equal(x, y, verbose=False):
    """
    Compares two networks. The networks are considered equal
    if they share the same keys and values, except of the
    'et' (elapsed time) entry which differs depending on
    runtime conditions.
    """
    eq = True

    # for dicts call equal on all values
    if isinstance(x, dict) or isinstance(x, ADict) and \
            isinstance(y, dict) or isinstance(y, ADict):
        try:
            # make sure both dicts have the same keys
            if len(set(x.keys()) - set(y.keys())) + \
                    len(set(y.keys()) - set(x.keys())) > 0:
                if verbose:
                    logger.debug(set(x.keys()) - set(y.keys()))
                    logger.debug(set(y.keys()) - set(x.keys()))
                    logger.debug(list(x.keys()))
                    logger.debug(list(y.keys()))
                return False

            # iter through keys
            for k in list(x.keys()):
                if (k != 'et' and not k.startswith("_")):
                    eq &= equal(x[k], y[k])
        except KeyError:
            return False
    else:
        # logger.debug type(x)
        # for lists or n-dimensional numpy arrays call equal on each element
        if isinstance(x, list) or isinstance(x, np.ndarray) and \
                isinstance(y, list) or isinstance(y, np.ndarray):
            for i in range(len(x)):
                eq &= equal(x[i], y[i])

            for i in range(len(y)):
                eq &= equal(x[i], y[i])
        # for DataFrames eval if all entries are equal
        elif isinstance(x, pandas.DataFrame) and isinstance(y, pandas.DataFrame):
            eq &= x.fillna(1).sort_index(axis=1).eq(y.fillna(1).sort_index(axis=1)).all().all()
            if not eq and verbose:
                logger.debug("_____________________________________________")
                logger.debug(x)
                logger.debug("---------")
                logger.debug(y)
                logger.debug("_____________________________________________")
        elif hasattr(x, "__dict__") and hasattr(y, "__dict__"):
            eq &= equal(x.__dict__, y.__dict__)
        else:  # else items are assumed to be single values, compare them
            eq &= x == y

    # if not eq:
    #     logger.debug x
    #     logger.debug y
    return eq
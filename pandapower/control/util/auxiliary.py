# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import sys

import numpy as np
from pandas import Index, Series

from pandapower.auxiliary import soft_dependency_error, ensure_iterability
from .characteristic import SplineCharacteristic

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)


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


def get_controller_index_by_type(net, ctrl_type, idx=[]):
    """
    Returns controller indices of a given type as list.
    """
    idx = idx if len(idx) else net.controller.index
    is_of_type = net.controller.object.apply(lambda x: isinstance(x, ctrl_type))
    return list(net.controller.index.values[net.controller.index.isin(idx) & is_of_type])

def get_controller_index_by_typename(net, typename, idx=[], case_sensitive=False):
    """
    Returns controller indices of a given name of type as list.
    """
    idx = idx if len(idx) else net.controller.index
    if case_sensitive:
        return [i for i in idx if str(net.controller.object.at[i]).split(" ")[0] == typename]
    else:
        return [i for i in idx if str(net.controller.object.at[i]).split(" ")[0].lower() ==
                typename.lower()]


def _controller_attributes_query(controller, parameters):
    """
    Returns a boolean if the controller attributes matches given parameter dict data
    """
    complete_match = True
    element_index_match = True
    for key in parameters.keys():
        if key not in controller.__dict__:
            logger.debug(str(key) + " is no attribute of controller object " + str(controller))
            return False
        try:
            match = bool(controller.__getattribute__(key) == parameters[key])
        except ValueError:
            try:
                match = all(controller.__getattribute__(key) == parameters[key])
            except ValueError:
                match = bool(len(set(controller.__getattribute__(key)) & set(parameters[key])))
        if key == "element_index":
            element_index_match = match
        else:
            complete_match &= match

    if complete_match and not element_index_match:
        intersect_elms = set(ensure_iterability(controller.__getattribute__("element_index"))) & \
                         set(ensure_iterability(parameters["element_index"]))
        if len(intersect_elms):
            logger.debug("'element_index' has an intersection of " + str(intersect_elms) +
                         " with Controller %i" % controller.index)

    return complete_match & element_index_match


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
        idx = Index(idx, dtype=np.int64)
        for df_key in df_keys:
            idx = idx.intersection(net.controller.index[net.controller[df_key] == parameters[df_key]])
        # query of parameters in controller object attributes
        matches = net.controller.object.apply(lambda ctrl: _controller_attributes_query(ctrl, attributes_dict))
        idx = list(net.controller.index.values[net.controller.index.isin(idx) & matches])
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
            net.controller = net.controller.drop(same_type_existing_ctrl)
            logger.debug("Controllers " + str(['%i' % idx for idx in same_type_existing_ctrl]) +
                         "got removed because of same type and matching parameters as new " +
                         "controller " + index + ".")
    else:
        logger.info("Creating controller " + index + " of type %s, " % this_ctrl_type +
                    "no matching parameters are given to check which " +
                    "same type controllers should be dropped.")


def plot_characteristic(characteristic, start, stop, num=20, xlabel=None, ylabel=None):
    if not MATPLOTLIB_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
    x = np.linspace(start, stop, num)
    y = characteristic(x)
    plt.plot(x, y, marker='x')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)


def create_trafo_characteristics(net, trafotable, trafo_index, variable, x_points, y_points):
    # create characteristics for the specified variable and set their indices in the trafo table
    col = f"{variable}_characteristic"
    # check if the variable is a valid attribute of the trafo table
    supported_columns = {"trafo": ["vk_percent_characteristic", "vkr_percent_characteristic"],
                         "trafo3w": [f"vk{r}_{side}_percent_characteristic" for side in ["hv", "mv", "lv"] for r in
                                     ["", "r"]]}
    if col not in supported_columns[trafotable]:
        raise UserWarning("Variable %s is not supported for table %s" % (variable, trafotable))

    # check inputs, check if 1 trafo or multiple, verify shape of x_points and y_points and trafo_index
    if hasattr(trafo_index, '__iter__'):
        single_mode = False
        if not (len(trafo_index) == len(x_points) == len(y_points)):
            raise UserWarning("The lengths of the trafo index and points do not match!")
    else:
        single_mode = True
        if (len(x_points) != len(y_points)):
            raise UserWarning("The lengths of the points do not match!")

    if 'tap_dependent_impedance' not in net[trafotable]:
        net[trafotable]['tap_dependent_impedance'] = Series(index=net[trafotable].index, dtype=np.bool_, data=False)

    if col not in net[trafotable]:
        net[trafotable][col] = Series(index=net[trafotable].index, dtype="Int64")

    # set the flag for the trafo table
    net[trafotable].loc[trafo_index, 'tap_dependent_impedance'] = True

    if single_mode:
        zip_params = zip([trafo_index], [x_points], [y_points])
    else:
        zip_params = zip(trafo_index, x_points, y_points)

    for tid, x_p, y_p in zip_params:
        # create the characteristic and set its index in the trafotable
        s = SplineCharacteristic(net, x_p, y_p)
        net[trafotable].at[tid, col] = s.index

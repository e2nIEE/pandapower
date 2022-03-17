# -*- coding: utf-8 -*-
__author__ = 'smeinecke'

import copy
# import numpy as np
import pandas as pd
import uuid

from pandapower.io_utils import JSONSerializableClass
from pandapower.auxiliary import get_free_id, _preserve_dtypes
from pandapower.create import create_empty_network
from pandapower.toolbox import ensure_iterability, pp_elements

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Group(JSONSerializableClass):
    """
    Base class for groups of elements of pandapower networks.
    """

    def __init__(self, net, elms_dict, elm_col=None, name="", index=None,
                 overwrite=False, **kwargs):
        """
        Add a group to net['group'] dataframe.

        INPUT:
            **net** - pandapower net

            **elms_dict** (dict: key as string, values as pd.Index (even if elm_col is not None)) -
            dict of indices belonging to this group. Within initialization or by update_elms_dict()
            also other input type than pd.Index are handled.
            Attention: If you declare a group but forget to declare all connected elements although
            you wants to (e.g. declaring lines but forgetting to mention the connected switches),
            you may get problems after using drop_elms_and_group()
            or other functions. There are different pandapower toolbox functions
            which may help you to define 'elms_dict', such as get_connecting_branches(),
            get_inner_branches(), get_connecting_elms_dict().

        OPTIONAL:
            **elm_col** (str, None) - If given, the elms_dict should not refer to DataFrames index
            but to another column. It is highly relevant that the elm_col exists in all
            DataFrames of the grouped elements and have the same dtype.

            **name** (str, "") - name of the group

            **index** (int, None) - index for the dataframe net.group

            **overwrite** (bool, False) - whether the entry in net.group with the same index should
            be overwritten

            ****kwargs** - key word arguments
        """

        super().__init__()

        index = index if index is not None else get_free_id(net.group)
        self.index = index
        self.elms_dict = elms_dict
        self.elm_col = elm_col
        self.define_elm_type(net)
        self.update_elms_dict(net, verbose=kwargs.get("verbose", True))
        self.add_group_to_net(net, name, index, overwrite)
        logger.debug(f"Initialized Group '{name}'")

    def __str__(self):
        s = self.__class__.__name__
        return s

    def __repr__(self):
        return f"Group '{self.index}'"

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)

        for attr in self.json_excludes:
            try:
                del state[attr]
            except:
                continue

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self, other):
        return self.compare_group(other)

    def add_group_to_net(self, net, name, index, overwrite):
        dtypes = net.group.dtypes

        # use base class method to raise an error if the object is in DF and overwrite = False
        super().add_to_net(net=net, element='group', index=index, overwrite=overwrite)

        columns = ['name', 'object']
        net.group.loc[index, columns] = (name, self)

        _preserve_dtypes(net.group, dtypes)

    def define_elm_type(self, net):
        """
        Defines self.elm_type with respect to the net[elm].dtypes
        """
        if self.elm_col is not None:
            for i, (elm, _) in enumerate(self.elms_dict.items()):
                if self.elm_col not in net[elm].columns:
                    raise ValueError(f"elm_col '{self.elm_col}' doesn't exist in net[{elm}].")
                if i == 0:
                    if pd.api.types.is_integer_dtype(net[elm][self.elm_col]):
                        self.elm_type = "int"
                    elif pd.api.types.is_numeric_dtype(net[elm][self.elm_col]):
                        self.elm_type = "float"
                    else:  # pd.api.types.is_object_dtype(net[elm][self.elm_col])
                        self.elm_type = "object"
                elif self.elm_type == "int" and not pd.api.types.is_integer_dtype(
                        net[elm][self.elm_col]) or self.elm_type == "float" and not \
                        pd.api.types.is_numeric_dtype(net[elm][self.elm_col]) or \
                        self.elm_type == "object" and not pd.api.types.is_object_dtype(
                            net[elm][self.elm_col]):
                    raise ValueError(
                        f"The dtypes of net[*].{self.elm_col} does not fit toghether (* in [" +
                        ", ".join(self.elms_dict.keys()) + "]).")
        else:
            self.elm_type = "int"

    def get_elm_type(self):
        if self.elm_type == "int":
            return int
        elif self.elm_type == "float":
            return float
        elif self.elm_type == "object":
            return object
        else:
            raise NotImplementedError(f"self.elm_type {self.elm_type} is unknown.")

    def _update_elms_dict(self, net, elms_dict, elm_col, elm_type, verbose=True):
        """
        Internal function of update_elms_dict().
        """
        elms = list(elms_dict.keys())
        for elm in elms:
            if elm not in net.keys() or not isinstance(net[elm], pd.DataFrame):
                if verbose:
                    logger.info(f"{elm} is in elms_dict but is no dataframe in net")
                del elms_dict[elm]
            elif not net[elm].shape[0]:
                if verbose:
                    logger.info(f"net.{elm} has no entries, so elms_dict[{elm}] is removed.")
                del elms_dict[elm]
            else:
                elms_dict[elm] = pd.Index(ensure_iterability(elms_dict[elm]),
                                          dtype=elm_type).dropna()
                if elm_col is None:
                    new_idx = elms_dict[elm].intersection(net[elm].index)
                else:
                    new_idx = elms_dict[elm].intersection(pd.Index(set(net[elm][elm_col].astype(
                        elm_type))))
                if len(new_idx) != len(elms_dict[elm]) and verbose:
                    logger.info("In %s, the number of indices has been corrected from %i to %i." % (
                        elm, len(elms_dict[elm]), len(new_idx)))
                if len(new_idx):
                    elms_dict[elm] = new_idx
                else:
                    del elms_dict[elm]

    def update_elms_dict(self, net, verbose=True):
        """
        * Set correct types of elms_dict values (pandas.Index)
        * Remove element keys and elements indices if these are not available in net.
        """
        self._update_elms_dict(net, self.elms_dict, self.elm_col, self.get_elm_type(),
                               verbose=verbose)

    def get_idx(self, net, elm):
        """
        Returns indices of the requested element dataframe, considering self.elm_col.
        """
        if elm in self.elms_dict.keys():
            return self.elms_dict[elm] if self.elm_col is None else net[elm].index[net[elm][
                self.elm_col].isin(self.elms_dict[elm])]
        else:
            return pd.Index([], dtype=int)

    def set_elm_col(self, net, elm_col):
        """
        Sets new self.elm_col and updates self.elms_dict. If self.elms_dict is not up-to-date,
        update_elms_dict() is needed before using this function.
        """
        dupl_elms = list()
        elms_dict = dict()
        for elm in self.elms_dict.keys():

            # fill nan values in net[elm][elm_col] with unique names
            if elm_col not in net[elm].columns:
                net[elm][elm_col] = pd.Series([None]*net[elm].shape[0], dtype=object)
            if pd.api.types.is_object_dtype(net[elm][elm_col]):
                idxs = net[elm].index[net[elm][elm_col].isnull()]
                net[elm][elm_col].loc[idxs] = ["%s_%i_%s" % (elm, idx, str(uuid.uuid4())) for idx in
                                               idxs]

            # determine duplicated values which would corrupt Groups functionality
            if (net[elm][elm_col].duplicated() | net[elm][elm_col].isnull()).any():
                dupl_elms.append(elm)

            # update elms_dict
            if not len(dupl_elms):
                elms_dict[elm] = pd.Index(net[elm].loc[self.get_idx(net, elm), elm_col])

        if len(dupl_elms):
            raise ValueError(f"In net[*].{elm_col} are duplicated or nan values - * in {dupl_elms}")

        # final update
        self.elm_col = elm_col
        self.elms_dict = elms_dict
        self.define_elm_type(net)

    def set_value(self, net, value, column, replace=True, append_column=True):
        """
        Sets for all elements of the group the same value to the column of the element tables.
        """
        for elm, idx in self.elms_dict.items():
            if append_column or column in net[elm].columns:
                if self.elm_col is None:
                    if replace or column not in net[elm].columns:
                        ix = idx
                    else:
                        ix = net[elm].loc[idx].index[net[elm][column].loc[idx].isnull()]
                else:
                    if replace or column not in net[elm].columns:
                        ix = net[elm].index[net[elm][self.elm_col].isin(idx)]
                    else:
                        ix = net[elm].index[net[elm][self.elm_col].isin(idx) &
                                            net[elm][column].isnull()]
                net[elm].loc[ix, column] = value

    def set_in_service(self, net):
        """
        Sets all elements of the group in service.
        """
        self.set_value(net, True, "in_service", replace=True, append_column=False)

    def set_out_of_service(self, net):
        """
        Sets all elements of the group out of service.
        """
        self.set_value(net, False, "in_service", replace=True, append_column=False)

    def drop_elms_and_group(self, net):
        """
        Drops all elements of the group and in net.group the group itself.
        """
        # functions like drop_trafos, drop_lines, drop_buses are not considered since all elements
        # should be included in elms_dict
        for elm in self.elms_dict.keys():
            idx = self.get_idx(net, elm)
            net[elm].drop(idx, inplace=True)
            res_elm = "res_" + elm
            if res_elm in net.keys() and net[res_elm].shape[0]:
                net[res_elm].drop(net[res_elm].index.intersection(idx), inplace=True)
        self.compare_group(net.group.object.at[self.index], raise_false_str="self.index must be "
                           "outdated, since this group object doesn't match.")
        net.group.drop(self.index, inplace=True)

    def return_group_as_net(self, net, keep_everything_else=False, **kwargs):
        """
        Returns a pandapower net consisting of the members of this group.
        """
        if keep_everything_else:
            group_net = copy.deepcopy(net)
            for elm in pp_elements():
                if group_net[elm].shape[0]:
                    keep = self.get_idx(net, elm) if elm in self.elms_dict.keys() else pd.Index(
                        [], dtype=self.get_elm_type())
                    group_net[elm].drop(group_net[elm].index.difference(keep), inplace=True)
            if net.group.shape[0] > 1:
                logger.warning("The returned net includes further groups which should probably be "
                               "updated.")
        else:
            self.compare_group(net.group.object.at[self.index], raise_false_str="self.index must "
                               "be outdated, since this group object doesn't match.")
            group_net = create_empty_network(
                name=net.group.name.at[self.index], f_hz=net.f_hz, sn_mva=net.sn_mva,
                add_stdtypes=kwargs.get("add_stdtypes", True))
            group_net["group"] = net.group.loc[[self.index]]
            for elm in self.elms_dict.keys():
                idx = self.get_idx(net, elm)
                group_net[elm] = net[elm].loc[idx]
        return group_net

    def append_to_group(self, elms_dict_to_append, net=None):
        """
        Appends the group by the elements given as dict of indices in 'elms_dict_to_append'.
        If net is given, elms_dict_to_append is checked and updated by _update_elms_dict().
        """
        if net is not None:
            self._update_elms_dict(net, elms_dict_to_append, self.elm_col, self.get_elm_type())
        for elm, idx in elms_dict_to_append.items():
            if elm in self.elms_dict.keys():
                self.elms_dict[elm] = pd.Index(ensure_iterability(
                    elms_dict_to_append[elm]), dtype=self.get_elm_type()).dropna().union(
                        self.elms_dict[elm])
            else:
                self.elms_dict[elm] = pd.Index(ensure_iterability(
                    elms_dict_to_append[elm]), dtype=self.get_elm_type()).dropna()

    def drop_from_group(self, elms_dict_to_drop):
        """
        Drops elements that are given in elms_dict_to_drop from this group. This is the reverse
        function of append_to_group().
        """
        for elm, idx2drop in elms_dict_to_drop.items():
            if elm in self.elms_dict.keys():
                self.elms_dict[elm] = self.elms_dict[elm].drop(self.elms_dict[elm].intersection(
                    pd.Index(ensure_iterability(idx2drop), dtype=self.get_elm_type()).dropna()))
                if not len(self.elms_dict[elm]):
                    del self.elms_dict[elm]

    def compare_group(self, other_group_object, raise_false_str=None):
        """ Returns a boolean whether this group has equal attributes as the passed
        'other_group_object'. """
        def return_raise_false(raise_false_str):
            if isinstance(raise_false_str, str):
                raise ValueError(raise_false_str)
            else:
                return False

        ogo = other_group_object
        for key, val in vars(self).items():
            if key not in vars(ogo).keys() or type(val) != type(vars(ogo)[key]):
                return_raise_false(raise_false_str)
            if key == "elms_dict":
                if not self.compare_elms_dict(vars(ogo)[key], verbose=False):
                    return_raise_false(raise_false_str)
            else:
                try:
                    if not val == vars(ogo)[key]:
                        return_raise_false(raise_false_str)
                except:
                    if not all(val == vars(ogo)[key]):
                        return_raise_false(raise_false_str)
        return True

    def compare_elms_dict(self, elms_dict_to_compare_with, verbose=True):
        """
        Compares the elements of this group with the given ones in elms_dict_to_compare_with.
        If both include the same, True is returned; otherwise False. However, if return_list is
        True, a list of differing elements is returned.

        OUTPUT:
            bool: whether self.elms_dict and elms_dict_to_compare_with include the same elements
        """
        ed1 = self.elms_dict
        ed2 = dict()
        for elm, idx in elms_dict_to_compare_with.items():
            pd_idx = pd.Index(ensure_iterability(idx), dtype=self.get_elm_type()).dropna()
            if len(pd_idx):
                ed2[elm] = pd_idx

        # --- determine differences
        key_diff1 = set(ed1.keys()) - set(ed2.keys())
        key_diff2 = set(ed2.keys()) - set(ed1.keys())
        idx_diff = [key for key in set(ed2.keys()) - key_diff2 if len(
            ed1[key].symmetric_difference(ed2[key]))]

        # --- loggings
        if len(key_diff1) and verbose:
            logger.info("Missing element keys in elms_dict_to_compare_with: " + str(key_diff1))
        if len(key_diff2) and verbose:
            logger.info("Missing element keys in self.elms_dict: " + str(key_diff2))
        if len(idx_diff) and verbose:
            logger.info("Differing indices in " + str(idx_diff))

        return not (len(key_diff1) or len(key_diff2) or len(idx_diff))

    def elm_counts(self):
        """ Returns a Series concluding the number of included elements in self.elms_dict """
        ser = pd.Series([], dtype=int)
        for elm, idx in self.elms_dict.items():
            ser.loc[elm] = len(idx)
        return ser

    def _sum_powers(self, net, formula_character, unit):
        power = 0.
        missing_res_idx = list()
        no_power_column_found = list()
        for elm in self.elms_dict.keys():
            if elm in ["switch", "measurement", "bus"]:
                continue
            idx = self.get_idx(net, elm)
            res_elm = "res_" + elm
            res_idx = net[res_elm].index.intersection(idx)
            sign = 1 if elm not in ["ext_grid", "gen", "sgen"] else -1
            if len(res_idx) != len(idx):
                missing_res_idx.append(elm)
            col1 = "%s_%s" % (formula_character, unit)
            col2 = "%sl_%s" % (formula_character, unit)
            if col1 in net[res_elm].columns:
                power += sign * net[res_elm][col1].loc[res_idx].sum()
            elif col2 in net[res_elm].columns:
                power += sign * net[res_elm][col2].loc[res_idx].sum()
            else:
                no_power_column_found.append(elm)

        if len(missing_res_idx):
            logger.warning("The resulting power may be wrong since in the results tables of these "
                           "elements lack of indices: " + str(missing_res_idx))
        if len(no_power_column_found):
            logger.warning("The resulting power may be wrong since in the results tables of these "
                           "elements no power column was found: " + str(no_power_column_found))
        return power

    def res_p_mw(self, net):
        return self._sum_powers(net, "p", "mw")

    def res_q_mvar(self, net):
        return self._sum_powers(net, "q", "mvar")


def update_group_indices(net):
    """ Updates the index value of all group objects in net.group. """
    if "group" in net:
        for idx in net.group.index:
            net.group.object.at[idx].index = idx


if __name__ == '__main__':
    from pandapower.networks import case24_ieee_rts

    net = case24_ieee_rts()
    gr1 = Group(net, {"gen": [0, 1], "sgen": [2, 3], "load": [0]}, name='1st Group', index=2)
    gr2 = Group(net, {"trafo": net.trafo.index}, name='Group of transformers')

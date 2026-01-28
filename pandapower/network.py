# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# Additional copyright for modified code by Brendan Curran-Johnson (ADict class):
# Copyright (c) 2013 Brendan Curran-Johnson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# (https://github.com/bcj/AttrDict/blob/master/LICENSE.txt)

import copy
import logging
from typing import overload
from typing_extensions import deprecated
from collections.abc import MutableMapping

import numpy as np
import pandas as pd

from pandapower.network_structure import get_structure_dict
from pandapower.std_types import add_basic_std_types

logger = logging.getLogger(__name__)


def plural_s(number):
    return "" if number == 1 else "s"


def _preserve_dtypes(df, dtypes):
    for item, dtype in list(dtypes.items()):
        if df.dtypes.at[item] != dtype:
            if (dtype == bool or dtype == np.bool_) and np.any(df[item].isnull()):
                raise UserWarning(f"Encountered NaN value(s) in a boolean column {item}! "
                                  f"NaN are casted to True by default, which can lead to errors. "
                                  f"Replace NaN values with True or False first.")
            try:
                df[item] = df[item].astype(dtype)
            except ValueError:
                df[item] = df[item].astype(float)


class ADict(dict, MutableMapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # to prevent overwrite of internal attributes by new keys
        # see _valid_name()
        self._setattr('_allow_invalid_attributes', False)

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

    # --- taken from AttrDict

    def __getstate__(self):
        return self.copy(), self._allow_invalid_attributes

    def __dir__(self):
        return list(self.keys())

    def __setstate__(self, state):
        mapping, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    @classmethod
    def _constructor(cls, mapping):
        return cls(mapping)

    # --- taken from MutableAttr

    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping (i.e. internals)
        """
        super(MutableMapping, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.

        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping (i.e. internals)
        """
        super(MutableMapping, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.

        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )

    def __call__(self, key):
        """
        Dynamically access a key-value pair.

        key: A key associated with a value in the mapping.

        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __deepcopy__(self, memo):
        """
        overloads the deepcopy function of pandapower if at least one DataFrame with column
        "object" is in net

        reason: some of these objects contain a reference to net which breaks the default deepcopy
        function. Also, the DataFrame doesn't deepcopy its elements if geodata changes in the
        lists, it affects both net instances
        This fix was introduced in pandapower 2.2.1

        """
        deep_columns = {'object', 'coords', 'geometry'}
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            if isinstance(v, pd.DataFrame) and not set(v.columns).isdisjoint(deep_columns):
                if k not in result:
                    result[k] = v.__class__(index=v.index, columns=v.columns)
                for col in v.columns:
                    if col in deep_columns:
                        result[k][col] = v[col].apply(lambda x: copy.deepcopy(x, memo))
                    else:
                        result[k][col] = copy.deepcopy(v[col], memo)
                _preserve_dtypes(result[k], v.dtypes)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        result._setattr('_allow_invalid_attributes', self._allow_invalid_attributes)
        return result

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.

        A key may be used as an attribute if:
         * It is a string
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
                isinstance(key, str) and
                not hasattr(cls, key)
        )


class pandapowerNet(ADict):
    """
    pandapowerNet constructor
    
    Parameters:
        name: Network name
        f_hz: power system frequency in hertz, default: 50.
        sn_mva: reference apparent power for per unit system, default: 1.
        add_stdtypes: Includes standard types to net, dafault: True
        custom_data: custom data to add to the network

    Example:
        net = pandapowerNet(name="My Network")
    """
    @overload
    def __init__(
            self, name: str, f_hz: float = 50., sn_mva: float = 1., add_stdtypes: bool = True, custom_data: dict = None
    ) -> None: ...
    
    @overload
    @deprecated("Calling pandapowerNet to copy a network is no longer supported. Use copy.deepcopy(net) instead.")
    def __init__(self, net: "pandapowerNet") -> None: ...
    
    
    def __init__(
            self,
            net: "pandapowerNet" = None,
            name: str = None,
            f_hz: float = 50.,
            sn_mva: float = 1.,
            add_stdtypes: bool = True,
            custom_data: dict = None,
            **kwargs
    ) -> None:
        # TODO: remove once deprecations are removed
        if net is not None:
            if name is not None or f_hz != 50. or sn_mva != 1. or not add_stdtypes or custom_data is not None:
                raise AttributeError(
                    'Passing net and other attributes is not supported. Do not pass a net to pandapowerNet()'
                )
            super().__init__(net, **kwargs)
            if isinstance(net, self.__class__):
                self.clear()
                self.update(**copy.deepcopy(net))
    
            for key in self:
                if isinstance(self[key], list) and len(self[key]) == 1:
                    self[key] = self[key][0]
        if net is None and name is not None:
            # ---- new code for the remaining version ----
            super().__init__(**kwargs)
            if name == "":
                logger.warning("When calling pandapowerNet() name should not be empty.")
            network_structure_dict = get_structure_dict()
            network_structure_dict["name"] = name
            network_structure_dict["f_hz"] = f_hz
            network_structure_dict["sn_mva"] = sn_mva
            
            # create dataframes from network_structure_dict
            data = pandapowerNet.create_dataframes(network_structure_dict)
            # set data on self
            for key, value in data.items():
                self[key] = value
                # to avoid creating dataframes where dicts are required they are stored in single entry lists and
                #  unpacked here
                if isinstance(value, list) and len(value) == 1:
                    self[key] = value[0]
            
            # add custom data
            if custom_data is not None:
                for key, value in custom_data.items():
                    self[key] = value
        
            self._empty_res_load_3ph = self._empty_res_load
            self._empty_res_sgen_3ph = self._empty_res_sgen
            self._empty_res_storage_3ph = self._empty_res_storage
        
            if add_stdtypes:
                add_basic_std_types(self)  # TODO: Test this
            else:
                self.std_types = {"line": {}, "line_dc": {}, "trafo": {}, "trafo3w": {}, "fuse": {}}  # TODO: this should not be set here. a function to return the empty std_types would be sensible
            # reset res_… objects:
            for suffix in [None, "est", "sc", "3ph"]:
                elements = []
                match suffix:
                    case "sc":
                        elements = ["bus", "line", "trafo", "trafo3w", "ext_grid", "gen", "sgen", "switch"]
                    case "est":
                        elements = ["bus", "line", "trafo", "trafo3w", "impedance", "switch", "shunt"]
                    case "3ph":
                        elements = [
                            "bus", "line", "trafo", "ext_grid", "shunt", "load", "sgen", "storage", "asymmetric_load",
                            "asymmetric_sgen"
                        ]
                    case None:
                        elements = [
                            "bus", "bus_dc", "line", "line_dc", "trafo", "trafo3w", "impedance", "ext_grid", "load",
                            "load_dc", "motor", "sgen", "storage", "shunt", "gen", "ward", "xward", "dcline",
                            "asymmetric_load", "asymmetric_sgen", "source_dc", "switch", "tcsc", "svc", "ssc", "vsc",
                            "b2b_vsc"
                        ]
                for element in elements:  # FIXME: is this related to @heckstrahler removing res_ if empty?
                    res_element = f"res_{element}" if suffix is None else f"res_{element}_{suffix}"
                    res_empty_element = f"_empty_{f'res_{element}' if suffix == 'est' else res_element}"
                    if res_empty_element in self:
                        self[res_element] = self[res_empty_element].copy()
                    else:
                        self[res_element] = pd.DataFrame(
                            columns=pd.Index([], dtype=object), index=pd.Index([], dtype=np.int64)
                        )
                if "res_cost" in self.keys():
                    del self["res_cost"]
            self.user_pf_options = {}
            

    @staticmethod
    def create_dataframes(data):
        for key in data:  # TODO: change index dtype to np.uint32
            if isinstance(data[key], dict):
                data[key] = pd.DataFrame(columns=data[key].keys(), index=pd.Index([], dtype=np.int64)).astype(data[key])
        return data

    def __repr__(self):  # pragma: no cover
        """
        See Also
        --------
        count_elements
        """
        par = []
        res = []
        for et in list(self.keys()):
            if not et.startswith("_") and isinstance(self[et], pd.DataFrame) and len(self[et]) > 0:
                n_rows = self[et].shape[0]
                if 'res_' in et:
                    res.append(f"   - {et} ({n_rows} element{plural_s(n_rows)})")
                elif et == 'group':
                    n_groups = len(set(self[et].index))
                    par.append(f"   - {et} ({n_groups} group{plural_s(n_groups)}, {n_rows} row{plural_s(n_rows)})")
                else:
                    par.append(f"   - {et} ({n_rows} element{plural_s(n_rows)})")
        res_cost = [" and the following result values:",
                    "   - %s" % "res_cost"] if "res_cost" in self.keys() else []
        if not len(par) + len(res):
            return "This pandapower network is empty"
        if len(res):
            res = [" and the following results tables:"] + res
        lines = ["This pandapower network includes the following parameter tables:"] + par + res + res_cost
        return "\n".join(lines)

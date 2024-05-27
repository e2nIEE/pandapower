# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.controller.const_control import ConstControl
try:
    from pandaplan.core import pplog
except:
    import logging as pplog
import numpy as np

logger = pplog.getLogger(__name__)


class PQController(ConstControl):
    """
    Parent class of all Controllers designed to control P and/or Q of an PQ-element.
    Can be used for loads, shunts and sgen.
    NOT applicable for gens.
    Contains general initialization and
    functions for getting and setting values from/to the data-structure.
    Furthermore it contains the convergence check in respect to the
    criteria it was created with.

    INPUT:
        **net** (attrdict) - pandapower network

        **gid** (int[]) - IDs of the controlled elements

    OPTIONAL:

        **max_p_error** (float, 0.0001) - Maximum error of active power

        **max_q_error** (float, 0.0001) - Maximum error of reactive power

        **p_ac** (float, 1.0) - Simultaneity factor applied to P and Q

        **f_sizing** (float, 1.0) - Sizing of the converter factor limiting P

        **data_source** ( , None) - A DataSource that contains profiles

        **profile_scale** (float, 1.0) - A scaling factor applied to the values of profiles

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **ts_absolute** (bool, True) - Whether the time step values are absolute power values or
        scaling factors

    """

    def __init__(self, net, gid, element="sgen", max_p_error=0.0001, max_q_error=0.0001, p_ac=1.,
                 f_sizing=1., data_source=None, profile_scale=1., in_service=True, ts_absolute=True,
                 order=0, level=0, **kwargs):
        super().__init__(net, element=element, variable="p_mw", element_index=gid,
                         in_service=in_service, order=order, level=level, **kwargs)

        # read attributes from net
        self.gid = gid
        self.element = element
        self.bus = net[self.element]["bus"][gid]
        self.p_mw = net[self.element]["p_mw"][gid]
        self.q_mvar = net[self.element]["q_mvar"][gid]
        self.sn_mva = net[self.element]["sn_mva"][gid]
        self.name = net[self.element]["name"][gid]
        self.gen_type = net[self.element]["type"][gid]
        self.element_in_service = net[self.element]["in_service"][gid]

        self.sign = 1
        if element == "sgen":
            self.sign *=-1
        # Simultaneity factor
        self.p_ac = p_ac

        # Sizing of the AC/DC converter
        self.f_sizing = f_sizing

        # attributes for profile
        self.data_source = data_source
        self.profile_scale = profile_scale
        self.p_profile = None
        self.q_profile = None
        self.ts_absolute = ts_absolute

        # Init variables for convergence check
        self.max_p_error = max_p_error
        self.max_q_error = max_q_error

        # Init curtailment
        self.p_curtailment = 0

        self.set_recycle(net)

    def set_p_profile(self, p_profile, profile_from_name):
        if profile_from_name:
            if p_profile:
                logger.warning("Given parameter 'p_profile' will be discarded "
                               "since 'profile_from_name' has been set to True.")
            self.p_profile = [f"P_{name}" for name in self.name]
        else:
            self.p_profile = p_profile

    def set_q_profile(self, q_profile, profile_from_name):
        if profile_from_name:
            if q_profile:
                logger.warning("Given parameter 'q_profile' will be discarded "
                               "since 'profile_from_name' has been set to True.")
            self.q_profile = [f"Q_{name}" for name in self.name]
        else:
            self.q_profile = q_profile

    def __repr__(self):
        rep = super(PQController, self).__repr__()

        for member in ["gid", "bus", "p_mw", "q_mvar", "sn_mva", "name", "gen_type",
                       "element_in_service", "p_ac",
                       "f_sizing", "data_source", "profile_scale", "p_profile", "q_profile",
                       "ts_absolute", "max_p_error",
                       "max_q_error"]:
            rep += ("\n" + member + ":").ljust(20)
            d = locals()
            exec('value = self.' + member, d)
            rep += str(d['value'])

        return rep

    def read_profiles(self, time):
        """
        Reads new data if a DataSource and profiles have been set.
        The simultaneity factor p_ac is being applied directly to the value
        retrieved from the DataSource. self.profile_scale in turn is being
        passed to get_time_step_value() and applied by the DataSource.
        """

        if self.data_source is not None:
            if self.p_profile or self.p_profile == 0:
                self.p_mw = self.p_ac * \
                            self.data_source.get_time_step_value(time_step=time,
                                                                 profile_name=self.p_profile,
                                                                 scale_factor=self.profile_scale)

            if self.q_profile or self.q_profile == 0:
                self.q_mvar = self.p_ac * \
                              self.data_source.get_time_step_value(time_step=time,
                                                                   profile_name=self.q_profile,
                                                                   scale_factor=self.profile_scale)

            if not self.ts_absolute:
                if self.sn_mva.isnull().any():
                    logger.error(f"There are PQ controlled elements with NaN sn_mva values.")
                self.p_mw = self.p_mw * self.sn_mva
                self.q_mvar = self.q_mvar * self.sn_mva

            # store provided power until controller converged and subtract in finalize_step
            self.p_curtailment = self.p_mw

    def write_to_net(self, net):
        # write p, q to bus within the net
        net[self.element].loc[self.gid, "p_mw"] = self.p_mw
        net[self.element].loc[self.gid, "q_mvar"] = self.q_mvar

    def finalize_control(self, net):
        self.calc_curtailment()

    def calc_curtailment(self):
        # p_curtailment contains the provided power up to here,
        # now the actual power is subtracted to store the curtailment
        try:
            self.p_curtailment -= self.p_mw
        except AttributeError:
            logger.error("No p_kw present at %s. Assuming no curtailment." % self)

    def limit_to_inverter_sizing(self, p_mw, q_mvar):
        """
        Returns the limited P and Q values in respect to the PVs inverter sizing

        INPUT:
            **p_kw** - Active power of [self.element]

            **q_kvar** - Reactive power of [self.element]

        OUTPUT:

            **p_kw** - Active power limited to inverter sizing

            **q_kvar** - Reactive power limited to inverter sizing
        """
        if n_nan_sn := sum(self.sn_mva.isnull()):
            logger.warning(f"Limiting to inverter size will result in NaN for the {n_nan_sn} PQ "
                           "controlled elements with NaN sn_mva values.")
        # limit output to inverter sizing
        if (np.sqrt(p_mw ** 2 + q_mvar ** 2) > self.f_sizing * self.sn_mva).any():
           # limit Q first
            # t = q_mvar
            # q_mvar = sign(q_mvar) * math.sqrt((self.f_sizing * self.sn_mva) ** 2 - p_mw ** 2)
            sn_mva = self.f_sizing *self.sn_mva
            max_q_mvar = np.sqrt(sn_mva ** 2 - p_mw ** 2)
            q_mvar = np.min((q_mvar*self.sign, max_q_mvar*self.sign))

            logger.debug("Note: Q_mvar has been limited"
                             "respect to the inverter sizing.")
            if np.isnan(max_q_mvar).any():  # limit P if it already exceeds Sn
                q_mvar = 0
                p_mw = self.f_sizing * self.sn_mva
                logger.debug("Note: P_mw has been limited and"
                             "q_mvar has been set to 0 in respect to sn_mva ")
        return p_mw, q_mvar

    def initialize_control(self, net):
        """
        Reading the actual P and Q state from the respective [self.element] in the net
        """
        self.p_mw = net[self.element].loc[self.gid, "p_mw"]
        self.q_mvar = net[self.element].loc[self.gid, "q_mvar"]

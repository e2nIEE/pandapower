# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.control.controller.pq_control import PQController
from pandapower.control.controller.DERController.DERBasics import BaseModel
from pandapower.control.controller.DERController.QModels import QModel
from pandapower.control.controller.DERController.PQVAreas import BasePQVArea

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class SArea(BaseModel):
    """
    Class to allow Sn based saturation.
    """

    def __init__(self, q_prio):
        self.q_prio = q_prio

    def step(self, p, q):
        # Saturation on SnMVA according to priority mode
        if p**2 + q**2 > 1:
            if self.q_prio:
                q = np.clip(q, -1, 1)
                p = np.sqrt(1-q**2)
            else:
                p = np.clip(p, 0, 1)
                q = np.sqrt(1-p**2) * np.sign(q)
        return p, q


# -------------------------------------------------------------------------------------------------
""" DERModel """
# -------------------------------------------------------------------------------------------------


class DERModel:
    """
    Class used to model a DERController
    """

    def __init__(self, sgen_sn_mva, q_model=None, pqv_area=None, saturate_sn_mva=True, q_prio=True):
        self.sn_mva = sgen_sn_mva
        self.q_model = None

        self.saturation = saturate_sn_mva or (pqv_area is not None)
        self.q_prio = q_prio
        self.pqv_area = None
        self.saturator_sn_mva = None
        if saturate_sn_mva:
            self.saturator_sn_mva = SArea(q_prio)

        if q_model is not None:
            if isinstance(q_model, QModel):
                self.q_model = q_model
            else:
                logger.error("Q model not available")

        if pqv_area is not None:
            if isinstance(pqv_area, BasePQVArea):
                self.pqv_area = pqv_area
            else:
                logger.error("Error! PQ saturation area not available")
                raise Exception("Error! PQ saturation area not available")

    def __str__(self):
        return "q_model:" + str(self.q_model) +\
                ", saturation:" + str(self.saturation) +\
                ", q_priority:" + str(self.q_prio) +\
                ", pq_area:" + str(self.pq_area)

    def step(self, vm_pu, p_series_mw=None, q_series_mvar=None,
             p_setpoint_mw=np.NaN, q_setpoint_mvar=np.NaN):
        if p_series_mw is None:
            p_series_mw = self.sn_mva
        # p_series_mw is forced to be greater/equal zero
        p_series_mw = 0 if p_series_mw < 0 else p_series_mw

        """First Step: Calculate/Select P, Q"""
        p = self._step_p(p_series_mw=p_series_mw, p_setpoint_mw=p_setpoint_mw)
        q = self._step_q(p_series_mw=p_series_mw, q_series_mvar=q_series_mvar,
                         q_setpoint_mvar=q_setpoint_mvar, vm_pu=vm_pu)

        """Second Step: Saturates P, Q according to SnMVA/PQ_AREA"""
        if self.saturation:
            p, q = self._saturate(p=p, q=q, vm_pu=vm_pu)

        """Third Step: Convert relative P, Q to P_mw, Q_mvar """
        p_mw, q_mvar = p * self.sn_mva, q * self.sn_mva
        if not np.isscalar(p_mw) or not np.isscalar(q_mvar):
            p_mw = np.asscalar(p_mw)
            q_mvar = np.asscalar(q_mvar)
        return p_mw, q_mvar

    def _step_p(self, p_series_mw=None, p_setpoint_mw=None):
        p = p_setpoint_mw/self.sn_mva if p_setpoint_mw is not None and np.isfinite(p_setpoint_mw)\
            else p_series_mw/self.sn_mva
        if p > p_series_mw/self.sn_mva:
            p = p_series_mw/self.sn_mva
        return p

    def _step_q(self, p_series_mw=None, q_setpoint_mvar=None, q_series_mvar=None, vm_pu=None):
        """Q priority: Q setpoint > Q model > Q series"""
        if q_setpoint_mvar is not None and np.isfinite(q_setpoint_mvar):
            q = q_setpoint_mvar/self.sn_mva
        elif self.q_model is not None:
            q = self.q_model.step(vm_pu=vm_pu, p=p_series_mw/self.sn_mva)
        else:
            if q_series_mvar is None:
                raise Exception("No Q_model and no q_profile available")
            q = q_series_mvar/self.sn_mva
        return q

    def _saturate(self, vm_pu=None, p=None, q=None):
        assert p is not None and q is not None

        # Saturation on given pq_area
        if self.pqv_area is not None:
            if not self.pqv_area.pq_in_range(p=p, q=q, vm_pu=vm_pu):
                q_flex = self.pqv_area.q_flexibility(p=p, vm_pu=vm_pu)
                q = np.clip(q, *q_flex)

        if self.saturator_sn_mva is not None:
            p, q = self.saturator_sn_mva.step(p, q)
        return p, q


# -------------------------------------------------------------------------------------------------
""" DERController """
# -------------------------------------------------------------------------------------------------


class DERController(PQController):
    """
    Flexible controller to model plenty types of DER control characteristics, such as Q(V), Q(P),
    cosphi(P), Q(P, V), and restrict the behavior to defined PQV areas.

    INPUT:
        **net** (pandapower net)

        **gid** (int) - index of the controlled element

    OPTIONAL:
        **element** (str, "sgen") - element type which is controlled

        **q_model"" (object, None) - an q_model, such as provided in this file, should be passed to
        model how the q value should be determined.

        **pqv_area** (object, None) - an pqv_area, such as provided in this file, should be passed
        to model q values are allowed.
    """

    def __init__(self, net, gid, element="sgen",
                 q_model=None, pqv_area=None,
                 saturate_sn_mva=True, q_prio=True, damping_coef=2,
                 max_p_error=1e-6, max_q_error=1e-6, p_ac=1., f_sizing=1.,
                 data_source=None, p_profile=None, profile_from_name=False,
                 profile_scale=1.0, in_service=True, ts_absolute=True,
                 order=0, level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"gid": gid}
        super().__init__(net, gid=gid, element=element, max_p_error=max_p_error,
                         max_q_error=max_q_error, p_ac=p_ac,
                         f_sizing=f_sizing, data_source=data_source,
                         profile_scale=profile_scale, in_service=in_service, initial_run=True,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_powerflow=False,
                         order=order, level=level, **kwargs)

        # Init DER Model
        self.der_model = DERModel(self.sn_mva,
                                  q_model=q_model, pqv_area=pqv_area,
                                  saturate_sn_mva=saturate_sn_mva, q_prio=q_prio)
        self.damping_coef = damping_coef
        self.set_p_profile(p_profile, profile_from_name)

    def time_step(self, net, time):
        # get new values from profiles
        self.read_profiles(time)
        self.p_series_mw = self.p_mw
        self.q_series_mvar = self.q_mvar

#        self.write_to_net(net)

    def is_converged(self, net):
        v = net.res_bus.at[self.bus, "vm_pu"]

        target_p_mw, target_q_mvar =\
            self.der_model.step(
                vm_pu=v, p_series_mw=self.p_mw if "p_series_mw" not in vars(self) else
                self.p_series_mw,
                q_series_mvar=self.q_mvar if "q_series_mvar" not in vars(self) else
                self.q_series_mvar)

        # Apply damping factor
        self.target_p_mw = self.p_mw + (target_p_mw - self.p_mw) / self.damping_coef
        self.target_q_mvar = self.q_mvar + (target_q_mvar - self.q_mvar) / self.damping_coef

        return np.isclose(self.target_q_mvar, self.q_mvar, atol=self.max_q_error) and\
            np.isclose(self.target_p_mw, self.p_mw, atol=self.max_p_error)

    def control_step(self, net):
        self.p_mw, self.q_mvar = self.target_p_mw, self.target_q_mvar

        self.write_to_net(net)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandapower as pp
    try:
        import simbench as sb
    except ImportError as e:
        e += "\nFor this example code, simbench is used. Please install it via `pip install simbench`."
        raise ImportError(e)

    def plot_voltage_range(res_data):
        assert ("res_bus.vm_pu") in res_data.keys()
        result = res_data["res_bus.vm_pu"]
        result.dropna(axis=1, how="all", inplace=True)
        result["average_v"] = result.values.mean(axis=1)
        result["max_v"] = result.drop(["average_v"], axis=1).values.max(axis=1)
        result["min_v"] = result.drop(["average_v"], axis=1).values.min(axis=1)
        fig = plt.figure()
        ax_v = fig.add_subplot()
        result.plot(y=["average_v", "max_v", "min_v"], ax=ax_v, color=["blue", "green", "red"])
        ax_v.fill_between(result.index, result["max_v"], result["min_v"], facecolor="lightgray")
        ax_v.set_xlabel("Zeitschritt [15 min]", fontsize=20)
        ax_v.set_ylabel(r'Spannung [p.u]', fontsize=20)
        ax_v.xaxis.set_tick_params(labelsize=18)
        ax_v.yaxis.set_tick_params(labelsize=18)
        ax_v.legend(fontsize=18, frameon=False, loc='best', ncol=len(result))
        ax_v.grid()
        plt.show()

    sb_code = "1-MV-urban--1-sw"
    num_calc = 100
    net = sb.get_simbench_net(sb_code)
    pp.runpp(net)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    for key, profile in profiles.items():
        profiles[key] = profile.iloc[4000:4100, :].reset_index(drop=True)
#        profiles[key] = profile.iloc[200:300, :].reset_index(drop=True)

    sb.apply_const_controllers(net, profiles)
    ds = pp.timeseries.DFData(profiles[("sgen", "p_mw")])
    time_steps = np.arange(num_calc)

    for idx in net.sgen.index:

        # Attetion q is NOT in MVAR but relative to SnMVA
        # c = DERController(net, idx, q_model=QModelConstQ(q=0.1),
        #               pqv_area=PQVArea4120V2(), saturate_sn_mva=True, q_prio=True,
        #               data_source=ds)

        c = DERController(net, idx, q_model=pp.toolbox.DERController.QModelQV(
            qv_curve=pp.toolbox.DERController.QVCurve(
                v_points_pu=(0, 0.93, 1.05, 1.1, 2),
                q_points=(0.312, 0.312, 0, -0.312, -0.312)
            )), pqv_area=pp.toolbox.DERController.PQVArea4120V2(), saturate_sn_mva=True,
            q_prio=True, data_source=ds)
        c.set_p_profile(idx, False)

    ow = pp.timeseries.output_writer.OutputWriter(net=net, time_steps=time_steps)
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_sgen', 'q_mvar')
    pp.timeseries.run_timeseries(net, time_steps)

    res_data = net.output_writer.iloc[0, 0].output
    # plot_voltage_range(res_data)

    result_q = res_data["res_sgen.q_mvar"]
    result_u = res_data['res_bus.vm_pu']
    plt.figure()
    ax = plt.gca()
    for s in net.sgen.index:
        idx_bus = net.sgen.bus.loc[s]
        plt.scatter(result_u[idx_bus], result_q[s]/net.sgen.sn_mva[s])
    ax.set_xlabel("u [pu]", fontsize=20)
    ax.set_ylabel(r'Q [pu]', fontsize=20)

    result_p = res_data["res_sgen.p_mw"]
    result_q = res_data["res_sgen.q_mvar"]
    plt.figure()
    ax = plt.gca()
    # for s in net.sgen.index:
    s = 1
    plt.scatter(result_p.loc[:, s]/net.sgen.sn_mva[s], result_q.loc[:, s]/net.sgen.sn_mva[s])
    ax.set_xlabel("P [pu]", fontsize=20)
    ax.set_ylabel(r'Q [pu]', fontsize=20)

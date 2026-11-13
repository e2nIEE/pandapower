# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
from itertools import pairwise

import networkx as nx
import numpy as np
import pandas as pd

from pandapower.auxiliary import read_from_net
from pandapower.control.controller.trafo.continuous_tap_control import ContinuousTapControl
from pandapower.control.controller.trafo.discrete_tap_control import DiscreteTapControl
from pandapower.io_utils import JSONSerializableClass
from pandapower.topology import create_nxgraph

logger = logging.getLogger(__name__)


class LineDropCompensation:
    """
    Mixin for tap changer controllers with line drop compensation (LDC).

    A tap changer with line drop compensation does not control the voltage at the bus of the transformer itself, but
    the (estimated) voltage at a remote point of the feeder. To this end, the power flowing through the transformer is
    used to calculate the voltage drop over a fictitious line between the transformer bus and the remote point:

    .. math::

        V_\\mathrm{comp} = V_\\mathrm{trafo}
        - \\frac{P \\cdot R_\\mathrm{comp} + Q \\cdot X_\\mathrm{comp}}{V_\\mathrm{trafo} \\cdot V_\\mathrm{nom}}

    with the power flow :math:`P`, :math:`Q` from the transformer into the feeder at the controlled side, the
    fictitious line impedance :math:`R_\\mathrm{comp}`, :math:`X_\\mathrm{comp}` in ohm, the measured voltage
    :math:`V_\\mathrm{trafo}` and the rated voltage :math:`V_\\mathrm{nom}` of the controlled bus in kV. The
    controller then regulates :math:`V_\\mathrm{comp}` to the set point instead of the measured bus voltage, which
    corresponds to regulating the voltage at the remote point of the feeder.

    The compensation is based on the magnitude of the voltage and the power flow, i.e. voltage angles are not
    required for the load flow calculation.

    This strategy is typically used in radial distribution grids with a single feeder. In meshed grids or grids with a
    high share of distributed generation the voltage at the transformer bus is not a good indicator for the voltage at
    other buses, so the compensation can be inaccurate there.

    Parameters:
        net (ADict): Pandapower struct
        element_index (int or array_like): ID(s) of the trafo(s) that is/are controlled
        side (string, "lv"): Side of the transformer at which the voltage is controlled (hv, mv or lv). The fictitious
            line is assumed to be connected to this side.
        r_ohm (float or array_like, None): Resistance of the fictitious line in ohm (per phase, referred to the voltage
            level of the controlled side). Can be omitted if only a reactance is given. A single value is applied to
            all controlled transformers.
        x_ohm (float or array_like, None): Reactance of the fictitious line in ohm (per phase, referred to the voltage
            level of the controlled side). Can be omitted if only a resistance is given. A single value is applied to
            all controlled transformers.
        remote_bus (int or array_like, None): Index of the bus whose voltage is to be controlled (the far end of the
            fictitious line). Only used for documentation purposes and for the topological calculation of the
            impedance, i.e. if calculate_impedance is True. A single bus is applied to all controlled transformers.
        calculate_impedance (bool, False): If True, the impedance of the fictitious line is calculated from the
            topology of the network as the shortest path between the bus at the controlled side of the transformer and
            ``remote_bus``. The path must not contain transformers, i.e. the remote bus must be on the same voltage
            level as the controlled bus. The impedance is calculated during initialization (and updated whenever the
            controller is initialized) and can be refreshed manually with :meth:`update_impedance`. If True, any given
            r_ohm and x_ohm are ignored and replaced by the calculated impedance.
    """

    def __init__(
        self,
        net,
        element_index,
        side,
        r_ohm=None,
        x_ohm=None,
        remote_bus=None,
        calculate_impedance=False,
        **kwargs,
    ):
        self.r_ohm = r_ohm
        self.x_ohm = x_ohm
        self.remote_bus = remote_bus
        self.calculate_impedance = calculate_impedance

        super().__init__(net, element_index=element_index, side=side, **kwargs)
        # the base class only initializes the trafo bus during initialize_control, but it is needed for the calculation
        # of the compensated voltage already before the first control step
        self._update_trafobus(net)

        if self.calculate_impedance:
            # the user values are replaced by the topology result, so a failure at creation must raise
            self.r_ohm = None
            self.x_ohm = None
            self.update_impedance(net)
        else:
            if self.r_ohm is None and self.x_ohm is None:
                raise ValueError(
                    "Line drop compensation requires the impedance of the fictitious line: either provide r_ohm "
                    "and/or x_ohm or set calculate_impedance=True to derive it from the network topology"
                )
            self.r_ohm = self._check_impedance("r_ohm", 0.0 if self.r_ohm is None else self.r_ohm)
            self.x_ohm = self._check_impedance("x_ohm", 0.0 if self.x_ohm is None else self.x_ohm)

    def _check_impedance(self, name, value):
        """
        Converts a given impedance to a float (single index) or an array (array-like element index) and checks that
        the number of values matches the number of controlled transformers.
        """
        if self._read_write_flag == "single_index":
            number_of_elements = 1
        else:
            number_of_elements = len(np.atleast_1d(self.element_index))

        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if arr.size == 1:
            # a single value is applied to all controlled transformers
            arr = np.repeat(arr, number_of_elements)
        if arr.size != number_of_elements:
            raise ValueError(
                f"Line drop compensation: {name} has {arr.size} values, but {number_of_elements} transformers are "
                f"controlled"
            )
        if self._read_write_flag == "single_index":
            return float(arr[0])
        return arr

    def initialize_control(self, net):
        super().initialize_control(net)
        # recalculate the fictitious line impedance in case the network topology has changed
        if not self.nothing_to_do(net):
            self.update_impedance(net)

    def update_impedance(self, net):
        """
        (Re-)calculates the impedance of the fictitious line from the network topology.

        Only has an effect if the controller was created with ``calculate_impedance=True``. Otherwise, the
        user-defined impedance values are kept.

        Parameters:
            net (ADict): Pandapower struct
        """
        if not self.calculate_impedance:
            return
        self._update_trafobus(net)
        try:
            r_ohm, x_ohm = self._calc_impedance_from_topology(net)
        except ValueError as e:
            if self.r_ohm is None or self.x_ohm is None:
                raise
            # the topology may change between (time series) runs, e.g. if lines or switches are opened, so the
            # previously calculated impedance is kept in this case
            logger.warning(
                f"Line drop compensation for {self.element} {self.element_index}: the impedance of the fictitious "
                f"line could not be recalculated from the network topology ({e}). The previously calculated "
                f"impedance r_ohm={self.r_ohm}, x_ohm={self.x_ohm} is used."
            )
            return
        self.r_ohm = r_ohm
        self.x_ohm = x_ohm
        logger.debug(
            f"Line drop compensation for {self.element} {self.element_index}: line impedance to remote bus "
            f"{self.remote_bus} calculated from topology: r_ohm={self.r_ohm}, x_ohm={self.x_ohm}"
        )

    def _calc_impedance_from_topology(self, net):
        if self.remote_bus is None:
            raise ValueError(
                "remote_bus has to be provided if the impedance of the fictitious line is to be calculated from the "
                "network topology (calculate_impedance=True)"
            )

        trafo_buses = np.atleast_1d(self.trafobus)
        remote_buses = np.atleast_1d(self.remote_bus)
        if remote_buses.size == 1:
            remote_buses = np.repeat(remote_buses, trafo_buses.size)
        elif remote_buses.size != trafo_buses.size:
            raise ValueError(
                f"Number of remote buses ({remote_buses.size}) does not match the number of controlled transformers "
                f"({trafo_buses.size})"
            )

        # transformers are excluded from the graph, because the fictitious line stays on one voltage level
        mg = self._create_graph(net)

        r_ohm = np.zeros(trafo_buses.size, dtype=np.float64)
        x_ohm = np.zeros(trafo_buses.size, dtype=np.float64)
        for i, (trafo_bus, remote_bus) in enumerate(zip(trafo_buses, remote_buses)):
            if not self._bus_in_graph(mg, trafo_bus):
                # the transformer bus is out of service or not connected, so no compensation is applied
                r_ohm[i], x_ohm[i] = 0.0, 0.0
                continue
            if not self._bus_in_graph(mg, remote_bus):
                raise ValueError(f"Line drop compensation: remote bus {remote_bus} is not part of the network graph")
            r_ohm[i], x_ohm[i] = self._impedance_between_buses(mg, trafo_bus, remote_bus)

        if self._read_write_flag == "single_index":
            return float(r_ohm[0]), float(x_ohm[0])
        return r_ohm, x_ohm

    # create_nxgraph(calc_branch_impedances=True) rebuilds these internal keys of the net
    _NET_STATE_KEYS = ("_ppc", "_options", "_pd2ppc_lookups")

    def _create_graph(self, net):
        """
        Creates a networkx graph with branch impedances in ohm, which does not contain transformers.

        The graph is only needed for the shortest path search. Since ``create_nxgraph`` with
        ``calc_branch_impedances=True`` rebuilds the internal structures of the net, the previous state is restored
        afterwards so that e.g. the result recycling of a time series simulation keeps working.
        """
        internal_state = {}
        for key in self._NET_STATE_KEYS:
            if key in net:
                value = net[key]
                # the dicts are modified in place, so a shallow copy of the content is stored as well
                internal_state[key] = (value, dict(value) if isinstance(value, dict) else None)
        try:
            # transformers are excluded from the graph, because the fictitious line stays on one voltage level
            return create_nxgraph(
                net,
                calc_branch_impedances=True,
                branch_impedance_unit="ohm",
                include_trafos=False,
                include_trafo3ws=False,
                include_dclines=False,
                include_line_dc=False,
            )
        finally:
            for key in self._NET_STATE_KEYS:
                if key not in internal_state:
                    net.pop(key, None)
                    continue
                value, content = internal_state[key]
                if content is not None:
                    value.clear()
                    value.update(content)
                net[key] = value

    @staticmethod
    def _bus_in_graph(mg, bus):
        try:
            return bus in mg
        except TypeError:
            return False

    @staticmethod
    def _impedance_between_buses(mg, from_bus, to_bus):
        if from_bus == to_bus:
            return 0.0, 0.0

        try:
            path = nx.shortest_path(mg, from_bus, to_bus, weight=LineDropCompensation._segment_weight)
        except nx.NodeNotFound as e:
            raise ValueError(
                f"Line drop compensation: bus {e} of the fictitious line is not part of the network graph"
            ) from e
        except nx.NetworkXNoPath as e:
            raise ValueError(
                f"Line drop compensation: no path without transformers between bus {from_bus} and remote bus "
                f"{to_bus}. The remote bus has to be on the same voltage level as the controlled bus."
            ) from e

        r_ohm = 0.0
        x_ohm = 0.0
        for u, v in pairwise(path):
            z_eq = LineDropCompensation._equivalent_segment_impedance(mg.get_edge_data(u, v))
            r_ohm += z_eq.real
            x_ohm += z_eq.imag
        return r_ohm, x_ohm

    @staticmethod
    def _segment_weight(from_bus, to_bus, edge_data):
        """
        Weight of a graph segment for the shortest path search, i.e. the magnitude of the equivalent impedance of all
        parallel branches between the two buses.
        """
        return abs(LineDropCompensation._equivalent_segment_impedance(edge_data))

    @staticmethod
    def _equivalent_segment_impedance(edge_data):
        """
        Returns the equivalent impedance of all parallel branches between two buses of the graph.
        """
        y_sum = 0j
        for attributes in edge_data.values():
            z = complex(attributes.get("r_ohm", 0.0), attributes.get("x_ohm", 0.0))
            if z == 0:
                # ideal connection, e.g. a closed bus-bus switch
                return complex(0.0, 0.0)
            y_sum += 1 / z
        if y_sum == 0:
            return complex(np.inf, 0.0)
        return 1 / y_sum

    def _get_side_power(self, net):
        """
        Returns the active and reactive power flowing from the transformer into the grid at the controlled side in MW
        and Mvar.
        """
        p_mw = read_from_net(net, f"res_{self.element}", self.element_index, f"p_{self.side}_mw", self._read_write_flag)
        q_mvar = read_from_net(
            net, f"res_{self.element}", self.element_index, f"q_{self.side}_mvar", self._read_write_flag
        )
        # p and q in the result tables are positive if the power flows into the transformer
        return -np.asarray(p_mw, dtype=np.float64), -np.asarray(q_mvar, dtype=np.float64)

    def _calc_voltage_drop_pu(self, net):
        """
        Returns the voltage drop over the fictitious line in pu of the nominal voltage of the controlled bus.
        """
        p_mw, q_mvar = self._get_side_power(net)
        vm_pu = np.asarray(
            read_from_net(net, "res_bus", self.trafobus, "vm_pu", self._read_write_flag), dtype=np.float64
        )
        vn_kv = np.asarray(read_from_net(net, "bus", self.trafobus, "vn_kv", self._read_write_flag), dtype=np.float64)
        # the voltage drop is related to the actual voltage at the transformer bus, which is approximated by the
        # magnitude of the measured voltage (standard line drop compensation formula)
        v_kv = vm_pu * vn_kv
        with np.errstate(invalid="ignore", divide="ignore"):
            drop_pu = (p_mw * self.r_ohm + q_mvar * self.x_ohm) / (v_kv * vn_kv)

        # if the trafo is out of service, the power flow results are NaN and no compensation should be applied
        drop_pu = np.asarray(drop_pu, dtype=np.float64)
        if drop_pu.ndim == 0:
            return 0.0 if pd.isna(drop_pu) else float(drop_pu)
        return np.where(pd.isna(drop_pu), 0.0, drop_pu)

    def _get_vm_pu(self, net):
        """
        Returns the compensated voltage in pu, i.e. the measured voltage at the bus of the transformer minus the
        calculated voltage drop over the fictitious line.
        """
        vm_pu = read_from_net(net, "res_bus", self.trafobus, "vm_pu", self._read_write_flag)
        return vm_pu - self._calc_voltage_drop_pu(net)

    def __eq__(self, other):
        if not isinstance(other, LineDropCompensation):
            return False
        # DiscreteTapControl defines its own __eq__ that ignores the line drop compensation parameters
        if not super().__eq__(other):
            return False
        return self._impedance_equal(self.r_ohm, other.r_ohm) and self._impedance_equal(self.x_ohm, other.x_ohm)

    @staticmethod
    def _impedance_equal(value, other_value):
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        other_arr = np.atleast_1d(np.asarray(other_value, dtype=np.float64))
        if arr.shape != other_arr.shape:
            return False
        return bool(np.allclose(arr, other_arr, equal_nan=True))

    # defining __eq__ would otherwise set __hash__ to None and make the controllers unhashable
    __hash__ = JSONSerializableClass.__hash__


class ContinuousTapControlLDC(LineDropCompensation, ContinuousTapControl):
    """
    Trafo controller with continuous tap changer voltage control and line drop compensation (LDC).

    Instead of the voltage at the bus of the transformer, the estimated voltage at a remote point of the feeder is
    controlled (see :class:`LineDropCompensation`).

    Parameters:
        net (ADict): Pandapower struct
        element_index (int or array_like): ID(s) of the trafo(s) that is/are controlled
        vm_set_pu (float): Voltage set point at the remote bus in pu
        side (string, "lv"): Side of the transformer where the voltage is controlled (hv, mv or lv)
        r_ohm (float or array_like, None): Resistance of the fictitious line between the transformer and the remote
            bus in ohm (per phase). A single value is applied to all controlled transformers
        x_ohm (float or array_like, None): Reactance of the fictitious line between the transformer and the remote bus
            in ohm (per phase). A single value is applied to all controlled transformers
        remote_bus (int or array_like, None): Index of the controlled remote bus. Only used for documentation purposes
            and for the topological calculation of the impedance. A single bus is applied to all controlled
            transformers
        calculate_impedance (bool, False): Calculate the impedance of the fictitious line from the network topology
            instead of using the given r_ohm and x_ohm
        tol (float, 0.001): Voltage tolerance band at bus in percent (default: 1% = 0.01pu)
        element (string, "trafo"): Trafo type ("trafo" or "trafo3w")
        in_service (bool, True): Indicates if the controller is currently in_service
        check_tap_bounds (bool, True): In case of true the tap_bounds will be considered
        drop_same_existing_ctrl (bool, False): Indicates if already existing controllers of the same type and with the
            same matching parameters (e.g. at same element) should be dropped

    Example:
        >>> import pandapower as pp
        >>> from pandapower.control import ContinuousTapControlLDC
        >>> net = pp.networks.simple_four_bus_system()
        >>> pp.runpp(net)
        >>> c = ContinuousTapControlLDC(net, 0, vm_set_pu=1.0, r_ohm=0.321, x_ohm=0.0415, remote_bus=3)
        >>> pp.runpp(net, run_control=True)
        >>> isinstance(c, ContinuousTapControlLDC)
        True
    """

    def __init__(
        self,
        net,
        element_index,
        vm_set_pu,
        side="lv",
        r_ohm=None,
        x_ohm=None,
        remote_bus=None,
        calculate_impedance=False,
        tol=1e-3,
        element="trafo",
        in_service=True,
        check_tap_bounds=True,
        level=0,
        order=0,
        drop_same_existing_ctrl=False,
        matching_params=None,
        **kwargs,
    ):
        if matching_params is None:
            matching_params = {"element_index": element_index, "element": element}
        super().__init__(
            net,
            element_index=element_index,
            side=side,
            r_ohm=r_ohm,
            x_ohm=x_ohm,
            remote_bus=remote_bus,
            calculate_impedance=calculate_impedance,
            vm_set_pu=vm_set_pu,
            tol=tol,
            element=element,
            in_service=in_service,
            check_tap_bounds=check_tap_bounds,
            level=level,
            order=order,
            drop_same_existing_ctrl=drop_same_existing_ctrl,
            matching_params=matching_params,
            **kwargs,
        )


class DiscreteTapControlLDC(LineDropCompensation, DiscreteTapControl):
    """
    Trafo controller with discrete tap changer voltage control and line drop compensation (LDC).

    Instead of the voltage at the bus of the transformer, the estimated voltage at a remote point of the feeder is
    controlled (see :class:`LineDropCompensation`).

    Parameters:
        net (ADict): Pandapower struct
        element_index (int or array_like): ID(s) of the trafo(s) that is/are controlled
        vm_lower_pu (float): Lower voltage limit at the remote bus in pu
        vm_upper_pu (float): Upper voltage limit at the remote bus in pu
        side (string, "lv"): Side of the transformer where the voltage is controlled (hv, mv or lv)
        r_ohm (float or array_like, None): Resistance of the fictitious line between the transformer and the remote
            bus in ohm (per phase). A single value is applied to all controlled transformers
        x_ohm (float or array_like, None): Reactance of the fictitious line between the transformer and the remote bus
            in ohm (per phase). A single value is applied to all controlled transformers
        remote_bus (int or array_like, None): Index of the controlled remote bus. Only used for documentation purposes
            and for the topological calculation of the impedance. A single bus is applied to all controlled
            transformers
        calculate_impedance (bool, False): Calculate the impedance of the fictitious line from the network topology
            instead of using the given r_ohm and x_ohm
        element (string, "trafo"): Trafo type ("trafo" or "trafo3w")
        tol (float, 0.001): Voltage tolerance band at bus in percent (default: 1% = 0.01pu)
        in_service (bool, True): Indicates if the controller is currently in_service
        hunting_limit (int, None): Maximum number of tap positions that are stored to detect hunting
        drop_same_existing_ctrl (bool, False): Indicates if already existing controllers of the same type and with the
            same matching parameters (e.g. at same element) should be dropped

    Example:
        >>> import pandapower as pp
        >>> from pandapower.control import DiscreteTapControlLDC
        >>> net = pp.networks.simple_four_bus_system()
        >>> pp.runpp(net)
        >>> c = DiscreteTapControlLDC(net, 0, vm_lower_pu=0.99, vm_upper_pu=1.01, r_ohm=0.321, x_ohm=0.0415,
        ...                           remote_bus=3)
        >>> pp.runpp(net, run_control=True)
        >>> isinstance(c, DiscreteTapControlLDC)
        True
    """

    def __init__(
        self,
        net,
        element_index,
        vm_lower_pu,
        vm_upper_pu,
        side="lv",
        r_ohm=None,
        x_ohm=None,
        remote_bus=None,
        calculate_impedance=False,
        element="trafo",
        tol=1e-3,
        in_service=True,
        hunting_limit=None,
        level=0,
        order=0,
        drop_same_existing_ctrl=False,
        matching_params=None,
        **kwargs,
    ):
        if matching_params is None:
            matching_params = {"element_index": element_index, "element": element}
        super().__init__(
            net,
            element_index=element_index,
            side=side,
            r_ohm=r_ohm,
            x_ohm=x_ohm,
            remote_bus=remote_bus,
            calculate_impedance=calculate_impedance,
            vm_lower_pu=vm_lower_pu,
            vm_upper_pu=vm_upper_pu,
            element=element,
            tol=tol,
            in_service=in_service,
            hunting_limit=hunting_limit,
            level=level,
            order=order,
            drop_same_existing_ctrl=drop_same_existing_ctrl,
            matching_params=matching_params,
            **kwargs,
        )

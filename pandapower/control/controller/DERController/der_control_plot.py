# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt


def plot_pq_area(pq_area, **kwargs):
    """
    Assumption: there is no q restriction at vm = 1 pu
    """
    p_samples = kwargs.pop("p_samples", None)
    if p_samples is None:
        p_samples = np.linspace(0, 1, 200)

    min_max_q = pq_area.q_flexibility(p_pu=p_samples, vm_pu=1)

    _plot_pq_area(min_max_q, p_samples=p_samples, **kwargs)


def _plot_pq_area(min_max_q, title=None, ax=None, saturate_sn_pu:float=np.nan, circle_segment=90,
                  p_samples=None, tex=False):
    if p_samples is None:
        p_samples = np.linspace(0, 1, 200)
    if tex:
        texts = "$Q_{min}$", "$Q_{max}$", \
            r"underexcited $\leftarrow------ Q/S_N ------\rightarrow$ overexcited", "$P/S_N$", \
            '$S_{max}$'
    else:
        texts = "Qmin", "Qmax", "underexcited < ----- Q/Sn ----- > overexcited", "P/Sn", 'Smax'
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(min_max_q[:, 0], p_samples, c="blue", alpha=0.9, label=texts[0])
    ax.plot(min_max_q[:, 1], p_samples, c="red", alpha=0.9, label=texts[1])
    ax.set_xlabel(texts[2])
    ax.set_ylabel(texts[3])
    ax.legend()
    ax.grid(alpha=0.8)
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(p_samples.min(), p_samples.max())

    if not np.isnan(saturate_sn_pu):
        # the output x, y of generate_circle_segment() is used in reverse, since a segment around
        # 0 degree => east is used instead of around north
        y, x = generate_circle_segment(0, 0, saturate_sn_pu, -circle_segment, circle_segment, 1)
        ax.plot(x, y, '--k', label=texts[4])
        y_circ = np.sqrt(saturate_sn_pu**2 - p_samples**2)
        x_diff = max(x) - min(x)
        ax.set_xlim(min(x)-0.05*x_diff, max(x)+0.05*x_diff)
    else:
        y_circ = np.ones(p_samples.shape)
        x_diff = max(min_max_q[:, 1]) - min(min_max_q[:, 0])
        ax.set_xlim(min(min_max_q[:, 0])-0.05*x_diff, max(min_max_q[:, 1])+0.05*x_diff)
    ax.fill_betweenx(p_samples, np.max(np.c_[min_max_q[:, 0], -y_circ], axis=1),
                     np.min(np.c_[min_max_q[:, 1], y_circ], axis=1), color="green", alpha=0.2)
    plt.tight_layout()


def plot_qv_area(qv_area, title=None, ax=None, prune_to_flexibility=False, vm_samples=None, tex=False):
    if vm_samples is None:
        vm_samples = np.linspace(0.8, 1.2, 200)
    if tex:
        texts = "$Q_{min}$", "$Q_{max}$", "$v_m$ in pu", "$Q/S_N$"
    else:
        texts = "Qmin", "Qmax", "Vm in pu", "Q/Sn"

    min_max_q = qv_area.q_flexibility(p_pu=0.5, vm_pu=vm_samples) # p should be irrelevant since
    # it is about QV areas, not about PQV areas

    len_ = min_max_q.shape[0]

    if prune_to_flexibility:
        start = min(np.arange(len_, dtype=int)[~np.isclose(min_max_q[:, 0], min_max_q[0, 0])])-1
        stop = max(np.arange(len_, dtype=int)[~np.isclose(min_max_q[:, 1], min_max_q[-1, 1])])+2
        vm_samples = vm_samples[start:stop]
        min_max_q = min_max_q[start:stop, :]

    # --- do plotting
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(vm_samples, min_max_q[:, 0], c="blue", alpha=0.9, label=texts[0])
    ax.plot(vm_samples, min_max_q[:, 1], c="red", alpha=0.9, label=texts[1])
    ax.set_xlabel(texts[2])
    ax.set_ylabel(texts[3])
    ax.legend()
    ax.grid(alpha=0.8)
    ax.fill_between(vm_samples, min_max_q[:, 0], min_max_q[:, 1], color="green", alpha=0.2)

    if title is not None:
        ax.set_title(title)
    plt.tight_layout()


def generate_circle_segment(center_x, center_y, radius, start, stop, step):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    generates x and y coordinates for a segment of a circle, centered at center_x, center_y. The
    segment (start, stop, step) is given in degree.
    Which degree means which direction?
    0: "east", 90: "north", 180: "west", 270: "south"

    Example
    -------
    >>> x, y = generate_circle_segment(0.5, 0.7, 1, 90, 225, 1)
    >>> plt.plot(x, y)
    >>> plt.axis('equal')
    >>> plt.show()
    Even a helix can be generated:
    >>> x, y = generate_circle_segment(0, 0, np.linspace(0, 1, 3*360+1), 0, 3*360, 1)
    >>> plt.plot(x, y)
    >>> plt.axis('equal')
    >>> plt.show()
    """
    degrees = np.array([start, stop + step/2, step])
    angles = np.radians(degrees)
    sample_angles = np.arange(angles[0], angles[1], angles[2])
    x = center_x + radius * np.cos(sample_angles)
    y = center_y + radius * np.sin(sample_angles)
    return x, y


if __name__ == "__main__":
    import pandapower.control.controller.DERController as DERModels

    pq_area = DERModels.PQVArea4120V1()
    plot_pq_area(pq_area, "PQVArea4120V1")
    plt.show()
    plot_qv_area(pq_area, "PQVArea4120V1")
    plt.show()

    pq_area = DERModels.PQVArea4120V2()
    plot_pq_area(pq_area, "PQVArea4120V2")
    plt.show()
    plot_qv_area(pq_area, "PQVArea4120V2")
    plt.show()

    pq_area = DERModels.PQVArea4120V3()
    plot_pq_area(pq_area, "PQVArea4120V3")
    plt.show()
    plot_qv_area(pq_area, "PQVArea4120V3")
    plt.show()

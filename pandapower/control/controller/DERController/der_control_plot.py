# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt


from pandaplan.core.control.controller.der_control import *

# Verify functions
def verify_pq_area(pq_area, title=None, ax=None, circle_segment=90):
    """
    Assumption: there is no q restriction at vm = 1 pu
    """
    p = np.linspace(0, 1, 200)
    min_max_q = pq_area.q_flexibility(p=p, vm_pu=1)
    plot_pq_area(p, min_max_q, title, ax)


def verify_qv_area(qv_area, title=None, p=1, ax=None):
    """
    Assumption: all relevant information is in vm = 0.8 .. 1.2 pu
    """
    vm_pu = np.linspace(0.8, 1.2, 200)
    min_max_q = qv_area.q_flexibility(p=p, vm_pu=vm_pu)
    len_ = min_max_q.shape[0]
    start = min(np.arange(len_, dtype=int)[~np.isclose(min_max_q[:, 0], min_max_q[0, 0])])-1
    stop = max(np.arange(len_, dtype=int)[~np.isclose(min_max_q[:, 1], min_max_q[-1, 1])])+2
    plot_qv_area(vm_pu, min_max_q, title, ax)


def plot_pq_area(p, min_max_q, title=None, ax=None, circle_segment=90):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(min_max_q[:, 0], p, c="blue", alpha=0.9, label="$Q_{min}$")
    ax.plot(min_max_q[:, 1], p, c="red", alpha=0.9, label="$Q_{max}$")
    ax.set_xlabel(r"underexcited $\leftarrow------ Q/S_N ------\rightarrow$ overexcited$")
    ax.set_ylabel("$P/S_N$")
    ax.legend()
    ax.grid(alpha=0.8)
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(p.min(), p.max())

    x, y = generate_semicircle(0, 0, 1, 90-circle_segment, 90+circle_segment, 1)
    ax.plot(y, x, '--k', label='$S_{max}$')

    y_circ = np.sqrt(1 - p**2)
    ax.fill_betweenx(p, np.max(np.c_[min_max_q[:, 0], -y_circ], axis=1), np.min(np.c_[min_max_q[:, 1], y_circ], axis=1),
                     color="green", alpha=0.2)
    plt.tight_layout()


def plot_qv_area(v, min_max_q, title=None, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(v, min_max_q[:, 0], c="blue", alpha=0.9, label="q_min")
    ax.plot(v, min_max_q[:, 1], c="red", alpha=0.9, label="q_max")
    ax.set_xlabel("$v_m$ in pu")
    ax.set_ylabel("$Q/S_N$")
    ax.legend()
    ax.grid(alpha=0.8)
    ax.fill_between(v, min_max_q[:, 0], min_max_q[:, 1], color="green", alpha=0.2)

    if title is not None:
        ax.set_title(title)
    plt.tight_layout()


def generate_circle_segment(center_x, center_y, radius, start, stop, step):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    generates x and y coordinates for a segment of a circle, centered at center_x, center_y. The
    segment is given in degree.
    Which degree means wich direction?
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
    pq_area = PQVArea4120V1()
    verify_pq_area(pq_area, "PQVArea4120V1")
    plt.show()
    verify_qv_area(pq_area, "PQVArea4120V1")
    plt.show()

    pq_area = PQVArea4120V2()
    verify_pq_area(pq_area, "PQVArea4120V2")
    plt.show()
    verify_qv_area(pq_area, "PQVArea4120V2")
    plt.show()

    pq_area = PQVArea4120V3()
    verify_pq_area(pq_area, "PQVArea4120V3")
    plt.show()
    verify_qv_area(pq_area, "PQVArea4120V3")
    plt.show()

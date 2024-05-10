# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt


from pandaplan.core.control.controller.der_control import *

# Verify functions
def verify_pq_area(pq_area, title, ax=None):
    all_p = np.linspace(0, 1, 200)
    q_min = [pq_area.q_flexibility(p=p, vm_pu=1)[0] for p in all_p]
    q_max = [pq_area.q_flexibility(p=p, vm_pu=1)[1] for p in all_p]
    plot_pq_area(all_p, q_min, q_max, title, ax)


def verify_qv_area(pqv_area, title, p=1, ax=None):
    all_v = np.linspace(0.8, 1.2, 200)
    q_min = [pqv_area.q_flexibility(p=p, vm_pu=v)[0] for v in all_v]
    q_max = [pqv_area.q_flexibility(p=p, vm_pu=v)[1] for v in all_v]
    plot_qv_area(all_v, q_min, q_max, title, ax)


def plot_pq_area(p, qmin, qmax, title, ax=None):
    ax = ax or plt.gca()
    ax.plot(qmin, p, c="blue", alpha=0.9, label="$Q_{min}$")
    ax.plot(qmax, p, c="red", alpha=0.9, label="$Q_{max}$")
    ax.set_xlabel("$underexcited \longleftarrow------ Q/Sn ------\longrightarrow overexcited$")
    ax.set_ylabel("P/Sn")
    ax.legend()
    ax.grid(alpha=0.8)
    ax.fill_betweenx(p, qmin, qmax, color="green", alpha=0.2)
    ax.set_title(title)
    ax.set_ylim(p.min(), p.max())

    x,y = generate_semicircle(0,0,max(p), 0.01)
    ax.plot(y, x, '--k', label='$S_{max}$')


def plot_qv_area(v, qmin, qmax, title, ax=None):
    ax = ax or plt.gca()
    ax.plot(v, qmin, c="blue", alpha=0.9, label="q_min")
    ax.plot(v, qmax, c="red", alpha=0.9, label="q_max")
    ax.set_xlabel("vm_pu")
    ax.set_ylabel("Q/Sn")
    ax.legend()
    ax.grid(alpha=0.8)
    ax.fill_between(v, qmin, qmax, color="green", alpha=0.2)

    ax.set_title(title)
    ax.set_xlim(v.min(), v.max())


def generate_semicircle(center_x, center_y, radius, stepsize=0.1):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    """

    x = np.arange(center_x, center_x+radius+stepsize, stepsize)
    y = np.sqrt(radius**2 - x**2)

    # since each x value has two corresponding y-values, duplicate x-axis.
    # [::-1] is required to have the correct order of elements for plt.plot.
    x = np.concatenate([x,x[::-1]])

    # concatenate y and flipped y.
    y = np.concatenate([y,-y[::-1]])

    return x, y + center_y


if __name__ == "__main__":
    pq_area = PQVArea4110User()
    verify_pq_area(pq_area, pq_area.name)
    plt.show()
    verify_qv_area(pq_area, pq_area.name)
    plt.show()

    pq_area = PQVArea4120V1()
    verify_pq_area(pq_area, pq_area.name)
    plt.show()
    verify_qv_area(pq_area, pq_area.name)
    plt.show()

    pq_area = PQVArea4120V2()
    verify_pq_area(pq_area, pq_area.name)
    plt.show()
    verify_qv_area(pq_area, pq_area.name)
    plt.show()

    pq_area = PQVArea4120V3()
    verify_pq_area(pq_area, pq_area.name)
    plt.show()
    verify_qv_area(pq_area, pq_area.name)
    plt.show()

    pq_area = PQVArea_STATCOM()
    verify_pq_area(pq_area, pq_area.name)
    plt.show()
    verify_qv_area(pq_area, pq_area.name)
    plt.show()

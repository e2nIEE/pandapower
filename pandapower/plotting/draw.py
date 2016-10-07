# -*- coding: utf-8 -*-
from __future__ import division
__author__ = "Alexander Scheidler"

import matplotlib.pyplot as plt
import copy

def draw_collections(collections, figsize=(10, 8), ax=None):
    """
    """

    if not ax:
        plt.figure(facecolor="white", figsize=figsize)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                            wspace=0.02, hspace=0.04)
    ax = ax or plt.gca()

    for c in collections:
        if c:
            cc = copy.copy(c)
            ax.add_collection(cc)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect('equal', 'datalim')
    ax.autoscale_view(True, True, True)
    ax.margins(.02)
    plt.tight_layout()
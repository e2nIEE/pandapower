# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:38:31 2019

@author: e2n035
"""

import numpy as np

from scipy.optimize import minimize, linprog

H = np.array([-10.0, 0, -10.0, 
              -10.0, 0.0, 0.0, 
              0.0, 0.0, 10.0, 
              -10.0, 0.0, 10.0, 
              30.0, -10.0, -10.0, 
              -10.0, 10.0, 0.0,
              0.0, 10.0, -10.0]).reshape((7, 3))

n, m = 3, 7
X0 = np.array([0, 1, 1]).reshape((-1, 1))
Z = H @ X0

z = np.array([0.6, 0.1, 0.5, -0.4, 0.6, 0.2, 0.1]).reshape((-1, 1))

zero_n = np.zeros((n, 1))
one_m = np.ones((m, 1))
c_T = np.r_[zero_n, zero_n, one_m, one_m]

A = np._r[H, -H, Im, -Im]
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
z = np.array([0.6, 0.1, 0.5, -0.4, 0.6, 0.2, 0.1]).reshape((-1, 1))

H = np.array([-10.0, 0, -10.0, 
              -10.0, 0.0, 0.0, 
              0.0, 0.0, 10.0, 
              -10.0, 0.0, 10.0, 
              30.0, -10.0, -10.0, 
              -10.0, 10.0, 0.0,
              0.0, 10.0, -10.0]).reshape((7, 3))
z = np.array([0.6, 0.1, 0.5, -0.4, 0.6, 0.2, 0.1]).reshape((-1, 1))

n, m = H.shape[1], H.shape[0]
x = np.array([0, 1, 1]).reshape((-1, 1)).astype(float)


Z = H @ x

delta_z =  z-Z

zero_n = np.zeros((n, 1))
one_m = np.ones((m, 1))
Im = np.eye(m)

c_T = np.r_[zero_n, zero_n, one_m, one_m]
A = np.c_[H, -H, Im, -Im]

res = linprog(c_T.ravel(), A_eq=A, b_eq=delta_z ,bounds=[(0, None) for _ in range(A.shape[1])])
delta_x = res.x
x -= res.x[n:2*n].reshape((-1, 1))
x += res.x[:n].reshape((-1, 1))
print(np.max(delta_x[:6]))

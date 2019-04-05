# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:23:30 2019

@author: Zhenqi Wang
"""

import numpy as np

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import estimate
from pandapower.estimation.toolbox import add_virtual_meas_from_loadflow
from copy import deepcopy

import torch


# 1. Create network
net = pp.create_empty_network()
pp.create_bus(net, name="bus1", vn_kv=1.)
pp.create_bus(net, name="bus2", vn_kv=1.)
pp.create_load(net, 1, p_mw=0.0111, q_mvar=0.06)
pp.create_ext_grid(net, 0, vm_pu=1.038)
pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1, x_ohm_per_km=0.5, c_nf_per_km=0,
                               max_i_ka=1)

pp.create_measurement(net, "p", "line", -0.0111, 0.003, 0, 1)  # p12
pp.create_measurement(net, "q", "line", -0.06, 0.003, 0, 1)    # q12

pp.create_measurement(net, "v", "bus", 1.038, 0.0001, 0)  # u1
pp.create_measurement(net, "v", "bus", 0.98, 0.1, 1)   # u2

# 2. Do state estimation
success = estimate(net, init='flat', algorithm="wls")


Ybus, Yf, Yt = (net._ppc['internal']['Ybus'].toarray(),\
               net._ppc['internal']['Yf'].toarray(),\
               net._ppc['internal']['Yt'].toarray())
yfr = torch.from_numpy(np.real(Yf).astype(np.double))
yfi = torch.from_numpy(np.imag(Yf).astype(np.double))
ytr = torch.from_numpy(np.real(Yt).astype(np.double))
yti = torch.from_numpy(np.imag(Yt).astype(np.double))
ybr = torch.from_numpy(np.real(Ybus).astype(np.double))
ybi = torch.from_numpy(np.imag(Ybus).astype(np.double))
baseMVA = net._ppc['baseMVA']

fbus = torch.from_numpy(np.array([0])).type(torch.LongTensor)
tbus = torch.from_numpy(np.array([1])).type(torch.LongTensor)
bus_baseKV = torch.from_numpy(np.array([1, 1]))
E = torch.from_numpy(np.array([0, 1.038, 1]))
Z = torch.from_numpy(np.array([-0.0111, -0.06, 1.038, 0.98]))
r_cov = torch.from_numpy(np.array([0.003, 0.003, 0.0001, 0.1]))
#non_nan_meas_mask = torch.Tensor([[False], [False], [False], [True], [False], [False], [False], 
#                                  [True], [True], [True], [False], [False]]).type(torch.ByteTensor)
non_nan_meas_mask = torch.Tensor([[False], [False], [False], [True], [False], [False], [False], 
                                  [True], [True], [True]]).type(torch.ByteTensor)
vi_slack = torch.zeros(1, 1, requires_grad=False, dtype=torch.double)

@torch.jit.script
def real_matmul(m1r, m1i, m2r, m2i):
    return m1r @ m2r - m1i @ m2i

@torch.jit.script
def imag_matmul(m1r, m1i, m2r, m2i):
    return m1r @ m2i + m1i @ m2r

@torch.jit.script
def real_mul(m1r, m1i, m2r, m2i):
    return m1r * m2r - m1i * m2i

@torch.jit.script
def imag_mul(m1r, m1i, m2r, m2i):
    return m1r * m2i + m1i * m2r


def model(vr, vi_grad):
    vi = torch.cat((vi_slack,
                    vi_grad), 0)
#    vi = vi_grad

    p_b = real_mul(vr, vi, real_matmul(ybr, ybi, vr, vi),
                           - imag_matmul(ybr, ybi, vr, vi))
    q_b = imag_mul(vr, vi, real_matmul(ybr, ybi, vr, vi),
                           - imag_matmul(ybr, ybi, vr, vi))

    vfr, vfi = torch.index_select(vr, 0, fbus), torch.index_select(vi, 0, fbus)
    p_f = real_mul(vfr, vfi, real_matmul(yfr, yfi, vr, vi),
                             - imag_matmul(yfr, yfi, vr, vi))
    q_f = imag_mul(vfr, vfi, real_matmul(yfr, yfi, vr, vi),
                             - imag_matmul(yfr, yfi, vr, vi))
#    i_f = torch.sqrt(p_f**2 + q_f**2) / torch.sqrt(vfr**2 + vfi**2)
   
    vtr, vti = torch.index_select(vr, 0, tbus), torch.index_select(vi, 0, tbus)
    p_t = real_mul(vtr, vti, real_matmul(ytr, yti, vr, vi),
                             - imag_matmul(ytr, yti, vr, vi))
    q_t = imag_mul(vtr, vti, real_matmul(ytr, yti, vr, vi),
                             - imag_matmul(ytr, yti, vr, vi))
#    i_t = torch.sqrt(p_t**2 + q_t**2) / torch.sqrt(vtr**2 + vti**2)

    hx_pq = torch.cat([p_b, 
                       p_f,
                       p_t,
                       q_b,
                       q_f,
                       q_t], 0) * baseMVA
    
    hx_v = torch.sqrt(vr**2 + vi**2)
#    hx_i = torch.cat([i_f * (baseMVA / torch.index_select(bus_baseKV, 0, fbus)).type(torch.DoubleTensor),
#                      i_t * (baseMVA / torch.index_select(bus_baseKV, 0, tbus)).type(torch.DoubleTensor)], 0) 
#    hx = torch.cat([hx_pq, hx_v, hx_i])
    hx = torch.masked_select(torch.cat([hx_pq, hx_v]), non_nan_meas_mask)
    return hx

#va = torch.from_numpy(np.array([[0], [np.deg2rad(3.02107)]]))
#vm = torch.from_numpy(np.array([[1.038], [0.995292]]))
#vr_test = vm * torch.cos(va)
#vi_test = vm * torch.sin(va)
#hx = model(vr_test, 
#           vi_test)
#V = np.array([[1.038], [0.9953]]) * np.exp(1j * np.array([[0], [np.deg2rad(3.02107)]]))
##V = np.array([1.038, 0.9953]) * np.exp(1j * np.array([[0, np.deg2rad(3.02107)]]))
#Sbus = V * np.conj(Ybus @ V)
#print(Sbus)
#print(np.angle(Sbus))

def weighted_mse_loss(input, target, weight):
    return torch.sum(1/weight * (input - target) ** 2)

optimizer = torch.optim.Adam([vr, vi_grad], lr=0.005)
for _ in range(1000):
    optimizer.zero_grad()
    
    output = model(vr, vi_grad)
#    print(output)
    loss = weighted_mse_loss(output, Z, r_cov)
#    print(loss)
    loss.backward(retain_graph=True)
    print(vr.grad.data)
    if (torch.abs(vr.grad.data) < 1e-10).all():
        break
    optimizer.step()


print(vr, vi_grad)



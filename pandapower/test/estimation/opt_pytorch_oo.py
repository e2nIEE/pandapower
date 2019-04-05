# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:07 2019

@author: Zhenqi Wang
"""

import numpy as np
import torch

from pandapower.estimation import estimate
from functools import partial


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


class StateEstimator(torch.nn.Module):
    def __init__(self, Ybus, Yf, Yt, baseMVA,
                 non_slack_bus, slack_bus, fbus, tbus,
                 non_nan_meas_mask):
        super(StateEstimator, self).__init__()
        self.baseMVA = baseMVA
        self.yfr = torch.from_numpy(np.real(Yf).astype(np.double))
        self.yfi = torch.from_numpy(np.imag(Yf).astype(np.double))
        self.ytr = torch.from_numpy(np.real(Yt).astype(np.double))
        self.yti = torch.from_numpy(np.imag(Yt).astype(np.double))
        self.ybr = torch.from_numpy(np.real(Ybus).astype(np.double))
        self.ybi = torch.from_numpy(np.imag(Ybus).astype(np.double))

        self.fbus = torch.from_numpy(fbus).type(torch.LongTensor)
        self.tbus = torch.from_numpy(tbus).type(torch.LongTensor)
        self.non_nan_meas_mask = non_nan_meas_mask
        self.vi_slack = torch.zeros(slack_bus.shape[0], 1, 
                                    requires_grad=False, dtype=torch.double)
        self.non_slack_bus = non_slack_bus
        self.slack_bus = slack_bus  
        self.vi_mapping = torch.from_numpy(np.r_[slack_bus, non_slack_bus]).type(torch.LongTensor)

    def forward(self, vr, vi_non_slack):
        vi = torch.cat((self.vi_slack,
                        vi_non_slack), 0)
        vi = torch.index_select(vi, 0, self.vi_mapping)

        p_b = real_mul(vr, vi, real_matmul(self.ybr, self.ybi, vr, vi),
                               - imag_matmul(self.ybr, self.ybi, vr, vi))
        q_b = imag_mul(vr, vi, real_matmul(self.ybr, self.ybi, vr, vi),
                               - imag_matmul(self.ybr, self.ybi, vr, vi))

        vfr, vfi = torch.index_select(vr, 0, self.fbus), torch.index_select(vi, 0, self.fbus)
        p_f = real_mul(vfr, vfi, real_matmul(self.yfr, self.yfi, vr, vi),
                                 - imag_matmul(self.yfr, self.yfi, vr, vi))
        q_f = imag_mul(vfr, vfi, real_matmul(self.yfr, self.yfi, vr, vi),
                                 - imag_matmul(self.yfr, self.yfi, vr, vi))

        vtr, vti = torch.index_select(vr, 0, self.tbus), torch.index_select(vi, 0, self.tbus)
        p_t = real_mul(vtr, vti, real_matmul(self.ytr, self.yti, vr, vi),
                                 - imag_matmul(self.ytr, self.yti, vr, vi))
        q_t = imag_mul(vtr, vti, real_matmul(self.ytr, self.yti, vr, vi),
                                 - imag_matmul(self.ytr, self.yti, vr, vi))

        hx_pq = torch.cat([p_b,
                           p_f,
                           p_t,
                           q_b,
                           q_f,
                           q_t], 0) * self.baseMVA
        hx_v = torch.sqrt(vr**2 + vi**2)
        hx = torch.masked_select(torch.cat([hx_pq, hx_v]), self.non_nan_meas_mask)
        return hx


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def optimize(model, floss):
    optimizer = torch.optim.Adam([vr, vi_non_slack], lr=5e-3)
    for t in range(1000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(vr, vi_non_slack)

        # Compute and print loss
        loss = weighted_mse_loss(y_pred, Z, 1/r_cov)
        print(t, loss.item())
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        if (torch.abs(vr.grad.data) < 1e-10).all() and (torch.abs(vi_non_slack.grad.data) < 1e-10).all():
            break
        optimizer.step()
    return vr, vi_non_slack


if __name__ == "__main__":
    import pandapower as pp
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
    
    E = torch.from_numpy(np.array([0, 1.038, 1]))
    Z = torch.from_numpy(np.array([-0.0111, -0.06, 1.038, 0.98]))
    r_cov = torch.from_numpy(np.array([0.003, 0.003, 0.0001, 0.1]))
    
    Ybus, Yf, Yt = (net._ppc['internal']['Ybus'].toarray(),\
                   net._ppc['internal']['Yf'].toarray(),\
                   net._ppc['internal']['Yt'].toarray())
    baseMVA = net._ppc['baseMVA']
    fbus = np.array([0])
    tbus = np.array([1])
    non_nan_meas_mask = torch.Tensor([[False], [False], [False], [True], [False], [False], [False], 
                                      [True], [True], [True]]).type(torch.ByteTensor)
    slack_bus = np.array([0])
    non_slack_bus = np.array([1])
    
    model = StateEstimator(Ybus, Yf, Yt, baseMVA=baseMVA, slack_bus=slack_bus, non_slack_bus=non_slack_bus,
                           fbus=fbus, tbus=tbus, non_nan_meas_mask=non_nan_meas_mask)

    vr = torch.tensor([[1.038], [1]], dtype=torch.double, requires_grad=True)
    vi_non_slack = torch.zeros(1, 1, requires_grad=True, dtype=torch.double)
    floss = partial(weighted_mse_loss, {"target": Z,
                                        "weight": 1/r_cov})
    vr, vi_non_slack = optimize(model, floss)

    vi = np.zeros(np.r_[slack_bus, non_slack_bus].shape)  
    vi[non_slack_bus] = vi_non_slack.detach().numpy().ravel()
    vr = vr.detach().numpy().ravel()
    V = vr + vi * 1j
    
# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import torch
from functools import partial
import warnings

from pandapower.estimation.algorithm.wls import WLSAlgorithm
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_bus import BUS_TYPE
from pandapower.pypower.idx_brch import F_BUS, T_BUS


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


class TorchEstimator(torch.nn.Module):
    def __init__(self, ppci,  non_nan_meas_mask):
        super(TorchEstimator, self).__init__()      
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
            ppci['internal']['Yf'], ppci['internal']['Yt'],\
                ppci['internal']['Ybus'] = Yf, Yt, Ybus
            

        self.baseMVA = ppci['baseMVA']
        Ybus, Yf, Yt = Ybus.toarray(), Yf.toarray(), Yt.toarray()
        self.yfr = torch.from_numpy(np.real(Yf).astype(np.double))
        self.yfi = torch.from_numpy(np.imag(Yf).astype(np.double))
        self.ytr = torch.from_numpy(np.real(Yt).astype(np.double))
        self.yti = torch.from_numpy(np.imag(Yt).astype(np.double))
        self.ybr = torch.from_numpy(np.real(Ybus).astype(np.double))
        self.ybi = torch.from_numpy(np.imag(Ybus).astype(np.double))

        self.slack_bus = np.argwhere(ppci["bus"][:, BUS_TYPE] == 3).ravel() 
        self.non_slack_bus = np.argwhere(ppci["bus"][:, BUS_TYPE] != 3).ravel()
        self.fbus = torch.from_numpy(np.abs(ppci["branch"][:, F_BUS])).type(torch.LongTensor)
        self.tbus = torch.from_numpy(np.abs(ppci["branch"][:, T_BUS])).type(torch.LongTensor)
        
        # ignore current measurement
        non_nan_meas_mask = non_nan_meas_mask[:-(self.fbus.shape[0] * 2)]
        self.non_nan_meas_mask = torch.tensor(non_nan_meas_mask.reshape(-1, 1).tolist()).type(torch.ByteTensor)
        self.vi_slack = torch.zeros(self.slack_bus.shape[0], 1, 
                                    requires_grad=False, dtype=torch.double)
        self.vi_mapping = torch.from_numpy(np.argsort(np.r_[self.slack_bus, 
                                                            self.non_slack_bus])).type(torch.LongTensor)
#        self.vi_mapping = torch.from_numpy(np.array([1,2,3,0,4])).type(torch.LongTensor)

    def forward(self, vr, vi_non_slack):
        vi = torch.cat((self.vi_slack,
                        vi_non_slack), 0)
        vi = vi.index_select(0, self.vi_mapping)

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
        hx = torch.masked_select(torch.cat([hx_pq, hx_v]),
                                 self.non_nan_meas_mask)
        return hx


@torch.jit.script
def weighted_mse_loss(input, target, weight):
    return torch.sum((input - target)**2)


def optimize(model, floss, vr, vi_non_slack):
    optimizer = torch.optim.LBFGS([vr, vi_non_slack], lr=1)
#    optimizer = torch.optim.Adam([vr, vi_non_slack], lr=0.005)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    for t in range(80):
        def closure():
#            scheduler.step()
            # Forward pass: Compute predicted y by passing x to the model
            hx = model(vr, vi_non_slack)
    
            # Compute and print loss
            loss = floss(hx)
            print(t, loss.data)
    
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            return loss
#        print(t, torch.min(torch.abs(vr.grad.data)))
#        if (torch.abs(vr.grad.data) < 1e-5).all() and\
#            (torch.abs(vi_non_slack.grad.data) < 1e-5).all():
#            return vr, vi_non_slack
        optimizer.step(closure)
    return vr, vi_non_slack


class TorchAlgorithm(WLSAlgorithm):
    def estimate(self, ppci, opt_vars=None):
#        assert 'estimator' in opt_vars and opt_vars['estimator'] in ESTIMATOR_MAPPING

        non_slack_buses, v_m, delta, delta_masked, E, r_cov, r_inv, z, non_nan_meas_mask =\
            self.wls_preprocessing(ppci)
        
        model = TorchEstimator(ppci, non_nan_meas_mask=non_nan_meas_mask)
        floss = partial(weighted_mse_loss, 
                        target=torch.tensor(z).type(torch.DoubleTensor), 
                        weight=torch.tensor(1/r_cov/100).type(torch.DoubleTensor))
        vr = torch.tensor(E[len(non_slack_buses):].reshape(-1, 1), 
                            dtype=torch.double, requires_grad=True)
        vi_non_slack = torch.tensor(E[:len(non_slack_buses)].reshape(-1, 1), 
                            dtype=torch.double, requires_grad=True)
        res = optimize(model, floss, vr, vi_non_slack)
        
        if res is not None:
            self.successful = True
            vr, vi_non_slack = res
            vi = np.zeros(np.r_[model.slack_bus, model.non_slack_bus].shape)  
            vi[model.non_slack_bus] = vi_non_slack.detach().numpy().ravel()
            vr = vr.detach().numpy().ravel()
            V = vr + vi * 1j
            return V
        else:
            raise Exception("Optimiaztion failed! State Estimation not successful!")

    
if __name__ == '__main__':
    pass

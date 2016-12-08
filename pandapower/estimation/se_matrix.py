__author__ = 'menke'

from past.utils import old_div
import numpy as np
import warnings
from pypower.makeYbus import makeYbus


class se_matrix:
    def __init__(self, ppc, slack, s_ref):
        np.seterr(divide='ignore', invalid='ignore')
        self.ppc = ppc
        self.s_ref = s_ref
        self.slack_bus = slack
        self.Y_bus = None
        self.G = None
        self.B = None
        self.G_series = None
        self.B_series = None
        self.G_shunt = None
        self.B_shunt = None
        self.keep_ix = None
        self.i_ij = None
        # Offset to accomodate pypower <-> pandapower differences (additional columns)
        self.br_col_offset = 6
        self.create_y()

    # Function which builds a node admittance matrix out of the topology data
    # In addition, it provides the series admittances of lines as G_series and B_series    
    def create_y(self):
        from_to = np.concatenate((self.ppc["branch"][:, 0].real, self.ppc["branch"][:, 1].real))\
            .astype(int)
        to_from = from_to[::-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, _, _ = makeYbus(self.s_ref, self.ppc["bus"], self.ppc["branch"])

        # create relevant matrices
        self.Y_bus = Ybus.toarray()
        self.G = self.Y_bus.real
        self.B = self.Y_bus.imag
        n = len(self.ppc["bus"])
        self.G_series = - self.G
        np.fill_diagonal(self.G_series, 0.)
        self.B_series = - self.B
        np.fill_diagonal(self.B_series, 0.)
        # In case that's becoming relevant later, G_shunt will not be removed
        self.G_shunt = np.zeros_like(self.G)
        self.B_shunt = np.zeros((n, n))
        self.B_shunt[from_to, to_from] = np.tile(0.5 * self.ppc["branch"][:, 4].real, 2)
        
    # Get Y as tuple (real, imaginary)
    def get_y(self):
        return self.G, self.B
    
    # Creates h(x), depending on the current U and delta and the static topology data
    def create_hx(self, v, delta):
        deltas = delta[:, np.newaxis] - delta
        cos_delta = np.cos(deltas)
        sin_delta = np.sin(deltas)
        vi_vj = np.outer(v, v)

        # Power flow from node i to node j
        p_ij = (np.multiply((self.G_series + self.G_shunt).T, v ** 2).T - vi_vj *
               (self.G_series * cos_delta + self.B_series * sin_delta))
        q_ij = (-1 * np.multiply((self.B_series + self.B_shunt).T, v ** 2).T - vi_vj *
               (self.G_series * sin_delta - self.B_series * cos_delta))

        # Bus powers:
        p_i = np.sum(vi_vj * (self.G * cos_delta + self.B * sin_delta), axis=1)
        q_i = np.sum(vi_vj * (self.G * sin_delta - self.B * cos_delta), axis=1)
        self.i_ij = np.divide(np.sqrt(np.float64(p_ij ** 2 + q_ij ** 2)).T, v).T

        # Build h(x) from measurements
        # [p_i p_ij q_i q_ij U i_ij]

        # P line
        not_nan = ~np.isnan(self.ppc["branch"][:, 16+self.br_col_offset])
        ix1_first_ix = self.ppc["branch"][not_nan, 17+self.br_col_offset].real.astype(int)
        ix1_second_ix = self.ppc["branch"][not_nan, 1].real.astype(int)
        tmp = [ix for ix in range(len(ix1_first_ix)) if ix1_first_ix[ix] == ix1_second_ix[ix]]
        ix1_fb = self.ppc["branch"][not_nan, 0].real
        ix1_second_ix[tmp] = ix1_fb[tmp]

        # Q line
        not_nan = ~np.isnan(self.ppc["branch"][:, 19+self.br_col_offset])
        ix2_first_ix = self.ppc["branch"][not_nan, 20+self.br_col_offset].real.astype(int)
        ix2_second_ix = self.ppc["branch"][not_nan, 1].real.astype(int)
        tmp = [ix for ix in range(len(ix2_first_ix)) if ix2_first_ix[ix] == ix2_second_ix[ix]]
        ix2_fb = self.ppc["branch"][not_nan, 0].real
        ix2_second_ix[tmp] = ix2_fb[tmp]

        # I line
        not_nan = ~np.isnan(self.ppc["branch"][:, 13+self.br_col_offset])
        ix3_first_ix = self.ppc["branch"][not_nan, 14+self.br_col_offset].real.astype(int)
        ix3_second_ix = self.ppc["branch"][not_nan, 1].real.astype(int)
        tmp = [ix for ix in range(len(ix3_first_ix)) if ix3_first_ix[ix] == ix3_second_ix[ix]]
        ix3_fb = self.ppc["branch"][not_nan, 0].real
        ix3_second_ix[tmp] = ix3_fb[tmp]

        v_bus_not_nan = ~np.isnan(self.ppc["bus"][:, 13])
        p_bus_not_nan = ~np.isnan(self.ppc["bus"][:, 15])
        q_bus_not_nan = ~np.isnan(self.ppc["bus"][:, 17])

        hx = np.hstack((p_i[p_bus_not_nan],
                        p_ij[ix1_first_ix, ix1_second_ix],
                        q_i[q_bus_not_nan],
                        q_ij[ix2_first_ix, ix2_second_ix],
                        v[v_bus_not_nan],
                        self.i_ij[ix3_first_ix, ix3_second_ix]))

        return hx

    # Create Jacobian matrix    
    def create_jacobian(self, v, delta):
        n = len(self.ppc["bus"])
        G = self.G
        B = self.B
        G_series = self.G_series
        B_series = self.B_series
        G_shunt = self.G_shunt
        B_shunt = self.B_shunt
        
        # Create Jacobi Matrix
        # Source: p.23; Power System State Estimation by Ali Abur
            
        deltas = delta[:, np.newaxis] - delta  # delta_i - delta_j
        cos_delta = np.cos(deltas)  # cos(delta_i - delta_j)
        sin_delta = np.sin(deltas)  # sin(delta_i - delta_j)
        vi_vj = np.outer(v, v)  # Ui * Uj
        diag_n = list(range(n))  # Used for indexing the diagonal of a n x n matrix

        # Submatrices d(Pinj)/d(theta) and d(Pinj)/d(V)
        H_dPinj_dth = vi_vj * (G * sin_delta - B * cos_delta)
        H_dPinj_dth[diag_n, diag_n] = np.sum(vi_vj * (-G * sin_delta + B * cos_delta), axis=1) \
                                      - (v ** 2 * B.diagonal())

        H_dPinj_dU = np.multiply((G * cos_delta + B * sin_delta).T, v).T
        H_dPinj_dU[diag_n, diag_n] = np.sum(v * (G * cos_delta + B * sin_delta), axis=1) \
                                     + (v * G.diagonal())

        # Submatrices d(Qinj)/d(theta) and d(Qinj)/d(V)
        H_dQinj_dth = vi_vj * (-G * cos_delta - B * sin_delta)
        H_dQinj_dth[diag_n, diag_n] = np.sum(vi_vj * (G * cos_delta + B * sin_delta), axis=1) \
                                      - (v ** 2 * G.diagonal())

        H_dQinj_dU = np.multiply((G * sin_delta - B * cos_delta).T, v).T
        H_dQinj_dU[diag_n, diag_n] = np.sum(v * (G * sin_delta - B * cos_delta), axis=1) \
                                     - (v * B.diagonal())

        # Submatrices d(Pij)/d(theta) and d(Pij)/d(V)	
        # d(P01)/d(theta0) is at position H_dPij_dth_i[0,1]
        # d(P23)/d(theta3) is at position H_dPij_dth_j[2,3]
        # d(P23)/d(theta1) is 0 and not stored in the matrix
        H_dPij_dth_i = vi_vj * (G_series * sin_delta - B_series * cos_delta)
        H_dPij_dth_j = - H_dPij_dth_i

        H_dPij_dU_i = (G_series * cos_delta + B_series * sin_delta) * -v + \
                      2 * np.multiply((G_series+G_shunt).T, v).T
        H_dPij_dU_j = np.multiply((G_series * cos_delta + B_series * sin_delta).T, -v).T

        # Submatrices d(Qij)/d(theta) and d(Qij)/d(V)
        # two columns, two derivatives (theta1, theta2) per line
        # d(Q12)/d(theta_1) | d(Q12)/d(theta_2)
        H_dQij_dth_i = -vi_vj * (G_series * cos_delta + B_series * sin_delta)
        H_dQij_dth_j = -H_dQij_dth_i

        H_dQij_dU_i = (G_series * sin_delta - B_series * cos_delta) * -v - \
                      2 * np.multiply((B_series+B_shunt).T, v).T
        H_dQij_dU_j = np.multiply((G_series * sin_delta - B_series * cos_delta).T, -v).T

        # Submatrices d(Vi)/d(Vi..j)
        H_dU_dU = np.eye(n)  # diagonally 1, otherwise 0
        H_dU_dth = np.zeros((n, n))  # always 0

        # Submatrices d(Iij)/d(theta) and d(Iij)/d(V)
        # two columns, two derivatives (theta1, theta2) per line
        # d(I12)/d(theta_1) | d(I12)/d(theta_2)
        if not np.all(self.i_ij == 0.):
            H_dIij_dth_i = np.divide((G_series**2 + B_series**2) * vi_vj * sin_delta, self.i_ij)
            H_dIij_dth_j = -H_dIij_dth_i
            H_dIij_dU_i = np.divide(G_series ** 2 + B_series ** 2, self.i_ij) * \
                          (v - (v * cos_delta).T).T
            H_dIij_dU_j = np.divide(G_series ** 2 + B_series ** 2, self.i_ij) * \
                          (v - np.multiply(cos_delta.T, v).T)

        # Build H dynamically from submatrices and measurements
        columns = 2 * n
        range_theta = list(range(self.slack_bus)) + list(range(self.slack_bus + 1, n))
        range_v = list(range(n))
        
        H = np.zeros((1, columns - 1))  # create matrix with dummy line so that we can append to it

        # if P bus measurements exist
        p_bus_not_nan = ~np.isnan(self.ppc["bus"][:, 15])
        if True in p_bus_not_nan:
            nodes = np.arange(n)[p_bus_not_nan]
            H_t = H_dPinj_dth[np.tile(nodes, len(range_theta)), np.repeat(range_theta, len(nodes))]\
                .reshape(len(range_theta), len(nodes)).T
            H_U = H_dPinj_dU[np.tile(nodes, len(range_v)), np.repeat(range_v, len(nodes))]\
                .reshape(len(range_v), len(nodes)).T
            H_ = np.hstack((H_t, H_U))
            H = np.vstack((H, H_))

        # if P line measurements exist
        # and so on ..
        p_line_not_nan = ~np.isnan(self.ppc["branch"][:, 16+self.br_col_offset])
        if True in p_line_not_nan:
            ix1_first_ix = self.ppc["branch"][p_line_not_nan, 17+self.br_col_offset].real.astype(int)
            ix1_second_ix = self.ppc["branch"][p_line_not_nan, 1].real.astype(int)
            tmp = [ix for ix in range(len(ix1_first_ix)) if ix1_first_ix[ix] == ix1_second_ix[ix]]
            ix1_fb = self.ppc["branch"][p_line_not_nan, 0].real
            ix1_second_ix[tmp] = ix1_fb[tmp]
            node1 = ix1_first_ix
            node2 = ix1_second_ix

            nr = range(0, len(node1))
            H_ = np.zeros((len(node1), columns))
            H_[nr, node1] = H_dPij_dth_i[node1, node2]
            H_[nr, node2] = H_dPij_dth_j[node1, node2]
            H_[nr, n+node1] = H_dPij_dU_i[node1, node2]
            H_[nr, n+node2] = H_dPij_dU_j[node1, node2]
            H_ = np.delete(H_, 0, 1)
            H = np.vstack((H, H_))

        q_bus_not_nan = ~np.isnan(self.ppc["bus"][:, 17])
        if True in q_bus_not_nan:
            nodes = np.arange(n)[q_bus_not_nan]
            H_t = H_dQinj_dth[np.tile(nodes, len(range_theta)), np.repeat(range_theta, len(nodes))]\
                .reshape(len(range_theta), len(nodes)).T
            H_U = H_dQinj_dU[np.tile(nodes, len(range_v)), np.repeat(range_v, len(nodes))]\
                .reshape(len(range_v), len(nodes)).T
            H_ = np.hstack((H_t, H_U))
            H = np.vstack((H, H_))

        q_line_not_nan = ~np.isnan(self.ppc["branch"][:, 19+self.br_col_offset])
        if True in q_line_not_nan:
            ix2_first_ix = self.ppc["branch"][q_line_not_nan, 20+self.br_col_offset].real.astype(int)
            ix2_second_ix = self.ppc["branch"][q_line_not_nan, 1].real.astype(int)
            tmp = [ix for ix in range(len(ix2_first_ix)) if ix2_first_ix[ix] == ix2_second_ix[ix]]
            ix2_fb = self.ppc["branch"][q_line_not_nan, 0].real
            ix2_second_ix[tmp] = ix2_fb[tmp]
            node1 = ix2_first_ix
            node2 = ix2_second_ix

            nr = range(0, len(node1))
            H_ = np.zeros((len(node1), columns))
            H_[nr, node1] = H_dQij_dth_i[node1, node2]
            H_[nr, node2] = H_dQij_dth_j[node1, node2]
            H_[nr, n+node1] = H_dQij_dU_i[node1, node2]
            H_[nr, n+node2] = H_dQij_dU_j[node1, node2]
            H_ = np.delete(H_, 0, 1)
            H = np.vstack((H, H_))

        v_bus_not_nan = ~np.isnan(self.ppc["bus"][:, 13])
        if True in v_bus_not_nan:
            nodes = np.arange(n)[v_bus_not_nan]
            H_t = H_dU_dth[np.repeat(range_theta, len(nodes)), np.tile(nodes, len(range_theta))]\
                .reshape(len(range_theta), len(nodes)).T
            H_U = H_dU_dU[np.repeat(range_v, len(nodes)), np.tile(nodes, len(range_v))]\
                .reshape(len(range_v), len(nodes)).T
            H_ = np.hstack((H_t, H_U))
            H = np.vstack((H, H_))

        i_line_not_nan = ~np.isnan(self.ppc["branch"][:, 13+self.br_col_offset])
        if True in i_line_not_nan:
            ix3_first_ix = self.ppc["branch"][i_line_not_nan, 14+self.br_col_offset].real.astype(int)
            ix3_second_ix = self.ppc["branch"][i_line_not_nan, 1].real.astype(int)
            tmp = [ix for ix in range(len(ix3_first_ix)) if ix3_first_ix[ix] == ix3_second_ix[ix]]
            ix3_fb = self.ppc["branch"][i_line_not_nan, 0].real
            ix3_second_ix[tmp] = ix3_fb[tmp]
            node1 = ix3_first_ix
            node2 = ix3_second_ix

            nr = range(0, len(node1))
            H_ = np.zeros((len(node1), columns))

            if not np.all(self.i_ij==0.):
                H_[nr, node1] = H_dIij_dth_i[node1, node2]
                H_[nr, node2] = H_dIij_dth_j[node1, node2]
                H_[nr, n+node1] = H_dIij_dU_i[node1, node2]
                H_[nr, n+node2] = H_dIij_dU_j[node1, node2]
            H_ = np.delete(H_, 0, 1)
            H = np.vstack((H, H_))    
        
        H = H[1:, :]  # delete dummy line
        return H

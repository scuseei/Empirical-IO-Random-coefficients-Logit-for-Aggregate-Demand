# -*- coding: utf-8 -*-
"""
Task
-------
RC Logit for Aggregate Demand

Version      |Author       |Affiliation                |Email
--------------------------------------------------------------------------
Feb 25, 2018 |Chenshuo Sun |Stern Business School, NYU |csun@stern.nyu.edu

Goal(s)
-------
Bayesian estimation approach
"""

import numpy as np
import pandas as pd
import line_profiler
profiler = line_profiler.LineProfiler()


class Bayesian(object):
    """Class for Bayesian estimation
    """

    def __init__(self, a0, A0, b0, B0, v0, S0, m, R):
        """Class initialization
        """
        self.a0 = a0
        self.A0 = A0
        self.b0 = b0
        self.B0 = B0
        self.v0 = v0
        self.S0 = S0
        self.m = m
        self.R = R

    def __data_loader(self):
        """Function for loading data
        """
        m = self.m
        file_name = 'Data_' + str(m)
        # print(file_name + ' loaded')
        data = pd.read_pickle(file_name)
        return data

    @profiler
    def __draw_theta(self, theta_r_1, w_r, SIGMA_r, data):
        """Function for drawing theta given w and SIGMA
        """
        # n_hat
        p_jt = data['p_jt']
        one_jt = pd.Series(1, index=np.arange(len(data['z_jt'])))
        Z_jt = pd.concat([one_jt, data['z_jt']], axis=1)
        n_hat_jt = p_jt - np.dot(Z_jt, w_r)
        # e and n and rho
        e_sig = np.sqrt(SIGMA_r[0][0])
        en_sig = SIGMA_r[0][1]
        n_sig = np.sqrt(SIGMA_r[1][1])
        rho = en_sig / e_sig / n_sig
        denom = np.sqrt((1 - rho**2) * e_sig**2)
        # X and A
        X_jt = pd.concat([one_jt, data['x_jt'], data['p_jt']], axis=1)
        X_hat_jt = X_jt / denom
        A_r = np.linalg.inv(np.dot(X_hat_jt.T, X_hat_jt))
        # delta and a
        v1_jt = np.random.normal(0, 1, 1)
        d_jt = np.dot(X_jt, theta_r_1) + e_sig / n_sig * rho * \
            n_hat_jt + np.sqrt((1 - rho**2) * e_sig**2) * v1_jt
        d_hat_jt = (d_jt - e_sig / n_sig * rho * n_hat_jt) / denom
        # OLS
        a_r = np.dot(
            np.linalg.inv(
                np.dot(
                    X_hat_jt.T, X_hat_jt)), np.dot(
                X_hat_jt.T, d_hat_jt))
        # draw
        theta_r = np.random.multivariate_normal(a_r, A_r)
        data['d_jt'] = d_jt
        return theta_r, data

    @profiler
    def __draw_w(self, w_r_1, theta_r, SIGMA_r, data):
        """Function for drawing w given theta and SIGMA
        """
        # e_hat
        one_jt = pd.Series(1, index=np.arange(len(data['z_jt'])))
        X_jt = pd.concat([one_jt, data['x_jt'], data['p_jt']], axis=1)
        Z_jt = pd.concat([one_jt, data['z_jt']], axis=1)
        e_hat_jt = data['d_jt'] - np.dot(X_jt, theta_r)
        # e and n and rho
        e_sig = np.sqrt(SIGMA_r[0][0])
        en_sig = SIGMA_r[0][1]
        n_sig = np.sqrt(SIGMA_r[1][1])
        rho = en_sig / e_sig / n_sig
        denom = np.sqrt((1 - rho**2) * n_sig**2)
        # Z and B
        Z_hat_jt = Z_jt / denom
        B_r = np.linalg.inv(np.dot(Z_hat_jt.T, Z_hat_jt))
        # p and b
        v2_jt = np.random.normal(0, 1, 1)
        p_jt = np.dot(Z_jt, w_r_1) + n_sig / e_sig * rho * \
            e_hat_jt + np.sqrt((1 - rho**2) * n_sig**2) * v2_jt
        p_hat_jt = (p_jt - n_sig / e_sig * rho * e_hat_jt) / denom
        b_r = np.dot(
            np.linalg.inv(
                np.dot(
                    Z_hat_jt.T, Z_hat_jt)), np.dot(
                Z_hat_jt.T, p_hat_jt))
        # draw
        w_r = np.random.multivariate_normal(b_r, B_r)
        return w_r

    @profiler
    def __draw_SIGMA(self, theta_r, w_r, data):
        """Function for drawing w given theta and w
        """
        # e_hat and n_hat
        one_jt = pd.Series(1, index=np.arange(len(data['z_jt'])))
        X_jt = pd.concat([one_jt, data['x_jt'], data['p_jt']], axis=1)
        Z_jt = pd.concat([one_jt, data['z_jt']], axis=1)
        e_hat = data['d_jt'] - np.dot(X_jt, theta_r)
        n_hat = data['p_jt'] - np.dot(Z_jt, w_r)
        # S_hat
        JT = len(n_hat)
        S_hat = np.zeros([2, 2])
        for jt in range(JT):
            e_hat_jt = e_hat[jt]
            n_hat_jt = n_hat[jt]
            e_n_jat_jt = np.matrix([e_hat_jt, n_hat_jt])
            S_hat += np.dot(e_n_jat_jt.T, e_n_jat_jt)
        S_hat /= JT
        # v1 and S1
        v1 = 2 + JT
        I = np.identity(2)
        S1 = (2 * I + JT * S_hat) / (2 + JT)
        # draw SIGMA
        from scipy.stats import invwishart
        SIGMA_r = invwishart.rvs(v1, S1)
        return SIGMA_r

    @profiler
    def draw_first_part(self):
        """Function for the first step draw
        """
        # get from class
        a0 = self.a0
        A0 = self.A0
        b0 = self.b0
        B0 = self.B0
        v0 = self.v0
        S0 = self.S0
        data = self.__data_loader()
        # store value
#        theta = pd.DataFrame(index=range(R), columns=['a0', 'a1', 'b_mean'])
#        w = pd.DataFrame(index=range(R), columns=['w0', 'w1'])
#        SIGMA = pd.DataFrame(
#            index=range(R), columns=[
#                'e_sig', 'en_sig', 'n_sig'])
        # initial value
        theta_ini = np.random.multivariate_normal(a0, A0)
        w_ini = np.random.multivariate_normal(b0, B0)
        from scipy.stats import invwishart
        SIGMA_ini = invwishart.rvs(v0, S0)

        theta_r, data = self.__draw_theta(
                theta_ini, w_ini, SIGMA_ini, data)
        # draw_w
        w_r = self.__draw_w(w_ini, theta_r, SIGMA_ini, data)
        # draw_SIGMA
        SIGMA_r = self.__draw_SIGMA(theta_r, w_r, data)
        print('First step draw')
        # return
        return theta_r, w_r, SIGMA_r

    @profiler
    def draw_second_part(self):
        """Function for the second step draw
        """
        # get from class


        # return
        return None
  
    @profiler
    def iteration(self):
        """Function for the main iteration
        """
        # get from class


        # return
        return None


@profiler
def main():
    """Main function for generating results
    """
    # set initial values
    a0 = [3.0, 1.5, -1.0]
    A0 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b0 = [5.0, 2.0]
    B0 = [[1, 0], [0, 1]]
    v0 = 2
    S0 = [[1, 0], [0, 1]]
    m = 1

    # get estimation results
    est = Bayesian(a0, A0, b0, B0, v0, S0, m, R=5)
    est.draw_first_part()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Task
-------
RC Logit for Aggregate Demand

Version      |Author       |Affiliation                |Email
--------------------------------------------------------------------------
Feb 26, 2018 |Chenshuo Sun |Stern Business School, NYU |csun@stern.nyu.edu

Goal(s)
-------
BLP GMM estimation approach
"""

import numpy as np
import pandas as pd
import itertools
import line_profiler
profiler = line_profiler.LineProfiler()


class GMM(object):
    """Class for Bayesian estimation
    """

    def __init__(self, m, D, J, T, e, delta_ini, theta2_ini):
        """Class initialization
        """
        self.m = m
        self.D = D
        self.J = J
        self.T = T
        self.e = e
        self.delta_ini = delta_ini
        self.theta2_ini = theta2_ini

    def __data_loader(self):
        """Function for loading data
        """
        m = self.m
        file_name = 'Data_' + str(m)
        # print(file_name + ' loaded')
        data = pd.read_pickle(file_name)
        return data

    @profiler
    def __update_delta(self, data, X_jt, s_o_jt, v_i, theta2, delta_ini):
        """Function for computing market share given delta and theta_2
        """
        D = self.D
        J = self.J
        T = self.T
        sum_s = pd.Series(0, index=data.index.values)
        p_jt = data['p_jt']
        
        for d in range(D):
            v = v_i[d]
            mu_ijt = theta2 * v * p_jt
            demon = pd.DataFrame(
                [],
                index=range(
                    1,
                    T + 1),
                columns=[0],
                dtype=object)
            for t in range(1, T + 1):
                tmp = 1
                for k in range(1, J + 1):
                    inx = (k, t)
                    tmp += np.exp(delta_ini.get_value(inx, 0) +
                                  mu_ijt.get_value(inx, 0))
                demon.set_value(t, 0, tmp)
            JT = list(itertools.product(range(1, J + 1), range(1, T + 1)))
            tmp_s = pd.Series(0, index=data.index.values)
            for (j, t) in JT:
                dem = demon.loc[t][0]
                inx = (j, t)
                nom = np.exp(
                    delta_ini.get_value(
                        inx,
                        0) +
                    mu_ijt.get_value(
                        inx,
                        0))
                tmp_s.loc[inx] = nom / dem
            sum_s += tmp_s
        s_jt = sum_s / D
        delta_l = delta_ini + \
            pd.DataFrame(np.log(s_o_jt) - np.log(s_jt), index=data.index.values)
        return delta_l

    @profiler
    def inner_loop(self, params):
        """Function for computing GMM obj function
        """
        # get from class
        theta2 = params
        data = self.__data_loader()
        data = data.set_index(['J', 'T'])
        delta_ini = self.delta_ini
        e = self.e
        # variables
        one_jt = pd.Series(1, index=data.index.values)
        X_jt = pd.concat([one_jt, data['x_jt'], data['p_jt']], axis=1)
        Z_jt = pd.concat([one_jt, data['z_jt']], axis=1)
        delta_ini = pd.DataFrame(delta_ini, index=data.index.values)
        s_o_jt = data['s_jt']
        v_i = np.random.normal(0, 1, self.D)
        # initial value before loop
        delta_l_1 = pd.DataFrame(delta_ini)
        delta_l = self.__update_delta(
            data, X_jt, s_o_jt, v_i, theta2, delta_ini)

        # begin loop
        print(
            '------------\n' +
            'inner loop for getting theta1 and GMM for theta2')
        while (np.max(np.abs(delta_l - delta_l_1)) > e).bool():
            print(np.max(np.abs(delta_l - delta_l_1)))
            # update delta
            delta_l_1 = pd.DataFrame(delta_l)
            delta_l = self.__update_delta(
                data, X_jt, s_o_jt, v_i, theta2, delta_l_1)  # delta_l
        # estimate theta_1
        theta_hat_1 = np.dot(
            np.linalg.inv(
                np.dot(
                    X_jt.T, X_jt)), np.dot(
                X_jt.T, delta_l))
        # compute e_hat_jt
        e_hat_jt = delta_l - np.dot(X_jt, theta_hat_1)
        # return obj
        g_theta2 = np.dot(e_hat_jt.T, Z_jt)
        Wn = np.dot(Z_jt.T, Z_jt)
        obj_value = np.dot(np.dot(g_theta2, Wn), g_theta2.T)
        return obj_value
    

@profiler
def main():
    """Main function for generating results
    """
    # set initial values
    m = 1
    D = 10
    J = 10
    T = 50
    e = 1e-1
    delta_ini = np.zeros(J * T)
    theta2_ini = 1.0

    # get estimation results
    est = GMM(m, D, J, T, e, delta_ini, theta2_ini)
    
    # outer loop
    from scipy.optimize import minimize
    params = theta2_ini
    theta2 = minimize(
        est.inner_loop,
        params,
        method='L-BFGS-B',
        options={
                'gtol': 0.01,
                'disp': True})
    obj_value, theta1 = est.inner_loop(theta2)
    
    # print result
    print('-------------\nResults are: ')
    print('\ntheta1: ' + str(theta1))
    print('\ntheta2: ' + str(theta2))

if __name__ == "__main__":
    main()

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
Simulate datasets, which consist of market share, product characteristics and
prices, for each product j, and market t.
"""

import numpy as np
import pandas as pd
import itertools
import line_profiler
profiler = line_profiler.LineProfiler()


class DataSimulator(object):
    """Class for data simulating
    """

    def __init__(self, J, T, params, x_jt, z_jt, M):
        """Class initialization
        """
        self.J = J
        self.T = T
        self.params = params
        self.x_jt = x_jt
        self.z_jt = z_jt
        self.M = M

    @profiler
    def characteristics(self):
        """Function for simulating the product characteristics for each (j, t)
        """
        # get params from class
        J = self.J
        T = self.T
        x_jt = self.x_jt
        # initialize
        JT = list(itertools.product(range(1, J + 1), range(1, T + 1)))
        data = pd.DataFrame(
            index=JT,
            columns=['x_jt'],
            dtype='float64').fillna(0)
        # generate x_jt
        x_mean = x_jt[1]
        x_std = x_jt[2]
        for jt in JT:
            data.loc[jt, 'x_jt'] = np.random.normal(x_mean, x_std, 1)
        # return
        return data

    @profiler
    def prices_and_share(self):
        """Function for simulating price and market share for each (j, t)
        """
        # get params and return from class
        J = self.J
        T = self.T
        JT = list(itertools.product(range(1, J + 1), range(1, T + 1)))
        params = self.params
        z_jt = self.z_jt
        data = self.characteristics()
        #
        data['e_jt'] = pd.Series(
            0,
            index=data.index.values,
            dtype='float64').fillna(0)
        data['n_jt'] = pd.Series(
            0,
            index=data.index.values,
            dtype='float64').fillna(0)
        data['z_jt'] = pd.Series(
            0,
            index=data.index.values,
            dtype='float64').fillna(0)
        data['p_jt'] = pd.Series(
            0,
            index=data.index.values,
            dtype='float64').fillna(0)
        data['d_jt'] = pd.Series(
            0,
            index=data.index.values,
            dtype='float64').fillna(0)
        data['u_ijt'] = pd.Series(
            0,
            index=data.index.values,
            dtype=object)
        data['nom_ijt'] = pd.Series(
            0,
            index=data.index.values,
            dtype=object)
        data['denom_it'] = pd.Series(
            0,
            index=data.index.values,
            dtype=object)
        data['s_jt'] = pd.Series(
            0,
            index=data.index.values,
            dtype='float64').fillna(0)
        # generate correlated error terms
        e_sig = params[-3]
        en_sig = params[-2]
        n_sig = params[-1]
        en_meam = [0, 0]
        en_cov = [[e_sig**2, en_sig], [en_sig, n_sig**2]]
        en = np.random.multivariate_normal(en_meam, en_cov, len(data))
        data['e_jt'] = en[:, 0]
        data['n_jt'] = en[:, 1]
        # generate z_jt and p_jt thereafter
        z_mean = z_jt[1]
        z_std = z_jt[2]
        for jt in JT:
            data.loc[jt, 'z_jt'] = np.random.normal(z_mean, z_std, 1)
        w0 = params[-5]
        w1 = params[-4]
        for (jt, r) in zip(JT, range(len(data))):
            data.loc[jt, 'p_jt'] = w0 + w1 * \
                data.iloc[r]['z_jt'] + data.iloc[r]['n_jt']
        # generate d_jt and u_ijt for s_jt
        a0 = params[0]
        a1 = params[1]
        beta_mean = params[2]
        beta_std = params[3]
        for (jt, r) in zip(JT, range(len(data))):
            data.loc[jt, 'd_jt'] = a0 + a1 * data.iloc[r]['x_jt'] + \
                beta_mean * data.iloc[r]['p_jt'] + data.iloc[r]['e_jt']  # d_jt
        n = 1000
        v_i = np.random.normal(0, 1, n)
        # Phi_v_i = norm.cdf(v_i)
        for (jt, r) in zip(JT, range(len(data))):
            p_jt = data.iloc[r]['p_jt']
            u_ijt = v_i * beta_std * p_jt
            data.set_value(jt, 'u_ijt', u_ijt)  # u_ijt
            d_jt = data.iloc[r]['d_jt']
            nom = np.exp([d_jt] * n + u_ijt)
            data.set_value(jt, 'nom_ijt', nom)  # nom
        data_ = data.reset_index()
        data_['J'] = data_['index'].str[0]
        data_['T'] = data_['index'].str[1]
        data_ = data_.set_index(['J', 'T'])
        demon = pd.DataFrame(
            [],
            index=range(
                1,
                T + 1),
            columns=['demon'],
            dtype=object)
        for t in range(1, T + 1):
            tmp = [1] * n
            for k in range(1, J + 1):
                tmp += np.exp([data_.loc[(k, t), 'd_jt']] * n +
                              data_.loc[(k, t), 'u_ijt'])
            demon.set_value(t, 'demon', tmp)
        for (j, t) in JT:
            tmp = demon.loc[t].loc['demon']
            data_.set_value((j, t), 'denom_it', tmp)  # denom
        # calculate s_jt
        for (jt, r) in zip(JT, range(len(data_))):
            fun = data_.iloc[r]['nom_ijt'] / data_.iloc[r]['denom_it']
            tmp = np.sum(fun) / n  # might be problematic
            data_.set_value(jt, 's_jt', tmp)
        data = data_.reset_index()
        data = data[['J', 'T', 's_jt', 'x_jt', 'z_jt', 'p_jt']]
        # return
        return data

    @profiler
    def make_data(self):
        """Function for simulating the utility for each (i, t)
        """
        M = self.M
        for m in range(M):
            file_name = 'Data_' + str(m + 1) 
            data = self.prices_and_share()
            data.to_pickle(file_name)
            file_name = file_name + '.csv'
            data.to_csv(file_name)

@profiler
def main():
    """Main function for generating results
    """
    # set params
    J = 10
    T = 50
    a0 = 3.0
    a1 = 1.5
    beta_mean = -1.0
    beta_sig = 0.5
    w0 = 5.0
    w1 = 2.0
    e_sig = 1.0
    en_sig = 0.5
    n_sig = 2.0
    params = [a0, a1, beta_mean, beta_sig, w0, w1, e_sig, en_sig, n_sig]
    x_jt = ['Normal', 1, 0.5]
    z_jt = ['Normal', 1, 0.5]
    M = 5

    # get simulation data
    ds = DataSimulator(J, T, params, x_jt, z_jt, M)
    ds.make_data()


if __name__ == "__main__":
    main()

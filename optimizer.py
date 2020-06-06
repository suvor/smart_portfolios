import numpy as np
import pandas as pd
import math
import cvxpy as cp
import itertools
import matplotlib.pyplot as plt


def get_vector(l):
    return np.array(l).reshape(len(l), 1)


# a_list = [1, 2]
# b_list = [3, 4]
#
# a = get_vector(a_list)
# b = get_vector(b_list)

data_old = {"SP": {"AM": 8.2/100, "GM": 6.23/100, "std": 19.8/100, "SR": 0.315},
            "10Y": {"AM": 2.2/100, "GM": 1.86/100, "std": 8.27/100, "SR": 0.225}}

data = {"SP": {"AM": 5/100, "GM": 3/100, "std": 19.8/100, "SR": 0.15},
        "10Y": {"AM": 1.6/100, "GM": 1.3/100, "std": 8.27/100, "SR": 0.15},
        "BOND": {"AM": 0.4/100, "GM": 0.38/100, "std": 2.07/100, "SR": 0.18}}

data_old_df = pd.DataFrame(data_old).transpose()
data_df = pd.DataFrame(data).transpose()


def get_vector(l):
    return np.array(l).reshape(len(l), 1)


def sigma_2(weights, cov):
    return weights.T.dot(cov).dot(weights)[0, 0]


def sigma(weights):
    return math.sqrt(sigma_2(weights, cov))


def r_geom(weights, cov, r):
    return (weights.T.dot(r) - 0.5 * sigma_2(weights, cov))[0, 0]


def sharpe_geom(weights, cov, r):
    return r_geom(weights, cov, r) / math.sqrt(sigma_2(weights, cov))


# calculates covariance matrix from correlation matrix
def cov(correl, std):
    return std.dot(std.T) * correl


r = get_vector(data_old_df.loc[:, "AM"])
std = get_vector(data_old_df.loc[:, "std"])
correl = np.array([[1, 0.05], [0.05, 1]])
weights = get_vector([0.6, 0.4])
cov = cov(correl, std)

# def r_geom_f(weights):
#     sigma_2 = ((weights.T).dot(cov)).dot(weights)
#     return weights.T.dot(r) - 0.5 * sigma_2
#
# w1 = cp.Variable()
# w2 = cp.Variable()
#
# constraints = [w1 >= 0, w1 <= 1]
# obj = cp.Maximize(r_geom_f(get_vector([w1, 1 - w1])))

n = 100
w1 = np.linspace(0, 1, n + 1)
w2 = 1 - w1

weights = [get_vector([w1[i], w2[i]]) for i in range(0, n + 1)]

r_geom_array = np.array([r_geom(w, cov, r) for w in weights])
sharpe_geom_array = np.array([sharpe_geom(w, cov, r) for w in weights])
sigma_array = np.array([sigma(w) for w in weights])

# max geom return
weights[np.argmax(r_geom_array)]

# max SR
weights[np.argmax(sharpe_geom_array)]

# min vol
weights[np.argmin(sigma_array)]

plt.figure(figsize=(12, 8))
plt.scatter(sigma_array, r_geom_array, c=sharpe_geom_array, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')



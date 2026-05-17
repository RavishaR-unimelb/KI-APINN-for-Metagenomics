import math
import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta

import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from IPython import display

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

import statsmodels.api as sm
import statsmodels.formula.api as smf

import networkx as nx

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import datetime


# method for initializing the constants
# n - number of bacteria
# alpha - for the power law distribution
# s - scaling component
# time - number of timepoints to generate

def init_sim(n=100, alpha=5, s=1, time=300, pr=False, ER=True, p=0.5, sigma=1):
    N = get_N(n)
    H = get_H(alpha, n)
    if (ER):
        G = get_G_ER(n, p)
    else:
        G = get_G(n)
    A = calc_A(N, H, G, s)
    r = get_r(n)

    if pr:
        print("n=", n, "alpha=", alpha, "s=", s, "time=", time)

    return A, time, r

def get_N(n, sigma=1):
    N = np.random.normal(0, sigma, (n, n))
    np.fill_diagonal(N, 0)
    return N

def get_H(alpha, n):
    h_ = np.random.power(alpha, n)
    h_ = h_/np.mean(h_)
    H = np.diag(h_)
    return H

def get_G(n):
    G = np.ones((n, n))
    np.fill_diagonal(G, 0)
    return G

def get_G_ER(n, p = 0.5):
    G = np.asarray(nx.to_numpy_matrix(nx.gnp_random_graph(n, p, directed = True)))
    np.fill_diagonal(G, 0)
    return G

def calc_A(N, H, G, s):
    NH = np.matmul(N, H)
    A = np.multiply(NH, G)
    A = A*s
    np.fill_diagonal(A, -1)
    return A

def init_abandunce_matrix(n, time):
    x = np.empty([n, time])
    x[:, 0] = np.random.power( 1, n)
    x[:, 0]
    return x

def get_r(n):
    r = np.random.uniform(0, 1, n)
    return r


def create_abundance_profile(n=10, timesteps=300, sigma=1, alpha=5, p=0.5):
    n = n
    pr = True
    ER = True
    time = timesteps
    # sending time and returning time is obsolete - just keeping it as it is for now

    A, time, r = init_sim(n=n, time=time, pr=pr, ER=ER, sigma=sigma, alpha=alpha, p=p)
    x = init_abandunce_matrix(n, time)

    # Propagating the new abundance profile
    for i in tqdm_notebook(range(1, time)):
        x_dot = np.matmul(np.diag(x[:, i - 1]), r + np.matmul(A, x[:, i - 1]))
        noise = 0
        noise = np.random.normal(0, 0.1, n)
        # noise = np.random.power(1.01, n)
        # print x[:, 0]
        # print x_dot
        x[:, i] = (x[:, i - 1] + x_dot + noise).clip(0)
        # print x[:, i]

    plotly.offline.init_notebook_mode()

    data = []

    for i in tqdm_notebook(range(0, n)):
        data.append(go.Scatter(
            x=range(0, time),
            y=x[i, :] / max(1, min(x[i, :])),
            mode='lines',
            name=str(i),
            hoverinfo='name+y',
        ))

    layout = dict(
        legend=dict(
            y=0.5,
            font=dict(
                size=16
            )
        )
    )

    fig = dict(data=data, layout=layout)
    plotly.offline.iplot(fig)

    # Normalize data before stackplot representation

    px = x
    # px is the percentaged x

    for i in range(timesteps):
        sum_1 = px[:, i][px[:, i] > 0].sum()
        px[:, i] = px[:, i] * 100 / sum_1

    # Stack plot representation

    ys = px.tolist()
    xs = range(0, timesteps)

    plt.figure(figsize=(100, 20))

    plt.stackplot(xs, *ys)

    plt.legend()
    plt.show()

    return x, px, A


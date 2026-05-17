# import statements

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import pandas as pd


def GLV(XC1, A, T, N):
    '''
    Run GLV to populate XC1 over T timepoints for N OTUs, with the interaction Matrix A
    :param XC1: Initial Population, vector with N elements
    :param A: NxN matrix, interaction parameters
    :param T: Timepoints
    :param N: Number of OTUs
    :return: X - the populated abundance profile
    '''

    time = T

    X = np.empty([N, time])
    X[:, 0] = XC1

    for i in range(1, time):
        for j in range(0, N):
            x_dot = 0
            for k in range(0, N):
                x_dot += X[j, i-1] * A[j, k] * X[k, i-1]

            X[j, i] = max( X[j, i-1] + x_dot, 0)

    return X

def GLV_with_percentage(XC1, A, T, N, r, DEBUG=False):
    '''
    Run GLV to populate XC1 over T timepoints for N OTUs, with the interaction Matrix A
    :param XC1: Initial Population, vector with N elements
    :param A: NxN matrix, interaction parameters
    :param T: Timepoints
    :param N: Number of OTUs
    :return: X - the populated abundance profile
    '''

    #print("Static GLV")

    time = T

    X = np.empty([N, time])
    X[:, 0] = XC1

    DEBUG = False

    if DEBUG:
        print(XC1)
        print(X[:, 0])

    for i in range(1, time):
        if DEBUG:
            print(i)
            print(X[:, i-1])
        for j in range(0, N):
            # r term
            x_dot = r[j] * X[j, i-1]
            for k in range(0, N):
                x_dot += X[j, i-1] * A[j, k] * X[k, i-1]
                if DEBUG:
                    print(x_dot, X[j, i-1], X[k, i-1], A[j, k])   # TODO should not ignore the birth rate
            X[j, i] = max( X[j, i-1] + x_dot, 0)

        summ = sum(X[:, i])
        if summ != 0:
            # This was * 100 / summ before - figure out why?!?!?!?!
            X[:, i] = X[:, i] * 100 / summ

        # print("sum", sum(X[:, i]))

    return X

def dynamic_GLV_with_percentage(XC1, A, T, N, r, DEBUG=False):
    '''
    Run GLV to populate XC1 over T timepoints for N OTUs, with the dynamic interaction Matrix A
    :param XC1: Initial Population, vector with N elements
    :param A: NxNxT matrix, dynamic interaction parameters
    :param T: Timepoints
    :param N: Number of OTUs
    :return: X - the populated abundance profile
    '''

    #print("Dynamic GLV")

    time = T

    X = np.empty([N, time])
    X[:, 0] = XC1

    DEBUG = False

    if DEBUG:
        print(XC1)
        print(X[:, 0])

    for i in range(1, time):
        if DEBUG:
            print(i)
            print(X[:, i-1])
        for j in range(0, N):
            # r term
            x_dot = r[j] * X[j, i-1]
            for k in range(0, N):
                x_dot += X[j, i-1] * A[j, k, i] * X[k, i-1]
                if DEBUG:
                    print(x_dot, X[j, i-1], X[k, i-1], A[j, k, i])   # TODO should not ignore the birth rate
            X[j, i] = max( X[j, i-1] + x_dot, 0)

        summ = sum(X[:, i])
        if summ != 0:
            # This was * 100 / summ before - figure out why?!?!?!?!
            X[:, i] = X[:, i] * 100 / summ

        # print("sum", sum(X[:, i]))

    return X

def BCD(X, K, T, N):

    average_BCD = 0

    for t in range(T):
        diff = 0
        sum = 0
        for i in range(N):
            diff += abs(X[i, t] - K[i, t])
            sum += (X[i, t] + K[i, t])
        #print(diff, sum)
        bcd = (diff*1.0) / (sum*1.0)
        #print(bcd)
        average_BCD += bcd

    average_BCD /= T
    return average_BCD

def InitPop_Random(N, pop_size=10000):
    population = []
    gene_size = int(N*N*0.1)
    for i in range(pop_size):
        gene = []
        for j in range(gene_size):
            I = np.random.randint(0, N)
            J = np.random.randint(0, N)
            A_ij = np.random.normal(0, 0.1)
            gene.append((I, J, A_ij))
        population.append(gene)
    return population

def printPopulation(population):
    for gene in population:
        for tuple in gene:
            print(tuple[0], tuple[1], tuple[2])


def draw_network(G, pos, ax, sg=None):
    for n in G:
        c = Circle(pos[n], radius=0.02, alpha=0.5, color='green')
        label = ax.annotate(G.node[n]['name'], pos[n], fontsize=10, ha="center")
        ax.add_patch(c)
        G.node[n]['patch'] = c
        x, y = pos[n]
    seen = {}
    for (u, v, d) in G.edges(data=True):
        n1 = G.node[u]['patch']
        n2 = G.node[v]['patch']
        rad = 0.1
        if (u, v) in seen:
            rad = seen.get((u, v))
            rad = (rad + np.sign(rad) * 0.1) * -1
        alpha = 0.5
        color = d['color']

        #print d['weight'], hasattr(G[u][v], 'weight')

        e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=%s' % rad,
                            mutation_scale=10.0,
                            lw=d['weight'],
                            alpha=alpha,
                            color=color)
        seen[(u, v)] = rad
        ax.add_patch(e)
    return e

fig_x = 20
fig_y = 10


def draw_netowrk_graph(path):

    plt.figure(figsize=(fig_x, fig_y))
    G1 = nx.MultiDiGraph()

    ground_truth = pd.read_csv(path).to_numpy()
    ground_truth = ground_truth[:, 1:]

    matrix = ground_truth
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    ls = []
    for i in range(rows):
        ls.append(str(i))

    for x in range(0, rows):
        G1.add_node(x, name=ls[x])

    for x in range(0, rows):
        for y in range(0, cols):
            if x != y:
                if(matrix[x][y] > 0):
                    G1.add_edge(x, y, weight=pow(matrix[x][y], 2)*2, color="blue")
                else:
                    G1.add_edge(x, y, weight=pow(matrix[x][y] * -1, 2)*2, color="red")

    pos = nx.shell_layout(G1)
    ax = plt.gca()
    draw_network(G1, pos, ax)
    ax.autoscale()
    # plt.show()
    plt.savefig(path+"_interactions_graph.svg")

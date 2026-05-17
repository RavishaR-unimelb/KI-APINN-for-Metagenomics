from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from .tools import GLV_with_percentage
from .tools import BCD
from .tools import draw_network
import numpy as np
import networkx as nx
import pandas as pd


class MIN:
    def __init__(self, a, r):
        self.A = a
        self.r = r

    def heat_map(self, figsize=(20, 10), img_path=None):
        plt.figure(figsize=figsize)
        plt.imshow(self.A, cmap='hot', interpolation='nearest')

        if img_path is None:
            plt.show()
        else:
            plt.savefig(img_path)

    def get_accuracy(self, dataset, start, delta_t):
        count = self.A.shape[0]
        x_c1 = dataset.matrix[:, start]

        # TODO: replace this with a proper GLV with percentage tool
        x = GLV_with_percentage(x_c1, self.A, delta_t, count, self.r)

        bcd = BCD(x, dataset.matrix[:, start:], delta_t, count)

        score = 1 - bcd
        if np.isnan(score):
            return 0
        else:
            return score

    def draw_interaction_graph_classic(self, dataset, figsize=(20, 10), img_path=None):
        # list of bacteria names
        ls = list(dataset.df.index)

        plt.figure(figsize=figsize)

        G = nx.MultiDiGraph()

        matrix = self.A
        rows = matrix.shape[0]
        cols = matrix.shape[1]


        for x in range(0, rows):
            str = ls[x]
            ind = str.rfind('_')
            str = str[ind+1:]
            G.add_node(x, name=str)

        for x in range(0, rows):
            for y in range(0, cols):
                if x != y:
                    if(matrix[x][y] > 0):
                        G.add_edge(x, y, weight=pow(matrix[x][y], 2), color="blue")
                    else:
                        G.add_edge(x, y, weight=pow(matrix[x][y] * -1, 2), color="red")

        pos = nx.shell_layout(G)
        ax = plt.gca()
        draw_network(G, pos, ax)
        ax.autoscale()

        if img_path is None:
            plt.show()
        else:
            plt.savefig(img_path)

    def get_interactions_as_DF(self, percentage=0.5, _abs=False):
        interactions_list = []
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if i != j:
                    _dict = {
                        'source': i,
                        'target': j,
                        'value': (self.A[i][j], abs(self.A[i][j]))[_abs]
                    }
                    interactions_list.append(_dict)
        interactions_df = pd.DataFrame(interactions_list)

        df_temp = interactions_df

        #n largest
        if percentage < 1.0:
            df_temp = df_temp.nlargest(min(df_temp.shape[0], int(self.A.shape[0]*self.A.shape[1]*percentage)), 'value')

        interactions_df = df_temp
        return interactions_df

    def compare_with_MSE(self, MIN):
        a1 = self.A
        a2 = MIN.A

        assert a1.shape == a2.shape, "Dimensions mismatch"

        sq_err = 0.0

        for i in range(0, a1.shape[0]):
            for j in range(0, a2.shape[1]):
                sq_err +=  pow(a1[i][j]*1.0 - a2[i][j]*1.0, 2)

        MSE = sq_err / (a1.shape[0] * a2.shape[1])

        return MSE


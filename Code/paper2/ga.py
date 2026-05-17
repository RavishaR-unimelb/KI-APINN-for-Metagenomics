import numpy as np
from .tools import GLV_with_percentage
from .tools import dynamic_GLV_with_percentage
from .tools import BCD
from .tools import draw_network
from .dataset import Dataset
from .datasim import init_sim
from .datasim import get_r

from sys import exit
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import datetime
import pandas as pd
import networkx as nx
import sys
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp

# IMPARO


DEBUG = False

fig_x = 20
fig_y = 10


def log(*string):
    global DEBUG
    if DEBUG:
        print(list(string))


class Chromosome:

    # REMOVE t from Chromosome

    # Chromosome is a list of genes
    def __init__(self, n, t):
        self.genes = []
        self.fitness = 0.0
        self.score = 0.0
        self.N = n
        self.A = np.zeros((n, n))
        self.T = t
        self.r = get_r(n)

        self.isDynamic = False

    def setDynamic(self, isDynamic):
        self.isDynamic = isDynamic
        print("Dynamic Activated")

    def add_gene(self, gene):
        self.genes.append(gene)

    def calculate_score(self, K, XC1):
        # K is the original Abundance Profile
        # XC1 is the first time point of the original abundance profile
        # REMOVE

        # self.A = np.zeros((self.N, self.N))

        if not self.isDynamic:
            for gene in self.genes:
                self.A[gene.i, gene.j] = gene.A

        # TODO : check whether setting diagonal to -1 improves
        #for i in range(self.N):
        #    self.A[i, i] = -0.5

        t = K.shape[1]
        if(self.isDynamic):
            X = dynamic_GLV_with_percentage(XC1, self.A, t, self.N, self.r)
            #print("Forwarding to dynamic GLV")
        else:
            X = GLV_with_percentage(XC1, self.A, t, self.N, self.r)
            #print("Forwarding to static GLV")

        #print("Forwarding to BCD calculation")
        bcd = BCD(X, K, t, self.N)

        self.score = (1 - bcd)
        if np.isnan(self.score):
            self.score = 0
        # log(self.score)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def mutate(self, chance):

        for i in range(len(self.r)):
            rand = np.random.rand()
            if rand < chance:
                # do the randomization
                self.r[i] += np.random.normal(0, 0.01)

        for gene in self.genes:
            gene.mutate(chance)

    def visualize(self):
        # TODO improve visualization
        plt.figure(figsize=(fig_x, fig_y))
        plt.imshow(self.A, cmap='hot', interpolation='nearest')
        plt.show()
        #plt.savefig(folder_path+"interactions.svg")

    def print_a(self):
        np.set_printoptions(precision=3)
        print(self.A)
        print(np.count_nonzero(self.A))

    def draw_netowrk_graph(self, dataset):
        ls = dataset.get_OTUs
        plt.figure(figsize=(fig_x, fig_y))
        G1 = nx.MultiDiGraph()

        matrix = self.A
        rows = matrix.shape[0]
        cols = matrix.shape[1]

        for x in range(0, rows):
            G1.add_node(x, name=ls[x])

        for x in range(0, rows):
            for y in range(0, cols):
                if x != y:
                    if(matrix[x][y] > 0):
                        G1.add_edge(x, y, weight=pow(matrix[x][y], 2), color="blue")
                    else:
                        G1.add_edge(x, y, weight=pow(matrix[x][y] * -1, 2), color="red")

        pos = nx.shell_layout(G1)
        ax = plt.gca()
        draw_network(G1, pos, ax)
        ax.autoscale()
        plt.show()
        #plt.savefig(folder_path + "interactions_graph.svg")

class Gene:

    # A gene is a tuple: i, j and A_i,j
    # N is the total number of OTUs

    def __init__(self, i, j, a, n):
        self.i = i
        self.j = j
        self.A = a

        self.N = n

    def mutate(self, chance):

        # mutate i
        r = np.random.rand()
        if r < chance:
            # do the randomization
            np.random.randint(0, self.N)

        # mutate j
        r = np.random.rand()
        if r < chance:
            # do the randomization
            np.random.randint(0, self.N)

        # mutate A
        r = np.random.rand()
        if r < chance:
            # TODO parametrize sigma
            self.A += np.random.normal(0, 0.01)


def initpoprandom(t, n, pop_size=10000, target_connections=0.1):
    population = []

    gene_size = int(n * n * target_connections)

    # log("InitPopRandom", T, N)
    # TODO Initialize the population with A = (NH)oG type of dataset

    # for x in range(pop_size):

        # Initpop_A, temp1, temp2 = init_sim(n=n, time=300, pr=False, ER=True, sigma=0.1, alpha=0.2, p=target_connections)
        # chrom = Chromosome(n, t)
        # for i in range(n):
        #     for j in range(n):
        #         if Initpop_A[i][j] != 0:
        #             _I = i
        #             _J = j
        #             A_ij = Initpop_A[i][j]
        #
        #             gene = Gene(_I, _J, A_ij, n)
        #             chrom.add_gene(gene)
        # r = get_r(n)
        # chrom.r = r
        # population.append(chrom)
        # print(len(chrom.genes), "asdf")

    for i in range(pop_size):
        chrom = Chromosome(n, t)
        for j in range(gene_size):
            _I = np.random.randint(0, n)
            _J = np.random.randint(0, n)
            # TODO parametrize sigma
            A_ij = np.random.normal(0, 0.5)

            gene = Gene(_I, _J, A_ij, n)
            chrom.add_gene(gene)
        r = get_r(n)
        chrom.r = r
        population.append(chrom)
    #     print(len(chrom.genes), "asdfasdf")
    #


    return population


def printpopulation(population):
    global DEBUG
    if DEBUG:

        for chrom in population:

            for gene in chrom.genes:
                print(gene.i, gene.j, gene.A)

            log("End Chromosome")

        log("End Population")


def visualize_x(x):
    plt.figure(figsize=(fig_x, fig_y))

    for i in range(x.shape[0]):
        plt.plot(x[i, :])
    plt.legend(bbox_to_anchor=(1.05, 1))
    # plt.show()


def calculate_fitness(pop):
    sum_score = 0.0

    for chrom in pop:
        sum_score += chrom.score

    for chrom in pop:
        chrom.set_fitness(chrom.score / sum_score)


def pick_random(pop):
    # type: (list) -> Chromosome
    tries = 0

    while tries < 5000:
        random_int = np.random.randint(0, len(pop))

        monte = np.random.rand()

        if pop[random_int].fitness > monte:
            return pop[random_int]
        tries += 1

    return pop[np.random.randint(0, len(pop))]


def crossover(chrom1, chrom2):
    # corssover point
    cp = np.random.randint(0, min(len(chrom1.genes), len(chrom2.genes)))

    # print len(chrom1.genes), len(chrom2.genes), cp

    #TODO : crossover r

    for i in range(cp):
        chrom1.genes[i] = deepcopy(chrom2.genes[i])

    cp2 = np.random.randint(0, chrom1.N)

    for i in range(cp2):
        chrom1.r[i] = chrom2.r[i]

    return chrom1


def run_ga(dataset, pop_size=100, mutate_chance=0.01, num_gen=1000, target_connections=0.1, print_gen=100):
    GA_Stats = []

    # pop: - current population
    # gen: no of generations
    # best_fitness: best fitness so far
    # pop_size: population size

    N = dataset.get_N
    T = dataset.get_time

    log("run_GA", T, N)

    # Initialize with initial population
    pop = initpoprandom(T, N, pop_size, target_connections)
    gen = 0

    # printPopulation(pop)

    K = dataset.matrix
    #TODO Figure ^ out

    best_score = 0.0
    best_chrom = None
    best_X = None

    while gen < num_gen:

        XC1 = dataset.matrix[:,0]

        for chrom in pop:
            chrom.calculate_score(K, XC1)

        # pop.sort(key=lambda x: x.score, reverse=True)
        best_chrom = max(pop, key=lambda x: x.score)
        GA_Stats.append(best_chrom.score)


        if gen % print_gen == 0:
            log("Gen: ", gen, best_chrom.score, (best_chrom.score - best_score) / best_chrom.score)
            best_score = best_chrom.score
            # best_chrom.print_a()

        # Last generation examination
        #if gen == num_gen-1:
        #    log("Last Gen: ", gen, len(pop))
        #    #pop = pop.sort(key=lambda x: x.score)
        #    for chrom in pop:
        #        log(chrom.score)


        best_X = GLV_with_percentage(XC1, best_chrom.A, T, N, best_chrom.r)

        calculate_fitness(pop)

        new_pop = [best_chrom]

        for i in range(0, pop_size // 2 - 1):
            parent1 = deepcopy(pick_random(pop))
            parent2 = pick_random(pop)

            child = deepcopy(pick_random(pop))
            child = crossover(parent1, parent2)
            child.mutate(mutate_chance)

            new_pop.append(child)

        alt_pop = initpoprandom(T, N, pop_size - ((pop_size // 2) - 1), target_connections)
        new_pop.extend(alt_pop)

        pop = new_pop
        gen += 1

    #print("Best:", best_chrom.score)
    return GA_Stats, best_chrom

def test_ga(dataset):
    return "Hello"
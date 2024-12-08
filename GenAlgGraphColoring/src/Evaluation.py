from typing import Union
import numpy as np

from GenAlgGraphColoring.src.Individual import Individual
from GraphAdjList import GraphAdjList
from GraphAdjMatrix import GraphAdjMatrix


class Evaluation:
    def __init__(self, graph: Union[GraphAdjMatrix, GraphAdjList]):
        self.graph = graph

    def evaluate_population_vectorized(self, population: list[Individual]):
        size = len(population)

        chromosomes = np.array([inv.chromosome for inv in population])
        conflicts = np.zeros(size, dtype=int)

        # Store conflicted edges for each individual
        conflicting_edges = {i: [] for i in range(size)}

        for a, b in self.graph.edges:
            c = np.array(chromosomes[:, a] == chromosomes[:, b])
            for i in range(size):
                if c[i]:
                    conflicts[i] += 1
                    conflicting_edges[i].append((a, b))

        for i, inv in enumerate(population):
            inv.fitness = conflicts[i]
            inv.conflicting_edges = conflicting_edges[i]

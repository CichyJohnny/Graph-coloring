import random
from typing import Union

from GraphAdjList import GraphAdjList
from GraphAdjMatrix import GraphAdjMatrix
from GenAlgGraphColoring.src.Individual import Individual


class Mutation:
    def __init__(self,
                 chromosome_size: int,
                 graph: Union[GraphAdjMatrix, GraphAdjList]):

        self.chromosome_size = chromosome_size
        self.graph = graph

    # Random mutation of the chromosome
    def mutation(self,
                 population: list[Individual],
                 number_of_colors: int
                 ) -> list[Individual]:

        for individual in population:

            chromosome = individual.chromosome

            for a, b in individual.conflicting_edges:
                if chromosome[a] == chromosome[b]:

                    if self.graph.representation == "matrix":
                        neigh_idx = [i for i, x in enumerate(self.graph.matrix[a]) if x == 1]
                    else:
                        neigh_idx = self.graph.list[a]

                    neigh_colors = set(chromosome[i] for i in neigh_idx)
                    available_colors = [
                        color for color in range(number_of_colors)
                        if color not in neigh_colors and color != chromosome[a]
                    ]

                    if available_colors:
                        chromosome[a] = random.choice(available_colors)
                    else:
                        chromosome[a] = random.randint(0, number_of_colors - 1)

        return population

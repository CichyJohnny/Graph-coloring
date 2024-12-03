import random
from typing import Union

from GraphAdjList import GraphAdjList
from GraphAdjMatrix import GraphAdjMatrix
from GenAlgGraphColoring.src.Individual import Individual


class Mutation:
    def __init__(self,
                 chromosome_size: int,
                 graph: Union[GraphAdjMatrix, GraphAdjList],
                 representation: str):

        self.chromosome_size = chromosome_size
        self.graph = graph
        self.representation = representation

    # Random mutation of the chromosome
    def mutation(self,
                 population: list[Individual],
                 number_of_colors: int,
                 mutation_rate: float,
                 ) -> None:

        for individual in population:
            p = random.random()

            if p < mutation_rate:

                if self.representation == "matrix":

                    for i in range(self.graph.v):
                        for j in range(i, self.graph.v):

                            if self.graph.matrix[i][j] == 1 and individual.chromosome[i] == individual.chromosome[j]:
                                individual.chromosome[i] = random.randint(1, number_of_colors)

                else:

                    for i in range(self.graph.v):
                        for j in range(len(self.graph.list[i])):

                            if individual.chromosome[i] == individual.chromosome[self.graph.list[i][j]]:
                                individual.chromosome[i] = random.randint(1, number_of_colors)

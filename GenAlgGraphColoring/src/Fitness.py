from typing import Union

from GenAlgGraphColoring.src.Individual import Individual
from GraphAdjMatrix import GraphAdjMatrix
from GraphAdjList import GraphAdjList


class Fitness:

    # Calculate the fitness of an individual
    @staticmethod
    def get_fitness(graph: Union[GraphAdjMatrix, GraphAdjList], inv: Individual, representation: str) -> int:
        for i in range(graph.v):

            if representation == "matrix":
                for j in range(i, graph.v):
                    if graph.matrix[i][j] == 1 and inv.chromosome[i] == inv.chromosome[j]:
                        inv.fitness += 1

            else:
                for j in range(len(graph.list[i])):
                    if inv.chromosome[i] == inv.chromosome[graph.list[i][j]]:
                        inv.fitness += 1

        return inv.fitness

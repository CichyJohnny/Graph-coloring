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
                 number_of_colors: int,
                 mutation_rate: float,
                 ) -> list[Individual]:

        selected_mutation = population[:int(len(population) * mutation_rate)]

        for individual in selected_mutation:

            chromosome = individual.chromosome

            for a, b in individual.conflicting_edges:
                if chromosome[a] == chromosome[b]:

                    if self.graph.representation == "matrix":
                        neigh_idx = [i for i, x in enumerate(self.graph.matrix[a]) if x == 1]
                    else:
                        neigh_idx = self.graph.list[a]

                    all_colors = list(range(0, number_of_colors))
                    neigh_colors = set(individual.chromosome[i] for i in neigh_idx)
                    available_colors = list(set(all_colors) - neigh_colors - {individual.chromosome[a]})

                    if available_colors:
                        individual.chromosome[a] = random.choice(available_colors)
                    else:
                        individual.chromosome[a] = random.choice(all_colors)


                    # chromosome[a] = random.randint(0, number_of_colors)

        return selected_mutation

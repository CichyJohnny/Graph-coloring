from concurrent.futures import ThreadPoolExecutor
from typing import Union

from GenAlgGraphColoring.src.Individual import Individual
from GraphAdjList import GraphAdjList
from GraphAdjMatrix import GraphAdjMatrix


class FitnessEvaluator:
    def __init__(self, graph: Union[GraphAdjMatrix, GraphAdjList], representation: str):
        self.graph = graph
        self.representation = representation

    # Threadpool fitness evaluation of the population
    def evaluate_population(self, population: list[Individual]):
        with ThreadPoolExecutor(max_workers=len(population)) as executor:
            executor.map(self.get_fitness, population)


    # Calculate the fitness of a single individual
    def get_fitness(self, inv: Individual) -> int:
        inv.fitness = 0

        for i in range(self.graph.v):

            if self.representation == "matrix":
                for j in range(i, self.graph.v):
                    if self.graph.matrix[i][j] == 1 and inv.chromosome[i] == inv.chromosome[j]:
                        inv.fitness += 1

            else:
                for j in range(len(self.graph.list[i])):
                    if inv.chromosome[i] == inv.chromosome[self.graph.list[i][j]]:
                        inv.fitness += 1

        return inv.fitness

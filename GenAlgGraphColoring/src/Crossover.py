import random
import numpy as np

from GenAlgGraphColoring.src.Individual import Individual


class Crossover:
    def __init__(self, population_size: int, chromosome_size: int, crossover_rate: float):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.crossover_rate = crossover_rate

    # Cross every two parent individuals to create two children individuals
    def crossover(self, population: list[Individual]) -> list[Individual]:
        children = []

        # Cross every two parents from top crossover_rate% individuals to create two children individuals
        for _ in range(int(self.population_size * self.crossover_rate)):
            parent1, parent2 = random.choices(population[:self.population_size // 2], k=2)

            child = self.mating(parent1, parent2)
            children.append(child)

        return children

    # Chromosome crossover
    def mating(self,
            parent1: Individual,
            parent2: Individual
    ) -> Individual:

        cross_point = random.randint(2, self.chromosome_size - 2)

        child = Individual()

        child.chromosome = np.concatenate((parent1.chromosome[:cross_point], parent2.chromosome[cross_point:]))

        return child

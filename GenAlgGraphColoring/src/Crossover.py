import random
import numpy as np

from GenAlgGraphColoring.src.Individual import Individual


class Crossover:
    def __init__(self, population_size: int, chromosome_size: int):
        self.population_size = population_size
        self.chromosome_size = chromosome_size

    # Cross every two parent individuals to create two children individuals
    def crossover(self, population: np.ndarray[Individual]) -> np.ndarray:
        new_population = np.array([], dtype=Individual)

        # Cross every two parent individuals to create two children individuals
        for i in range(0, self.population_size - 1, 2):
            child1, child2 = Crossover.mating(population[i], population[i + 1], self.chromosome_size)

            new_population = np.append(new_population, [child1, child2])

        return new_population

    # Chromosome crossover
    @staticmethod
    def mating(inv1: Individual, inv2: Individual, chromosome_size: int) -> tuple[Individual, Individual]:
        cross_point = random.randint(2, chromosome_size - 2)

        child1 = Individual()
        child2 = Individual()

        child1.chromosome = np.concatenate((inv1.chromosome[:cross_point], inv2.chromosome[cross_point:]))
        child2.chromosome = np.concatenate((inv2.chromosome[:cross_point], inv1.chromosome[cross_point:]))

        return child1, child2

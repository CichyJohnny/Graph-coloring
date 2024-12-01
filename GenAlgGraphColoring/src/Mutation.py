import random
import numpy as np


class Mutation:
    def __init__(self, chromosome_size: int):
        self.chromosome_size = chromosome_size

    # Random mutation of the chromosome
    def mutation(self, population: np.ndarray, number_of_colors: int, mutation_rate: float):

        for individual in population:
            p = random.random()

            if p < mutation_rate:
                position = random.randint(0, self.chromosome_size - 1)
                individual.chromosome[position] = random.randint(0, number_of_colors)

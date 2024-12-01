import random
import numpy as np


class Individual:
    def __init__(self):
        self.chromosome = np.array([], dtype=int)
        self.fitness = 0

    def create_chromosome(self, chromosome_size: int, upper_bound: int) -> None:
        for i in range(chromosome_size):
            self.chromosome = np.append(self.chromosome, random.randint(0, upper_bound))

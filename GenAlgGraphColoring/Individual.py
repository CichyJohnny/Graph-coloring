import random


class Individual:
    def __init__(self):
        self.chromosome = []
        self.fitness = 0

    def create_chromosome(self, chromosome_size, upper_bound):
        for i in range(chromosome_size):
            self.chromosome.append(random.randint(0, upper_bound))

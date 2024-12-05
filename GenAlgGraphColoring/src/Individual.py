import numpy as np

class Individual:
    def __init__(self):
        self.chromosome = np.array([])
        self.fitness = 0
        self.conflicting_edges = []

    def create_chromosome(self, chromosome_size: int, number_of_colors: int) -> None:
        self.chromosome = np.random.randint(0, number_of_colors, size=chromosome_size)
import random
from .Individual import Individual

class Crossover:

    # Chromosome crossover
    @staticmethod
    def crossover(inv1: Individual, inv2: Individual, chromosome_size: int) -> tuple[Individual, Individual]:
        cross_point = random.randint(2, chromosome_size - 2)

        child1 = Individual()
        child2 = Individual()

        child1.chromosome = inv1.chromosome[:cross_point] + inv2.chromosome[cross_point:]
        child2.chromosome = inv2.chromosome[:cross_point] + inv1.chromosome[cross_point:]

        return child1, child2

import random
from GenAlgGraphColoring.src.Individual import Individual


class Crossover:
    def __init__(self, population_size: int, chromosome_size: int, crossover_rate: float):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.crossover_rate = crossover_rate

    # Cross every two parent individuals to create two children individuals
    def crossover(self, population: list[Individual]) -> list[Individual]:
        new_population = []

        # Cross every two parents from top 50% individuals to create two children individuals
        for _ in range(int(self.population_size * self.crossover_rate)):
            parent1, parent2 = random.choices(population[:self.population_size // 2], k=2)

            child1, child2 = Crossover.mating(parent1, parent2, self.chromosome_size)
            new_population.append(child1)
            new_population.append(child2)

        return new_population

    # Chromosome crossover
    @staticmethod
    def mating(
            parent1: Individual,
            parent2: Individual,
            chromosome_size: int
    ) -> tuple[Individual, Individual]:

        cross_point = random.randint(2, chromosome_size - 2)

        child1 = Individual()
        child2 = Individual()

        child1.chromosome = parent1.chromosome[:cross_point] + parent2.chromosome[cross_point:]
        child2.chromosome = parent2.chromosome[:cross_point] + parent1.chromosome[cross_point:]

        return child1, child2

import random

from GenAlgGraphColoring.src.Individual import Individual


class Mutation:
    def __init__(self, chromosome_size: int):
        self.chromosome_size = chromosome_size

    # Random mutation of the chromosome
    def mutation(self,
                 population: list[Individual],
                 number_of_colors: int,
                 mutation_rate: float
                 ) -> None:

        for individual in population:
            p = random.random()

            if p < mutation_rate:
                position = random.randint(0, self.chromosome_size - 1)
                individual.chromosome[position] = random.randint(1, number_of_colors)

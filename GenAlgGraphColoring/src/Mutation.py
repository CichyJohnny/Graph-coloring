import random

class Mutation:

    # Random mutation of the chromosome
    @staticmethod
    def mutation(population, chromosome_size, number_of_colors, mutation_rate):

        for individual in population:
            p = random.random()

            if p < mutation_rate:
                position = random.randint(0, chromosome_size - 1)
                individual.chromosome[position] = random.randint(1, number_of_colors)

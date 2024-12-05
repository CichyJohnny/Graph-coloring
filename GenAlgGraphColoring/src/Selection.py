import random

from GenAlgGraphColoring.src.Individual import Individual


class Selection:
    def __init__(self, population_size: int, crossover_rate: float):
        self.population_size = population_size
        self.crossover_rate = crossover_rate

    # Selection of individuals using roulette-wheel approach
    def roulette_wheel_selection(self, population: list[Individual]) -> list[Individual]:

        relative_fitness = [1 / (1 + i.fitness) for i in population]
        total_fitness = sum(relative_fitness)
        cumulative_fitness = [sum(relative_fitness[:i + 1]) / total_fitness for i in range(len(relative_fitness))]

        new_population = []
        while len(new_population) < int(self.population_size * (1 - self.crossover_rate)):
            roulette = random.uniform(0, 1)

            for i, cumulative_value in enumerate(cumulative_fitness):
                if roulette <= cumulative_value:
                    new_population.append(population[i])

                    break

        random.shuffle(new_population)

        return new_population

    # Return top % of the population
    def elitism_selection(self, population: list[Individual]) -> list[Individual]:
        top = int(self.population_size * (1 - self.crossover_rate)) + 1

        new_population = population[:top]

        return new_population

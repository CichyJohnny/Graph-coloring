import random
from typing import Callable

from GenAlgGraphColoring.src.Individual import Individual


class Selection:

    # Selection of individuals using roulette-wheel approach
    @staticmethod
    def roulette_wheel_selection(
            population: list[Individual],
            fitness_values: list[int]
    ) -> list[Individual]:

        relative_fitness = [1 / (1 + i) for i in fitness_values]
        total_fitness = sum(relative_fitness)
        cumulative_fitness = [sum(relative_fitness[:i + 1]) / total_fitness for i in range(len(relative_fitness))]

        new_population = []
        while len(new_population) < len(population):
            roulette = random.uniform(0, 1)

            for i, cumulative_value in enumerate(cumulative_fitness):
                if roulette <= cumulative_value:
                    new_population.append(population[i])

                    break

        random.shuffle(new_population)

        return new_population

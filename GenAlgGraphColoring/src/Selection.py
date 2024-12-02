import random
from typing import Callable

from GenAlgGraphColoring.src.Individual import Individual


class Selection:

    # Elitism selection of individuals using roulette-wheel approach
    @staticmethod
    def roulette_wheel_selection(
            population: list[Individual],
            get_fitness: Callable[[Individual], int]
    ) -> list[Individual]:

        fitness_values = [1 / (1 + get_fitness(individual)) for individual in population]
        total_fitness = sum(fitness_values)
        cumulative_fitness = [sum(fitness_values[:i + 1]) / total_fitness for i in range(len(fitness_values))]

        new_population = []
        while len(new_population) < len(population):
            roulette = random.uniform(0, 1)

            for i, cumulative_value in enumerate(cumulative_fitness):
                if roulette <= cumulative_value:
                    new_population.append(population[i])

                    break

        random.shuffle(new_population)

        return new_population

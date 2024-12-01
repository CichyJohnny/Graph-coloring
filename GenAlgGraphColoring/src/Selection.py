import random
import numpy as np
from typing import Callable

from GenAlgGraphColoring.src.Individual import Individual


class Selection:

    # Elitism selection of individuals using roulette-wheel approach
    @staticmethod
    def roulette_wheel_selection(population: np.ndarray, get_fitness: Callable[[Individual], int]):
        fitness_values = np.array(
            [
                1 / (1 + get_fitness(individual)) for individual in population
            ],
            dtype=float
        )
        total_fitness = np.sum(fitness_values)
        cumulative_fitness = np.array(
            [
                np.sum(fitness_values[:i + 1]) / total_fitness for i in range(fitness_values.size)
            ],
            dtype=float
        )

        new_population = np.array([], dtype=Individual)
        while new_population.size < population.size:
            roulette = random.uniform(0, 1)

            for i, cumulative_value in np.ndenumerate(cumulative_fitness):
                if roulette <= cumulative_value:
                    new_population = np.append(new_population, population[i])

                    break

        np.random.shuffle(new_population)

        return new_population

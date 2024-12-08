import time
import numpy as np
from copy import deepcopy
from typing import Union

from GraphAdjMatrix import GraphAdjMatrix
from GraphAdjList import GraphAdjList

from src.Individual import Individual
from src.Selection import Selection
from src.Crossover import Crossover
from src.Mutation import Mutation
from src.Evaluation import Evaluation
from src.Visualization import Visualization
from GreedyGraphColoring.GreedyGC import GreedyGraphColoring


# Genetic Algorithm for Graph Coloring with adjustable parameters
class GeneticAlgorithmGraphColoring:
    def __init__(self,
                 graph: Union[GraphAdjMatrix, GraphAdjList],
                 population_size: int,
                 mutation_rate: float,
                 crossover_rate: float,
                 randomness_rate: float,
                 visualise: bool=False,
                 star_with_greedy: bool=False
                 ):

        # Graph settings
        self.graph = graph
        self.chromosome_size = graph.v

        # Genetic algorithm parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.randomness_rate = randomness_rate
        self.const_randomness_rate = randomness_rate

        # Settings for genetic run
        self.population = None
        self.next_population = None
        self.number_of_colors = -1

        # Additional settings
        self.visualise = visualise
        self.start_from_greedy = star_with_greedy

        # Initialization of classes
        self.selector = Selection(self.population_size, self.crossover_rate)
        self.crossover = Crossover(self.population_size, self.chromosome_size, self.crossover_rate)
        self.mutator = Mutation(self.chromosome_size, self.graph)
        self.evaluator = Evaluation(self.graph)


    # Main method to start the genetic algorithm for graph coloring
    def start(self) -> None:
        # Select the starting point
        if self.start_from_greedy:
            self.start_greedy()

        else:
            self.start_normal()


        self.number_of_colors -= 1

        print(f"--------------------------------------------------")
        print(f"Trying for {self.number_of_colors} or less colors")

        t = time.perf_counter()

        # Genetic algorithm for each number of colors
        while True:

            # Start the genetic algorithm
            self.genetic_run(t)


    # Single run of the genetic algorithm
    def genetic_run(self, t: float) -> None:
        # Generate the initial population
        self.generate_population()

        best_individual = None
        best_fit = float("inf")
        best_fitness_list = []

        # Start from base randomness rate
        self.randomness_rate = self.const_randomness_rate

        generation = 0
        # While loop for genetic algorithm generations
        while best_fit != 0:
            generation += 1

            # Sort population by fitness and crop it to the population size
            self.population = sorted(self.population, key=lambda x: x.fitness)[:self.population_size]

            best_individual = self.population[0]
            best_fit = best_individual.fitness
            best_fitness_list.append(best_fit)

            # If found solution, break the loop
            if best_fit == 0:
                break

            # Initialization of the previous population and next population
            copy_population = deepcopy(self.population) # Ensure changes won't change self.population
            self.next_population = []

            # Standard genetic:
            self.start_crossover(copy_population)

            self.start_mutation(copy_population)


            self.end_generation(generation, best_fit, best_fitness_list)


        # If found a solution, print the summary and go next
        if best_fit == 0:
            # Set the number of colors to the best found solution
            self.number_of_colors = len(set(best_individual.chromosome))

            # Visualize the solution if needed
            if self.visualise:
                Visualization.visualize(generation, best_fitness_list, self.number_of_colors)


            self.summarise_generation(generation, t)

            self.number_of_colors -= 1


    # Initialize and perform the crossover process
    def start_crossover(self, copy_population: list[Individual]) -> None:
        parents_size = int(self.population_size // 4)
        selection_parents = self.selector.roulette_wheel_selection(copy_population, parents_size)
        random_parents = self.create_random_individuals(int(parents_size * self.randomness_rate))

        # Parents are randomness_rate % of parents_size, the rest is selected by roulette wheel
        parents = list(random_parents + selection_parents)[:parents_size]

        self.next_population.extend(self.crossover.crossover(parents))

    # Initialize and perform the mutation process
    def start_mutation(self, copy_population: list[Individual]) -> None:
        mutation_size = int(self.population_size * self.mutation_rate)
        selection_mutates = self.selector.roulette_wheel_selection(copy_population, mutation_size)
        random_mutate = self.create_random_individuals(int(mutation_size * self.randomness_rate))
        self.evaluator.evaluate_population_vectorized(random_mutate)

        # Mutates are randomness_rate % of mutation_size, the rest is selected by roulette wheel
        mutates = list(random_mutate + selection_mutates)[:mutation_size]

        self.next_population.extend(self.mutator.mutation(mutates, self.number_of_colors))


    # End of the generation, add the new population to the old one
    def end_generation(self, generation: int, best_fit: int, best_fitness_list: list[int]) -> None:
        self.evaluator.evaluate_population_vectorized(self.next_population)
        self.population.extend(self.next_population)

        step = 100
        if generation % step == 0:
            print(f"{generation}:{best_fit}", end=" | ")

            if len(set(best_fitness_list[:step:-1])) == 1:
                # If the best fitness is the same for the last 10 generations
                # and randomness_rate is less than 90%,
                # increase the randomness rate
                # to avoid premature convergence to a local minimum
                if self.randomness_rate < 0.9:
                    self.randomness_rate += 0.1

            else:
                # If improvement is made, reset the randomness rate
                self.randomness_rate = self.const_randomness_rate

    # Print the summary of the generation
    def summarise_generation(self, generation: int, t: float) -> None:
        print(f"==================================================")
        print(f"Succeeded for {self.number_of_colors} colors")
        print(f"In {generation} generations")
        print(f"Time taken: {round(time.perf_counter() - t, 2)}")
        print(f"--------------------------------------------------")
        print(f"Trying for {self.number_of_colors - 1} or less colors")


    # Start genetic algorithm where the greedy algorithm ended
    def start_greedy(self) -> None:
        greedy = GreedyGraphColoring(self.graph)
        greedy.start_coloring()

        self.number_of_colors = greedy.n

        self.population = []
        for _ in range(self.population_size):
            inv = Individual()
            inv.chromosome = greedy.colors
            self.population.append(inv)

        print(f"==================================================")
        print(f"Greedy algorithm ended with {self.number_of_colors} colors")

    # Start genetic algorithm with the max number of colors
    def start_normal(self) -> None:
        self.number_of_colors = self.graph.get_max_colors()
        print(f"==================================================")
        print(f"Starting with {self.number_of_colors} colors")


    # Generate the initial population
    def generate_population(self):
        # If the previous population exists, adjust and evaluate it
        if self.population:
            # Adjust the existing population to ensure it doesn't contain illegal number of colors
            for i, inv in enumerate(self.population):
                inv.chromosome = np.clip(inv.chromosome, 0, self.number_of_colors - 1)

            self.evaluator.evaluate_population_vectorized(self.population)
            self.population = sorted(self.population, key=lambda x: x.fitness)

        else:
            # If the population doesn't exist, create a new random one
            self.population = self.create_random_individuals(self.population_size)
            self.evaluator.evaluate_population_vectorized(self.population)

    # Create a population of random individuals
    def create_random_individuals(self, size: int) -> list[Individual]:
        new_population = []

        for _ in range(size):
            individual = Individual()
            individual.create_chromosome(self.chromosome_size, self.number_of_colors)
            new_population.append(individual)

        return new_population


if __name__ == "__main__":
    g = GraphAdjList()
    # g = GraphAdjMatrix()
    g.load_from_file('../tests/miles250.txt', 1)

    gen_alg = GeneticAlgorithmGraphColoring(g,
                                            100,
                                            0.5,
                                            0.5,
                                            0.2,
                                            visualise=True,
                                            star_with_greedy=True)

    gen_alg.start()

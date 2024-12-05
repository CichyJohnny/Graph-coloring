import time
import numpy as np
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
                 visualise: bool=False,
                 star_with_greedy: bool=False
                 ):

        self.graph = graph
        self.chromosome_size = graph.v
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = None
        self.next_population = None
        self.number_of_colors = -1

        self.visualise = visualise
        self.start_with_greedy = star_with_greedy

    # Main method to start the genetic algorithm
    def start(self):
        # Initialize starting settings

        if self.start_with_greedy:
            # Start where the greedy algorithm ended
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

        else:
            # Start with the maximum number of colors
            self.number_of_colors = self.graph.get_max_colors()
            print(f"==================================================")
            print(f"Starting with {self.number_of_colors} colors")

        self.number_of_colors -= 1

        print(f"--------------------------------------------------")
        print(f"Trying for {self.number_of_colors} colors")

        selector = Selection(self.population_size, self.crossover_rate)
        crossover = Crossover(self.population_size, self.chromosome_size, self.crossover_rate)
        mutator = Mutation(self.chromosome_size, self.graph)
        evaluator = Evaluation(self.graph, self.population_size)

        t = time.perf_counter()

        # Genetic algorithm for each number of colors
        while True:
            generation = 0

            self.generate_population()

            best_fit = float("inf")
            best_individual = None

            evaluation_times = []
            selection_times = []
            crossover_times = []
            mutation_times = []

            best_fitness_list = []

            # Genetic algorithm while loop for each generation
            while best_fit != 0:
                generation += 1
                self.next_population = []



                # Calculate fitness
                start = time.perf_counter()

                # evaluator.evaluate_population(self.population)
                evaluator.evaluate_population_vectorized(self.population)

                evaluation_times.append(time.perf_counter() - start)



                # Sort population by fitness
                self.population = sorted(self.population, key=lambda x: x.fitness)

                best_individual = self.population[0]
                best_fit = best_individual.fitness
                best_fitness_list.append(best_fit)

                if best_fit == 0:
                    break



                # Standard genetic: selection, crossover, mutation
                start = time.perf_counter()

                # next_population.extend(selector.roulette_wheel_selection(self.population))
                self.next_population.extend(selector.elitism_selection(self.population))

                selection_times.append(time.perf_counter() - start)


                start = time.perf_counter()
                self.next_population.extend(crossover.crossover(self.population))
                crossover_times.append(time.perf_counter() - start)


                start = time.perf_counter()
                mutator.mutation(self.next_population, self.number_of_colors, self.mutation_rate)
                mutation_times.append(time.perf_counter() - start)



                self.population = self.next_population


                if generation % 100 == 0:
                    print(f"{generation}:{best_fit}", end=" | ")

            if best_fit == 0:
                if self.visualise:
                    Visualization.visualize(generation, best_fitness_list, self.number_of_colors)

                print(f"==================================================")
                print(f"Succeeded for {self.number_of_colors} colors")
                print(f"In {generation} generations")
                print(f"Best chromosome: {best_individual.chromosome}")
                print(f"Time taken: {round(time.perf_counter() - t, 2)}")
                print(f"--------------------------------------------------")
                print("Avg Evaluation time: ", sum(evaluation_times) / len(evaluation_times))
                print("Avg Selection time: ", sum(selection_times) / len(selection_times))
                print("Avg Crossover time: ", sum(crossover_times) / len(crossover_times))
                print("Avg Mutation time: ", sum(mutation_times) / len(mutation_times))
                print(f"--------------------------------------------------")
                print(f"Trying for {self.number_of_colors - 1} colors")

                self.number_of_colors -= 1


    # Generate the initial population
    def generate_population(self):
        if self.population:
            evaluator = Evaluation(self.graph, self.population_size)
            evaluator.evaluate_population_vectorized(self.population)
            self.population = sorted(self.population, key=lambda x: x.fitness)


            # Adjust the existing population to ensure it doesn't contain illegal number of colors
            for i, individual in enumerate(self.population):
                if i < self.population_size // 10:
                    individual.chromosome = np.clip(individual.chromosome, 0, self.number_of_colors)
                else:
                    individual.create_chromosome(self.chromosome_size, self.number_of_colors)

        else:
            self.population = []
            for i in range(self.population_size):
                individual = Individual()
                individual.create_chromosome(self.chromosome_size, self.number_of_colors)
                self.population.append(individual)


if __name__ == "__main__":
    g = GraphAdjList()
    # g = GraphAdjMatrix()
    g.load_from_file('../tests/queen6.txt', 1)

    gen_alg = GeneticAlgorithmGraphColoring(g,
                                            10,
                                            0.2,
                                            crossover_rate=0.9,
                                            visualise=True,
                                            star_with_greedy=True)

    gen_alg.start()

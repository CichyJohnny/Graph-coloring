import time
from typing import Union

from GraphAdjMatrix import GraphAdjMatrix
from GraphAdjList import GraphAdjList

from src.Individual import Individual
from src.Selection import Selection
from src.Crossover import Crossover
from src.Mutation import Mutation
from src.Fitness import Fitness
from src.Visualization import Visualization


# Genetic Algorithm for Graph Coloring with adjustable parameters
class GeneticAlgorithmGraphColoring:
    def __init__(self, graph: Union[GraphAdjMatrix, GraphAdjList], population_size: int, mutation_rate: float, visualise: bool=False):
        self.representation = "matrix" if isinstance(graph, GraphAdjMatrix) else "list"
        self.graph = graph
        self.chromosome_size = graph.v
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = None
        self.number_of_colors = -1
        self.visualise = visualise

    # Main method to start the genetic algorithm
    def start(self):
        # Initialize starting settings
        self.get_num_of_colors()

        crossover = Crossover(self.population_size, self.chromosome_size)
        mutator = Mutation(self.chromosome_size)

        t = time.perf_counter()

        # Genetic algorithm for each number of colors
        while True:
            self.population = self.generate_population()
            generation = 0
            best_fitness_list = []
            best_fit = float("inf")
            best_individual = None

            selection_times = []
            crossover_times = []
            mutation_times = []

            # Genetic algorithm while loop for each generation
            while best_fit != 0:
                generation += 1

                # Standard genetic: selection, crossover, mutation
                start = time.perf_counter()
                self.population = Selection.roulette_wheel_selection(self.population, self.get_fitness)
                selection_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                self.population = crossover.crossover(self.population)
                crossover_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                mutator.mutation(self.population, self.number_of_colors, self.mutation_rate)
                mutation_times.append(time.perf_counter() - start)

                # Find the best individual in the population
                for individual in self.population:
                    fit = self.get_fitness(individual)
                    if fit < best_fit:
                        best_fit = fit
                        best_individual = individual

                best_fitness_list.append(best_fit)

            else:
                if self.visualise:
                    Visualization.visualize(generation, best_fitness_list, self.number_of_colors)

                print(f"==================================================")
                print(f"Succeeded for {self.number_of_colors} colors")
                print(f"In {generation} generations")
                print(f"Best chromosome: {best_individual.chromosome}")
                print(f"Time taken: {round(time.perf_counter() - t, 2)}")
                print(f"--------------------------------------------------")
                print("Avg Selection time: ", sum(selection_times) / len(selection_times))
                print("Avg Crossover time: ", sum(crossover_times) / len(crossover_times))
                print("Avg Mutation time: ", sum(mutation_times) / len(mutation_times))
                print(f"--------------------------------------------------")
                print(f"Trying for {self.number_of_colors - 1} colors")

                self.number_of_colors -= 1


    # Get the number of colors needed for the graph
    def get_num_of_colors(self) -> int:
        for i in range(self.graph.v):

            if self.representation == "matrix":
                if sum(self.graph.matrix[i]) > self.number_of_colors:
                    self.number_of_colors = sum(self.graph.matrix[i]) + 1

            else:
                if len(self.graph.list[i]) > self.number_of_colors:
                    self.number_of_colors = len(self.graph.list[i]) + 1

        return self.number_of_colors

    # Calculate the fitness of an individual
    def get_fitness(self, inv: Individual) -> int:
        return Fitness.get_fitness(self.graph, inv, self.representation)

    # Generate the initial population
    def generate_population(self) -> list[Individual]:
        population = []

        for i in range(self.population_size):
            individual = Individual()
            individual.create_chromosome(self.chromosome_size, self.number_of_colors)
            population.append(individual)

        return population


if __name__ == "__main__":
    g = GraphAdjList()
    # g = GraphAdjMatrix()
    g.load_from_file('GraphInput.txt', 1)

    gen_alg = GeneticAlgorithmGraphColoring(g, 100, 0.2, visualise=True)
    gen_alg.start()

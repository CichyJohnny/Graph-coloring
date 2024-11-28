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



selection_times = []
crossover_times = []
mutation_times = []



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
        self.population = self.generate_population()
        generation = 0
        best_fitness_list = []
        best_fit = float("inf")
        best_individual = None

        # Genetic algorithm while loop until local minimum is found
        while best_fit != 0:
            generation += 1

            # Standard genetic: selection, crossover, mutation
            start = time.perf_counter()
            self.population = Selection.roulette_wheel_selection(self.population, self.get_fitness)
            selection_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            self.mating()
            crossover_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            Mutation.mutation(self.population, self.chromosome_size, self.number_of_colors, self.mutation_rate)
            mutation_times.append(time.perf_counter() - start)

            # Find the best individual in the population
            for individual in self.population:
                fit = self.get_fitness(individual)
                if fit < best_fit:
                    best_fit = fit
                    best_individual = individual

            best_fitness_list.append(best_fit)

            # Summarize the results of the generation
            print("Current generation: ", generation)
            print("Best fitness: ", best_fit)
            print("Best chromosome: ", best_individual.chromosome)
            print("Number of colors: ", self.number_of_colors)
            print("--------------------------------------------------")

            if generation % 50 == 0 and self.visualise:
                Visualization.visualize(generation, best_fitness_list)

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

    # Cross every two parent individuals to create two children individuals
    def mating(self) -> None:
        new_population = []

        # Cross every two parent individuals to create two children individuals
        for i in range(0, self.population_size - 1, 2):
            child1, child2 = Crossover.crossover(self.population[i], self.population[i + 1], self.chromosome_size)
            new_population.append(child1)
            new_population.append(child2)

        self.population = new_population

if __name__ == "__main__":
    # g = GraphAdjList()
    g = GraphAdjMatrix()
    g.load_from_file('GraphInput.txt', 1)

    gen_alg = GeneticAlgorithmGraphColoring(g, 100, 0.2)
    gen_alg.start()

    print("Total Selection time: ", sum(selection_times))
    print("Total Crossover time: ", sum(crossover_times))
    print("Total Mutation time: ", sum(mutation_times))

    print("Avg Selection time: ", sum(selection_times)/len(selection_times))
    print("Avg Crossover time: ", sum(crossover_times)/len(crossover_times))
    print("Avg Mutation time: ", sum(mutation_times)/len(mutation_times))

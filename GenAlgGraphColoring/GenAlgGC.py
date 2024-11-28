import random
import time

from typing import Union
from matplotlib import pyplot as plt

from GraphAdjMatrix import GraphAdjMatrix
from GraphAdjList import GraphAdjList
from Individual import Individual



selection_times = []
crossover_times = []
mutation_times = []



# Genetic Algorithm for Graph Coloring with adjustable parameters
class GeneticAlgorithmGraphColoring:
    def __init__(self,
                 graph: Union[GraphAdjMatrix, GraphAdjList],
                 population_size: int,
                 mutation_rate: float,
                 visualise: bool=False):

        self.representation = "matrix" if isinstance(graph, GraphAdjMatrix) else "list"
        self.graph = graph
        self.chromosome_size = graph.v

        self.population_size = population_size
        self.mutation_rate = mutation_rate

        self.population = None
        self.number_of_colors = -1

        self.visualise = visualise


    # Main function to start the genetic algorithm
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
            self.roulette_wheel_selection()
            selection_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            self.mating()
            crossover_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            self.mutation()
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

            # If visualization is enabled, plot the best fitness through every 50 generations
            if generation % 50 == 0 and self.visualise:
                self.visualize(generation, best_fitness_list)


    # Get the number of colors needed for the graph
    def get_num_of_colors(self) -> int:
        for i in range(self.graph.v):

            if self.representation == "matrix":

                # The Maximum number of colors is the maximum degree of the graph + 1
                if sum(self.graph.matrix[i]) > self.number_of_colors:
                    self.number_of_colors = sum(self.graph.matrix[i]) + 1

            else:

                if len(self.graph.list[i]) > self.number_of_colors:
                    self.number_of_colors = len(self.graph.list[i]) + 1

        return self.number_of_colors

    # Calculate the fitness of an individual
    def get_fitness(self, inv: Individual) -> int:
        for i in range(self.graph.v):

            if self.representation == "matrix":

                for j in range(i, self.graph.v):

                    # Penalty for the same color of adjacent vertices
                    if self.graph.matrix[i][j] == 1 and inv.chromosome[i] == inv.chromosome[j]:
                        inv.fitness += 1

            else:

                for j in range(len(self.graph.list[i])):

                    # Penalty for the same color of adjacent vertices
                    if inv.chromosome[i] == inv.chromosome[self.graph.list[i][j]]:
                        inv.fitness += 1

        return inv.fitness


    # Generate the initial population
    def generate_population(self) -> list[Individual]:
        population = []

        # Initialize the population with individuals with random chromosomes
        for i in range(self.population_size):
            individual = Individual()

            individual.create_chromosome(self.chromosome_size, self.number_of_colors)

            population.append(individual)

        return population


    # Genetic mating process
    def mating(self) -> None:
        new_population = []

        # Cross every two parent individuals to create two children individuals
        for i in range(0, self.population_size - 1, 2):
            child1, child2 = self.crossover(self.population[i], self.population[i + 1])

            new_population.append(child1)
            new_population.append(child2)

        self.population = new_population

    # Chromosome crossover
    def crossover(self, inv1: Individual, inv2: Individual) -> tuple[Individual, Individual]:
        cross_point = random.randint(2, self.chromosome_size - 2)

        child1 = Individual()
        child2 = Individual()

        # Cross chromosomes in random chosen cross-point
        child1.chromosome = inv1.chromosome[:cross_point] + inv2.chromosome[cross_point:]
        child2.chromosome = inv2.chromosome[:cross_point] + inv1.chromosome[cross_point:]

        return child1, child2


    # Random mutation of the chromosome
    def mutation(self) -> None:
        for individual in self.population:

            # If probability p in mutation rate range, mutate random gene in the chromosome
            p = random.random()
            if p < self.mutation_rate:
                position = random.randint(0, self.chromosome_size - 1)

                individual.chromosome[position] = random.randint(1, self.number_of_colors)


    # Elitism selection of individuals using roulette-wheel approach
    def roulette_wheel_selection(self):
        fitness_values = [1 / (1 + self.get_fitness(individual)) for individual in self.population]
        total_fitness = sum(fitness_values)
        cumulative_fitness = [sum(fitness_values[:i + 1]) / total_fitness for i in range(len(fitness_values))]

        new_population = []
        while len(new_population) < self.population_size:
            roulette = random.uniform(0, 1)

            for i, cumulative_value in enumerate(cumulative_fitness):
                if roulette <= cumulative_value:
                    new_population.append(self.population[i])

                    break

        random.shuffle(new_population)
        self.population = new_population


    # Visualize the best fitness through generations
    @staticmethod
    def visualize(generation: int, fitness: list[int]) -> None:
        generations = list(range(1, generation + 1))

        # Plot chart with x: generations and y: best fitness
        plt.plot(generations, fitness)

        plt.xlabel("generation")
        plt.ylabel("best-fitness")

        plt.show()



# Main script
if __name__ == "__main__":
    g = GraphAdjMatrix()
    # g = GraphAdjList()

    g.load_from_file('GraphInput.txt', 1)

    gen_alg = GeneticAlgorithmGraphColoring(g, 100, 0.2)
    gen_alg.start()

    print("Total Selection time: ", sum(selection_times))
    print("Total Crossover time: ", sum(crossover_times))
    print("Total Mutation time: ", sum(mutation_times))

    print("Avg Selection time: ", sum(selection_times)/len(selection_times))
    print("Avg Crossover time: ", sum(crossover_times)/len(crossover_times))
    print("Avg Mutation time: ", sum(mutation_times)/len(mutation_times))

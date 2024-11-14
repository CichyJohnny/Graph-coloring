import random
from matplotlib import pyplot as plt

from Graph import Graph
from Individual import Individual


class GenAlg:
    def __init__(self, graph, population_size, mutation_rate, visualise=False):
        self.graph = graph
        self.chromosome_size = graph.v

        self.population_size = population_size
        self.mutation_rate = mutation_rate

        self.population = None
        self.number_of_colors = -1

        self.visualise = visualise


    def start(self):
        self.get_num_of_colors()
        self.population = self.generate_population()

        generation = 0
        best_fitness_list = []
        best_fit = float("inf")
        best_individual = None
        while best_fit != 0:
            generation += 1

            self.roulette_wheel_selection()

            self.mating()

            self.mutation()

            for individual in self.population:
                fit = self.get_fitness(individual)

                if fit < best_fit:
                    best_fit = fit
                    best_individual = individual

            best_fitness_list.append(best_fit)

            print("Current generation: ", generation)
            print("Best fitness: ", best_fit)
            print("Best chromosome: ", best_individual.chromosome)
            print("Number of colors: ", self.number_of_colors)
            print("--------------------------------------------------")

            if generation % 50 == 0 and self.visualise:
                self.visualize(generation, best_fitness_list)


    def get_num_of_colors(self) -> int:
        for i in range(self.graph.v):
            if sum(self.graph.matrix[i]) > self.number_of_colors:
                self.number_of_colors = sum(self.graph.matrix[i]) + 1

        return self.number_of_colors

    def get_fitness(self, inv: Individual) -> int:
        for i in range(self.graph.v):
            for j in range(i, self.graph.v):
                if self.graph.matrix[i][j] == 1 and inv.chromosome[i] == inv.chromosome[j]:
                    inv.fitness += 1

        return inv.fitness


    def generate_population(self) -> list[Individual]:
        population = []

        for i in range(self.population_size):
            individual = Individual()

            individual.create_chromosome(self.chromosome_size, self.number_of_colors)

            population.append(individual)

        return population


    def mating(self) -> None:
        new_population = []

        for i in range(0, self.population_size - 1, 2):
            child1, child2 = self.crossover(self.population[i], self.population[i + 1])

            new_population.append(child1)
            new_population.append(child2)

        self.population = new_population

    def crossover(self, inv1: Individual, inv2: Individual) -> tuple[Individual, Individual]:
        cross_point = random.randint(2, self.chromosome_size - 2)

        child1 = Individual()
        child2 = Individual()

        child1.chromosome = inv1.chromosome[:cross_point] + inv2.chromosome[cross_point:]
        child2.chromosome = inv2.chromosome[:cross_point] + inv1.chromosome[cross_point:]

        return child1, child2


    def mutation(self) -> None:
        for individual in self.population:

            p = random.random()
            if p < self.mutation_rate:

                position = random.randint(0, self.chromosome_size - 1)
                individual.chromosome[position] = random.randint(1, self.number_of_colors)


    def roulette_wheel_selection(self):
        total_fitness = 0

        for individual in self.population:
            total_fitness += 1 / (1 + self.get_fitness(individual))

        cumulative_fitness = []
        cumulative_fitness_sum = 0

        for i in range(len(self.population)):
            cumulative_fitness_sum += 1 / (1 + self.get_fitness(self.population[i])) / total_fitness
            cumulative_fitness.append(cumulative_fitness_sum)

        new_population = []
        while len(new_population) < self.population_size:
            roulette = random.uniform(0, 1)
            for j in range(len(self.population)):
                if roulette <= cumulative_fitness[j]:
                    new_population.append(self.population[j])
                    break

        random.shuffle(new_population)
        self.population = new_population

    @staticmethod
    def visualize(generation, fitness) -> None:
        generations = list(range(1, generation + 1))

        plt.plot(generations, fitness)

        plt.xlabel("generation")
        plt.ylabel("best-fitness")

        plt.show()



if __name__ == "__main__":
    g = Graph()
    g.load_from_file('../GreedyGraphColoring/GraphInput.txt')

    gen_alg = GenAlg(g, 100, 0.2)
    gen_alg.start()

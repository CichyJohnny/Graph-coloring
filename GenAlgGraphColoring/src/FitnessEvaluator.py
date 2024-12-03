from concurrent.futures import ThreadPoolExecutor

from GenAlgGraphColoring.src.Individual import Individual


class FitnessEvaluator:
    def __init__(self, graph, representation):
        self.graph = graph
        self.representation = representation

    # Threadpool fitness evaluation of the population
    def evaluate_population(self, population: list[Individual]) -> list[int]:
        with ThreadPoolExecutor(max_workers=100) as executor:
            fitness_values = list(executor.map(self.get_fitness, population))

        return fitness_values

    # Calculate the fitness of a single individual
    def get_fitness(self, inv):
        for i in range(self.graph.v):

            if self.representation == "matrix":
                for j in range(i, self.graph.v):
                    if self.graph.matrix[i][j] == 1 and inv.chromosome[i] == inv.chromosome[j]:
                        inv.fitness += 1

            else:
                for j in range(len(self.graph.list[i])):
                    if inv.chromosome[i] == inv.chromosome[self.graph.list[i][j]]:
                        inv.fitness += 1

        return inv.fitness

from GraphAdjList import GraphAdjList as AdjList
from GraphAdjMatrix import GraphAdjMatrix as AdjMatrix
from GenAlgGC import GeneticAlgorithmGraphColoring as GenAlg


if __name__ == "__main__":
    ################################################################
    # Adjustable
    representation = AdjMatrix
    # representation = AdjList
    filename = '../tests/queen6.txt'

    population_size = 100
    mutation_rate = 0.2
    crossover_rate = 0.8

    visualize = True
    greedy = True
    ################################################################

    g = representation()
    g.load_from_file(filename, start_index=1)

    gen_alg = GenAlg(g, population_size, mutation_rate, crossover_rate, visualise=visualize, star_with_greedy=greedy)
    gen_alg.start()

from GraphAdjList import GraphAdjList as AdjList
from GraphAdjMatrix import GraphAdjMatrix as AdjMatrix
from GenAlgGC import GeneticAlgorithmGraphColoring as GenAlg


if __name__ == "__main__":
    ################################################################
    # Adjustable
    representation = AdjMatrix
    # representation = AdjList
    filename = '../tests/test_100-0.3-1405.txt'

    population_size = 100
    mutation_rate = 0.3

    visualize = True
    ################################################################

    g = representation()
    g.load_from_file('../tests/test_100-0.3-1405.txt')

    gen_alg = GenAlg(g, population_size, mutation_rate, visualise=visualize)
    gen_alg.start()

from GraphAdjMatrix import GraphAdjMatrix
from GenAlgAdjMatrixGColoring import GeneticAlgorithmGraphColoring as GenAlg

if __name__ == "__main__":
    ################################################################
    # Adjustable
    filename = '../tests/test_100-0.3-1405.txt'

    population_size = 200
    mutation_rate = 0.4

    visualize = True
    ################################################################

    g = GraphAdjMatrix()
    g.load_from_file('../tests/test_100-0.3-1405.txt')

    gen_alg = GenAlg(g, population_size, mutation_rate, visualise=visualize)
    gen_alg.start()

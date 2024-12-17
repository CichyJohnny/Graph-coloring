from GraphAdjList import GraphAdjList as AdjList
from GraphAdjMatrix import GraphAdjMatrix as AdjMatrix
from GenAlgGC import GeneticAlgorithmGraphColoring as GenAlg

import threading


if __name__ == "__main__":
    ################################################################
    # Adjustable
    representation = AdjMatrix
    # representation = AdjList
    filename = '../tests/gc500.txt'

    population_size = 100
    mutation_rate = 0.2
    crossover_rate = 0.8

    visualize = True
    greedy = True
    ################################################################

    g = representation()
    g.load_from_file(filename, start_index=1)

    def run_algorithm(gen_alg):
        # Replace this with your algorithm
        print("Algorithm started")

        try:
            gen_alg.start()
        except SystemExit:
            print("Algorithm terminated")
            print(f"Finished with {gen_alg.number_of_colors} colors")


    def run_with_timeout(target, timeout):
        gen_alg = GenAlg(g,
                         50,
                         0.5,
                         0.5,
                         0.2,
                         100,
                         visualise=False,
                         start_with_greedy=True,
                         num_threads=1)

        thread = threading.Thread(target=target, args=[gen_alg], daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            print(f"Timeout reached. Terminating the algorithm.")
            c = gen_alg.number_of_colors
            print(f"Finished with {c} colors")

            with open('results.txt', 'a') as f:
                f.write(f"{c}\n")

            raise SystemExit


    # Example usage:
    while True:
        try:
            run_with_timeout(run_algorithm, timeout=150)  # Run for a maximum of 5 seconds
        except SystemExit:
            pass

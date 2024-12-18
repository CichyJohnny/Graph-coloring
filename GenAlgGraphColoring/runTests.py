from GraphAdjList import GraphAdjList as AdjList
from GraphAdjMatrix import GraphAdjMatrix as AdjMatrix
from GenAlgGC import GeneticAlgorithmGraphColoring as GenAlg

import threading
import csv

import os
import sys
from contextlib import contextmanager


# Suppress stdout of the algorithm
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Create an algorithm for each instance and run it with a timeout
def run_with_timeout(instance_, timeout, result_path="../results/result.csv"):
    g = instance_[1]()
    g.load_from_file(instance_[0], 1)
    gen_alg = GenAlg(g, *instance_[2:])

    print("Running algorith")

    with suppress_stdout():
        thread = threading.Thread(target=gen_alg.start, daemon=True)
        thread.start()
        thread.join(timeout)

    print(f"Finished with: {gen_alg.number_of_colors}")


    if thread.is_alive():
        c = gen_alg.number_of_colors

        with open(result_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([c, *instance_])

        raise SystemExit


if __name__ == "__main__":
    ################################################################
    # Adjustable
    time_duration = 60  # seconds
    num_tries = 5

    test_name = "miles250"

    test_path = f"../tests/{test_name}.txt"

    # Instance: [path, Graph class, population_size, mutation_rate, crossover_rate,
    # randomness_rate, increase_randomness_rate, visualize, start_with_greedy, num_threads, use_seed]
    instances = [
        [test_path, AdjMatrix, 20, 0.5, 0.5, 0.2, 100, False, True, 1, False],
        [test_path, AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 1, False],
        [test_path, AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 1, False],

        [test_path, AdjList, 20, 0.5, 0.5, 0.2, 100, False, True, 1, False],
        [test_path, AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 1, False],
        [test_path, AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 1, False],

        [test_path, AdjMatrix, 20, 0.5, 0.5, 0.2, 100, False, True, 1, True],
        [test_path, AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 1, True],
        [test_path, AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 1, True],

        [test_path, AdjList, 20, 0.5, 0.5, 0.2, 100, False, True, 1, True],
        [test_path, AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 1, True],
        [test_path, AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 1, True],
                 ]

    ################################################################


    for instance in instances:
        print(f"Running instance: {instance}")
        for _ in range(num_tries):
            try:
                run_with_timeout(instance, time_duration, f"../results/{test_name}.csv")
            except SystemExit:
                pass
        print("Finished instance\n\n")

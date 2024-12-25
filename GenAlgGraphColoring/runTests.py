from GraphAdjList import GraphAdjList as AdjList
from GraphAdjMatrix import GraphAdjMatrix as AdjMatrix
from GenAlgGC import GeneticAlgorithmGraphColoring as GenAlg

import threading
import csv
import time

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
    gen_alg.thread_timeout = time.time() + timeout

    print("Running algorith")

    with suppress_stdout():
        thread = threading.Thread(target=gen_alg.start, daemon=True)
        thread.start()
        thread.join(timeout)

    c = gen_alg.number_of_colors + 1
    # print(f"Finished with: {c}")

    with open(result_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([c, *instance_])


if __name__ == "__main__":
    ################################################################
    # Adjustable
    time_duration = 60  # seconds
    num_tries = 3

    test_names = ["anna", "david", "gc500", "gc1000", "homer", "le450_5a", "miles250",
                  "myciel4", "myciel5", "myciel6", "queen6", "queen7"]

    # TODO: Read instances from file
    # Instance: [path, Graph class, population_size, mutation_rate, crossover_rate,
    # randomness_rate, increase_randomness_rate, visualize, start_with_greedy, num_threads, mutate_parents]
    instances = [
        [AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 1, False],
        [AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 1, False],

        [AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 2, False],
        [AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 2, False],

        [AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 3, False],
        [AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 3, False],


        [AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 1, True],
        [AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 1, True],

        [AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 2, True],
        [AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 2, True],

        [AdjMatrix, 50, 0.5, 0.5, 0.2, 100, False, True, 3, True],
        [AdjMatrix, 100, 0.5, 0.5, 0.2, 100, False, True, 3, True],



        [AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 1, False],
        [AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 1, False],

        [AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 2, False],
        [AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 2, False],

        [AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 3, False],
        [AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 3, False],


        [AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 1, True],
        [AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 1, True],

        [AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 2, True],
        [AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 2, True],

        [AdjList, 50, 0.5, 0.5, 0.2, 100, False, True, 3, True],
        [AdjList, 100, 0.5, 0.5, 0.2, 100, False, True, 3, True],
                 ]

    ################################################################

    for i, test_name in enumerate(test_names):
        print("Running test: ", test_name)

        for j, instance in enumerate(instances):
            print(f"Running instance: {instance}")
            instance = [f"../tests/{test_name}.txt", *instance]

            print(f"Time left: {round(len(test_names[i:]) * time_duration * num_tries * len(instances[j:]) / 60, 2)} minutes")
            for _ in range(num_tries):
                run_with_timeout(instance, time_duration, f"../results/{test_name}.csv")

            print("Finished instance\n\n")

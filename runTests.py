import os
import time
from GreedyGraphColoring.GreedyGC import GreedyGraphColoring
from GraphAdjMatrix import GraphAdjMatrix


def run_all_tests(directory):
    paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            paths.append(filepath)

    return paths


if __name__ == "__main__":
    directory = './tests/'
    paths = run_all_tests(directory)

    iterations = 3
    scores = []
    for path in paths:
        print("Running test for\t", path)
        G = GraphAdjMatrix()
        G.load_from_file(path)
        s = []

        for _ in range(iterations):
            gc = GreedyGraphColoring(G)

            start = time.perf_counter()
            gc.start_coloring()
            elapsed = time.perf_counter() - start

            print("Elapsed\t", elapsed)
            s.append(elapsed)

        scores.append(sum(s) / len(s))

    with open("scores.txt", 'w') as f:
        for p, s in zip(paths, scores):
            f.write(f"{p}\t\t\t{s}\n")
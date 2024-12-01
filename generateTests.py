import random
import time

def generate_random_graph(v, density):
    random.seed(time.time())

    matrix = [[0 for _ in range(v)] for _ in range(v)]
    edges = []
    e = 0

    for i in range(v):
        for j in range(i + 1, v):
            if random.random() < density:
                matrix[i][j] = 1
                matrix[j][i] = 1

                edges.append((i, j))
                e += 1

    path = f"./tests/test_{v}-{density}-{e}.txt"
    with open(path, 'w') as f:
        f.write(f"{v} {e}\n")

        for a, b in edges:
            f.write(f"{a} {b}\n")



generate_random_graph(50, 0.3)

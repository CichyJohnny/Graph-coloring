import random
import time


# Simple graph class with adjacency matrix representation
class Graph:
    def __init__(self):
        self.v = 0
        self.e = 0
        self.edges = []
        self.matrix = []

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            self.v, self.e = map(int, f.readline().split())

            self.matrix = [[0 for _ in range(self.v)] for _ in range(self.v)]
            for i in range(self.e):
                a, b = map(int, f.readline().split())
                self.edges.append((a, b))
                self.matrix[a][b] = 1
                self.matrix[b][a] = 1

    def generate_random_graph(self, v, density):
        random.seed(time.time())

        self.matrix = [[0 for _ in range(v)] for _ in range(v)]
        self.v = v

        for i in range(v):
            for j in range(i + 1, v):
                if random.random() < density:
                    self.matrix[i][j] = 1
                    self.matrix[j][i] = 1

                    self.edges.append((i, j))
                    self.e += 1

        path = f"./tests/test_{v}-{density}-{self.e}.txt"
        with open(path, 'w') as f:
            f.write(f"{self.v} {self.e}\n")

            for a, b in self.edges:
                f.write(f"{a} {b}\n")

import random
import time


# Simple graph class with adjacency matrix representation
class GraphAdjList:
    def __init__(self):
        self.v = 0
        self.e = 0
        self.edges = []
        self.list = []

    def load_from_file(self, filename, start_index=0):
        with open(filename, 'r') as f:
            self.v, self.e = map(int, f.readline().split())

            self.list = [[] for _ in range(self.v)]
            for i in range(self.e):
                a, b = map(int, f.readline().split())

                if start_index != 0:
                    a -= start_index
                    b -= start_index

                self.edges.append((a, b))
                self.list[a].append(b)

    def generate_random_graph(self, v, density):
        random.seed(time.time())

        self.list = [[] for _ in range(v)]
        self.v = v

        for i in range(v):
            for j in range(i + 1, v):
                if random.random() < density:
                    self.list[i].append(j)

                    self.edges.append((i, j))
                    self.e += 1

        path = f"./tests/test_{v}-{density}-{self.e}.txt"
        with open(path, 'w') as f:
            f.write(f"{self.v} {self.e}\n")

            for a, b in self.edges:
                f.write(f"{a} {b}\n")

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

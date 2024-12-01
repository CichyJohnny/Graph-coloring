# Simple graph class with adjacency list representation
class GraphAdjMatrix:
    def __init__(self):
        self.v = 0
        self.e = 0
        self.edges = []
        self.matrix = []

    def load_from_file(self, filename, start_index=0):
        with open(filename, 'r') as f:
            self.v, self.e = map(int, f.readline().split())

            self.matrix = [[0 for _ in range(self.v)] for _ in range(self.v)]
            for i in range(self.e):
                a, b = map(int, f.readline().split())

                if start_index != 0:
                    a -= start_index
                    b -= start_index

                self.edges.append((a, b))
                self.matrix[a][b] = 1
                self.matrix[b][a] = 1


    # Get the maximum number of colors needed for the graph
    def get_max_colors(self):
        number_of_colors = -1

        for i in range(self.v):
            if sum(self.matrix[i]) > number_of_colors:
                number_of_colors = sum(self.matrix[i]) + 1

        return number_of_colors

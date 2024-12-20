# Simple graph class with adjacency matrix representation
class GraphAdjList:
    def __init__(self):
        self.representation = "list"
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
                self.list[b].append(a)


    # Get the maximum number of colors needed for the graph
    def get_max_colors(self) -> int:
        number_of_colors = -1

        for i in range(self.v):
            if len(self.list[i]) > number_of_colors:
                number_of_colors = len(self.list[i]) + 1

        return number_of_colors

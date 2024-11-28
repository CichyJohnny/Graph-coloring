from GraphAdjMatrix import GraphAdjMatrix


# Greedy graph coloring algorithm
class GreedyGraphColoring:
    def __init__(self, g):
        self.graph = g
        self.colors = [0] * g.v
        self.n = 0

    # Method that colors the graph using the greedy algorithm
    # Start with the first vertex and assign it the first color
    # For each vertex, check the colors of its adjacent vertices
    # Assign the smallest color that is not used by any of the adjacent vertices
    def start_coloring(self):
        self.colors[0] = 1
        self.n += 1

        for curr_v in range(1, self.graph.v):
            adj_index = [i for i in range(self.graph.v) if self.graph.matrix[curr_v][i] == 1]
            adj_colors = set([self.colors[i] for i in adj_index]) - {0}

            if len(adj_colors) == self.n:
                # New Color
                self.n += 1
                self.colors[curr_v] = self.n
            else:
                # Reuse color
                left_colors = set(range(1, self.n + 1)) - adj_colors
                self.colors[curr_v] = min(left_colors)


if __name__ == "__main__":
    g = GraphAdjMatrix()
    g.load_from_file('GraphInput.txt')

    gc = GreedyGraphColoring(g)
    gc.start_coloring()

    for i in range(g.v):
        print(i, '-->', gc.colors[i])

    print('Number of colors:', gc.n)

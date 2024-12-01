import networkx as nx
import matplotlib.pyplot as plt
import time
from GraphAdjMatrix import GraphAdjMatrix

# Import here your algorithms
from GreedyGraphColoring.GreedyGC import GreedyGraphColoring

################################################################
# Adjustable
filename = 'tests/test_100-0.3-1405.txt'
algorithm = GreedyGraphColoring
################################################################

myGraph = GraphAdjMatrix()
myGraph.load_from_file(filename)

gc = algorithm(myGraph)

start = time.perf_counter()
gc.start_coloring()
print("Time: ", time.perf_counter() - start)
print("Colors: ", gc.n)

# Create color map
# Add as many colors as needed
node_colors = ['red', 'green', 'blue', 'orange', 'yellow', 'purple', 'brown', 'pink', 'cyan', 'magenta',
               'white', 'gray', 'lightblue', 'lightgreen', 'lightcoral', 'lightcyan', 'lightgray', 'lightpink',
               'lightyellow', 'darkblue', 'darkgreen', 'darkcoral', 'darkcyan', 'darkgray', 'darkpink', 'darkyellow']

if gc.n < len(node_colors):
    # Create networkx graph
    G = nx.Graph()
    for i in range(myGraph.e):
        G.add_edge(*myGraph.edges[i])

    color_map = [node_colors[i] for i in [gc.colors[node] for node in G.nodes()]]

    # Draw graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=color_map, font_weight='bold')

    plt.show()

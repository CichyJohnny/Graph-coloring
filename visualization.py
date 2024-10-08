import networkx as nx
import matplotlib.pyplot as plt
from Graph import Graph

# Import here your algorithms
from GreedyGraphColoring.GreedyGraphColoring import GreedyGraphColoring

################################################################
# Adjustable
filename = 'GreedyGraphColoring/GraphInput.txt'
algorithm = GreedyGraphColoring
################################################################

myGraph = Graph()
myGraph.load_from_file(filename)

gc = algorithm(myGraph)
gc.start_coloring()

# Create networkx graph
G = nx.Graph()
for i in range(myGraph.e):
    G.add_edge(*myGraph.edges[i])

# Create color map
# Add as many colors as needed
node_colors = ['red', 'green', 'blue', 'orange', 'yellow', 'purple', 'brown', 'pink', 'cyan', 'magenta']
color_map = [node_colors[i] for i in [gc.colors[node] for node in G.nodes()]]

# Draw graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=color_map, font_weight='bold')

plt.show()

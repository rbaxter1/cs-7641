from nxpd import nxpdParams
import networkx as nx
from nxpd import draw

G = nx.DiGraph()
G.graph['rankdir'] = 'LR'
G.graph['dpi'] = 220
G.add_cycle(range(4))
G.add_node(0, color='red', style='filled', fillcolor='pink')
G.add_node(1, shape='square')
G.add_node(3, style='filled', fillcolor='#00ffff')
G.add_edge(0, 1, color='red', style='dashed')
G.add_edge(3, 3, label='a')
draw(G)
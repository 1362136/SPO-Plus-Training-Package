import networkx as nx

def create_grid_graph(d,c):
    '''
    :param d: denotes the dimensions of the grid graph
    :param c: denotes the weight vector for the graph
    :return: A n x n directed grid graph with weights from c.
    '''
    grid = nx.DiGraph()
    grid.add_edges_from([(j * d + i, j * d + i + 1) for j in range(d) for i in range(d-1)])
    grid.add_edges_from([(i + d * j, i + d * (j + 1)) for i in range(d) for j in range(d-1)])
    k = 0
    for e in list(grid.edges()):
        grid[e[0]][e[1]]['weight'] = float(c[k])
        k += 1
    return grid

G = create_grid_graph(3,[1,2,3,4,5,6,7,8,9,10,11,12])


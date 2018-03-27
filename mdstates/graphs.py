from itertools import chain

import networkx as nx

__all__ = ['combined_graph_edges', 'combine_graphs']


def combined_graph_edges(G, H):
    intersect = set(G.edges) & set(H.edges)
    disj = set(G.edges) ^ set(H.edges)

    for u, v in chain(intersect, disj):
        if (u, v) in G.edges and (u, v) in H.edges:
            attr = G[u][v].copy()
            hdata = H[u][v]
            attr.update((key, attr[key] + hdata[key]) for key in attr.keys())
        elif (u, v) in G.edges:
            attr = G[u][v]
        else:
            attr = H[u][v]

        yield u, v, attr


def combine_graphs(G, H):
    graph = nx.DiGraph()
    graph.add_nodes_from(G.nodes(data=True))
    graph.add_nodes_from(node for node in H.nodes(data=True)
                         if node not in G.nodes(data=True))
    graph.add_edges_from(combined_graph_edges(G, H))
    return graph

from itertools import chain
from numbers import Number

import networkx as nx
import numpy as np

__all__ = ['combined_graph_nodes', 'combined_graph_edges', 'combine_graphs']


def combined_graph_nodes(G, H):
    intersect = set(G.nodes) & set(H.nodes)
    disj = set(G.nodes) ^ set(H.nodes)

    for n in chain(intersect, disj):
        if n in G.nodes and n in H.nodes:
            attr = G.node[n].copy()
            hdata = H.node[n]

            # All node attributes that are shared.
            shared = set(attr) & set(hdata)
            for key in shared:
                if isinstance(attr[key], Number):
                    attr.update({key: attr[key] + hdata[key]})

            # All node attributes that are not shared.
            not_shared = set(hdata) - set(attr)
            attr.update((key, hdata[key]) for key in not_shared)
        elif n in G.nodes:
            attr = G.node[n]
        else:
            attr = H.node[n]

        yield n, attr


def combined_graph_edges(G, H):
    intersect = set(G.edges) & set(H.edges)
    disj = set(G.edges) ^ set(H.edges)

    for u, v in chain(intersect, disj):
        if (u, v) in G.edges and (u, v) in H.edges:
            attr = G[u][v].copy()
            hdata = H[u][v]
            shared = set(attr) & set(hdata)
            for key in shared:
                if isinstance(attr[key], Number):
                    attr.update({key: attr[key] + hdata[key]})

            not_shared = set(hdata) - set(attr)
            attr.update((key, hdata[key]) for key in not_shared)
        elif (u, v) in G.edges:
            attr = G[u][v]
        else:
            attr = H[u][v]

        yield u, v, attr


def combine_graphs(G, H):
    graph = nx.DiGraph()
    graph.add_nodes_from(combined_graph_nodes(G, H))
    # graph.add_nodes_from(G.nodes(data=True))
    # graph.add_nodes_from(node for node in H.nodes(data=True)
    #                      if node not in G.nodes(data=True))
    graph.add_edges_from(combined_graph_edges(G, H))
    return graph

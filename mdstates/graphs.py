from itertools import chain
from numbers import Number

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np

__all__ = ['combined_graph_nodes', 'combined_graph_edges', 'combine_graphs']


def combined_graph_nodes(G, H):
    """Generator for both graph nodes with summed node attributes.

    Generator that yields the a node and the sum of all shared
    attributes and copies of all non-shared attributes.  All attributes
    with numeric values will be summed, any other type will not be
    summed. If both graphs have an edge with the same attribute that
    contains a non-numeric value, then preference will be given to the
    second graph, `H`. Any nodes that are shared but do not share a
    given attribute will have that attribute migrated to the summed
    node.

    Parameters
    ----------
    G, H : networkx.DiGraph
        Graphs that will be combined.

    Yields
    ------
    n : str
        Directed graph node.
    attr : dict
        Dictionary containing all node attributes.
    """
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
    """Generator for both graph edges with summed edge attributes.

    Generator that yields the two nodes that form an edge and the sum
    of all shared attributes and copies of all non-shared attributes.
    All attributes with numeric values will be summed, any other type
    will not be summed. If both graphs have an edge with the same
    attribute that contains a non-numeric value, then preference will
    be given to the second graph, `H`. Any edges that are shared but do
    not share a given attribute will have that attribute migrated to
    the summed edge.

    Parameters
    ----------
    G, H : networkx.DiGraph
        Graphs that will be combined.

    Yields
    ------
    u, v : str
        Two directed graph nodes that form an edge.
    attr : dict
        Dictionary containing all edge attributes
    """
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


def combine_graphs(G, H, directed=True):
    """Disjoint and intersection of 2 graphs with attributes summed.
    
    Combines two graphs into a single graph that has the intersection
    and disjoint of the two graphs. All intersecting, numeric node and
    edge attributes that are shared are summed. All intersecting,
    non-numeric node and edge attributes that are shared are only
    contributed by `H`. All non-shared attributes at copied into the
    new graph, as well as all disjoint node and edge attributes.

    Parameters
    ----------
    G, H : networkx.DiGraph
        Graphs that will be combined.

    Returns
    -------
    graph : networkx.DiGraph
        Combination of two graphs with all node and edge numeric
        attributes summed.

    """
    graph = nx.DiGraph()
    graph.add_nodes_from(combined_graph_nodes(G, H))
    graph.add_edges_from(combined_graph_edges(G, H))

    if not directed:
        graph = graph.to_undirected()
    else:
        pass

    return graph

from itertools import chain
from numbers import Number
import os

import networkx as nx
import numpy as np

from .util import Scaler

# __all__ = ['combined_graph_nodes', 'combined_graph_edges', 'combine_graphs']


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
    directed : bool, optional
        If `False`, return an undirected graph. Default is `True`.

    Returns
    -------
    graph : networkx.DiGraph
        Combination of two graphs with all node and edge numeric
        attributes summed.
    """

    graph = nx.DiGraph()
    graph.add_nodes_from(_combined_graph_nodes(G, H))
    graph.add_edges_from(_combined_graph_edges(G, H))

    if not directed:
        graph = graph.to_undirected()
    else:
        pass

    return graph


def _combined_graph_nodes(G, H):
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
            attr = H.node[n].copy()
            gdata = G.node[n]

            # All node attributes that are shared.
            shared = set(attr) & set(gdata)
            for key in shared:
                if isinstance(attr[key], Number):
                    attr.update({key: attr[key] + gdata[key]})

            # All node attributes that are not shared.
            not_shared = set(gdata) - set(attr)
            attr.update((key, gdata[key]) for key in not_shared)
        elif n in G.nodes:
            attr = G.node[n]
        else:
            attr = H.node[n]

        yield n, attr


def _combined_graph_edges(G, H):
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
            attr = H[u][v].copy()
            gdata = G[u][v]
            shared = set(attr) & set(gdata)
            for key in shared:
                if isinstance(attr[key], Number):
                    attr.update({key: attr[key] + gdata[key]})

            not_shared = set(gdata) - set(attr)
            attr.update((key, gdata[key]) for key in not_shared)
        elif (u, v) in G.edges:
            attr = G[u][v]
        else:
            attr = H[u][v]

        yield u, v, attr


def _prepare_graph(G, edge_attr=None, drop_all_below=None, style_edge=False,
                   scaler_range=(0, 1), show_labels=False,
                   image_loc='SMILESimages'):
    """Prepares a graph for visualization with Graphviz.

    Parameters
    ----------
    G : nx.DiGraph
    edge_attr : str, optional
        Name of edge attribute of interest. The specified edge
        attribute will be used to style any graph edges, if desired,
        and can be used to filter out low or high values. Default is
        `None`.
    drop_all_below : float or int, optional
        If specified, all edges with attributes of `edge_attr` with
        values less than `drop_all_below` will not be carried over into
        the new graph for visualization. Default is `None`.
    style_edge : bool, optional
        If `True`, then the final graph will use `edge_attr` to style
        edges with `penwidth`. Default is `False`.
    scaler_range : tuple of int or float, optional
        Tells the program what range to scale the values in `edge_attr`
        to. Default is (0, 1).
    show_labels : bool
        If `True`, node labels will be placed at the bottom of each
        node. Default is `False`.

    Returns
    -------
    graph : nx.DiGraph
        NetworkX directed graph built to user specifications.
    """

    if edge_attr is None and style_edge:
        raise AssertionError("If edge attribute is not provided, " +
                             "edges cannot be styled.")
    elif edge_attr is None and drop_all_below is not None:
        raise AssertionError("If edge attribute is not provided, " +
                             "`drop_all_below` cannot be specified.")
    else:
        pass

    graph = nx.DiGraph()

    first_node = [n for n, data in G.nodes.data('rank') if data == 0][0]
    graph.add_node(first_node, rank=0)

    graph.add_nodes_from([n for n in G.nodes(data=False) if n != first_node])

    for n in graph.nodes:
        graph.node[n]['image'] = os.path.join(image_loc, str(n) + '.png')
        if show_labels:
            graph.node[n]['label'] = n
            graph.node[n]['labelloc'] = 'b'
        else:
            graph.node[n]['label'] = ''

    if style_edge:
        if drop_all_below is not None:
            data_range = np.array([data[2] for data in G.edges.data(edge_attr)
                                   if not data[2] < drop_all_below])
        else:
            data_range = np.array([data[2] for data
                                   in G.edges.data(edge_attr)])
        scaler = Scaler(*scaler_range)
        scaler.set_data_range(data_range.min(), data_range.max())
    else:
        pass

    for u, v in G.edges:
        if drop_all_below is not None:
            if G.edges[u, v][edge_attr] < drop_all_below:
                continue
            else:
                pass
        else:
            pass

        if style_edge:
            if (u, v) in graph.edges:
                pass
            else:
                graph.add_edge(u, v, penwidth=  # noqa
                               scaler.transform(G.edges[u, v][edge_attr]))
        else:
            if graph.has_edge(u, v):
                pass
            elif edge_attr is None and graph.has_edge(v, u):
                graph.edges[v, u]['dir'] = 'both'
            else:
                graph.add_edge(u, v)
                if edge_attr is None:
                    pass
                else:
                    graph.edges[u, v][edge_attr] = G.edges[u, v][edge_attr]

    return graph

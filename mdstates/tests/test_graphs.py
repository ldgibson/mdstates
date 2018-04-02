import os

import networkx as nx
from numpy.testing import assert_almost_equal

from ..graphs import combine_graphs, _combined_graph_edges,\
    _combined_graph_nodes, _prepare_graph


def test_combine_graphs():
    # Generate test graphs.
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.node[1]['color'] = 'red'
    G.node[2]['color'] = 'blue'
    G.node[2]['shape'] = 'circle'
    G.node[2]['size'] = 8
    G.add_edges_from([(1, 2, {'weight': 1, 'color': 'blue'}),
                      (2, 3, {'weight': 1, 'color': 'blue',
                              'show': 'true'})])

    H = nx.Graph()
    H.add_nodes_from([2, 3, 4])
    H.node[2]['color'] = 'white'
    H.node[2]['bold'] = 'true'
    H.node[2]['size'] = 6
    H.node[3]['color'] = 'blue'
    H.add_edges_from([(2, 3, {'weight': 1, 'color': 'red',
                              'style': 'dotted'}),
                      (3, 4, {'weight': 1, 'color': 'red'})])

    # Correct graph properties
    true_nodes = [(1, {'color': 'red'}),
                  (2, {'color': 'white', 'shape': 'circle',
                       'size': 14, 'bold': 'true'}),
                  (3, {'color': 'blue'}),
                  (4, {})]

    true_edges = [(1, 2, {'weight': 1, 'color': 'blue'}),
                  (2, 3, {'weight': 2, 'color': 'red',
                          'style': 'dotted', 'show': 'true'}),
                  (3, 4, {'weight': 1, 'color': 'red'})]

    combined = combine_graphs(G, H, directed=True)

    # Check specific node attributes.
    assert combined.node[1]['color'] == 'red',\
        "Not propagating node attributes correctly."
    assert combined.node[2]['color'] == 'white',\
        "Not propagating node attributes correctly."
    assert combined.node[3]['color'] == 'blue',\
        "Not propagating node attributes correctly."

    # Check node attribute summation.
    assert combined.node[2]['size'] == 14,\
        "Not properly summing numeric node attributes."

    # Check edge attribute summation.
    assert combined[2][3]['weight'] == 2,\
        "Not properly summing numeric edge attributes."

    # Ensure all nodes are correctly named and paired
    # with their attributes.
    for n, data in combined.nodes(data=True):
        assert (n, data) in true_nodes,\
            "Incorrect node and/or node information."

    # Ensure all edges are correctly named and paired
    # with their attributes.
    for u, v, data in combined.edges(data=True):
        assert (u, v, data) in true_edges,\
            "Incorrect edge and/or edge information."

    # Test generation of directed/undirected graphs.
    assert combined.is_directed(), "Directed graphs not being generated."

    undirected = combine_graphs(G, H, directed=False)
    assert not undirected.is_directed(),\
        "Undirected graphs not being generated."
    return


def test__combined_graph_edges():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2, {'weight': 1, 'color': 'blue'}),
                      (2, 3, {'weight': 1, 'color': 'blue',
                              'show': 'true'})])

    H = nx.Graph()
    H.add_nodes_from([2, 3, 4])
    H.add_edges_from([(2, 3, {'weight': 1, 'color': 'red',
                              'style': 'dotted'}),
                      (3, 4, {'weight': 1, 'color': 'red'})])

    true_edges = [(1, 2, {'weight': 1, 'color': 'blue'}),
                  (2, 3, {'weight': 2, 'color': 'red',
                          'style': 'dotted', 'show': 'true'}),
                  (3, 4, {'weight': 1, 'color': 'red'})]

    for u, v, data in _combined_graph_edges(G, H):
        assert G.has_edge(u, v) or H.has_edge(u, v),\
            "New edges are being created."
        assert (u, v, data) in true_edges,\
            "Either missing or incorrect edge information."
    return


def test__combined_graph_nodes():
    G = nx.Graph()
    G.add_nodes_from([1, 2])
    G.node[1]['color'] = 'red'
    G.node[2]['color'] = 'blue'
    G.node[2]['shape'] = 'circle'
    G.node[2]['size'] = 8

    H = nx.Graph()
    H.add_nodes_from([2, 3])
    H.node[2]['color'] = 'white'
    H.node[2]['bold'] = 'true'
    H.node[2]['size'] = 6
    H.node[3]['color'] = 'blue'

    true_nodes = [(1, {'color': 'red'}),
                  (2, {'color': 'white', 'shape': 'circle',
                       'size': 14, 'bold': 'true'}),
                  (3, {'color': 'blue'})]

    for n, data in _combined_graph_nodes(G, H):
        assert G.has_node(n) or H.has_node(n),\
            "New nodes are being created."
        assert (n, data) in true_nodes,\
            "Either missing or incorrect node information."

    return


def test__prepare_graph():
    G = nx.DiGraph()
    G.add_node(2, rank=0)
    G.add_edges_from([(1, 2, {'counts': 4}),
                      (2, 3, {'counts': 2}),
                      (2, 1, {'counts': 1})])

    # Check that errors are raised with bad input.
    try:
        _prepare_graph(G, style_edge=True)
    except(AssertionError):
        pass
    else:
        raise Exception("Edges cannot be styled if " +
                        "edge_attr is not specified.")

    try:
        _prepare_graph(G, drop_all_below=4)
    except(AssertionError):
        pass
    else:
        raise Exception("Cannot drop edges with attribute values less than " +
                        str(drop_all_below) +
                        " if `edge_attr` is not specified.")

    # Test function with only `edge_attr` and `image_loc` flags.
    graph = _prepare_graph(G, edge_attr='counts', image_loc="smiles")

    true_edges = [(1, 2, {'counts': 4}),
                  (2, 3, {'counts': 2}),
                  (2, 1, {'counts': 1})]
    
    assert graph.node[2]['rank'] == 0, "Node 2 must have rank=0"
    
    for n in graph:
        assert graph.node[n]['image'] == os.path.join("smiles",
                                                      str(n) + '.png'),\
            "Incorrect image path specified."

    assert graph.edges[1, 2]['counts'] == 4, "Incorrect edge attribute value."
    assert graph.edges[2, 3]['counts'] == 2, "Incorrect edge attribute value."
    assert graph.edges[2, 1]['counts'] == 1, "Incorrect edge attribute value."

    # Test new flags.
    graph2 = _prepare_graph(G, edge_attr='counts', drop_all_below=2)

    assert graph2.node[2]['rank'] == 0, "Node 2 must have rank=0"
    
    for n in graph2:
        assert graph2.node[n]['image'] == os.path.join("SMILESimages",
                                                      str(n) + '.png'),\
            "Incorrect image path specified."

    assert graph2.edges[1, 2]['counts'] == 4, "Incorrect edge attribute value."
    assert graph2.edges[2, 3]['counts'] == 2, "Incorrect edge attribute value."
    assert not graph2.has_edge(2, 1), "Edge should have been dropped."

    # Test `style_edge` flag.
    graph3 = _prepare_graph(G, edge_attr='counts', style_edge=True)

    assert graph3.node[2]['rank'] == 0, "Node 2 must have rank=0"
    
    for n in graph3:
        assert graph3.node[n]['image'] == os.path.join("SMILESimages",
                                                      str(n) + '.png'),\
            "Incorrect image path specified."

    assert_almost_equal(graph3.edges[1, 2]['penwidth'], 1.0,
                        err_msg="Incorrect styling.")
    assert_almost_equal(graph3.edges[2, 3]['penwidth'], 1 / 3,
                        err_msg="Incorrect styling.")
    assert_almost_equal(graph3.edges[2, 1]['penwidth'], 0.0,
                        err_msg="Incorrect styling.")

    return

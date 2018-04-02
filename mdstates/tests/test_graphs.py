import networkx as nx

from ..graphs import combine_graphs, _combined_graph_edges


def test_combine_graphs():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.node[2]['color'] = 'blue'
    G.add_edges_from([(1, 2, {'weight': 1}), (2, 3, {'weight': 1})])

    H = nx.Graph()
    H.add_nodes_from([2, 3, 4])
    H.node[3]['color'] = 'red'
    H.add_edges_from([(2, 3, {'weight': 1}), (3, 4, {'weight': 1})])

    combined = combine_graphs(G, H)
    true_nodes = [1, 2, 3, 4]

    for n in true_nodes:
        assert combined.has_node(n), "Node lists are not the same."

    assert combined.node[2]['color'] == 'blue',\
        "Not propagating node attributes."
    assert combined.node[3]['color'] == 'red',\
        "Not propagating node attributes."

    assert combined[2][3]['weight'] == 2,\
        "Not properly summing edge attributes."

    true_edges = [(1, 2, {'weight': 1}),
                  (2, 3, {'weight': 2}),
                  (3, 4, {'weight': 1})]

    for edge in combined.edges(data=True):
        assert edge in true_edges, "Incorrect edges."
    return


def test__combined_graph_edges():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2, {'weight': 1}), (2, 3, {'weight': 1})])

    H = nx.Graph()
    H.add_nodes_from([2, 3, 4])
    H.add_edges_from([(2, 3, {'weight': 1}), (3, 4, {'weight': 1})])

    true_edges = [(1, 2, {'weight': 1}),
                  (2, 3, {'weight': 2}),
                  (3, 4, {'weight': 1})]

    for u, v, data in _combined_graph_edges(G, H):
        assert (u, v) in set(G.edges) | set(H.edges),\
            "Edge pair not found in original graphs."
        assert (u, v, data) in true_edges,\
            "Not all edges are found."
    return

# -*- coding: utf-8 -*-
"""
Generators and metrics for selected network graphs for WSC paper.

"""

import numpy as np
import networkx as nx
import random


def get_metrics(G):
    """Returns dict of selected metrics on digraph G."""
    res = dict()
    nnodes = G.number_of_nodes()

    res['average out-degree'] = sum([x[1] for x in G.out_degree()])/float(nnodes)

    res['assortativity coefficient'] = nx.algorithms.degree_pearson_correlation_coefficient(G)
    res['reciprocity'] = nx.algorithms.reciprocity(G)

    # all_pairs_short... only returns reachable pairs, which is what I want
    shortest_path_length_list = nx.algorithms.all_pairs_shortest_path_length(G)
    shortest_path_lengths = []
    for i in shortest_path_length_list:
        shortest_path_lengths.extend(filter(lambda x: x > 0, i[1].values()))
    res['global efficiency'] = 1/(nnodes * (nnodes-1)) * sum([1/l for l in shortest_path_lengths])

    return res

def directed_random_tree(n, arrows_from_root, seed):
    G = nx.random_tree(n, seed)
    betweeness = nx.betweenness_centrality(G, normalized=False, seed=0)
    root = max(betweeness, key=lambda key: betweeness[key])
    T = nx.bfs_tree(G, root)  # Arrows point away from root

    if arrows_from_root:
        return T
    else:
        return nx.DiGraph(nx.reverse(T, copy=False))  # Cast from ReverseView

def erdos_renyi_random(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed, directed=True)

def scale_free(n, alpha, beta, gamma, delta_in, delta_out, seed):
    G = nx.scale_free_graph(n, alpha, beta, gamma, delta_in, delta_out, seed=seed)

    # The parameters & seeds I chose for this experiment can produce self-loops
    # on the scale free digraph, so I remove them.
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return nx.DiGraph(G)  # Flatten multi-digraph structure

def random_k_out(n, k, seed):
    np.random.seed(seed)
    G = nx.DiGraph()

    nodes = list(range(0, n))
    for n in nodes:
        neighbors = np.random.choice([i for i in nodes if i != n], size=k, replace=False)
        edges = [(n, i) for i in neighbors]
        G.add_edges_from(edges)

    return G

def get_network_instance(n, inst_num):
    if n in saved_instances:
        if inst_num in saved_instances[n]:
            return saved_instances[n][inst_num]
    else:
        saved_instances[n] = {}

    func, args = instance_definitions[n][inst_num]
    G = func(*args)

    leafs = [k for k, v in G.out_degree if v == 0]
    if len(leafs) > 0:
        connect_leafs(G, leafs)

    saved_instances[n][inst_num] = G
    return G

def connect_leafs(G, leafs):
    """Some graph generators make nodes with no neighbors. This fixes that."""
    random.seed(1)  # This is a side application, so I'm okay using different seed than generator
    for leaf in leafs:
        G.add_edge(leaf, random.choice(list(nx.non_neighbors(G, leaf))))


# Memoization for network instances; allows model to reuse across replications
# Note: this gets wiped when running as main().
saved_instances = {}

# All of these instances are weakly connected and have no self-loops.
# The last value in the parameter list is always the RNG seed.
# Network parameters were selected in attempt to produce good spread of metrics
instance_definitions = {
    100: {
        1: (directed_random_tree, (100, True, 1)),
        2: (directed_random_tree, (100, False, 1)),
        3: (erdos_renyi_random, (100, 0.08, 1)),
        4: (erdos_renyi_random, (100, 0.08, 2843)),
        5: (erdos_renyi_random, (100, 0.3, 30)),
        6: (erdos_renyi_random, (100, 0.3, 100)),
        7: (scale_free, (100, 0.41, 0.54, 0.05, 0.1, 0.1, 13)),
        8: (scale_free, (100, 0.41, 0.54, 0.05, 0.1, 0.1, 61)),
        9: (scale_free, (100, 0.80, 0.05, 0.15, 0.0, 0.1, 1)),
        10: (scale_free, (100, 0.80, 0.05, 0.15, 0.0, 0.1, 69)),
        11: (random_k_out, (100, 3, 11)),
        12: (random_k_out, (100, 3, 52)),
        13: (random_k_out, (100, 10, 74)),
        14: (random_k_out, (100, 10, 75)),
    },
    500: {
        1: (directed_random_tree, (500, True, 1)),
        2: (directed_random_tree, (500, False, 1)),
        3: (erdos_renyi_random, (500, 0.05, 45)),
        4: (erdos_renyi_random, (500, 0.05, 100)),
        5: (erdos_renyi_random, (500, 0.15, 8)),
        6: (erdos_renyi_random, (500, 0.15, 80)),
        7: (scale_free, (500, 0.41, 0.54, 0.05, 0.1, 0.1, 18)),
        8: (scale_free, (500, 0.41, 0.54, 0.05, 0.1, 0.1, 23)),
        9: (scale_free, (500, 0.80, 0.05, 0.15, 0.0, 0.1, 27)),
        10: (scale_free, (500, 0.80, 0.05, 0.15, 0.0, 0.1, 69)),
        11: (random_k_out, (500, 3, 38)),
        12: (random_k_out, (500, 3, 50)),
        13: (random_k_out, (500, 10, 9)),
        14: (random_k_out, (500, 10, 49)),
    },
    1000: {
        1: (directed_random_tree, (1000, True, 1)),
        2: (directed_random_tree, (1000, False, 1)),
        3: (erdos_renyi_random, (1000, 0.05, 6)),
        4: (erdos_renyi_random, (1000, 0.05, 100)),
        5: (erdos_renyi_random, (1000, 0.068, 10)),
        6: (erdos_renyi_random, (1000, 0.068, 90)),
        7: (scale_free, (1000, 0.41, 0.54, 0.05, 0.1, 0.1, 62)),
        8: (scale_free, (1000, 0.41, 0.54, 0.05, 0.1, 0.1, 68)),
        9: (scale_free, (1000, 0.80, 0.05, 0.15, 0.0, 0.1, 106)),
        10: (scale_free, (1000, 0.80, 0.05, 0.15, 0.0, 0.1, 150)),
        11: (random_k_out, (1000, 3, 44)),
        12: (random_k_out, (1000, 3, 49)),
        13: (random_k_out, (1000, 10, 33)),
        14: (random_k_out, (1000, 10, 83)),
    },
}

instance_name_map = {
    1: 'Directed Random Tree',
    2: 'Directed Random Tree (Reversed)',
    3: 'Erdos-Renyi Random #1a',
    4: 'Erdos-Renyi Random #1b',
    5: 'Erdos-Renyi Random #2a',
    6: 'Erdos-Renyi Random #2b',
    7: 'Scale-Free #1a',
    8: 'Scale-Free #1b',
    9: 'Scale-Free #2a',
    10: 'Scale-Free #2b',
    11: 'Random k-out (k=3) #a',
    12: 'Random k-out (k=3) #b',
    13: 'Random k-out (k=10) #a',
    14: 'Random k-out (k=10) #b',
}


if __name__ == '__main__':
    # Code for testing instance definitions and creating data table
    import pandas as pd

    data = []
    for N, v in instance_definitions.items():
        for i in v:
            G = get_network_instance(N, i)
            if not nx.is_weakly_connected(G):
                print(N, i, " NOT connected")
            if nx.number_of_selfloops(G) > 0:
                print(N, i, " has {} selfloops".format(nx.number_of_selfloops(G)))
            bad_nodes = [k for k, v in G.out_degree if v == 0]
            if len(bad_nodes) > 0:
                print(N, i, "has {} bad nodes".format(len(bad_nodes)))

            met = get_metrics(G)
            x = [N, i, v[i][0].__name__, *met.values()]
            data.append(x)

    df = pd.DataFrame(data, columns=['N', 'Instance', 'Family', 'Avg. Out-Degree', 'Assortativity', 'Reciprocity', 'Efficiency'])


# Helper code for making figure of example network
#G = nx.erdos_renyi_graph(15, 0.15, directed=True)
#pos = nx.spring_layout(G, k=2)
#labels={i:i+1 for i in list(G.nodes)}
#nx.drawing.draw(G, pos=pos, font_size=18, labels=labels, arrowsize=16, node_color='#dddddd', node_size=750)

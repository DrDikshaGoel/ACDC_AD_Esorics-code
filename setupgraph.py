from networkx.readwrite.gpickle import read_gpickle
import random
from utility import (
    dist_in_hops,
    remove_dead_nodes,
    trace_non_splitting_paths,
    gen_non_splitting_paths_cache,
    gen_edge_to_non_splitting_paths_cache,
    gen_blockworthy,
    gen_non_splitting_path_logic,
)
from numpy.random import multivariate_normal
import numpy.random
import settings
import pickle


def random_setup(fn, seed, start_node_number, budget, story_name, distribution):
    global CG

    CG = read_gpickle(f"data/{fn}.gpickle")
    #CG = pickle.load(f"data/{fn}.gpickle")
    random.seed(seed)
    CG.graph["budget"] = budget

    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    start_nodes_candidates = []
    for v in CG.nodes():
        dist = dist_in_hops(CG, v)
        assert dist is not None
        start_nodes_candidates.append((dist, v))
    start_nodes = [
        x[1] for x in sorted(start_nodes_candidates)[-2 * start_node_number :]
    ]
    random.shuffle(start_nodes)
    start_nodes = start_nodes[-1 * start_node_number :]
    if CG.graph["DA"] in start_nodes:
        print("DA is not supposed to be a start node")
        assert False
    assert len(start_nodes) == start_node_number
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"

    numpy.random.seed(seed)

    max_dist = max(start_nodes_candidates)[0]
    for u, v in CG.edges():
        dist = dist_in_hops(CG, u)
        assert dist is not None
        CG[u][v]["blockable"] = random.random() <= dist / max_dist
        if settings.distribution == "I":
            p1 = 0.2 * random.random()
            p2 = 0.2 * random.random()
        elif settings.distribution == "P":
            p1, p2 = multivariate_normal(
                [0.1, 0.1], [[0.05 ** 2, 0.5 * 0.05 ** 2], [0.5 * 0.05 ** 2, 0.05 ** 2]]
            )
        elif settings.distribution == "N":
            p1, p2 = multivariate_normal(
                [0.1, 0.1],
                [[0.05 ** 2, -0.5 * 0.05 ** 2], [-0.5 * 0.05 ** 2, 0.05 ** 2]],
            )
        else:
            assert False
        p1 = min(1, p1)
        p2 = min(1, p2)
        p1 = max(0, p1)
        p2 = max(0, p2)
        CG[u][v]["detect_p"] = p1
        CG[u][v]["fail_p"] = p2

    has_blockable = False
    for u, v in CG.edges():
        if CG[u][v]["blockable"]:
            has_blockable = True
    assert has_blockable

########################
    # Initialize counters
    num_edges_with_session = 0
    num_edges_without_session = 0

    # Iterate through edges
    for u, v in CG.edges():
        if CG[u][v]["access_type"] == "HasSession":
            # Increment the counter for edges with "HasSession"
            num_edges_with_session += 1
        else:
            # Increment the counter for edges without "HasSession"
            num_edges_without_session += 1

    print("Number of edges with 'HasSession':", num_edges_with_session)
    print("Number of edges without 'HasSession':", num_edges_without_session)
    #for u, v in CG.edges():
    #    if CG[u][v]["access_type"] == "HasSession":
    #        print("egde is has session")
    #    else:
    #        print("not has session")

#################

    for u in CG.nodes():
        if CG.nodes[u]["node_type"] == "S":
            for prev in list(CG.predecessors(u)):
                CG.remove_edge(prev, u)
    remove_dead_nodes(CG)

    # remove in degree 0
    keep_deleting = True
    while keep_deleting:
        keep_deleting = False
        for u in list(CG.nodes()):
            if u == CG.graph["DA"]:
                assert CG.in_degree(u) > 0
            if not CG.nodes[u]["node_type"] == "S" and CG.in_degree(u) == 0:
                CG.remove_node(u)
                keep_deleting = True

    start_nodes = set()
    for u in CG.nodes():
        if CG.nodes[u]["node_type"] == "S":
            start_nodes.add(u)
    assert len(start_nodes) >= 1

    # set of start nodes
    CG.graph["start_nodes"] = start_nodes

    # set of non splitting paths
    non_splitting_paths_sorted = sorted(list(trace_non_splitting_paths(CG)))
    CG.graph["non_splitting_paths_sorted"] = non_splitting_paths_sorted
    CG.graph["initial_state_tuple"] = tuple(0 for _ in non_splitting_paths_sorted)

    # dict that maps nsp to full path
    CG.graph["non_splitting_paths_cache"] = gen_non_splitting_paths_cache(CG)

    # dict that maps edge to all nsps using it
    CG.graph[
        "edge_to_non_splitting_paths_cache"
    ] = gen_edge_to_non_splitting_paths_cache(CG)

    # dict that maps blockworthy edges to all nsps using it
    CG.graph["blockworthy"] = gen_blockworthy(CG)

    # a few logic shortcuts for speeding up state transition
    (
        CG.graph["new_success"],
        CG.graph["new_failure"],
        CG.graph["last_steps"],
    ) = gen_non_splitting_path_logic(CG)

########################
    # Initialize counters
    num_edges_with_session = 0
    num_edges_without_session = 0

    # Iterate through edges
    for u, v in CG.edges():
        if CG[u][v]["access_type"] == "HasSession":
            # Increment the counter for edges with "HasSession"
            num_edges_with_session += 1
        else:
            # Increment the counter for edges without "HasSession"
            num_edges_without_session += 1

    print("Number of edges with 'HasSession':", num_edges_with_session)
    print("Number of edges without 'HasSession':", num_edges_without_session)
    #for u, v in CG.edges():
    #    if CG[u][v]["access_type"] == "HasSession":
    #        print("egde is has session")
    #    else:
    #        print("not has session")

#################

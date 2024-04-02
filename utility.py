from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.exception import NetworkXNoPath
from math import log

# failure = {"AdminTo": 0.10, "MemberOf": 0.10, "HasSession": 0.2}
# detection = {"AdminTo": 0.05, "MemberOf": 0.05, "HasSession": 0.1}


def trans(x):
    return (-1) * log(1 - x)


def remove_dead_nodes(g):
    keep_going = True
    while keep_going:
        keep_going = False
        for v in set(g.nodes):
            if dist_in_hops(g, v) is None:
                g.remove_node(v)
                keep_going = True


def dist_in_hops(g, v):
    try:
        path = shortest_path(g, v, g.graph["DA"])
        return len(path) - 1
    except NetworkXNoPath:
        return None


def report(g):
    print()
    n = g.number_of_nodes()
    m = g.number_of_edges()
    print("n,m", n, m)
    d = [g.out_degree(v) for v in g.nodes() if g.out_degree(v) > 1] + [1]
    print("out degrees", d[:-1])
    print("max out degree", max(d))
    print("number of splitting nodes", len(d) - 1)
    print("number of non-tree edges", m - n + 1)
    # budget being there means it is already set up by setupgraph
    if "budget" in g.graph:
        print("state cardinality", len(g.graph["initial_state_tuple"]))
        print("max dp size", 3 ** len(g.graph["initial_state_tuple"]))
    print()


def follow_non_splitting_path(g, u, v):
    if g.out_degree(v) > 1 or g.graph["DA"] == v:
        return [u, v]
    else:
        future = list(g.successors(v))
        assert len(future) == 1
        return [u] + follow_non_splitting_path(g, v, future[0])


def is_path_blockable(g, path):
    for i in range(len(path) - 1):
        if g[path[i]][path[i + 1]]["blockable"]:
            return True
    return False


def last_blockable(g, path):
    res = []
    for i in range(len(path) - 1):
        if g[path[i]][path[i + 1]]["blockable"]:
            res.append((path[i], path[i + 1]))
    if len(res) >= 1:
        return res[-1]
    else:
        return None


def trace_non_splitting_paths(g):
    non_splitting_paths = set()
    finished = set()
    working_nodes = list(g.graph["start_nodes"])
    while len(working_nodes) > 0:
        u = working_nodes.pop()
        for v in g.successors(u):
            dest = follow_non_splitting_path(g, u, v)[-1]
            non_splitting_paths.add((u, v))
            if dest not in working_nodes and dest not in finished:
                working_nodes.append(dest)
        finished.add(u)
    return non_splitting_paths


def gen_non_splitting_paths_cache(g):
    non_splitting_paths = trace_non_splitting_paths(g)
    res = {}
    for u, v in non_splitting_paths:
        res[(u, v)] = follow_non_splitting_path(g, u, v)
    return res


def gen_edge_to_non_splitting_paths_cache(g):
    non_splitting_paths = trace_non_splitting_paths(g)
    res = {}
    for u, v in non_splitting_paths:
        path = follow_non_splitting_path(g, u, v)
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge not in res:
                res[edge] = {(u, v)}
            else:
                res[edge].add((u, v))
    return res


def gen_blockworthy(g):
    bw = {}
    non_splitting_paths = trace_non_splitting_paths(g)
    for u, v in non_splitting_paths:
        path = follow_non_splitting_path(g, u, v)
        bw_edge = last_blockable(g, path)
        if bw_edge is not None:
            if bw_edge not in bw:
                bw[bw_edge] = {(u, v)}
            else:
                bw[bw_edge].add((u, v))
    return bw


def gen_non_splitting_path_logic(g):
    non_splitting_paths = trace_non_splitting_paths(g)
    new_success = {}
    for (u, v) in non_splitting_paths:
        dest = follow_non_splitting_path(g, u, v)[-1]
        res = set()
        for (uprime, vprime) in non_splitting_paths:
            if (u, v) == (uprime, vprime):
                continue
            destprime = follow_non_splitting_path(g, uprime, vprime)[-1]
            if destprime == dest:
                res.add((uprime, vprime))
        # if (u,v) is successful, then all nsps in new_success[(u,v)] are successful
        new_success[(u, v)] = res
    new_failure = {}
    for (u, v) in non_splitting_paths:
        dest = follow_non_splitting_path(g, u, v)[-1]
        from_dest = set()
        other_sources = set()
        for (uprime, vprime) in non_splitting_paths:
            if (u, v) == (uprime, vprime):
                continue
            if uprime == dest:
                from_dest.add((uprime, vprime))
            destprime = follow_non_splitting_path(g, uprime, vprime)[-1]
            if destprime == dest:
                other_sources.add((uprime, vprime))
        # if all nsps TO a node x fail, then all nsps FROM node x fail
        new_failure[(u, v)] = (other_sources, from_dest)
    # all nsps that directly lead to DA
    last_steps = set()
    for (u, v) in non_splitting_paths:
        dest = follow_non_splitting_path(g, u, v)[-1]
        if dest == g.graph["DA"]:
            last_steps.add((u, v))
    return new_success, new_failure, last_steps
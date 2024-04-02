from collections import deque
import setupgraph

# from utility import failure, detection


# state: 0 = ?, 1 = F, 2 = S


def tuple_to_state(tupl):
    state = {}
    for i in range(len(tupl)):
        state[setupgraph.CG.graph["non_splitting_paths_sorted"][i]] = tupl[i]
    return state


def state_to_tuple(state):
    return tuple(
        state[nsp] for nsp in setupgraph.CG.graph["non_splitting_paths_sorted"]
    )


# singular
def fix_state_with_new_success(state, success_nsp):
    state = state.copy()
    assert state[success_nsp] == 0
    state[success_nsp] = 2
    for nsp in setupgraph.CG.graph["new_success"][success_nsp]:
        assert state[nsp] != 2
        state[nsp] = 2
    return state


# plural
def fix_state_with_new_failures(state, fail_nsps):
    state = state.copy()
    for nsp in fail_nsps:
        assert state[nsp] != 2
        state[nsp] = 1
    queue = deque(fail_nsps)
    while len(queue) > 0:
        nsp = queue.popleft()
        assert state[nsp] == 1
        other_sources, from_dest = setupgraph.CG.graph["new_failure"][nsp]
        all_failed = True
        for y in other_sources:
            if state[y] != 1:
                all_failed = False
                break
        if all_failed:
            for y in from_dest:
                if state[y] == 0:
                    state[y] = 1
                    queue.append(y)
    return state


# whenever enter state, already check the corner case where no further action is possible
def admissible_actions(state):
    all_failed = True
    for x in setupgraph.CG.graph["last_steps"]:
        if state[x] == 2:
            return 1
        elif state[x] == 0:
            all_failed = False
    if all_failed:
        return 0

    secured_nodes = set(setupgraph.CG.graph["start_nodes"])
    for nsp, status in state.items():
        if status == 2:
            secured_nodes.add(setupgraph.CG.graph["non_splitting_paths_cache"][nsp][-1])
    actions = set()
    for nsp, status in state.items():
        if status == 0 and nsp[0] in secured_nodes:
            actions.add(nsp)
    if len(actions) == 0:
        return 0
    else:
        return actions


def state_transition(state, nsp):
    # assert that admissible_actions(state) is a set
    
    #print("state[nsp]", state, type(state), nsp, type(nsp))
    #print("state[nsp] == 0", state[nsp])
    assert state[nsp] == 0 and nsp in admissible_actions(state)
    future_states = {}
    continue_p = 1
    total_detect_p = 0
    path = setupgraph.CG.graph["non_splitting_paths_cache"][nsp]
    for i in range(len(path) - 1):
        u, v = (path[i], path[i + 1])
        # detect_p = detection[setupgraph.CG[u][v]["access_type"]]
        detect_p = setupgraph.CG[u][v]["detect_p"]
        total_detect_p += continue_p * detect_p
        # fail_p = failure[setupgraph.CG[u][v]["access_type"]]
        fail_p = setupgraph.CG[u][v]["fail_p"]
        fail_nsps = setupgraph.CG.graph["edge_to_non_splitting_paths_cache"][(u, v)]
        fail_state = fix_state_with_new_failures(state, fail_nsps)
        fail_state_tuple = state_to_tuple(fail_state)
        if fail_state_tuple in future_states:
            future_states[fail_state_tuple] += fail_p * continue_p
        else:
            future_states[fail_state_tuple] = fail_p * continue_p
        continue_p *= 1 - detect_p - fail_p
    success_state = fix_state_with_new_success(state, nsp)
    success_state_tuple = state_to_tuple(success_state)
    assert success_state_tuple not in future_states
    future_states[success_state_tuple] = continue_p
    total_future_p = sum(future_states.values())
    assert abs(1 - total_future_p - total_detect_p) < 0.000001
    return future_states

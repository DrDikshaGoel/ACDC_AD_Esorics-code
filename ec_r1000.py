import os
from random import shuffle, random, sample, seed
from states import tuple_to_state, state_to_tuple
import setupgraph
import numpy.random
import settings
import sys

def gen_initial_pop():
    print("\n++generating initial population++++\n\n")
    pop = []
    budget = setupgraph.CG.graph["budget"]
    ec_state_length = len(setupgraph.CG.graph["blockworthy"])
    if budget > ec_state_length:
        setupgraph.CG.graph["budget"] = ec_state_length
        budget = ec_state_length
    while (len(pop) != settings.pop_size):
        tmp_ec_state = [1] * budget + [0] * (ec_state_length - budget)
        shuffle(tmp_ec_state)
        tmp_ec_state_tuple = tuple(tmp_ec_state)
        if tmp_ec_state_tuple not in pop:
            pop += [tmp_ec_state_tuple]
    for i, j in zip(pop, range(settings.pop_size)):
        pop_file = str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_'+str(j)+'.txt'
        #os.makedirs(os.path.dirname(pop_file), exist_ok=True)
        with open(pop_file, 'w') as f:
            for k in i:
               print(k, file=f)
    pop_val = {}
    for i in pop:
        pop_val[i] = random()
    return pop_val

def ec_state_tuple_to_dp_state_tuple(ec_state_tuple):
    blockworthy = setupgraph.CG.graph["blockworthy"]
    blockworthy_edges_sorted = sorted(list(blockworthy.keys()))
    tmp_dp_state = tuple_to_state(setupgraph.CG.graph["initial_state_tuple"])
    assert len(ec_state_tuple) == len(blockworthy_edges_sorted)
    for i in range(len(ec_state_tuple)):
        if ec_state_tuple[i] == 1:
            for nsp in blockworthy[blockworthy_edges_sorted[i]]:
                assert tmp_dp_state[nsp] == 0
                tmp_dp_state[nsp] = 1
    return state_to_tuple(tmp_dp_state)


def ec_apply_val_func(ec_state_tuple, val_func):
    return val_func(ec_state_tuple)

def general_ec(reject_func, val_func, pop, ec_iterations=settings.ec_iterations):
    if pop is None:
        pop = gen_initial_pop(val_func)
    for ec_iteration in range(ec_iterations):
        pop_list = sorted(list(pop.keys()))
        action_type = random()
        times = numpy.random.poisson(lam=1, size=1)[0]
        if action_type <= settings.mutate_p:
            instance = sample(pop_list, 1)[0]
            instance = mutate(instance, times)
        else:
            if len(pop_list) < 2:
                continue
            samples = sample(pop_list, 2)
            instance1 = samples[0]
            instance2 = samples[1]
            instance = crossover(instance1, instance2, times)
        pop = reject_func(pop, instance, val_func)
    return pop


def mutate(instance, times):
    instance = list(instance)
    indices0 = []
    indices1 = []
    for i, v in enumerate(instance):
        if v == 0:
            indices0.append(i)
        else:
            indices1.append(i)
    samples0 = sample(indices0, min([times, len(indices0), len(indices1)]))
    samples1 = sample(indices1, min([times, len(indices0), len(indices1)]))
    for i in samples0:
        instance[i] = 1
    for i in samples1:
        instance[i] = 0
    assert sum(instance) == setupgraph.CG.graph["budget"]
    assert len(instance) == len(setupgraph.CG.graph["blockworthy"])
    return tuple(instance)


def crossover(instance1, instance2, times):
    instance = list(instance1)
    indices0 = []
    indices1 = []
    for i in range(len(instance1)):
        if instance1[i] == 0 and instance2[i] == 1:
            indices0.append(i)
        if instance1[i] == 1 and instance2[i] == 0:
            indices1.append(i)
    samples0 = sample(indices0, min([times, len(indices0), len(indices1)]))
    samples1 = sample(indices1, min([times, len(indices0), len(indices1)]))
    for i in samples0 + samples1:
        instance[i] = 1 - instance[i]
    assert sum(instance) == setupgraph.CG.graph["budget"]
    assert len(instance) == len(setupgraph.CG.graph["blockworthy"])
    return tuple(instance)

def reject_diversity(pop, instance, val_func):
    if instance in pop:
        return pop
    opt = min(pop.values())
    instance_val = ec_apply_val_func(instance, val_func)
    if settings.absolute_diff + opt < instance_val:
        worst_ec_state_tuple, worst_ec_val = worst_defense_from_pop(pop)
        if instance_val < worst_ec_val:
            pop[instance] = instance_val
            del pop[worst_ec_state_tuple]
        return pop
    pop[instance] = instance_val
    if len(pop) <= settings.pop_size:
        return pop
    s = [0] * len(instance)
    for key in pop:
        for i, v in enumerate(key):
            s[i] += v
    res = []
    for key in pop:
        s1 = list(s)
        for i, v in enumerate(key):
            s1[i] -= v
        res.append((sorted(s1, reverse=True), key))
    _, badkey = min(res)
    # always keep the best solution
    if pop[badkey] > opt + 0.000001:
        del pop[badkey]
    else:
        worst_ec_state_tuple, _ = worst_defense_from_pop(pop)
        del pop[worst_ec_state_tuple]
    return pop


def reject_value(pop, instance, val_func):
    if instance in pop:
        return pop
    instance_val = ec_apply_val_func(instance, val_func)
    pop[instance] = instance_val
    if len(pop) > settings.pop_size:
        worst_key, _ = worst_defense_from_pop(pop)
        del pop[worst_key]
    return pop


def greedy_defense(val_func, ec_state_tuple=None):
    # greedy starts from not blocking any edge and then greedily blocks the best blockworthy edge for b times
    if ec_state_tuple is None:
        ec_state_length = len(setupgraph.CG.graph["blockworthy"])
        ec_state_tuple = tuple([0] * ec_state_length)
    ec_state = list(ec_state_tuple)
    if sum(ec_state) == setupgraph.CG.graph["budget"]:
        return ec_state_tuple
    min_val, min_ec_state_tuple = 1000000, None
    for i in range(len(ec_state)):
        if ec_state[i] == 0:
            ec_state[i] = 1
            tmp_ec_state_tuple = tuple(ec_state)
            res = ec_apply_val_func(tmp_ec_state_tuple, val_func)
            if res < min_val:
                min_val = res
                min_ec_state_tuple = tmp_ec_state_tuple
            ec_state[i] = 0
    return greedy_defense(val_func, min_ec_state_tuple)

def best_defense_from_pop(pop):
    return min(pop.items(), key=lambda k: k[1])

def worst_defense_from_pop(pop):
    return max(pop.items(), key=lambda k: k[1])

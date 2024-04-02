#from ppo_policy import get_args
import argparse
import numpy as np
from tianshou.policy import PPOPolicy
import torch
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
import os
import sys
from AgentEnv_file_r1000 import AgentEnv
from tianshou.data import Collector

from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net

from states import tuple_to_state, state_to_tuple
import setupgraph
import numpy.random

from ec_r1000 import *
import settings


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.0005)#0.0001
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=settings.epoch)
    parser.add_argument('--step-per-epoch', type=int, default=80000) #80000 ##In each epoch, the policy is updated --step-per-epoch times using the collected data.
    parser.add_argument('--step-per-collect', type=int, default=2000) #2000  ##how many environment steps are performed before collecting experiences for training.#after these steps, data collected is used for training
    parser.add_argument('--repeat-per-collect', type=int, default=10)#10
    parser.add_argument('--batch-size', type=int, default=1000)#500
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128]) #256
    #parser.add_argument('--training-num', type=int, default=10)#1000
    parser.add_argument('--test-num', type=int, default=1000)#1000
    parser.add_argument('--logdir', type=str, default=str(settings.args["distribution"])+'_log_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--epoch_last', type=int, default=100)#100
    parser.add_argument('--test-num_last', type=int, default=100)#1000
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_known_args()[0]
    return args

def load_plans():
    key0 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_0.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key0= key0 + [int(i)]

    key1 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_1.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key1= key1 + [int(i)]

    key2 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_2.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key2= key2 + [int(i)]

    key3 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_3.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key3= key3 + [int(i)]

    key4 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_4.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key4= key4 + [int(i)]

    key5 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_5.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key5= key5 + [int(i)]

    key6 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_6.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key6= key6 + [int(i)]

    key7 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_7.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key7= key7 + [int(i)]

    key8 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_8.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key8= key8 + [int(i)]

    key9 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_9.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key9= key9 + [int(i)]
    key10 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_10.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key10= key10 + [int(i)]

    key11 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_11.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key11= key11 + [int(i)]

    key12 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_12.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key12= key12 + [int(i)]

    key13 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_13.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key13= key13 + [int(i)]

    key14 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_14.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key14= key14 + [int(i)]

    key15 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_15.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key15= key15 + [int(i)]

    key16 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_16.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key16= key16 + [int(i)]

    key17 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_17.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key17= key17 + [int(i)]

    key18 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_18.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key18= key18 + [int(i)]

    key19 = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_19.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key19= key19 + [int(i)]

    return key0, key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11, key12, key13, key14, key15, key16, key17, key18, key19



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


def new_population():
    print("+++++++++++++GENERTATION++++++++++++++")
    population = load_plans()
    pop_old = {}
    for i in population:
        i = tuple(i)
        pop_old[i] = compute_value((i))
    val_func = compute_value
    print("vaue computed")
    reject_func = settings.args["story_name"]
    print("before rej ct fucntion")
    if reject_func == "reject_diversity":
        pop = general_ec(reject_diversity, val_func, pop_old, settings.ec_iterations)
        for i, j in zip(pop, range(settings.pop_size)):
            pop_file = str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_'+str(j)+'.txt'
            with open(pop_file, 'w') as f:
                for k in i:
                   print(k, file=f)
    """
    elif reject_func == "reject_value":
        pop = general_ec(reject_value, val_func, pop_old, settings.ec_iterations)
        for i, j in zip(pop, range(settings.pop_size)):
            pop_file = str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_'+str(j)+'.txt'
            with open(pop_file, 'w') as f:
                for k in i:
                   print(k, file=f)

    elif reject_func == "greedy_defense":
        ec_tuple = greedy_defense(val_func)
        pop = []
        for i in range(settings.pop_size):
            pop = pop + [ec_tuple]
        for i, j in zip(pop, range(settings.pop_size)):
            pop_file = str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_'+str(j)+'.txt'
            with open(pop_file, 'w') as f:
                for k in i:
                   print(k, file=f)
    """
    return pop


def compute_value(state):
    env = AgentEnv(state)
    args = get_arg()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    policy_file= str(settings.args["distribution"])+'_log_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/ppo/policy.pth'
    log_path = os.path.join(args.logdir, 'ppo')
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
            Actor(net, args.state_shape, device=None).to(args.device)
        )
        critic = DataParallelNet(Critic(net, device=None).to(args.device))
    else:
        actor = Actor(net, args.state_shape, device=args.device).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)
    #actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    #critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)
    dist = torch.distributions.Categorical

    #policy = PPOPolicy(actor, critic, optim, dist)
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        #action_space=train_envs.action_space,
        deterministic_eval=False,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv)

    policy.load_state_dict(torch.load(policy_file))
    state = ec_state_tuple_to_dp_state_tuple(state)
    policy.eval()
    with torch.no_grad():
        #print(state, critic(obs=np.array([state])))
        return critic(obs=np.array([state]))

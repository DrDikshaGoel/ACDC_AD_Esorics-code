import math
import argparse
import os
import pprint
import numpy as np
import torch
import settings

from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, Batch, to_numpy
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Critic

import sys
from time import sleep
import warnings
from tianshou.env.worker import SubprocEnvWorker
warnings.filterwarnings('ignore')
from AgentEnv_file_r1000 import AgentEnv
from  utility_file_r1000 import *
import time
import random
from states import (
    state_to_tuple,
    tuple_to_state,
    admissible_actions,
    state_transition,)

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import Counter

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP

#global already_blocked
time1 = time.time()

def is_between_multiples_of_five(number):
    lower_limit = (number // settings.NNprun_formin) * settings.NNprun_formin
    upper_limit = lower_limit + 0.1
    return lower_limit, upper_limit

class Actor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self.softmax_output = softmax_output

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state)
        state = tuple_to_state(obs[0])
        addmissble_act = admissible_actions(state)
        nsp = list(state.keys())
       
        mask = torch.zeros(len(state), dtype=torch.bool)
        if type(addmissble_act) is not set:
            mask = torch.zeros(len(state), dtype=torch.bool)

        for i in state.keys():
            if type(addmissble_act) is not set:
                for i in range(len(state)):
                    mask[i] = False

            elif i in addmissble_act:
                index_true = nsp.index(i)
                mask[index_true] = True

        logits = self.last(logits)
        logits_modified = logits.clone()

        for i in range(len(state)):
            if mask[i] == False:
                logits_modified[0][i] = -9999999
        
        """
        min_non_zero_value = float('inf')
        min_non_zero_index = -1


        for i in range(len(state)):
            if logits[0][i] > 0 and logits[0][i] < min_non_zero_value:
                min_non_zero_value = logits[0][i]
                min_non_zero_index = i
        """
        
        if self.softmax_output:
            logits_modified = F.softmax(logits_modified, dim=-1)
        """
        lower, upper = is_between_multiples_of_five((time.time() - time1)/60)
        if (lower <= (time.time() - time1)/60 <= upper):
            #print("WE ARE HERE IN LOOP", lower, upper, (time.time() - time1)/60)
            ori_value = compute_values(obs[0])
            new_values = {}
            if not addmissble_act:
                logits = logits_modified
                return logits, hidden

            for acti in addmissble_act:
                state_new = obs[0]
                key, index_key = next(((i, k) for i, (k, v) in enumerate(state.items()) if np.array_equal(k, acti)), None)
                state_new[key] = 1
                new_values[key] = compute_values(state_new)
            closest_key = min(new_values, key=lambda k: torch.abs(new_values[k] - ori_value).item())
            logits_modified[0][closest_key] = settings.NNweight_of_ori * logits_modified[0][closest_key]
        """
        logits = logits_modified
        return logits, hidden

def get_args():
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

#nonlocal already_blocked
already_blocked = []


def test_ppo(plans, args=get_args(), policy_file=None):
    plans = load_plans()
    already_blocked = []
    env = AgentEnv(plans[0]) #create 20 defenseive plans and create for each defense 50 graphs and save snapshots to corresponsing folder
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    env_func = [lambda i=i: AgentEnv(plans[i], i) for i in range(settings.train_env)]
    train_envs = SubprocVectorEnv(env_func)
    env_func_test = [lambda i=i: AgentEnv(plans[i], i) for i in range(settings.test_env)]
    test_envs = SubprocVectorEnv(env_func_test)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
            Actor(net, args.state_shape, device=None).to(args.device)
        )
        critic = DataParallelNet(Critic(net, device=None).to(args.device))
    else:
        actor = Actor(net, args.state_shape, device=args.device).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)

    actor_critic = ActorCritic(actor, critic)

    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical

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
        deterministic_eval=False,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv)

    # collector
    buffer_store = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs,  buffer_store, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    #train_collector.collect(n_step=args.batch_size * settings.train_env)
    log_path = os.path.join(args.logdir, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        torch.save(optim.state_dict(), os.path.join(log_path, 'optim.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        return ckpt_path

    def train_fn(epoch,  env_step):
        iter = 0
        if (epoch  < settings.stop_after_epoch and epoch > 1):
            if ((epoch % settings.reset_after_epoch == 0 and env_step % args.step_per_epoch == 0)):## and (env_step % args.step_per_epoch == 0)):# or epoch == 1):
                print("*****************I am in THERE RESET environemnt")
                #nonlocal already_blocked
                global already_blocked1
                global already_blocked
                already_blocked = []
                already_blocked1 = []

                st = time.time()
                new_pop = new_population()
                print("pop")
                plans = load_plans()
                #nonlocal train_envs
                #nonlocal test_envs
                train_envs = SubprocVectorEnv([lambda i=i: AgentEnv(plans[i], i) for i in range(settings.train_env)])
                test_envs = SubprocVectorEnv([lambda i=i: AgentEnv(plans[i], i) for i in range(settings.test_env)])
                print("time2", (time.time() - st)/60)
                train_envs.reset()
                test_envs.reset()
    """
    def test_fn(epoch,  env_step):
        iter = 0
        if (epoch  < settings.stop_after_epoch and epoch > 1):
            iter = 0
            epoch_number = (epoch // settings.reset_after_epoch) * settings.reset_after_epoch
            if (epoch == epoch_number + 30 or epoch == epoch_number + 40):#### or epoch == epoch_number + 25 or epoch == epoch_number + 30) :
            #if (epoch % settings.simplify_after_epoch == 0 and epoch % settings.reset_after_epoch  != 0):
                print("*****************I am  Environment Pruning", env_step)
                policy.load_state_dict(torch.load(policy_file))
                policy.eval()
                with torch.no_grad():
                    replaybuffer = VectorReplayBuffer(args.buffer_size, settings.train_env)
                    collector = Collector(policy, train_envs, replaybuffer)
                    result = collector.collect(n_episode = settings.simplify_collect_data_for_episodes, render=args.render)
                    #print("\n\ncollector.data", collector.data)
                    to_sample = len(replaybuffer) #####int(len(replaybuffer)/2)
                    #sample_batch = replaybuffer.sample(len(replaybuffer)) #len_to_Sample
                    sample_batch = replaybuffer.sample(to_sample) #len_to_Sample
                    print("sample_batch", len(sample_batch))
                    batch_data, _ = sample_batch
                    unique_obs_dict = {}
                    max_actions_dict = {}
                    for obs, action in zip(batch_data.obs.tolist(), batch_data.act.tolist()):
                        obs_key = tuple(obs)
                        if obs_key not in unique_obs_dict:
                            unique_obs_dict[obs_key] = [action]
                        else:
                            unique_obs_dict[obs_key].append(action)


                    actions_taken_atleast_once = [] 
                    Ten_irrelevant_nsps = []
                    Twenty_irrelevant_nsps = []
                    max_nsps_used = set()
                    global already_blocked
                    global already_blocked1

                    already_blocked1 = already_blocked

                    addmissble_act = []
            
                    for obs_key, actions in unique_obs_dict.items():
                        addmissble_act.extend(admissible_actions(tuple_to_state(obs_key)))
                        s1 = tuple_to_state(obs_key)
                        actions_taken_atleast_once.extend(actions)
                        Ten_irrelevant_nsps = list(set(Ten_irrelevant_nsps + [item for item, count in Counter(actions).items() if count / len(actions) < 0.005]))
                        Twenty_irrelevant_nsps = list(set(Twenty_irrelevant_nsps + [item for item, count in Counter(actions).items() if count / len(actions) < 0.01]))

                    actions_taken_atleast_once  = list(set(actions_taken_atleast_once))
                    addmissble_act = list(set(addmissble_act))
                    addmissble_act = [i for i, key in enumerate(s1.keys()) for item in addmissble_act if key == item]

                    non_used_nsps = [x for x in addmissble_act if x not in actions_taken_atleast_once]
                    if not non_used_nsps:
                        non_used_nsps = [x for x in addmissble_act if x not in Ten_irrelevant_nsps]
                        if not non_used_nsps:
                            non_used_nsps = non_used_nsps = [x for x in addmissble_act if x not in Twenty_irrelevant_nsps]            
                    if not non_used_nsps:
                        return
                    
                    print("unique_obs_dict", len(unique_obs_dict))
                    print("non_used_nsps", non_used_nsps, len(non_used_nsps))
                    #sys.exit()
                    scores_old = {}
                    for obs_key, actions in unique_obs_dict.items():
                        compute_value_result = compute_values(tuple(obs_key))
                        scores_old[obs_key] = compute_value_result

                    state_len=len(list(unique_obs_dict.keys())[0])
                    start_time = time.time()
                    graph_block = []
                    while (iter < 1):
                        print("already blocked nsps", already_blocked)
                        iter = iter + 1
                        print("iter", iter, "time:", time.time() - start_time)
                        random.seed(time.time()) ###random.shuffle(list(non_used_nsps))
                        non_used_nsps = [x for x in non_used_nsps if x not in already_blocked]
                        print("non_used_nsps", non_used_nsps)
                        new_unique_obs_dict = {}
                        scores_new = {}
                        i = 0
                        for i in non_used_nsps:
                            print("i", i)
                            scores_new = {}
                            for obs_key, actions in unique_obs_dict.items():
                                modified_obs_key_list = list(obs_key)
                                index  = i
                                modified_obs_key_list[index] = 1   # set to failed
                                modified_obs_key = tuple(modified_obs_key_list)
                                scores_new[modified_obs_key] = compute_values(tuple(modified_obs_key))
                            values_old = scores_old.values()
                            values_new = scores_new.values()
                            differences = [abs(old_value - new_value).item() for old_value, new_value in zip(values_old, values_new)]
                            print("index to block", index, "\ndifference", differences)
                            if all(diff <= settings.score_difference_threshold for diff in differences) or sum(1 for diff in differences if settings.score_difference_threshold < diff <= settings.score_difference_threshold + 1) == 1:
                                already_blocked = already_blocked + [index]
                                print("selected_nsps blocked ", [index])
                    graph_block = [x for x in already_blocked if x not in already_blocked1]
                    print("now blocking edges ", graph_block)
                    if len(graph_block) != 0:
                        load_graphs(state_len, graph_block)
                        train_envs.reset()
                        test_envs.reset()
                        graph_block = []
    """
    policy_file= str(settings.args["distribution"])+'_log_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/ppo/policy.pth'
    log_path = os.path.join(args.logdir, 'ppo')
    policy.load_state_dict(torch.load(policy_file))##,  map_location=torch.device('cpu'))
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        train_fn=train_fn,
        #test_fn=test_fn,
        #resume_from_log=args.resume,
        save_best_fn=save_best_fn,
        logger=logger,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    print("")
    pprint.pprint(result)

    print("\n===========Load best plan for testing============")
    last_pop = load_plans()
    final_pop = {}
    for i in last_pop:
        i = tuple(i)
        final_pop[i] = compute_value((i))

    best  = min(final_pop, key=final_pop.get)
    print("best defensive plan", best)
    pop_file = str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_100.txt'
    with open(pop_file, 'w') as f:
        for k in best:
           print(k, file=f)

    key = []
    with open(str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/pop_100.txt') as f:
        b_plan = f.read()
    for i in b_plan:
        if (i == str(1) or i==str(0)):
            key= key + [int(i)]


    env = AgentEnv(key)
    policy.eval()
    collector = Collector(policy, env)  # call reset()
    result = collector.collect(n_episode=5000, render=args.render)#50
    rews, lens = result["rews"], result["lens"]
    print(f"\n\nFinal reward: {rews.mean()}, length: {lens.mean()}")
    return rews.mean(), lens.mean()


def test_ppo_resume(args=get_args()):
    args.resume = True
    test_ppo(args)

if __name__ == '__main__':
    test_ppo(plan)


def load_graphs(state_len, to_block_nsps):
    for env_index in range(math.ceil(settings.train_env / 2)):
        for snapshot_index  in range(settings.num_snapshots):
            with open(f"{settings.args['distribution']}_population_{settings.args['fn']}/{settings.args['story_name']}/{settings.args['seed']}/env_graph{env_index}/graph_{snapshot_index}.txt", 'r') as f:
                b_plan = [int(line.strip()) for line in f.readlines()]####b_plan = f.read()
            for j in range(state_len):
                if j in to_block_nsps:
                    b_plan[j] = 1

            with open(f"{settings.args['distribution']}_population_{settings.args['fn']}/{settings.args['story_name']}/{settings.args['seed']}/env_graph{env_index}/graph_{snapshot_index}.txt", 'w') as f:
                for k in b_plan:
                        print(k, file=f)

def compute_values(state):
    plan = load_plans()
    env = AgentEnv(plan[0])
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

    policy = PPOPolicy(actor, critic, optim, dist)
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

    policy.load_state_dict(torch.load(policy_file))###,  map_location=torch.device('cpu'))
    #state = ec_state_tuple_to_dp_state_tuple(state)
    policy.eval()
    with torch.no_grad():
        #print(state, critic(obs=np.array([state])))
        return critic(obs=np.array([state]))


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

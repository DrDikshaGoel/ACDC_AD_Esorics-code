import gym
import os
from gym import spaces
from random import random
from setupgraph import random_setup
import setupgraph
from utility import report
import numpy as np
import torch
import sys
import random
import settings
import numpy.random
from ec_r1000 import ec_state_tuple_to_dp_state_tuple
from random import choices
from states import (
    state_to_tuple,
    tuple_to_state,
    admissible_actions,
    state_transition,)

args = settings.args
random_setup(**args)
report(setupgraph.CG)
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)



class Make_AgentEnv(gym.Env):
    def __init__(self, plan, id = None):
        self.seed_state_tuples = ec_state_tuple_to_dp_state_tuple(plan)
        initial_state_tuple = self.seed_state_tuples
        self.state_length = len(initial_state_tuple)
        self.action_space = spaces.MultiDiscrete([self.state_length], dtype=np.int32)
        self.observation_space = spaces.MultiDiscrete([3 for _ in range(self.state_length)],  dtype=np.int32)
        self.env_idx = id
        self.state = self.seed_state_tuples

        self.reset_iter = 0 # used
        attacker_states = []
        self.def_plans = load_plan_txt()
        for i in range(len(self.def_plans)):
            attacker_states.append(ec_state_tuple_to_dp_state_tuple(self.def_plans[i]))
        probability_range = (settings.base_prob, settings.upper_prob)  #PARAMETER
        for counter, attacker_state in enumerate(attacker_states):
            for i in range(settings.num_snapshots):                     #PARAMETER
                modified_attacker_state = tuple(1 if random.random() < random.uniform(*probability_range) and value == 0 else value for value in attacker_state)
                save_to = str(settings.args["distribution"])+'_population_'+str(settings.args["fn"])+'/'+str(settings.args["story_name"])+'/'+str(settings.args["seed"])+'/'+'env_graph'+str(counter)+'/'+'graph_'+str(i)+'.txt'
                #os.makedirs(os.path.dirname(save_to), exist_ok=True)
                with open(save_to, 'w') as f:
                    for k in modified_attacker_state:
                        print(k, file=f)



    """
    def reset(self, id = None):#, **kwargs
        self.plans = load_plan_txt()
        if id == 0:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[0])
        elif id == 1:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[1])
        elif id == 2:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[2])
        elif id == 3:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[3])
        elif id == 4:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[4])
        elif id == 5:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[5])
        elif id == 6:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[6])
        elif id == 7:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[7])
        elif id == 8:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[8])
        elif id == 9:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[9])
        elif id == 10:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[10])
        elif id == 11:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[11])
        elif id == 12:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[12])
        elif id == 13:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[13])
        elif id == 14:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[14])
        elif id == 15:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[15])
        elif id == 16:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[16])
        elif id == 17:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[17])
        elif id == 18:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[18])
        elif id == 19:
            self.state = ec_state_tuple_to_dp_state_tuple(self.plans[19])
        else:
            self.state = self.seed_state_tuples
        return list(self.state)

    def reset(self, id=None):
        #print("in reset function", id)
        self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []###sys.exit()###[]

        for i in range(settings.num_snapshots):
            start_step = i * settings.each_snap_step + 1
            end_step = (i + 1) * settings.each_snap_step

            if start_step <= self.current_step <= end_step:
                self.state = self.plans[i] if self.plans else self.seed_state_tuples
                break
        #print("list(self.state)", list(self.state))
        return list(self.state)


    def reset(self, id=None):
        self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
        current_graph_to_load = self.reset_iter
        self.state = self.plans[current_graph_to_load] if self.plans else self.seed_state_tuples
        self.reset_iter = self.reset_iter + 1
        if (self.reset_iter == settings.num_snapshots):
            self.reset_iter = 0
        return list(self.state)


    """

    def reset(self, id = None):#, **kwargs
        current_graph_to_load = self.reset_iter
        if id == 0:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 1:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 2:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 3:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 4:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 5:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 6:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 7:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 8:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 9:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 10:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 11:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 12:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 13:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 14:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 15:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 16:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 17:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 18:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        elif id == 19:
            self.plans = load_graphs(id) if id is not None and 0 <= id <= 19 else []
            self.state = self.plans[current_graph_to_load]
        self.reset_iter = self.reset_iter + 1
        if (self.reset_iter == settings.num_snapshots):
            self.reset_iter = 0
        return list(self.state)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        else:
            # Set a default seed value if needed
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)

    def step(self, action, id=None):
        #self.current_step += 1
        state = tuple_to_state(self.state)
        
        nsp = list(state.keys())
        status = list(state.values())
        
        if (nsp[action] not in admissible_actions(state)):
            nsp_for_action = random.choice(list(admissible_actions(state)))
            action = nsp.index(nsp_for_action)
            """
            while True:
                if (action == (self.state_length - 1)):
                    action = -1
                action = action + 1
                print(action)
                if (nsp[action] in admissible_actions(state)):
                    break
            """
        assert status[action] == 0
        nsp_for_action = nsp[action]

        future_state_tuples = []
        future_ps = []
        for future_state_tuple, future_p in state_transition(state, nsp_for_action).items():
            future_state_tuples.append(future_state_tuple)
            future_ps.append(future_p)
        total_future_ps = sum(future_ps)
        future_state_tuples.append(None)
        future_ps.append(1 - total_future_ps)
        #max_index = future_ps.index(max(future_ps))
        #new_state = future_state_tuples[max_index]
        new_state = choices(future_state_tuples, weights=future_ps)[0]
        
        if new_state is None:
            new_state = [1 for _ in range(self.state_length)]
            new_state = tuple(new_state)
            return np.asarray(new_state, dtype=np.int32), 0, True, {}
        
        admissible_nsps = admissible_actions(tuple_to_state(new_state))
        if type(admissible_nsps) is not set:
            label = int(torch.tensor([admissible_nsps], dtype=torch.float32))
            if label == int(1):
                return  np.asarray(new_state, dtype=np.int32), 1, True, {}
            elif label == int(0):
                return  np.asarray(new_state, dtype=np.int32), 0, True, {}
        else:
            self.state = new_state
            return  np.asarray(new_state,  dtype=np.int32), 0, False, {} 




def load_graphs(id):
    keys = []
    for i in range(settings.num_snapshots):
        key = []
        with open(f"{settings.args['distribution']}_population_{settings.args['fn']}/{settings.args['story_name']}/{settings.args['seed']}/env_graph{id}/graph_{i}.txt", 'r') as f:
            b_plan = f.read()
        for j in b_plan:
            if j in {'0', '1'}:
                key.append(int(j))
        keys.append(key)
    return keys

def load_plan_txt():
    keys = []
    for i in range(20):
        key = []
        with open(f"{settings.args['distribution']}_population_{settings.args['fn']}/{settings.args['story_name']}/{settings.args['seed']}/pop_{i}.txt") as f:
            b_plan = f.read()
        for j in b_plan:
            if j in {'0', '1'}:
                key.append(int(j))
        keys.append(key)
    return keys


def AgentEnv(plan, id=None):
    env = Make_AgentEnv(plan, id)
    env.seed(id)
    return env


if __name__ == "__main__":
    env = AgentEnv([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],0)
    graphs = env.reset(0)
    print(graphs)
    for i in range(20):
        obs, reward, done, _ = env.step(2)
        #if done == True:
        graphs = env.reset(0)
        print("i", i, graphs)
    sys.exit()
    done = False
    #sys.exit()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(2)##action)
        #sys.exit()
        print("obs", obs,  "rewards", reward, "done", done)

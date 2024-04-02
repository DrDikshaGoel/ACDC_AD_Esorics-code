from setupgraph import random_setup
import setupgraph
from utility import report
import timeit
import numpy.random
import torch
import settings
from pprint import pprint
from ec_r1000 import *
from ppo_policy_r1000 import test_ppo as ppo
import main_r1000 as main_from_module_one
import sys
import statistics
from utility_file_r1000 import load_plans
import time


args = settings.args
print("Args", args)
fn, seed = args["fn"], args["seed"]
random_setup(**args)
report(setupgraph.CG)
#random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)
reject_func = settings.args["story_name"]

if __name__ == '__main__':
    start_time = time.time()
    print("start_time", start_time)
    population = gen_initial_pop() #load_plans()
    best_plan = ppo(population)
    print("\n\nArgs", args, "Reject function is", settings.args["story_name"], settings.distribution, "\n\n")
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time

    # Convert seconds to hours
    elapsed_time_hours = elapsed_time_seconds / 3600
    print(f"Total Elapsed time: {elapsed_time_hours:.4f} hours")
